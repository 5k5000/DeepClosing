import torch
import torch.nn as nn
from monai.networks.nets import UNet
import monai
import torch.nn.functional as F
import random
import numpy as np
from monai.transforms import SpatialPad, RandSpatialCrop
from torch.optim import Adam, AdamW
import sys
import os.path
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers, seed_everything, Trainer
import yaml
from monai.losses import DiceLoss
from monai.transforms import Compose, AddChannel, Orientation, ScaleIntensityRange, LoadImage, EnsureChannelFirst, \
    EnsureType, Resize, AsDiscrete, inverse, LoadImaged, EnsureChannelFirstd, Orientationd, EnsureTyped, Invertd, KeepLargestConnectedComponent
import torch.multiprocessing
import warnings
from pytorch_lightning import LightningModule
from Simple_Point_Erosion_Module import Simple_Point_Erosion_module
from monai.inferers import sliding_window_inference
import yaml
from Dataset.Drive import DriveDataset_MIM_PL


def Masked_Shape_Reconstruction(config_path, device = torch.device("cuda:0")): 
    """
    Masked Shape Reconstruction (Training Stage): to embed the shape prior of tubular structures into an AutoEncoder, via masked image modeling.
    """
    
    f = open(config_path, 'r', encoding='utf-8')
    cont = f.read()
    config = yaml.load(cont, Loader=yaml.FullLoader)
    
    save_dir = "./Logs"
    exp_save_dir = os.path.join(save_dir, config["Name"])
    os.makedirs(exp_save_dir, exist_ok=True)
    logger = WandbLogger(project="DeepClosing",name=config["Name"], save_dir=exp_save_dir)
    
    
    model = DeepClosing(config, device)

    dataset_name = config["Dataset"]["dataset_name"]
    if dataset_name == "DRIVE":
        dataset = DriveDataset_MIM_PL(config["Dataset"], data_dir="/root/DeepClosing-private/Data/Drive")
    else:   
        raise NotImplementedError
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    check_point = ModelCheckpoint(save_last=True)

    
    trainer = Trainer(
        max_epochs=config["Epoch"],
        log_every_n_steps=1,
        accelerator="gpu",
        devices=config["Devices"],
        logger=logger,
        callbacks=[lr_monitor, check_point],
        check_val_every_n_epoch=10,
    )
    trainer.fit(model, datamodule=dataset)


class DeepClosing(LightningModule):

    def __init__(self, config, device = torch.device("cuda:0")):
        super(DeepClosing, self).__init__()


        """reconstruction_mode: full or masked region only
          full: compute loss on the whole image
          masked region only: compute loss on the masked region only """
        self.reconstruction_mode = config.get("reconstruction_mode", "full")
        self.residue_union_mode = config.get("residue_union_mode", "GT only")
        assert self.reconstruction_mode in ["full","masked region only"]
        assert self.residue_union_mode  in ["GT only","pred only","both"]

        self.config = config
        NetConfig = config["Model"]
        DatasetConfig = config["Dataset"]
        SPE_Config = NetConfig["SPE"]
        
        # initialize the AutoEncoder
        self.net = Unet_MIM(**NetConfig["args"])
        self.input_type = NetConfig.get("input_type", "image")  # input_type: image or label

        self.test_sw_window_size = self.config.get("ts_sw_size", (224, 224))      # val/test shift window size
        self.test_sw_batch_size = self.config.get("ts_sw_batchsize", 4)          # val/test shift window batchsize
        self.test_sw_overlap = self.config.get("ts_sw_overlap", 0.5)        # val/test shift window overlap
        

        # topoloss weight (alpha)
        self.topoloss_weight = NetConfig.get("topo_weight", 1.0e-1 )


        # Simple Point Erosion Module   
        target_H_W = SPE_Config.get("target_HW", (224,224))
        self.SPE = Simple_Point_Erosion_module(target_H_W=target_H_W)
        self.tubescaler_range = (0,2)
        self.if_Recurrent_TubeScaler = True
        
        self.to(device)



    def forward(self, x):  
        # x : B,1,H,W, binary, non-masked shape
        
        # perform Recurrent TubeScaler here
        if self.if_Recurrent_TubeScaler and self.training:    
            # random sample a k from the range of self.tubescaler_range
            k_r = random.randint(self.tubescaler_range[0],self.tubescaler_range[1])
            if k_r ==0:
                print(k_r)
            x,_ = self.SPE.RSPE(T=x,M_T=x,max_K=k_r)
            self.log("train/k_r",k_r,prog_bar=True,on_step=True)
        

        
        result_dict = self.net(x)
        pred = result_dict["pred"]
        masked_img = result_dict["masked_img"]
        residue = pred - masked_img
        gt = result_dict["origin_imgs"]
        True_residue = gt - masked_img


        pred_clone = pred.clone().detach()
        # threshold with 0.5
        pred_clone[pred_clone>0.5] = 1
        simple_points_eroded_results,_ = self.SPE.RSPE(T=gt,M_T=True_residue,max_K=10)
        critical_points = simple_points_eroded_results - masked_img
        
        result_dict["critical_points"] = critical_points
    
        return result_dict

    def inference(self, data=None, is_infer_sliding_window=None, verbose=False,sw_roi_size=None):
        """

        :param data:  the input data,
        :param is_infer_sliding_window:
        :param verbose:
        :return:
        """
        self.eval()

        if is_infer_sliding_window:
            sw_batch_size = 4
            roi_size = sw_roi_size if sw_roi_size !=None else self.roi_size
            if verbose:
                print("using sliding window inference with roi_size:{}".format(roi_size))
            val_outputs = sliding_window_inference(
                data, roi_size = roi_size, sw_batch_size= sw_batch_size, predictor= self.net.net)
            val_outputs = val_outputs
            # print(val_outputs)
        else:
            if verbose:
                print("not using sliding window inference".format(self.roi_size))
            val_outputs = self.net.net(data)
        return val_outputs

    def training_step(self, batch_data, batch_idx):
        input = batch_data[self.input_type]
        result_dict = self(input)
        # additional topo loss
        topo_loss = F.mse_loss(result_dict["pred"]*result_dict["critical_points"], result_dict["origin_imgs"]*result_dict["critical_points"])
        result_dict["loss"] = result_dict["loss"] + topo_loss * self.topoloss_weight
        result_dict["topo_ls"] = topo_loss.detach()
        self.log_something(result_dict,"train")
        return result_dict

    def validation_step(self, batch_data, batch_idx):
        input = batch_data[self.input_type]

        result_dict = self(input)
        self.log_something(result_dict,"val")
        return result_dict
    

    def test_step(self, batch_data, batch_idx):
        input = batch_data[self.input_type]
        result_dict = self(input)
        result_dict["input"] = input
        return result_dict


    def configure_optimizers(self):
        """
        configure optimizer and lr_scheduler
        :return:
        """
        # 1.configure optimizer
        OptimizerConfig = self.config["Model"]["optimizer"]
        optimizer_name = OptimizerConfig["name"].lower()
        if optimizer_name == "adam":
            optimizer = Adam(self.parameters(), **OptimizerConfig["args"])
        elif optimizer_name == "adamw":
            optimizer = AdamW(self.parameters(), **OptimizerConfig["args"])

        # 2.configure lr_scheduler
        Lr_schedulerConfig = self.config["Model"]["lr_scheduler"]
        lr_scheduler_name = Lr_schedulerConfig["name"].lower()
        if lr_scheduler_name == "steplr":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, **Lr_schedulerConfig["args"])
        elif lr_scheduler_name == "cosine annealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  
                optimizer, **Lr_schedulerConfig["args"])

        return [optimizer], [lr_scheduler]
    
    
        

    def log_something(self, result_dict, stage):
        """
        log something
        1. mask_ratio
        2. patch_size
        3. loss  (together with topoloss)
        :param result_dict: something stored in this dict, which should include patchsize, mask_ratio,
        patch_size: the patchsize used for generating the mask,  could be an instance of float, which means the normal cubic mask type. 
        mask_ratio: the ratio of mask
        loss: train or val loss
        """
        assert stage == "train" or stage == "val", "the stage must be either train or val"
        self.log("{}/loss".format(stage),
                 result_dict["loss"], prog_bar=True, on_epoch=True, on_step=True)
    


        if "topo_ls" in result_dict :  
            self.log("{}/topoloss".format(stage),result_dict["topo_ls"], prog_bar=True, on_epoch=True, on_step=True)
        self.log("{}/mask_ratio".format(stage),
                 result_dict["mask_ratio"], on_step=True, on_epoch=False)
        patch_size = result_dict["patch_size"]

        if type(patch_size) == float:
            self.log("{}/patch_size".format(stage),
                     patch_size, on_step=True, on_epoch=False)
        elif type(patch_size) == tuple or type(patch_size) == list:
            self.log("{}/patch_size_x".format(stage),
                     patch_size[0], on_step=True, on_epoch=False)
            self.log("{}/patch_size_y".format(stage),
                     patch_size[1], on_step=True, on_epoch=False)
            if len(patch_size) == 3:
                self.log("{}/patch_size_z".format(stage),
                         patch_size[2], on_step=True, on_epoch=False)
        else:
            raise TypeError

    def Simple_Component_Erosion(self, T, M_T):
        """
        Simple Component Erosion
        T: binary structure, shape: 1,1,H,W
        M_T: the mask of specified region, binary,  shape 1,1,H,W
        """
        import skimage
        # compute connected components of M_T
        
        M_T_numpy = M_T.squeeze().detach().cpu().numpy().astype(np.uint8)
        Lcc = skimage.measure.label(M_T_numpy, connectivity=2)
        
        out, _ = self.SPE.RSPE(T=T, M_T=M_T, max_K=np.inf)
        remained = Lcc * out.squeeze().detach().cpu().numpy()
        remained_cc_label = np.unique(remained)
        
        result_array = (T - M_T).detach().squeeze().cpu().numpy()   # H,W
        
        # Retrieve
        for label in remained_cc_label:
            if label == 0:
                continue
            result_array = result_array + (Lcc == label) * 1
        
        result_array = torch.from_numpy(result_array).to(T.device)
        return result_array



    @torch.no_grad()
    def DeepDilation(self,T,is_infer_sliding_window=True, sw_roi_size=(224,224),sw_batch_size=4,verbose=False):
        """ DeepDilation
        using pre-trained AutoEncoder to indicate regions of disconnection
        T: presegmented image,  binary, shape: shape, B,1,H,W
        is_infer_sliding_window: whether to use sliding window inference
        verbose: whether to print the information
        sw_roi_size: the size of sliding window
        """
        self.eval()
        
        if is_infer_sliding_window:
            sw_batch_size = 4
            roi_size = sw_roi_size if sw_roi_size !=None else self.roi_size
            if verbose:
                print("using sliding window inference with roi_size:{}".format(roi_size))
            T_dd = sliding_window_inference(
                T, roi_size = roi_size, sw_batch_size= sw_batch_size, predictor= self.net.net)
        else:
            if verbose:
                print("not using sliding window inference".format(self.roi_size))
            T_dd= self.net.net(T)
        return T_dd

    @torch.no_grad()
    def DeepClosing(self,T,is_infer_sliding_window=True, sw_roi_size=(224,224),sw_batch_size=4,verbose=False):
        if type(T) == torch.Tensor:
            # assert T shape is B,1,H,W
            assert len(T.shape) == 4, "T shape must be B,1,H,W"
            # assert binay 
            assert T.unique() == torch.tensor([0,1]), "T must be binary"
        elif type(T) == str: # the path of the image
            transform = Compose(
                [
                    LoadImage(image_only=True),
                    AddChannel(),
                    ScaleIntensityRange(a_min=0, a_max=255,
                                        b_min=0, b_max=1, clip=True),
                    Orientation(axcodes="RAS"),
                    EnsureType(),
                ]
                )
            T = transform(T)[0] 
            if len(T.shape) ==4:  # RGB image, the last channel is 3
                # to check all the channels are the same
                assert (T[:,:,:,0] == T[:,:,:,1]).all()
                assert (T[:,:,:,0] == T[:,:,:,2]).all()
                T = T[:,:,:,0]
            
            T = T.unsqueeze(0).to(self.device)
        T_dd = self.DeepDilation(T=T,is_infer_sliding_window=True, sw_roi_size=(224,224),sw_batch_size=4,verbose=False)
        T_dd = transform.inverse(T_dd.squeeze().cpu())
        T_dd = AsDiscrete(threshold=0.2)(T_dd).int()
        
        M_T = T_dd.squeeze() - T.squeeze().cpu()
        M_T[M_T<0] =0
        M_T = M_T.squeeze()
        T_dc = self.Simple_Component_Erosion(T=T_dd.unsqueeze(0).unsqueeze(0),M_T=M_T.unsqueeze(0).unsqueeze(0))    
        
        return {"T_dd":T_dd,"T_dc":T_dc,"M_T":M_T,"T":T}
    
        








class Unet_MIM(nn.Module):
    "Unet with Masked Image Modeling"

    def __init__(self, in_chans=3, mask_ratio=0.75, loss_type="L2", image_size=224, patch_size=([2,8],[2,8]),
                 is_random_rotate=False, dropout_ratio=0.0, network_setting="default", final_act=None):
        """

        :param in_chans:  the input channels number of Unet
        :param mask_ratio:  the ratio of patches to be blanked
        :param loss_type:  the loss
        :param image_size: the input size of image （not in use)
        :param patch_size:  the size of mask
        :param is_random_rotate:    whether to rotate mask when training
        :param dropout_ratio:     the dropout ratio when training
        :param network_setting:
        :param final_act: the final activation in the end of network forward, default None
        :param mask_generator_version: the version of mask generator, default v1, supporting v1 and v2
        """
        super(Unet_MIM, self).__init__()
        self.patch_size = patch_size
        self.image_channel = in_chans  # add by Fivethousand
        self.mask_ratio = mask_ratio
        self.loss_type = loss_type
        self.is_random_rotate = is_random_rotate
        self.dropout_ratio = dropout_ratio
        self.network_setting = network_setting
        self.final_act = final_act
        self.net = self.get_net()

    def get_net(self):
        if self.network_setting == "default":  # default unet setting
            net = monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=self.image_channel,
                out_channels=self.image_channel,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                dropout=self.dropout_ratio)
        else:
            raise NotImplementedError

        # final activation
        if self.final_act is None:
            pass
        elif self.final_act == "sigmoid":
            net = nn.Sequential(net, nn.Sigmoid())
        else:
            raise NotImplementedError

        return net

    def get_mask_ratio_and_patch_size(self):
        if type(self.patch_size) == tuple and type(self.patch_size[0]) == list and type(
            self.patch_size[1]) == list:
            range0 = list(range(*self.patch_size[0]))
            range1 = list(range(*self.patch_size[1]))
            patch_h = random.choice(range0)
            patch_w = random.choice(range1)
        
        elif type(self.patch_size) == int:
            patch_h = self.patch_size
            patch_w = self.patch_size
        else:
            raise TypeError
        

        """mask ratio"""

        if isinstance(self.mask_ratio, float):
            mask_ratio = self.mask_ratio
        elif isinstance(self.mask_ratio, list):
            mask_ratio = random.choice(self.mask_ratio)
        elif isinstance(self.mask_ratio, tuple):
            mask_ratio = np.random.uniform(*self.mask_ratio)
        else:
            raise NotImplementedError
        return mask_ratio, (patch_h, patch_w)

    def forward(self, imgs):
        mask_ratio, patch_size = self.get_mask_ratio_and_patch_size()
        masked_img, mask = self.random_mask_FT(imgs, mask_ratio, patch_size)

        pred = self.net(masked_img)
        loss = self.forward_loss(pred, imgs, mask)  


        return {"loss": loss, "pred": pred, "mask": mask, "masked_img": masked_img, "mask_ratio": mask_ratio, "patch_size": patch_size,"origin_imgs":imgs}

    def forward_loss(self, pred, target, mask):
        if self.loss_type == "L2":
            # loss = (pred - target) ** 2
            loss = F.mse_loss(input=pred, target=target)
        else:
            raise NotImplementedError
        return loss

    def random_mask_FT(self, imgs, mask_ratio, patch_size):
        mask = mask_generator2D_v2(input_tensor=imgs, mask_ratio=mask_ratio,
                                    patch_sizes=patch_size, if_random_affine=self.is_random_rotate)
        mask_multiply = torch.abs(1 - mask)  # Now 1->0, 0->1
        masked_img = imgs * mask_multiply
        return masked_img, mask

    def patchify(self, imgs, patch_size):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.image_channel, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(
            shape=(imgs.shape[0], h * w, p ** 2 * self.image_channel))
        return x

    def unpatchify(self, x, patch_size):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)  # (N, image_channel, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.image_channel))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.image_channel, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is removed
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def test_display(self, output_results):
        # select first sample for display
        input = output_results["input"].detach().cpu()
        input = torch.einsum('nchw->nhwc', input)
        pred = output_results["pred"]
        mask = output_results["mask"]
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()
        channel = pred.shape[-1]
        # visualize the mask
        mask = mask.detach()
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        return input, mask, pred


def mask_generator2D_v2(input_tensor, mask_ratio, patch_sizes, if_random_affine=False):
    """
    generate 2D mask
    :param input_tensor:  input_tensor [B, C, H, W]
    :param mask_ratio: the ratio of patches to be masked
    :param patch_size: the patchsize of cubic mask [h,w]
    :param is_random_affine: whether to apply random rotation onto the generated 2D mask
    :return: mask with shape [B,1,H,W], 0 is to keep, 1 is to remove
    """
    assert type(patch_sizes) == tuple and len(patch_sizes) == 2
    p_h, p_w = patch_sizes
    """
    pad original H,W to new sizes
    """
    B, C, H, W = input_tensor.shape
    new_H = int(p_h * np.ceil(H/p_h))
    new_W = int(p_w * np.ceil(W/p_w))
    assert new_H >= H and new_W >= W
    mask = torch.ones(1, 1, new_H, new_W)

    h = new_H // p_h
    w = new_W // p_w
    mask = mask.reshape(shape=(1, 1, h, p_h, w, p_w))
    mask = torch.einsum('bchqwj->bhwqjc', mask)
    mask = mask.reshape(shape=(1, h * w, p_h * p_w * 1))
    shuffled_index = torch.randperm(h * w)

    # how many blocks should be kept
    len_keep = int(len(shuffled_index) * (1 - mask_ratio))
    shuffled_index = shuffled_index[:len_keep]
    mask[:, shuffled_index, :] = 0
    mask = mask.reshape(shape=(1, h, w, p_h, p_w, 1))
    mask = torch.einsum('bhwpqc->bchpwq', mask)
    mask = mask.reshape(shape=(1, 1, h * p_h, w * p_w))
    if if_random_affine:  # apply random affine

        # result, _ = monai.transforms.Affine(rotate_params=[np.pi / 4, np.pi / 4, np.pi / 4], mode="nearest")(mask[0])
        result = monai.transforms.RandAffine(rotate_range=[(-np.pi, np.pi), (-np.pi, np.pi)],
                                             prob=0.8, mode="nearest")(mask[0])
        mask[0] = torch.round(result)

    new_mask = RandSpatialCrop(
        roi_size=(H, W), random_size=False)(mask[0]).unsqueeze(0)
    mask = new_mask.repeat(B, 1, 1, 1)

    mask = mask.to(input_tensor.device)
    return mask



if __name__ == '__main__':
    Masked_Shape_Reconstruction(config_path="./Config/DRIVE.yaml")

    
