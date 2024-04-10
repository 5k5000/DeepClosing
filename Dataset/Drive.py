import os
import numpy as np
from os import path as osp
import torch
# import cv2
from torch.utils.data import Dataset
from PIL import Image
from monai.transforms import *
from monai.data import CacheDataset
import pytorch_lightning as pl
import os
import numpy as np
from os import path as osp
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset
from PIL import Image
from monai.transforms import *
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch





class DriveDataset_MIM():
    def __init__(self, root_dir,is_train=True, is_test=False, is_MakePatches=True, transform_id=0):
        self.is_train = is_train
        self.is_test = is_test
        self.is_MakePatches = is_MakePatches
        self.transform_id = transform_id
        
        self.root_dir = root_dir
        self.train_dir = osp.join(self.root_dir, 'training')
        self.test_dir = osp.join(self.root_dir, 'test')
        
        self.data_dir = self.train_dir if is_train else self.test_dir
        for r, dirs, images in os.walk(osp.join(self.data_dir, 'images')):
            self.images = map(lambda x: osp.join(r, x), filter(
                lambda x: x.endswith('tif'), images))
            self.images = sorted(list(self.images))
            break

        for r, dirs, images in os.walk(osp.join(self.data_dir, 'mask')):
            self.masks = map(lambda x: osp.join(r, x), filter(
                lambda x: x.endswith('gif'), images))
            self.masks = sorted(list(self.masks))
            break

        for r, dirs, images in os.walk(osp.join(self.data_dir, '1st_manual')):
            self.seg = map(lambda x: osp.join(r, x), images)
            self.seg = sorted(list(self.seg))
            break
        self.data_dict = [{"image": image_path, "mask": mask_path, "label": label_path} for
                          (image_path, mask_path, label_path) in zip(self.images, self.masks, self.seg)]
        self.init_transforms()

    def init_transforms(self):
        if self.transform_id == 0:
            self.train_transforms = Compose([
                LoadImaged(keys=["image", "mask", "label"],
                           reader="PILReader"),
                EnsureChannelFirstd(keys=["image", "mask", "label"]),
                MaskIntensityd(
                    keys=["image", "mask", "label"], mask_key="mask"),
                ResizeWithPadOrCropd(
                    keys=["image", "mask", "label"], spatial_size=(565, 565)),
                ScaleIntensityRanged(keys=["image", "mask", "label"], a_min=0, a_max=255,
                                     b_min=0, b_max=1, clip=True),
                RandCropByPosNegLabeld(
                    keys=["image", "label"], label_key="label", spatial_size=[224, 224], pos=1, neg=1, num_samples=4
                ) if self.is_MakePatches else None,
                RandFlipd(keys=["image", "mask", "label"], prob=0.5),
                RandRotate90d(keys=["image", "mask", "label"], prob=0.5),
                AsDiscreted(keys=["mask", "label"], rounding="torchrounding"),
                EnsureTyped(keys=["image", "mask", "label"])]
                                    
            )

            self.test_transforms = Compose([
                LoadImaged(keys=["image", "mask", "label"],
                           reader="PILReader"),
                EnsureChannelFirstd(keys=["image", "mask", "label"]),
                MaskIntensityd(
                    keys=["image", "mask", "label"], mask_key="mask"),
                ScaleIntensityRanged(keys=["image", "mask", "label"], a_min=0, a_max=255,
                                     b_min=0, b_max=1, clip=True),
                RandCropByPosNegLabeld(
                    keys=["image", "label"], label_key="label", spatial_size=[224, 224], pos=1, neg=1, num_samples=4
                ) if self.is_MakePatches else None,
                AsDiscreted(keys=["mask", "label"], rounding="torchrounding"),
                EnsureTyped(keys=["image", "mask", "label"])]
            )


    def return_TrainDataset(self):
        assert self.is_train
        assert not self.is_test
        ds = CacheDataset(
            self.data_dict, transform=self.train_transforms, cache_rate=1)
        return ds

    def return_TestDataset(self):
        assert self.is_test
        assert not self.is_train
        ds = CacheDataset(
            self.data_dict, transform=self.test_transforms, cache_rate=1)
        return ds






class DriveDataset_MIM_PL(pl.LightningDataModule):
    def __init__(self, Config, data_dir='./data/DRIVE'):
        self.config = Config
        self.data_dir = data_dir
        pass

    def train_dataloader(self):
        train_ds = DriveDataset_MIM(root_dir=self.data_dir,is_train=True, is_test=False).return_TrainDataset()
        train_dataloader = DataLoader(train_ds, batch_size=self.config["batch_size"], shuffle=True,num_workers=2)
        return train_dataloader

    def val_dataloader(self):
        val_ds = DriveDataset_MIM(root_dir=self.data_dir,is_train=False, is_test=True).return_TestDataset()
        val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False,num_workers=1)
        return val_dataloader

    def test_dataloader(self):
        test_ds = DriveDataset_MIM(root_dir=self.data_dir,is_train=False, is_test=True).return_TestDataset()
        test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False,num_workers=1)
        return test_dataloader
    


# main
if __name__ == "__main__":
    config = {
        "batch_size": 2,
    }
    data_dir = "/root/DeepClosing/Data/Drive"
    PL_dataset = DriveDataset_MIM_PL(config, data_dir=data_dir)
    train_loader = PL_dataset.train_dataloader()
    
    for data in train_loader:
        print(data)
    
    
    