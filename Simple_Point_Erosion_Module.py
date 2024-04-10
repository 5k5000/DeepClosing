import os
import skimage
import numpy as np
import torch
import torch.nn as nn

def whether_center_point_is_simple_point(patch):   #( 8, 4)
    """
    The criterion is derived from the following paper:
    
    [1]  X. Hu, “Structure-aware image segmentation with homotopy warping,” Advances in Neural Information Processing Systems, vol. 35, pp.
    24046–24059, 2022.
    [2]  T. Y. Kong and A. Rosenfeld, “Digital topology: Introduction and
        survey,” Computer Vision, Graphics, and Image Processing, vol. 48,
        no. 3, pp. 357–393, 1989.
        
    """

    """ Input:
            patch: a 3x3 binary patch
        Return:
            True: if the center point is a simple point
            False: if the center point is not a simple point
    """
    patch_copy1 = patch.copy()      # used for condition 1
    patch_copy2 = patch.copy()      # used for condition 2

    # （8，4） connectivity
    # count foreground 8 neighbors connected components
    count, num = skimage.measure.label(
        patch_copy1, connectivity=2, return_num=True)   # 8-adjacency connectivity
    # count background 4 neighbors connected components
    patch_copy1 = 1-patch_copy1
    count2, num2 = skimage.measure.label(
        patch_copy1, connectivity=1, return_num=True)   # 4-adjacency connectivity
    
    # flip the center point
    patch_copy2[1, 1] = 1-patch_copy2[1, 1]
    # count foreground 8 neighbors connected components
    count3, num3 = skimage.measure.label(
        patch_copy2, connectivity=2, return_num=True)   # 8-adjacency connectivity
    # count background 4 neighbors connected components
    patch_copy2 = 1-patch_copy2
    count4, num4 = skimage.measure.label(
        patch_copy2, connectivity=1, return_num=True)   # 4-adjacency connectivity
    
    if num == num3 and num2 == num4:        # whether the center point is a simple point
        return True
    else:
        return False
    
def patchify(binary_image):
    B,C,H,W = binary_image.shape
    image_patch =nn.Unfold(kernel_size=(3,3),padding=1,stride=1)(binary_image.float())      # divide the image into patches via sliding window
    # image patch, shape [B,9,H*W]
    
    # transfer to [B*H*W, 9]
    image_patch = image_patch.permute(0,2,1)    # B, H*W, 9
    image_patch = image_patch.reshape(B*H*W,9)  # B*H*W, 9

    return image_patch


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self,train_ratio=1,type='train'):
        
        self.total_num = 2**9
        self.train_ratio = train_ratio
        self.train_num = int(self.total_num*train_ratio)
        self.test_num = self.total_num - self.train_num
        self.train_dict_list,self.test_dict_list = self.create_train_dict_list_and_test_dict_list()
        self.type = type
        self.simple_list = []

        if self.type == 'train':
            self.dict_list = self.train_dict_list
        elif self.type == 'test':
            self.dict_list = self.test_dict_list
        else:
            raise ValueError('type must be train or test')
        
    def get_cc_num_and_simple_point_label(self,patch):
        _, num1 = skimage.measure.label(
            patch, connectivity=1, return_num=True)   # 4-adjacency connectivity
        _, num2 = skimage.measure.label(
            patch, connectivity=2, return_num=True)   # 8-adjacency connectivity
        simple_point_label =  None #whether_center_point_is_simple_point(patch)
        return  num1, num2, simple_point_label
    

    # a function to exhaust all the possible patches
    def create_train_dict_list_and_test_dict_list(self):
        total_list = []
        train_dict_list = []
        test_dict_list = []
        simple_count = 0
        # initialize a pytorch lookuptable
        simple_count = 0
        self.simple_list = []
        for i in range(2**9):

            # convert i to binary string and ensure the length of the string is 9
            binary_string = bin(i)[2:].zfill(9)
            # convert the binary string to a list of 0s and 1s
            binary_list = [int(x) for x in binary_string]
            # convert the list of 0s and 1s to a 3x3 array
            patch = np.array(binary_list).reshape(3,3)     
            criterion = whether_center_point_is_simple_point(patch)

            if criterion:
                simple_count = simple_count + 1
                self.simple_list.append(patch)

            total_list.append({'patch':patch,'num1':0,'num2':0,'simple_point_label':criterion})
        # chech no patch are same
        for i in range(len(total_list)):
            for j in range(i+1,len(total_list)):
                if np.array_equal(total_list[i]['patch'],total_list[j]['patch']):
                    raise ValueError('two patches are same')

        
        # shuffle the total_list
        np.random.shuffle(total_list)
        train_dict_list = total_list[:self.train_num]
        test_dict_list = total_list[self.train_num:]

        # numpy save self.simple_list 
        np.save('simple_list.npy',self.simple_list)
        return train_dict_list,test_dict_list

    def __len__(self):
        return len(self.dict_list)
    
    def __getitem__(self, idx):
        return self.dict_list[idx]
    
    def get_train_num(self):
        return self.train_num
    
    def get_test_num(self):
        return self.test_num
    
    
class Simple_Point_Erosion_module():
    def __init__(self,target_H_W=(960,960),device = torch.device("cuda:0")) -> None:
        self.simple_point_dataset = PatchDataset()
        # self.train_num = self.patch_dataset.get_train_num()
        self.H,self.W = target_H_W
        self.device = device
        self.Construct_a_LookupTable_of_SimplePoints()
        self.build_order_masks()

    def Construct_a_LookupTable_of_SimplePoints(self): # build a simple point lookup table
        # creata a
        self.SPLT = nn.Embedding(2**9,1)
        # initialize all the value to 0
        
        self.SPLT.weight.data.fill_(0)
        for data_dict in self.simple_point_dataset:
            patch = data_dict['patch']
            num1 = data_dict['num1']
            num2 = data_dict['num2']
            simple_point_label = data_dict['simple_point_label']
            if simple_point_label:
                # convert patch to a 9 bit number
                patch = patch.flatten()
                patch = patch.dot(2**np.arange(8, -1, -1))
                self.SPLT.weight.data[patch] = 1.0
        self.SPLT = self.SPLT.to(self.device)
        self.lookup_table = self.SPLT
    
    def build_order_masks(self):  # this method could be replaced with a more efficient way
        self.order_mask_list = []
        m1 = torch.zeros(self.H,self.W)
        m2 = torch.zeros(self.H,self.W)
        m3 = torch.zeros(self.H,self.W)
        m4 = torch.zeros(self.H,self.W)
        for i in range(self.H):
            for j in range(self.W):
                i_mod2 = i%2
                j_mod2 = j%2
                if i_mod2 == 0 and j_mod2 == 0:
                    m1[i,j] = 1
                elif i_mod2 == 0 and j_mod2 == 1:
                    m2[i,j] = 1
                elif i_mod2 == 1 and j_mod2 == 0:
                    m3[i,j] = 1
                elif i_mod2 == 1 and j_mod2 == 1:
                    m4[i,j] = 1
        self.order_mask_list.append(m1.unsqueeze(0).unsqueeze(0).to(self.device))
        self.order_mask_list.append(m2.unsqueeze(0).unsqueeze(0).to(self.device))
        self.order_mask_list.append(m3.unsqueeze(0).unsqueeze(0).to(self.device))
        self.order_mask_list.append(m4.unsqueeze(0).unsqueeze(0).to(self.device))
           
           
    def lookup_in_SPLT(self, x):
        patchfied_image = patchify(x)
        # shape of patchfied_image: B*H*W, 9
        

        patchfied_image = patchfied_image.matmul(2**torch.arange(8, -1, -1, device=patchfied_image.device).float()).long()
        embeddings = self.lookup_table(patchfied_image)
        
        return embeddings
    
             
    def PSPC(self,T,M_T,i):
        
        
        
        # T shape: B, 1, H, W
        assert T.shape == M_T.shape
        if T.shape[-2:] == (self.H,self.W):
            pass
        else:
            self.H,self.W = T.shape[-2:]
            self.build_order_masks()
            print("rebuild order masks to fit the shape of T, new shape: ",T.shape)
            
        picked_order_mask = self.order_mask_list[i]
        
        B,_,H,W = T.shape
        
        global_simple_point_label = self.lookup_in_SPLT(T)
        # shape of global_simple_point_label: B*H*W, 1
        global_simple_point_label = global_simple_point_label.reshape(B,1,H,W)
        global_simple_point_label = global_simple_point_label * picked_order_mask
        
        return global_simple_point_label
    
    @torch.no_grad()
    def RSPE(self,T,M_T,max_K=1000):
        T = T.to(self.device)
        M_T = M_T.to(self.device)
        k=0
        intermidiate_T = []
        last_sum_of_T = T.sum()
        while True:
            intermidiate_T.append(T)
            # print(k)
            k+=1
            S1 = self.PSPC(T,M_T,0)
            T = T - S1 * T * M_T
            S2 = self.PSPC(T,M_T,1)
            T = T - S2 * T * M_T
            S3 = self.PSPC(T,M_T,2)
            T = T - S3 * T * M_T
            S4 = self.PSPC(T,M_T,3)
            T = T - S4 * T * M_T
            if k>=max_K:            # termination condition 1: the number of iterations reaches the maximum
                break
            sum_of_T = T.sum()
            if sum_of_T == last_sum_of_T:       # termination condition 2: the sum of T does not change
                break
            last_sum_of_T = sum_of_T
            
        return T,intermidiate_T
    
    
    loop_forward = RSPE
    
        

    
    
    
    

    
