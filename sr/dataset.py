import numpy as np
import os,sys,math
from skimage.util import shape
from torch.utils.data import Dataset
import torch
from utils import is_png_file, is_mat_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random

import scipy.io
from skimage.measure import block_reduce

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 




##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        
        # gt_dir = 'groundtruth'
        # input_dir = 'input'
        gt_dir = 'vmodel_train'
        input_dir = 'georec_train'
        self.dataname = 'input'
        self.truthname = 'label'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        
        # self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        # self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)       for x in noisy_files if is_png_file(x)]
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_mat_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)       for x in noisy_files if is_mat_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

        #采样比例
        self.data_dsp_blk = (1,1,1)
        self.label_dsp_blk = (1,1,1)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        # clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        data_V = scipy.io.loadmat(self.clean_filenames[tar_index])
        clean = torch.from_numpy(np.float32(data_V[str(self.truthname)]))
        data_R = scipy.io.loadmat(self.noisy_filenames[tar_index])
        noisy = torch.from_numpy(np.float32(data_R[str(self.dataname)]))
        
        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)#转换维度
        # print(clean.size)


        noisy   = block_reduce(noisy,block_size=self.data_dsp_blk,func=np.max)
        clean   = block_reduce(clean,block_size=self.label_dsp_blk,func=np.max)
        # clean = clean/1000

        # ones = np.ones([256,256])
        # ones[:201, :256] = clean[:201, :256]
        # clean = ones
        # noisy = noisy[:, :256, :256]

        # print('noisy.shape:')
        # print(noisy.shape)
        # print('clean.shape:')
        # print(clean.shape)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean_dsp_dim = clean.shape
        noisy_dsp_dim = noisy.shape

        # #Crop Input and Target
        # ps = self.img_options['patch_size']
        # H = clean.shape[1]
        # W = clean.shape[2]
        # # r = np.random.randint(0, H - ps) if not H-ps else 0
        # # c = np.random.randint(0, W - ps) if not H-ps else 0
        # if H-ps==0:
        #     r=0
        #     c=0
        # else:
        #     r = np.random.randint(0, H - ps)
        #     c = np.random.randint(0, W - ps)
        # clean = clean[:, r:r + ps, c:c + ps]
        # noisy = noisy[:, r:r + ps, c:c + ps]
        # 
        # apply_trans = transforms_aug[random.getrandbits(3)]

        # clean = getattr(augment, apply_trans)(clean)
        # noisy = getattr(augment, apply_trans)(noisy) 
        # 
        return clean, noisy, clean_filename, noisy_filename, clean_dsp_dim, noisy_dsp_dim     

        #return clean, noisy, clean_filename, noisy_filename

##################################################################################################

class DataLoaderTrain_Gaussian(Dataset):
    def __init__(self, rgb_dir, noiselevel=5, img_options=None, target_transform=None):
        super(DataLoaderTrain_Gaussian, self).__init__()

        self.target_transform = target_transform
        #pdb.set_trace()
        clean_files = sorted(os.listdir(rgb_dir))
        #noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        #clean_files = clean_files[0:83000]
        #noisy_files = noisy_files[0:83000]
        self.clean_filenames = [os.path.join(rgb_dir, x) for x in clean_files if is_png_file(x)]
        #self.noisy_filenames = [os.path.join(rgb_dir, 'input', x)       for x in noisy_files if is_png_file(x)]
        self.noiselevel = noiselevel
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        print(self.tar_size)
    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        #print(self.clean_filenames[tar_index])
        clean = np.float32(load_img(self.clean_filenames[tar_index]))
        #noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        # noiselevel = random.randint(5,20)
        noisy = clean + np.float32(np.random.normal(0, self.noiselevel, np.array(clean).shape)/255.)
        noisy = np.clip(noisy,0.,1.)
        
        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.clean_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)

        return clean, noisy, clean_filename, noisy_filename
##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'vmodel_train'
        input_dir = 'georec_train'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_mat_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_mat_file(x)]
        

        self.tar_size = len(self.clean_filenames) 

        self.dataname = 'input'
        self.truthname = 'label'
        self.data_dsp_blk = (1,1,1)
        self.label_dsp_blk = (1,1,1)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        #clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        #noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        data_V = scipy.io.loadmat(self.clean_filenames[tar_index])
        clean = torch.from_numpy(np.float32(data_V[str(self.truthname)]))
        data_R = scipy.io.loadmat(self.noisy_filenames[tar_index])
        noisy = torch.from_numpy(np.float32(data_R[str(self.dataname)]))



        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        noisy   = block_reduce(noisy,block_size=self.data_dsp_blk,func=np.max)
        clean   = block_reduce(clean,block_size=self.label_dsp_blk,func=np.max)
        # clean = clean/1000

        clean_dsp_dim = clean.shape
        noisy_dsp_dim = noisy.shape
        # print(clean_dsp_dim)
        # print(noisy_dsp_dim)

        # zeros = np.zeros([256,256])
        # zeros[:201, :256] = clean[:201, :256]
        # clean = zeros
        # noisy = noisy[:, :256, :256]
        # ones = np.ones([256,256])
        # ones[:201, :256] = clean[:201, :256]
        # clean = ones
        # noisy = noisy[:, :256, :256]

        # print('test noisy.shape:')
        # print(noisy.shape)
        # print('test clean.shape:')
        # print(clean.shape)


        # return clean, noisy, clean_filename, noisy_filename
        return clean, noisy, clean_filename, noisy_filename, clean_dsp_dim, noisy_dsp_dim     

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))


        self.noisy_filenames = [os.path.join(rgb_dir, 'input', x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.noisy_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        noisy = noisy.permute(2,0,1)

        return noisy, noisy_filename


##################################################################################################

class DataLoaderTestSR(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTestSR, self).__init__()

        self.target_transform = target_transform

        LR_files = sorted(os.listdir(os.path.join(rgb_dir)))


        self.LR_filenames = [os.path.join(rgb_dir, x) for x in LR_files if is_png_file(x)]
        

        self.tar_size = len(self.LR_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        LR = torch.from_numpy(np.float32(load_img(self.LR_filenames[tar_index])))
                
        LR_filename = os.path.split(self.LR_filenames[tar_index])[-1]

        LR = LR.permute(2,0,1)

        return LR, LR_filename
