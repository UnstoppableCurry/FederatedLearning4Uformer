import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
from utils.loader import get_validation_data
import utils

from model import UNet,Uformer,Uformer_Cross,Uformer_CatCross

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from skimage.metrics import mean_squared_error as mse_loss


if __name__ == '__main__':




    sys.path.append('/home/wangzd/uformer/')



    parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
    parser.add_argument('--input_dir', default='/home/hyzb/user_wanghongzhou/SR/datasets/test/',
        type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='/home/hyzb/user_wanghongzhou/SR/results/denoising_input1/',
        type=str, help='Directory for results')##输出结果的目录
    parser.add_argument('--weights', default='/home/hyzb/user_wanghongzhou/SR/log/Uformer_32_0701_1/models/model_epoch_150.pth',#model_latest
        type=str, help='Path to weights')##输出权重的目录
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default='Uformer', type=str, help='arch')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
    parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
    parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
    parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
    # args for vit
    parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
    parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
    parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
    parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
    parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
    parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
    parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
    parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

    parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    utils.mkdir(args.result_dir)

    test_dataset = get_validation_data(args.input_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

    model_restoration= utils.get_arch(args)
    model_restoration = torch.nn.DataParallel(model_restoration)

    utils.load_checkpoint(model_restoration,args.weights)
    print("===>Testing using weights: ", args.weights)

    model_restoration.cuda()
    model_restoration.eval()
    with torch.no_grad():
        mse_val_rgb = []
        ssim_val_rgb = []
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            # rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))
            rgb_gt = data_test[0].numpy().squeeze()
            rgb_noisy = data_test[1].cuda()
            filenames = data_test[2]

            rgb_restored = model_restoration(rgb_noisy)# 数据进入模型
            # rgb_restored = torch.clamp(rgb_restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
            # rgb_restored = rgb_restored.cpu().numpy().squeeze().transpose((1,2,0))
            rgb_restored = rgb_restored.cpu().numpy().squeeze()
            # print(rgb_restored.shape)
            # print(rgb_gt.shape)
            mse_val_rgb.append(mse_loss(rgb_restored, rgb_gt))
            print(mse_val_rgb)
            ssim_val_rgb.append(ssim_loss(rgb_restored, rgb_gt, multichannel=True))

            if args.save_images:
                # utils.save_img(os.path.join(args.result_dir,filenames[0]), img_as_ubyte(rgb_restored))
                save_img_path = os.path.join(args.result_dir,filenames[0])
                data = {}
                data['Prediction'] = rgb_restored
                sio.savemat(save_img_path,data)
                print(save_img_path)
                print("already save as mat file !") 

    mse_val_rgb = sum(mse_val_rgb)/len(test_dataset)
    ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
    print("MSE: %f, SSIM: %f " %(mse_val_rgb,ssim_val_rgb))

