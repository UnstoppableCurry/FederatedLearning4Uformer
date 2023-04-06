import numpy as np
import os, sys, math
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

from model import UNet, Uformer, Uformer_Cross, Uformer_CatCross

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from skimage.metrics import mean_squared_error as mse_loss
from losses import CharbonnierLoss, L2Loss


def expand2square(timg, factor=16.0, inchannels=5):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)  # ceil(x) 函数返回一个大于或等于 x 的的最小整数。

    img = torch.zeros(1, inchannels, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

    return img, mask


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
    parser.add_argument('--input_dir', default='/home/hyzb/user_wanghongzhou/SR/datasets/test/',
                        type=str, help='Directory of validation images')  # 输入的测试数据地址
    parser.add_argument('--result_dir', default='/home/hyzb/user_wanghongzhou/SR_Test/results/denoising/',
                        type=str, help='Directory for results')  ## 输出结果的目录
    parser.add_argument('--result_dir2', default='/home/hyzb/user_wanghongzhou/SR_Test/results/input/',
                        type=str, help='Directory for results')  ## 输出结果的目录
    parser.add_argument('--weights',
                        # default='/home/hyzb/user_wanghongzhou/SR_Test/resume_True_num_comm_2000_client_16_gpuclient_4_batchsize_14_nepoch_1'
                        #         '_IID_False_Neo等差_warmup_True/test_num_comm638_E1_B14_lr0.0002_num_clients16_cf0.25.pth',
                        # model_latest
                        # parser.add_argument('--weights',
                        default='/home/hyzb/user_wanghongzhou/SR/log/Uformer_32_0701_1/models/model_epoch_500.pth',#model_latest
                        type=str, help='Path to weights')  ##导入的模型参数目录
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default='Uformer', type=str, help='arch')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
    parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
    parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
    parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
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
    parser.add_argument('--inchannels', default=5, type=int, help='Batch size for dataloader')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    utils.mkdir(args.result_dir)

    test_dataset = get_validation_data(args.input_dir)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

    model_restoration = utils.get_arch(args)
    # model_restoration = torch.nn.DataParallel(model_restoration)

    utils.load_checkpoint(model_restoration, args.weights)
    torch.save({'epoch': 0,
                'state_dict': model_restoration.state_dict(),
                'optimizer': 0
                },  "demo.pth")
    model_restoration.train()
    torch.cuda.empty_cache()
    utils.load_checkpoint(model_restoration, "demo.pth")
    print("===>Testing using weights: ", args.weights)

    model_restoration.cuda()
    model_restoration.eval()

    criterion = CharbonnierLoss().cuda()

    with torch.no_grad():
        mse_val_rgb = []
        ssim_val_rgb = []
        for ii, data_test in enumerate(tqdm(test_loader), 0):

            ## TEST THE EFFECT IN DIFFERENT SIZE
            # xsize = (456,234)
            # rgb_gt = F.interpolate(data_test[0],size=xsize).numpy().squeeze().transpose((1,2,0))
            # rgb_noisy, mask = expand2square(F.interpolate(data_test[1].cuda(),size=xsize), factor=64)

            clean_dsp_dim = data_test[4]
            clean_dsp_dim_1 = clean_dsp_dim[1][0].item()
            clean_dsp_dim_2 = clean_dsp_dim[2][0].item()
            noisy_dsp_dim = data_test[5]
            noisy_dsp_dim_1 = noisy_dsp_dim[1][0].item()
            noisy_dsp_dim_2 = noisy_dsp_dim[2][0].item()
            # print('------')
            print(clean_dsp_dim_1)
            print(clean_dsp_dim_2)
            print(noisy_dsp_dim_1)
            print(noisy_dsp_dim_2)

            # rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))
            # The factor is calculated (window_size(8) * down_scale(2^4) in this case)
            rgb_noisy, mask = expand2square(data_test[1].cuda(), factor=128, inchannels=5)
            filenames = data_test[2]

            rgb_restored = model_restoration(rgb_noisy, 1 - mask)
            # print(rgb_restored,mask.bool().size)
            rgb_restored = torch.masked_select(rgb_restored, mask.bool()).reshape(args.batch_size, 2, noisy_dsp_dim_1,
                                                                                  noisy_dsp_dim_2)
            # rgb_restored = torch.clamp(rgb_restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
            rgb_restored = rgb_restored[:, :, 0:0 + clean_dsp_dim_1, 0:0 + clean_dsp_dim_2]
            loss = criterion(rgb_restored.to('cuda:0'), data_test[0].to('cuda:0'))
            rgb_gt = data_test[0].numpy().squeeze()

            # print('rgb_restored.size')
            # print('rgb_gt.size')
            # print(rgb_restored.size)
            # print(rgb_gt.size)
            # loss = utils.batch_MSE(rgb_restored, rgb_gt, True).item()
            # MAE = utils.batch_MAE(rgb_restored, rgb_gt, True).item()
            # batch_PSNR = utils.batch_PSNR(rgb_restored, rgb_gt, True).item()
            # print(loss, MAE, batch_PSNR)
            # rgb_restored = rgb_restored.cpu().numpy().squeeze()
            # mse_val_rgb.append(mse_loss(rgb_restored, rgb_gt))
            print('mse->', loss)
            # ssim_val_rgb.append(ssim_loss(rgb_restored, rgb_gt, multichannel=True))
            ssim_val_rgb.append(loss.cpu().numpy())
            args.save_images = True
            if args.save_images:
                # utils.save_img(os.path.join(args.result_dir,filenames[0]), img_as_ubyte(rgb_restored))
                save_img_path = os.path.join(args.result_dir, 'prediction' + filenames[0])
                data = {}
                data = {'predict': rgb_restored.cpu().numpy().squeeze()}
                sio.savemat(save_img_path, data)

                data2 = {}
                save_img_path2 = os.path.join(args.result_dir2, 'target' + filenames[0])

                # data2['Prediction'] = rgb_gt
                data2 = {'target': rgb_gt}
                sio.savemat(save_img_path2, data2)
                print(save_img_path)
                print("already save as mat file !")

    # mse_val_rgb = sum(mse_val_rgb)/len(test_dataset)
    # ssim_val_rgb = sum(np.array(ssim_val_rgb))/len(test_dataset)
    # print("PSNR: %f, SSIM: %f " %(mse_val_rgb,ssim_val_rgb))
    # print("PSNR: %f " %(mse_val_rgb))
    print('mean los->', np.array(ssim_val_rgb).mean())
