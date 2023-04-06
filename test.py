
import numpy as np
import os, sys
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
# from ptflops import get_model_complexity_info
import scipy
import scipy.io as sio
from utils.loader import get_validation_data
import utils
import math
from losses import CharbonnierLoss, L2Loss

from model import UNet, Uformer, Uformer_Cross, Uformer_CatCross


def expand2square(timg, factor=16.0, batch_size=2, inchannels=29):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)  # ceil(x) 函数返回一个大于或等于 x 的的最小整数。

    img = torch.zeros(batch_size, inchannels, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

    return img, mask


if __name__ == '__main__':
    sys.path.append('/home/wangzd/uformer/')

    parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
    parser.add_argument('--input_dir', default='/home/hyzb/user_wanghongzhou/SR/datasets/test/',

                        # parser.add_argument('--input_dir', default='/home/hyzb/user_wanghongzhou/SR_test/datasets/train/',
                        type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='./results/denoising/',
                        type=str, help='Directory for results')  ##输出结果的目录
    parser.add_argument('--weights',
                        default='/home/hyzb/user_wanghongzhou/SR_Test/right_resume_False_num_comm_2000_client_16_gpuclient_4_batchsize_14_nepoch_1_IID_False_Neo等差_warmup_True/test_num_comm986_E1_B14_lr0.0002_num_clients16_cf0.25.pth',
                        # model_latest
                        type=str, help='Path to weights')  ##输出权重的目录



    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default='Uformer', type=str, help='arch')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
    parser.add_argument('--save_images', default=True, action='store_true',
                        help='Save denoised images in result directory')
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
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    utils.mkdir(args.result_dir)

    test_dataset = get_validation_data(args.input_dir)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

    # train_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [int(0.8 * len(test_dataset)),
    #                                                                           int(0.2 * len(test_dataset))])

    # train_dataset=val_dataset#测试用
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=0, pin_memory=True, drop_last=False)

    # val_dataset = get_validation_data(opt.val_dir)  ##获取验证集
    # val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
    #                         num_workers=opt.eval_workers, pin_memory=False, drop_last=False)
    val_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False, drop_last=True)
    model_restoration = utils.get_arch(args)
    # model_restoration = torch.nn.DataParallel(model_restoration)

    utils.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)
    criterion = CharbonnierLoss().cuda()

    model_restoration.cuda()
    model_restoration.eval()
    with torch.no_grad():
        mse_val_rgb = []
        ssim_val_rgb = []
        for ii, data_val in enumerate(tqdm(val_loader), 0):
            # # rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))
            # rgb_gt = data_test[0].numpy().squeeze()
            # rgb_noisy = data_test[1].cuda()
            filenames = data_val[2]
            # print('rgb_noisy',rgb_noisy.shape)
            # rgb_restored = model_restoration(rgb_noisy)# 数据进入模型
            # # rgb_restored = torch.clamp(rgb_restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
            # # rgb_restored = rgb_restored.cpu().numpy().squeeze().transpose((1,2,0))
            # rgb_restored = rgb_restored.cpu().numpy().squeeze()
            # # print(rgb_restored.shape)
            # # print(rgb_gt.shape)
            # mse_val_rgb.append(mse_loss(rgb_restored, rgb_gt))
            # print(mse_val_rgb)
            # ssim_val_rgb.append(ssim_loss(rgb_restored, rgb_gt, multichannel=True))

            # print('server 评估进度为-->', 100 * ii / len(val_loader), '  ', ii, len(val_loader))
            ###########################################
            target = data_val[0].to('cuda:0')

            clean_dsp_dim = data_val[4]
            clean_dsp_dim_1 = clean_dsp_dim[1][0].item()
            clean_dsp_dim_2 = clean_dsp_dim[2][0].item()
            noisy_dsp_dim = data_val[5]
            noisy_dsp_dim_1 = noisy_dsp_dim[1][0].item()
            noisy_dsp_dim_2 = noisy_dsp_dim[2][0].item()
            # print(noisy_dsp_dim_1)
            # print(noisy_dsp_dim_2)
            # print(clean_dsp_dim_1)
            # print(clean_dsp_dim_2)

            # The factor is calculated (window_size(8) * down_scale(2^4) in this case)
            input, mask = expand2square(data_val[1].to('cuda:0'), factor=128, batch_size=args.batch_size,
                                        inchannels=5)
            input2 = 1 - mask
            # restored = model_restoration(data_val[1].cuda())
            restored = model_restoration(input.to('cuda:0'), input2.to('cuda:0'))

            # restored = model_restoration(input.to('cuda:0'), input2.to('cuda:0'))
            restored = torch.masked_select(restored.to('cuda:0'), mask.bool()).reshape(args.batch_size, 2,
                                                                                       noisy_dsp_dim_1,
                                                                                       noisy_dsp_dim_2).to('cuda:0')
            restored = restored[:, :, 0:0 + clean_dsp_dim_1, 0:0 + clean_dsp_dim_2].to('cuda:0')
            # loss = utils.batch_MSE(restored, target, False).item()
            loss = criterion(target, restored)
            MAE = utils.batch_MAE(restored, target, False).item()
            batch_PSNR = utils.batch_PSNR(restored, target, False).item()
            print(loss)
            # print(loss, MAE, batch_PSNR)
            if args.save_images:
                # utils.save_img(os.path.join(args.result_dir,filenames[0]), img_as_ubyte(rgb_restored))
                save_img_path = os.path.join(args.result_dir, filenames[0])
                save_img_path2 = os.path.join('results/input/', 'target_' + filenames[0])
                data = {}
                # data['Prediction'] = restored
                data = {'predict': restored.cpu().numpy().squeeze()}
                # print('save_img_path->', save_img_path)
                data2 = {}
                # data2['Prediction'] = target
                data2 = {'target': target.cpu().numpy().squeeze()}
                sio.savemat(save_img_path, data)
                sio.savemat(save_img_path2, data2)
                # print(save_img_path)
                # print("already save as mat file !")


