import os
import sys
import math


import argparse
import options


import utils
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx

from losses import CharbonnierLoss,L2Loss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import  get_training_data,get_validation_data

import syft as sy

def expand2square(timg,factor=16.0,batch_size=2,inchannels=29):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)#ceil(x) 函数返回一个大于或等于 x 的的最小整数。

    img = torch.zeros(batch_size,inchannels,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(batch_size,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1.0)
    
    return img, mask



if __name__ == '__main__':

    # add dir
    dir_name = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(dir_name,'./auxiliary/'))
    print(dir_name)

    ######### parser ###########
    opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
    print(opt)

    ######### Set GPUs ###########
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    torch.backends.cudnn.benchmark = True

    ######### Logs dir ###########
    log_dir = os.path.join(dir_name,'log', opt.arch+opt.env)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
    logname = os.path.join(log_dir, '.txt') 
    print("Now time is : ",datetime.datetime.now().isoformat())
    result_dir = os.path.join(log_dir, 'results')
    model_dir  = os.path.join(log_dir, 'models')
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    # ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    ######### Model ###########
    model_restoration = utils.get_arch(opt)##定义模型

    with open(logname,'a') as f:
        f.write(str(opt)+'\n')
        f.write(str(model_restoration)+'\n')

    ######### Optimizer ###########
    start_epoch = 1
    if opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
    else:
        raise Exception("Error optimizer...")


    ######### DataParallel ###########
    model_restoration = torch.nn.DataParallel (model_restoration)
    model_restoration.cuda()

    ######### Resume ###########
    if opt.resume:
        path_chk_rest = opt.pretrain_weights
        utils.load_checkpoint(model_restoration,path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        lr = utils.load_optim(optimizer, path_chk_rest)

        for p in optimizer.param_groups: p['lr'] = lr
        warmup = False
        new_lr = lr
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:",new_lr)
        print('------------------------------------------------------------------------------')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

    ######### Scheduler ###########
    if opt.warmup:
        print("Using warmup and cosine strategy!")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 50
        print("Using StepLR,step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()


    ######### Loss ###########
    criterion = CharbonnierLoss().cuda()
    #criterion = L2Loss().cuda()#会产生更加模糊的效果

    hook = sy.TorchHook(torch)
    bob = sy.VirtualWorker(hook, id='bob')
    alice = sy.VirtualWorker(hook, id='alice')

    ######### DataLoader ###########
    print('===> Loading datasets')
    img_options_train = {'patch_size':opt.train_ps}
    train_dataset = get_training_data(opt.train_dir, img_options_train)##获取数据集
    #train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
    #        num_workers=opt.train_workers, pin_memory=True, drop_last=False)
    
    federated_train_loader = sy.FederatedDataLoader(train_dataset.federate((bob,alice)), batch_size=opt.batch_size, shuffle=True, **kwargs)

    val_dataset = get_validation_data(opt.val_dir)##获取验证集
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, 
            num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

    len_trainset = train_dataset.__len__()
    len_valset = val_dataset.__len__()
    print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)
    ######### validation ###########
    with torch.no_grad():
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            filenames = data_val[2]
        #     psnr_val_rgb.append(utils.batch_PSNR(input_, target, False).item())
        # psnr_val_rgb = sum(psnr_val_rgb)/len_valset
        # print('Input & GT (PSNR) -->%.4f dB'%(psnr_val_rgb))

    ######### train ###########
    print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
    best_psnr = 0
    best_epoch = 0
    best_iter = 0
    eval_now = len(train_loader)//1#指定多长时间进行一次测试
    print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

    loss_scaler = NativeScaler()#loss_scaler 函数，它的作用本质上是 loss.backward(create_graph=create_graph) 和 optimizer.step()。
    # loss_scaler 继承 NativeScaler 这个类。这个类的实例在调用时需要传入 loss, optimizer, clip_grad, parameters, create_graph 等参数，在 __call__ () 函数的内部实现了 loss.backward(create_graph=create_graph) 功能和 optimizer.step() 功能。

    torch.cuda.empty_cache()
    for epoch in range(start_epoch, opt.nepoch + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1

        for i, data in enumerate(federated_train_loader, 0): 
            model_restoration.send(data.location)
            # zero_grad
            optimizer.zero_grad()

            target = data[0].cuda()

            clean_dsp_dim = data[4]
            clean_dsp_dim_1=clean_dsp_dim[1][0].item()
            clean_dsp_dim_2=clean_dsp_dim[2][0].item()
            noisy_dsp_dim = data[5]
            noisy_dsp_dim_1=noisy_dsp_dim[1][0].item()
            noisy_dsp_dim_2=noisy_dsp_dim[2][0].item()
            # print(noisy_dsp_dim_1)
            # print(noisy_dsp_dim_2)
            # print(clean_dsp_dim_1)
            # print(clean_dsp_dim_2)

            # The factor is calculated (window_size(8) * down_scale(2^4) in this case) 
            input, mask = expand2square(data[1].cuda(), factor=128, batch_size=opt.batch_size, inchannels=5) 



            # 数据增强
            # if epoch>5:
            #     target, input_ = utils.MixUp_AUG().aug(target, input_)
            with torch.cuda.amp.autocast():# 给用户提供了较为方便的混合精度训练机制
                # restored = model_restoration(input_)# 数据进入模型
                ## restored = torch.clamp(restored,0,1)#将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。 
                # print(input.shape)
                restored = model_restoration(input, 1 - mask)
                restored = torch.masked_select(restored,mask.bool()).reshape(opt.batch_size,2,noisy_dsp_dim_1,noisy_dsp_dim_2)
                restored = restored[:,:,0:0+clean_dsp_dim_1,0:0+clean_dsp_dim_2]
                # print(restored.shape)
                # restored = restored.cpu().numpy().squeeze()
                loss = criterion(restored, target)# 计算损失
            loss_scaler(
                    loss, optimizer,parameters=model_restoration.parameters())# 更新参数
            model_restoration.get()
            
            epoch_loss +=loss.item()# item()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回。
            epoch_loss = epoch_loss.get()
            # #### Evaluation ####
            # if (i+1)%eval_now==0 and i>0:
            #     with torch.no_grad():
            #         model_restoration.eval()#进入测试模式
            #         psnr_val_rgb = []
            #         for ii, data_val in enumerate((val_loader), 0):
            #             target = data_val[0].cuda()
            #             input_ = data_val[1].cuda()
            #             filenames = data_val[2]
            #             with torch.cuda.amp.autocast():
            #                 restored = model_restoration(input_)
            #             # restored = torch.clamp(restored,0,1)  
            #             psnr_val_rgb.append(utils.batch_MSE(restored, target, False).item())

            #         psnr_val_rgb = sum(psnr_val_rgb)/len_valset
                    
            #         if psnr_val_rgb < best_psnr:
            #             best_psnr = psnr_val_rgb
            #             best_epoch = epoch
            #             best_iter = i 
            #             torch.save({'epoch': epoch, 
            #                         'state_dict': model_restoration.state_dict(),
            #                         'optimizer' : optimizer.state_dict()
            #                         }, os.path.join(model_dir,"model_best.pth"))

            #         print("[Ep %d it %d\t MSE VAL: %.4f\t] ----  [best_Ep_VAL %d best_it_VAL %d Best_MSE_VAL %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
            #         with open(logname,'a') as f:
            #             f.write("[Ep %d it %d\t MSE VAL: %.4f\t] ----  [best_Ep_VAL %d best_it_VAL %d Best_MSE_VAL %.4f] " \
            #                 % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
            #         model_restoration.train()
            #         torch.cuda.empty_cache()
        scheduler.step()
        
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss/len_trainset, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")
        with open(logname,'a') as f:
            f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss/len_trainset, scheduler.get_lr()[0])+'\n')

        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))   

        if epoch%opt.checkpoint == 0:
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
    print("Now time is : ",datetime.datetime.now().isoformat())






