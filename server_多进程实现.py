import os
import sys
import math

import argparse
from threading import Thread

import options

import utils
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import random
import time
import numpy as np
import datetime
from pdb import set_trace as stx

from losses import CharbonnierLoss, L2Loss
import torch.multiprocessing as mp

from tqdm import tqdm
from warmup_scheduler.scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import get_training_data, get_validation_data
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from clients import ClientsGroup, client


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


class Train_client(Thread):
    def __init__(self, obj, train_dataset, net, criterion, global_parameters, tb_writer, clent_name, clent_drive):
        Thread.__init__(self)
        self.result = None
        self.obj = obj
        self.train_dataset = train_dataset
        self.net = net
        self.criterion = criterion
        self.global_parameters = global_parameters
        self.tb_writer = tb_writer
        self.clent_name = clent_name
        self.clent_drive = clent_drive

    def run(self):
        self.result = self.obj.localUpdate(self.train_dataset, self.net, self.criterion, self.global_parameters,
                                           self.tb_writer, self.clent_name, self.clent_drive)

    def get_result(self):
        return self.result


if __name__ == "__main__":

    dir_name = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(dir_name, './auxiliary/'))
    print(dir_name)

    ######### parser ###########
    opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
    print(opt)
    tb_writer = SummaryWriter(opt.save_path)

    ######### Set GPUs ###########
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    # torch.backends.cudnn.benchmark = True

    ######### Logs dir ###########
    log_dir = os.path.join(dir_name, 'log', opt.arch + opt.env)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt')
    logname = os.path.join(log_dir, '.txt')
    print("Now time is : ", datetime.datetime.now().isoformat())
    result_dir = os.path.join(log_dir, 'results')
    model_dir = os.path.join(log_dir, 'models')
    print('modeldir', model_dir)
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    # ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    # torch.cuda.manual_seed_all(1234)
    ######### Model ###########
    model_restoration = utils.get_arch(opt)  ##定义模型

    with open(logname, 'a') as f:
        f.write(str(opt) + '\n')
        f.write(str(model_restoration) + '\n')

    ######### Optimizer ###########
    start_epoch = 1
    if opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=opt.weight_decay)
    else:
        raise Exception("Error optimizer...")

    ######### DataParallel ###########  有待商榷
    # model_restoration = torch.nn.DataParallel(model_restoration)
    # model_restoration.cuda()
    model_restoration.to('cuda:0')

    ######### Resume ###########
    if opt.resume:
        path_chk_rest = opt.pretrain_weights
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        lr = utils.load_optim(optimizer, path_chk_rest)

        for p in optimizer.param_groups: p['lr'] = lr
        warmup = False
        new_lr = lr
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - start_epoch + 1, eta_min=1e-6)

    ######### Scheduler ###########
    if opt.warmup:
        print("Using warmup and cosine strategy!")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 50
        print("Using StepLR,step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()

    ######### Loss ###########
    criterion = CharbonnierLoss().cuda()
    # criterion = L2Loss().cuda()#会产生更加模糊的效果

    ######### DataLoader ###########
    print('===> Loading datasets')
    img_options_train = {'patch_size': opt.train_ps}
    train_dataset = get_training_data(opt.train_dir, img_options_train)  ##获取数据集
    # iid_create(train_dataset,2)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.8 * len(train_dataset)),
                                                                               int(0.2 * len(train_dataset))])

    # train_dataset=val_dataset#测试用
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.train_workers, pin_memory=True, drop_last=False)

    # val_dataset = get_validation_data(opt.val_dir)  ##获取验证集
    # val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
    #                         num_workers=opt.eval_workers, pin_memory=False, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.eval_workers, pin_memory=False, drop_last=True)
    len_trainset = train_dataset.__len__()
    len_valset = val_dataset.__len__()
    print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)

    net = model_restoration

    myClients = ClientsGroup(train_dataset, opt.IID, opt.num_of_clients, opt.gpu, val_loader)

    # ---------------------------------------以上准备工作已经完成------------------------------------------#
    # 每次随机选取10个Clients
    num_in_comm = int(max(opt.num_of_clients * opt.cfraction, 1))

    # 得到全局的参数
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    steps = 0  # 预测步长
    # num_comm 表示通信次数，此处设置为1k
    # 通讯次数一共1000次
    for i in range(opt.num_comm):
        print("communicate round {}".format(i + 1))

        # 对随机选的将100个客户端进行随机排序
        order = np.random.permutation(opt.num_of_clients)
        # 生成个客户端
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
        sum_parameters = None
        # 这里的clients_
        all_local_parameters = []
        thds=[]
        # for client_name_index in range(1):
        mp.set_start_method(method='forkserver', force=True)
        pool = mp.Pool(processes=num_in_comm)

        all_result=[]
        for client_name_index in range(len(clients_in_comm)):
            print('client_name_index',client_name_index)
            client_name=clients_in_comm[client_name_index]
            # thd1=Train_client(myClients.clients_set[client_name],
            #                   train_dataset,
            #                   net.to('cuda:1'),
            #                   criterion,
            #                   global_parameters,
            #                   tb_writer,
            #                   client_name + '第' + str(i) + '次通讯',
            #                   'cuda:'+str(client_name_index))
            all_result.append(pool.apply_async(myClients.clients_set[client_name].localUpdate,
                                    ( train_dataset,
                              net.to('cuda:1'),
                              criterion,
                              global_parameters,
                              tb_writer,
                              client_name + '第' + str(i) + '次通讯',
                              'cuda:'+str(client_name_index)))
            )
        pool.close()
        pool.join()
        # pool.shutdown()
        for thd in all_result:
            all_local_parameters.append(thd.get())
        # thd2 = Train_client(myClients.clients_set[clients_in_comm[1]], train_dataset, net.to('cuda:2'),
        #                     criterion, global_parameters,
        #                     tb_writer,
        #                     clients_in_comm[0] + '第' + str(
        #                         i) + '次', 'cuda:2')
        for thd_local_parameters in all_local_parameters:
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in thd_local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + thd_local_parameters[var]
        # for client in tqdm(clients_in_comm):
        #     local_parameters = myClients.clients_set[client].localUpdate(train_dataset, net,
        #                                                                  criterion, global_parameters, tb_writer,
        #                                                                  client + '第' + str(i) + '次')
        #     # 对所有的Client返回的参数累加（最后取平均值）
        #     if sum_parameters is None:
        #         sum_parameters = {}
        #         for key, var in local_parameters.items():
        #             sum_parameters[key] = var.clone()
        #     else:
        #         for var in sum_parameters:
        #             sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        # 取平均值，得到本次通信中Server得到的更新后的模型参数
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)
        print('本次通讯结束开始评估')
        net.load_state_dict(global_parameters, strict=True)
        #### Evaluation ####
        with torch.no_grad():
            net.eval()  # 进入测试模式
            net.to('cuda:0')
            psnr_val_rgb = []
            MAE_val_rgb = []
            PSNR_val_rgb = []
            for ii, data_val in enumerate(val_loader, 0):
                print('server 评估进度为-->',100*ii/len(val_loader),  '  ',ii ,len(val_loader))
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
                input, mask = expand2square(data_val[1].to('cuda:0'), factor=128, batch_size=opt.batch_size,
                                            inchannels=5)
                with torch.cuda.amp.autocast():  # 给用户提供了较为方便的混合精度训练机制
                    input2 = 1 - mask
                    restored = net(input.to('cuda:0'), input2.to('cuda:0'))
                    restored = torch.masked_select(restored.to('cuda:0'), mask.bool()).reshape(opt.batch_size, 2,
                                                                                  noisy_dsp_dim_1,
                                                                                  noisy_dsp_dim_2).to('cuda:0')
                    restored = restored[:, :, 0:0 + clean_dsp_dim_1, 0:0 + clean_dsp_dim_2].to('cuda:0')
                    loss = utils.batch_MSE(restored, target, False).item()
                    MAE = utils.batch_MAE(restored, target, False).item()
                    batch_PSNR = utils.batch_PSNR(restored, target, False).item()
                    psnr_val_rgb.append(loss)
                    MAE_val_rgb.append(MAE)
                    PSNR_val_rgb.append(batch_PSNR)
                    psnr_val_rgb.append(loss)
                    steps += 1
                    if tb_writer:
                        tb_writer.add_scalar("server_epoch", steps, steps)
                        tb_writer.add_scalar("serve_eval_each_MSE_loss", loss, steps)
                        tb_writer.add_scalar("serve_eval_each_PSNR_loss", batch_PSNR, steps)
                        tb_writer.add_scalar("serve_eval_each_MAE_loss", MAE, steps)
            psnr_val_rgb = sum(psnr_val_rgb) / len_valset
            MAE_val_rgb = sum(MAE_val_rgb) / len_valset
            PSNR_val_rgb = sum(PSNR_val_rgb) / len_valset
            if tb_writer:
                tb_writer.add_scalar("server_epoch_mean_MSE_evl", psnr_val_rgb, i)
                tb_writer.add_scalar("server_epoch_mean_MAE_evl", MAE_val_rgb, i)
                tb_writer.add_scalar("server_epoch_mean_PSNR_evl", PSNR_val_rgb, i)
            print('本轮测试完毕 进度-->',i*100/opt.num_comm,'%   第',i,'次通讯')
            net.train()
            torch.cuda.empty_cache()
            if not os.path.exists(opt.save_path):
                os.mkdir(opt.save_path)
            torch.save(net, os.path.join(opt.save_path,
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}.pt'.format(opt.model_name,
                                                                                                   i, opt.nepoch,
                                                                                                   opt.batch_size,
                                                                                                   opt.lr_initial,
                                                                                                   opt.num_of_clients,
                                                                                                   opt.cfraction)))
