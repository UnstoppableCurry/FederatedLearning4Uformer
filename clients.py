import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import time
import math
# from getData import GetDataSet
from warmup_scheduler.scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
import argparse
import options
import torch.optim as optim
import utils


class client(object):
    def __init__(self, trainDataSet, dev,eval_dataset):
        self.dev = dev
        self.train_dl = trainDataSet
        self.local_parameters = None
        self.opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
        self.eval_dataset=eval_dataset
        self.net=None

    def expand2square(self, timg, factor=16.0, batch_size=2, inchannels=29):
        _, _, h, w = timg.size()

        X = int(math.ceil(max(h, w) / float(factor)) * factor)  # ceil(x) 函数返回一个大于或等于 x 的的最小整数。

        img = torch.zeros(batch_size, inchannels, X, X).type_as(timg)  # 3, h,w
        mask = torch.zeros(1, 1, X, X).type_as(timg)

        # print(img.size(),mask.size())
        # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
        img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
        mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

        return img, mask

    def localUpdate(self, train_dl,net, lossFun, global_parameters,tb_writer,client_name,drive=None,client_dataset_lens=0):
        '''
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前通讯中最全局参数
            tb_writer:tensorboard
            client_name:名称
            return: 返回当前Client基于自己的数据训练得到的新的模型参数
        '''



        if drive is not None:
            Net = utils.get_arch(self.opt)
            # for key, var in global_parameters.items():
            #     global_parameters[key] = var.to(drive)
            Net.to(drive)#模型切换至指定设置
            net.to(drive)#模型切换至指定设置
            global_parameters = {}
            for key, var in net.state_dict().items():
                global_parameters[key] = var.clone()
            Net.load_state_dict(global_parameters, strict=True)
        else:
            Net = utils.get_arch(self.opt)
            Net.load_state_dict(global_parameters, strict=False)
        if self.net is None:
            self.net=Net
        if self.opt.optimizer.lower() == 'adam':
            opti = optim.Adam(Net.parameters(), lr=self.opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                                   weight_decay=self.opt.weight_decay)
        elif self.opt.optimizer.lower() == 'adamw':
            opti = optim.AdamW(Net.parameters(), lr=self.opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                                    weight_decay=self.opt.weight_decay)
        else:
            raise Exception("Error optimizer...")
        # 载入Client自有数据集
        # 加载本地数据
        self.train_dl = DataLoader(dataset=train_dl, batch_size=self.opt.batch_size, shuffle=True,
                              num_workers=self.opt.train_workers, pin_memory=True, drop_last=True)
        loss_scaler = NativeScaler()  # loss_scaler 函数，它的作用本质上是 loss.backward(create_graph=create_graph) 和 optimizer.step()。

        if self.opt.warmup:
            print("Using warmup and cosine strategy!")
            warmup_epochs = self.opt.warmup_epochs
            scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(opti, self.opt.nepoch - warmup_epochs,
                                                                         eta_min=1e-6)
            scheduler = GradualWarmupScheduler(opti, multiplier=1, total_epoch=warmup_epochs,
                                               after_scheduler=scheduler_cosine)
            scheduler.step()
        else:
            step = 50
            print("Using StepLR,step={}!".format(step))
            scheduler = StepLR(opti, step_size=step, gamma=0.5)
            scheduler.step()
        # 设置迭代次数
        # for epoch in range(localEpoch):
        #     for data, label in self.train_dl:
        #         # 加载到GPU上
        #         data, label = data.to(self.dev), label.to(self.dev)
        #         # 模型上传入数据
        #         preds = Net(data)
        #         # 计算损失函数
        #         '''
        #             这里应该记录一下模型得损失值 写入到一个txt文件中
        #         '''
        #         loss = lossFun(preds, label)
        #         # 反向传播
        #         loss.backward()
        #         # 计算梯度，并更新梯度
        #         opti.step()
        #         # 将梯度归零，初始化梯度
        #         opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        step=0
        steps = 0
        for epoch in range(self.opt.nepoch):
            print(client_name,' 进度为-->',epoch*100/self.opt.nepoch,'  ',epoch)
            epoch_start_time = time.time()
            epoch_loss = 0

            for i, data in enumerate(self.train_dl, 0):
                print(client_name, 'epoch 进度为-->', i,len(self.train_dl),i * 100 / len(self.train_dl), '%')

                # data.to(self.dev)
                # zero_grad
                opti.zero_grad()
                if drive is not None:
                    target = data[0].to(drive)
                    data[1]=data[1].to(drive)

                else:
                    target = data[0].cuda()
                    data[1].cuda()

                clean_dsp_dim = data[4]
                clean_dsp_dim_1 = clean_dsp_dim[1][0].item()
                clean_dsp_dim_2 = clean_dsp_dim[2][0].item()
                noisy_dsp_dim = data[5]
                noisy_dsp_dim_1 = noisy_dsp_dim[1][0].item()
                noisy_dsp_dim_2 = noisy_dsp_dim[2][0].item()
                # print(noisy_dsp_dim_1)
                # print(noisy_dsp_dim_2)
                # print(clean_dsp_dim_1)
                # print(clean_dsp_dim_2)

                # The factor is calculated (window_size(8) * down_scale(2^4) in this case)
                input, mask = self.expand2square(data[1], factor=128, batch_size=self.opt.batch_size,
                                                 inchannels=5)

                # 数据增强
                # if epoch>5:
                #     target, input_ = utils.MixUp_AUG().aug(target, input_)
                with torch.cuda.amp.autocast():  # 给用户提供了较为方便的混合精度训练机制
                    # restored = model_restoration(input_)# 数据进入模型
                    ## restored = torch.clamp(restored,0,1)#将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
                    # print(input.shape)
                    input2=1-mask
                    if drive is not None:
                        input2=input2.to(drive)
                        # print('Net.device1', drive, input.device, input2.device)
                        # input.to(drive)
                    # print('Net.device',client_name,drive,input.device,input2.device)
                    restored = Net(input, input2)
                    restored = torch.masked_select(restored, mask.bool()).reshape(self.opt.batch_size, 2,
                                                                                  noisy_dsp_dim_1,
                                                                                  noisy_dsp_dim_2)
                    restored = restored[:, :, 0:0 + clean_dsp_dim_1, 0:0 + clean_dsp_dim_2]
                    # print(restored.shape)
                    # restored = restored.cpu().numpy().squeeze()
                    loss = lossFun(restored, target)  # 计算损失
                    MAE = utils.batch_MAE(restored, target, False).item()
                    batch_PSNR = utils.batch_PSNR(restored, target, False).item()
                    if tb_writer:
                        tb_writer.add_scalar(client_name+'_train_each_MSE', loss, step)
                        tb_writer.add_scalar(client_name+'_train_epoch', epoch, step)
                        tb_writer.add_scalar(client_name+'_train_each_MAE', MAE, step)
                        tb_writer.add_scalar(client_name+'_train_each_PSNR', batch_PSNR, step)

                loss_scaler(
                    loss, opti, parameters=Net.parameters())  # 更新参数
                epoch_loss += loss.item()  # item()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回。
                step+=1

            scheduler.step()

            with torch.no_grad():
                Net.eval()  # 进入测试模式
                psnr_val_rgb = []
                MAE_val_rgb = []
                PSNR_val_rgb = []
                for ii, data_val in enumerate(self.eval_dataset, 0):
                    print(client_name,'epoch 进度为-->',ii*100/len(self.eval_dataset),'%')
                    if data_val is not None:
                        data_val[1].to(drive)  # 模型切换至指定设置
                        target = data_val[0].to(drive)

                    else:
                        data_val[1].cuda()
                        target = data_val[0].cuda()

                    clean_dsp_dim = data_val[4]
                    clean_dsp_dim_1 = clean_dsp_dim[1][0].item()
                    clean_dsp_dim_2 = clean_dsp_dim[2][0].item()
                    noisy_dsp_dim = data_val[5]
                    noisy_dsp_dim_1 = noisy_dsp_dim[1][0].item()
                    noisy_dsp_dim_2 = noisy_dsp_dim[2][0].item()
                    input, mask = self.expand2square(data_val[1].to(drive), factor=128, batch_size=self.opt.batch_size,
                                                     inchannels=5)
                    with torch.cuda.amp.autocast():  # 给用户提供了较为方便的混合精度训练机制
                        input2 = 1 - mask
                        if drive is not None:
                            input2 = input2.to(drive)
                        # print('input1,input2->',input.device,input.device)
                        restored = Net(input.to(drive), input2.to(drive))
                        restored = torch.masked_select(restored, mask.bool()).reshape(self.opt.batch_size, 2,
                                                                                      noisy_dsp_dim_1,
                                                                                      noisy_dsp_dim_2)
                        restored = restored[:, :, 0:0 + clean_dsp_dim_1, 0:0 + clean_dsp_dim_2]
                        # loss = utils.batch_MSE(restored, target, False).item()
                        loss=lossFun(restored, target)
                        MAE = utils.batch_MAE(restored, target, False).item()
                        batch_PSNR = utils.batch_PSNR(restored, target, False).item()
                        psnr_val_rgb.append(loss.cpu().numpy())
                        MAE_val_rgb.append(MAE)
                        PSNR_val_rgb.append(batch_PSNR)
                        steps += 1
                        if tb_writer:
                            tb_writer.add_scalar(client_name + "_test_step", epoch, steps)
                            tb_writer.add_scalar(client_name + "_test_MSE_Loss", loss, steps)
                            tb_writer.add_scalar(client_name + "_test_PSNR_Loss", batch_PSNR, steps)
                            tb_writer.add_scalar(client_name + "_test_MAE_Loss", MAE, steps)
                # psnr_val_rgb = sum(psnr_val_rgb) / (len(self.eval_dataset)//self.opt.batch_size)
                # MAE_val_rgb = sum(MAE_val_rgb) / (len(self.eval_dataset)//self.opt.batch_size)
                # PSNR_val_rgb = sum(PSNR_val_rgb) / (len(self.eval_dataset)//self.opt.batch_size)

                if tb_writer:
                    tb_writer.add_scalar(client_name + "_test_epoch_evl_MSEc", np.array(psnr_val_rgb).mean(), epoch)
                    tb_writer.add_scalar(client_name + "_test_epoch_evl_MAE", np.array(MAE_val_rgb).mean(), epoch)
                    tb_writer.add_scalar(client_name + "_test_epoch_evl_psnr", np.array(PSNR_val_rgb).mean(), epoch)

                Net.train()
                torch.cuda.empty_cache()
        Net.to('cuda:0')
        return Net.state_dict(),client_dataset_lens

    def local_val(self):
        pass


class ClientsGroup(object):
    '''
        param: train_dataset 训练集
        param: isIID 是否是IID  这个只针对特定数据集排序去做没有通用性  改后 大致就是数据切分的方式对范化性能的评估
        param: numOfClients 客户端的数量
        param: dev 设备(GPU)  这个暂时就单卡实现了   多卡数据并行貌似保留的意义没有了, 每个客户端分配指定设备运行也是分布式
        param: clients_set 客户端

    '''

    def __init__(self, train_dataset, isIID, numOfClients, dev,test_dataset=None):
        self.train_dataset = train_dataset
        self.test_dataset=test_dataset
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.all_Data=[]
        self.test_dataset=test_dataset
        self.split_Data_historoy=[]

        self.dataSetBalanceAllocation()
    def iid_create(self,input_data,nums):
        if nums==0:
            return self.all_Data.append(input_data)
        else:
            nums=nums-1
            data1,data2=torch.utils.data.random_split(input_data, [int(len(input_data)*0.5),int(len(input_data)*0.5)])
            self.iid_create(data1,nums)
            self.iid_create(data2,nums)

    def create_list(self,all_len, numbers):
        s_e = all_len * 2 // numbers
        for i in range(1, 1000):
            rate = i * 0.001
            one = int(all_len * rate)
            end = s_e - one
            result = 0
            d = (end - one) / (numbers - 1)
            all_Datas = []
            if d.is_integer() and d > 0 and one > 0 and end > 0 and d != 0:
                print(s_e, one, end, d)
                for index in range(numbers):
                    value = one + index * d
                    result += value
                    print(index, '--> ', value)
                    all_Datas.append(int(value))
                print(result)
                return all_Datas
        return None
    def dataSetBalanceAllocation(self):
        all_dataset=None
        all_lens = []
        if self.is_iid==1:
            #数据切片   数据总量N   num_of_clients 不应大于 2^N
            # self.iid_create(self.train_dataset,self.num_of_clients//2)
            next_index=0
            mid = self.num_of_clients // 2 #3520 全长
            all_len = len(self.train_dataset) // self.num_of_clients
            all_clients_num=self.num_of_clients
            for i in range(self.num_of_clients):

                input_dataset, self.train_dataset = torch.utils.data.random_split(self.train_dataset,
                                                                                  [int(len(
                                                                                       self.train_dataset)//all_clients_num),
                                                                                   int(len(
                                                                                      self.train_dataset) *(1-1/all_clients_num))])
                # 创建一个客户端
                all_clients_num=all_clients_num-1
                all_lens.append(len(input_dataset))
                someone = client(input_dataset, self.dev,self.test_dataset)
                self.clients_set['client{}'.format(i)] = someone
            print(all_lens)
        elif self.is_iid==-1:

            all_len = len(self.train_dataset)
            all_clients_num=self.num_of_clients
            split_Data=self.create_list(all_len,all_clients_num)
            self.split_Data_historoy = split_Data
            if split_Data is None:
                # 数据切片   数据总量N   num_of_clients 不应大于 2^N
                # self.iid_create(self.train_dataset,self.num_of_clients//2)
                next_index = 0
                mid = self.num_of_clients // 2  # 3520 全长
                all_len = len(self.train_dataset) // self.num_of_clients
                all_clients_num = self.num_of_clients
                for i in range(self.num_of_clients):
                    input_dataset, self.train_dataset = torch.utils.data.random_split(self.train_dataset,
                                                                                      [int(len(
                                                                                          self.train_dataset) // all_clients_num),
                                                                                       int(len(
                                                                                           self.train_dataset) * (
                                                                                                       1 - 1 / all_clients_num))])
                    # 创建一个客户端
                    all_clients_num = all_clients_num - 1
                    all_lens.append(len(input_dataset))
                    someone = client(input_dataset, self.dev, self.test_dataset)
                    self.clients_set['client{}'.format(i)] = someone
                print(all_lens)
            for index in range(len(split_Data)):
                i=split_Data[index]
                if index ==all_clients_num:
                    someone = client(self.train_dataset, self.dev, self.test_dataset)
                    self.clients_set['client{}'.format(index)] = someone
                else:

                    input_dataset, self.train_dataset = torch.utils.data.random_split(self.train_dataset,
                                                                                      [i,
                                                                                       int(len(
                                                                                           self.train_dataset) -i)])
                    all_lens.append(len(input_dataset))
                    someone = client(input_dataset, self.dev, self.test_dataset)
                    self.clients_set['client{}'.format(index)] = someone
        elif self.is_iid==0:
            #数据不切片
            for i in range(self.num_of_clients):
                ## shards_id1

                # 创建一个客户端
                someone = client(self.train_dataset, self.dev,self.test_dataset)
                # 为每一个clients 设置一个名字
                # client10
                self.clients_set['client{}'.format(i)] = someone
        print(all_lens)


