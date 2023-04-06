import os
import torch


class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self, parser):
        # # data settings
        # parser.add_argument('--record_size', type=int, default=2, help='batch size')##默认为32
        # parser.add_argument('--vmodel_size', type=int, default=500, help='training epochs')##默认为250       
        # global settings
        parser.add_argument('--batch_size', type=int, default=14, help='batch size')  ##默认为32
        parser.add_argument('--nepoch', type=int, default=1, help='training epochs')  ##默认为250
        parser.add_argument('--train_workers', type=int, default=0, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=0, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default='SIDD')
        # parser.add_argument('--pretrain_weights',type=str, default='/home/hyzb/user_wanghongzhou/SR/log/Uformer_32_0701_1/models/model_epoch_100.pth', help='path of pretrained_weights')
        parser.add_argument('--pretrain_weights', type=str, default='', help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0,1,2,3', help='GPUs')
        parser.add_argument('--arch', type=str, default='Uformer', help='archtechture')
        parser.add_argument('--mode', type=str, default='denoising', help='image restoration mode')

        # args for saving
        parser.add_argument('--save_dir', type=str, default='./log', help='save dir')
        parser.add_argument('--save_images', action='store_true', default=False)
        parser.add_argument('--env', type=str, default='_32_0701_1', help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

        # args for Uformer
        parser.add_argument('--norm_layer', type=str, default='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')  ##自注意力窗的大小
        parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')

        # args for vit
        parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
        parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
        parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
        parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
        parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
        parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
        parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
        parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

        # args for training
        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')  ##贴片的大小，默认为128
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--train_dir', type=str, default='/home/hyzb/user_wanghongzhou/SR_test/datasets/train',
                            help='dir of train data')  # './datasets/train'
        parser.add_argument('--val_dir', type=str, default='/home/hyzb/user_wanghongzhou/SR_test/datasets/test',
                            help='dir of train data')  # './datasets/val'
        # parser.add_argument('--train_dir', type=str, default ='/www/dataset/联邦',  help='dir of train data')#'./datasets/train'
        # parser.add_argument('--val_dir', type=str, default ='datasets/val',  help='dir of train data')#'./datasets/val'
        parser.add_argument('--warmup', action='store_true', default=True, help='warmup')
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')

        # 联邦学习的参数
        parser.add_argument('--IID', type=int, default=-1, help='采样方式')  # -1 随机采样等差分布 0全部数据 1平均分布
        parser.add_argument('--num_of_clients', type=int, default=16, help='客户端数量')
        parser.add_argument('--cfraction', type=float, default=0.25, help='随机挑选的客户端的数量')
        parser.add_argument('--num_comm', type=int, default=2000, help='number of communications')
        parser.add_argument('--save_path', type=str, default='./debug',
                            help='the saving path of checkpoints')
        parser.add_argument('--model_name', type=str, default='test', help='the model to train')
        parser.add_argument('--drop_out_client', type=int, default=2, help='客户端随机失活数量')
        parser.add_argument('--drop_out', type=int, default=3, help='客户端随机失活策略 0 1 2 3')
        # 默认0 没策略 1挂掉的客户端不跑  2 只挂掉每次通讯中全部客户端中的指定数量 在剩余客户端中挑选指定数量参与训练 3 例 选出4个客户端 挂掉2个，后补两个
        parser.add_argument('--sr_weight', type=int, default=1, help='权重分配策略')  # 0 分权分配 ，1平均分配

        return parser
