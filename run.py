import argparse
import torch
from exp.exp_forecasting import Exp_Forecast
from utils.print_args import print_args
import random
import numpy as np
from utils.tools import name_with_datetime
import sys
import os

# Add the directory containing 'exp' to sys.path
exp_folder_path = '/path/to/exp'  # Replace with the actual path to the exp folder
sys.path.append(exp_folder_path)

# Import the module
from exp.exp_forecasting import Exp_Forecast


import wandb
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--task_name', type=str, required=True, default='forecast',
                        help='task name, options:[forecast]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Linear',
                        help='model name, options: [Linear, DeepAR, Transformer]')
    parser.add_argument('--from_pretrain', type=str, help='to load a pretrained model as encoder, e.g., GPT2')
    parser.add_argument('--reinit_pretrain', action='store_true', help='reinit the weights of a pretrained model')
    parser.add_argument('--pretrain_warmup', type=float, default=0,
                        help='keep the pretrained encoder completely frozen')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--data_version', type=str, required=True, default='v2',
                        help='Dataset version. Specific to Green_Energy.')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--load_checkpoint', type=str, help='load a model checkpoint')
    parser.add_argument("--track", action='store_true',
                        help="if toggled, this experiment will be tracked with Weights and Biases")

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    # DLinear specific
    parser.add_argument("--all_cov", action='store_true',
                        help="if toggled, use all input channels and time covariates")
    parser.add_argument("--individual", action='store_true',
                        help="if toggled, use a different set of weights for each output channel")

    # PatchTST specific
    parser.add_argument('--patch_len', type=int, default=16, help='number of timesteps per patch')
    parser.add_argument('--stride', type=int, default=8, help='stride size between two patches')

    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--cls_sz', type=int, default=0, help='num of CLS tokens')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--use_norm', action='store_true', help='whether to normalize in subwindow')

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--accum_iter', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--total_iters', type=int, default=20000, help='total number of training iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--clip_grad', type=float, default=0.1, help='maximum gradient norm to clip')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='percentage of warmup steps')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='optimizer learning rate during linear probing')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    if args.task_name == 'forecast':
        Exp = Exp_Forecast
    else:
        raise ValueError(f'Unknown task name: {args.task_name}')

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            detailed_model_name = args.model
            setting = f'{detailed_model_name}_sl{args.seq_len}_{args.data_version}_' \
                      f'dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_{args.des}_{name_with_datetime()}'
            args.setting = setting

            if args.track:
                wandb_tags = []
                wandb.init(entity="augustin-t-pelleau",
                           project="capstoneATP",  # TODO: change to your project name
                           name=setting,
                           sync_tensorboard=True,
                           monitor_gym=True, config={
                            'model_id': args.model_id,
                            'model': args.model,
                            'data': args.data,
                            'data_version': args.data_version,
                            'seq_len': args.seq_len,
                            'patch_len': args.patch_len,
                            'stride': args.stride,
                            'label_len': args.label_len,
                            'pred_len': args.pred_len,
                            'd_model': args.d_model,
                            'n_heads': args.n_heads,
                            'e_layers': args.e_layers,
                            'd_layers': args.d_layers,
                            'd_ff': args.d_ff,
                            'enc_in': args.enc_in,
                            'batch_size': args.batch_size,
                            'dropout': args.dropout,
                            'learning_rate': args.learning_rate,
                            'des': args.des},
                           # notes="",
                           tags=wandb_tags,
                           save_code=True
                           )
                writer = SummaryWriter(f"runs/{setting}")
                writer.add_text(
                    "hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
                )
            else:
                writer = None

            exp = Exp(args, writer)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            torch.cuda.empty_cache()
    else:  # TODO: update
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        writer = None

    if writer is not None:
        writer.close()
