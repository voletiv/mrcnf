import argparse
import csv
import datetime
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import shutil
import sys
import time
import warnings
import yaml

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.utils as vutils

from multiprocessing import Process
from torch.utils.data import TensorDataset
from tqdm import tqdm
try:
    from torchvision.transforms.functional import resize, InterpolationMode
    interp = InterpolationMode.NEAREST
except:
    from torchvision.transforms.functional import resize
    interp = 0

import dist_utils
import lib.utils as utils

from lib import layers
from lib.dataloader import get_dataloaders
from lib.multiscale import CNFMultiscale
from lib.regularization import get_regularization, append_regularization_to_log
from lib.regularization import append_regularization_keys_header, append_regularization_csv_dict
from lib.utils import logit_logpx_to_image_bpd, convert_base_from_10, vis_imgs_laps, convert_time_stamp_to_hrs, logpx_to_bpd
from misc import set_cnf_options, count_nfe, count_training_parameters, count_parameters

cudnn.benchmark = True
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'adaptive_heun', 'bosh3']


def get_args():
    parser = argparse.ArgumentParser("Multi-Resolution Continuous Normalizing Flow")

    # Mode
    parser.add_argument("--mode", type=str, default="image", choices=["wavelet", "mrcnf"])

    # Multi-res
    parser.add_argument("--normal_resolution", type=int, default=64, help="Resolution at which z is standard normal. (def: 64)")
    parser.add_argument('--std_scale', type=eval, default=True, choices=[True, False], help="Add AffineTx layer at end of CNF to scale output acc to z_std")
    # Data
    parser.add_argument("--data", type=str, default="mnist", choices=["mnist", "svhn", "cifar10", "lsun_church", "celebahq", "imagenet", "imagenet64_cf", "zap50k", "fashion_mnist"])
    parser.add_argument("--data_path", default="./data/", help="mnist: `./data/`, cifar10: `./data/CIFAR10`, imagenet: `./data/ilsvrc2012.hdf5`")
    parser.add_argument("--imagenet_classes", type=str, default="")
    parser.add_argument("--nworkers", type=int, default=8)
    parser.add_argument("--im_size", type=int, default=32)
    parser.add_argument('--ds_idx_mod', type=int, default=None, help="In case we want to train on only subset of images, e.g. mod=10 => images [0, 1, ..., 9]")
    parser.add_argument('--ds_idx_skip', type=int, default=0, help="In case we want to train on only subset of images, e.g. mod=10 and skip=10 => images [10, 11, ..., 19]")
    parser.add_argument('--ds_length', type=int, default=None, help="Total length of dataset, to decide number of batches per epoch")
    parser.add_argument('--test_ds_idx_mod', type=int, default=None, help="In case we want to test on only subset of images, e.g. mod=10 => images [0, 1, ..., 9]")
    parser.add_argument('--test_ds_idx_skip', type=int, default=0, help="In case we want to test on only subset of images, e.g. mod=10 and skip=10 => images [10, 11, ..., 19]")
    parser.add_argument('--test_ds_length', type=int, default=None, help="Total length of test dataset, to decide number of batches per epoch")

    # Save
    parser.add_argument("--save_path", type=str, default="experiments/cnf")

    # Model
    parser.add_argument("--dims", type=str, default="64,64,64")
    parser.add_argument("--strides", type=str, default="1,1,1,1")
    parser.add_argument("--num_blocks", type=str, default="2,2", help='Number of stacked CNFs, per scale. Should have 1 item, or max_scales number of items.')
    parser.add_argument('--bn', type=eval, default=False, choices=[True, False], help="Add BN to coarse")
    parser.add_argument("--layer_type", type=str, default="concat", choices=["ignore", "concat"])
    parser.add_argument("--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu", "swish", "square", "identity"])
    parser.add_argument('--zero_last', type=eval, default=True, choices=[True, False])

    # Data characteristics
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument('--max_scales', type=int, default=2, help="# of scales for image pyramid")
    parser.add_argument('--scale', type=int, default=0, help='freeze all parameters but this scale; start evaluating loss from this scale')
    parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument('--logit', type=eval, default=True, choices=[True, False])
    parser.add_argument("--alpha", type=float, default=0.05, help="if logit is true, alpha is used to convert from pixel to logit (and back)")
    parser.add_argument('--concat_input', type=eval, default=True, choices=[True, False], help="To concat the image input to odefunc or not.")

    # ODE Solver
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-5, help='only for adaptive solvers')
    parser.add_argument('--rtol', type=float, default=1e-5, help='only for adaptive solvers')
    parser.add_argument('--step_size', type=float, default=0.25, help='only for fixed step size solvers')
    parser.add_argument('--first_step', type=float, default=0.166667, help='only for adaptive solvers')

    # ODE Solver for test
    parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
    parser.add_argument('--test_atol', type=float, default=None)
    parser.add_argument('--test_rtol', type=float, default=None)
    parser.add_argument('--test_step_size', type=float, default=None)
    parser.add_argument('--test_first_step', type=float, default=None)

    # ODE stop time
    parser.add_argument('--time_length', type=float, default=1.0)
    parser.add_argument('--train_T', type=eval, default=False)
    parser.add_argument('--steer_b', type=float, default=0.0)

    # Train
    parser.add_argument('--joint', type=eval, default=False, choices=[True, False], help="Joint training of all scales (else train each scale separately)")
    parser.add_argument("--num_epochs", type=int, default=100, help="# of epochs in case of JOINT training only.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--epochs_per_scale', type=str, default=None, help="# of epochs per scale in case NOT JOINT training; if not specified, will default to `num_epochs/max_scales`. Eg. `100` or `40,30,30`")
    parser.add_argument("--batch_size_per_scale", type=str, default=None, help="Batch sizes to use for every scale. # mentioned can be 1, or must match max_scales. Will default to batch_size if not specified. Eg. `256` or `1024,512,256`")
    parser.add_argument("--test_batch_size", type=int, default=-1)

    parser.add_argument("--lr", type=float, default=0.001, help="LR of different scales")
    parser.add_argument("--lr_per_scale", type=str, default=None, help="LR of different scales; if not specified, will default to `lr")
    parser.add_argument("--lr_warmup_iters", type=int, default=1000)
    parser.add_argument('--lr_gamma', type=float, default=0.999)
    parser.add_argument('--lr_scheduler', type=str, choices=["plateau", "step", "multiplicative"], default="plateau")
    parser.add_argument('--plateau_factor', type=float, default=0.1)
    parser.add_argument('--plateau_patience', type=int, default=4)
    parser.add_argument('--plateau_threshold', type=float, default=0.0001)
    parser.add_argument('--plateau_threshold_mode', type=str, choices=["abs", "rel"], default="abs")
    parser.add_argument('--lr_step', type=int, default=10, help="Not valid for plateau or multiplicative")
    parser.add_argument('--min_lr', type=float, default=1.01e-8, help="Min LR")
    parser.add_argument('--min_lr_max_iters', type=int, default=100, help="Max iters to run at min_lr")

    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=100.0, help="Max norm of gradients")
    parser.add_argument("--grad_norm_patience", type=int, default=10, help="Max norm of gradients")

    # Regularizations
    parser.add_argument('--kinetic-energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--jacobian-norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--div_samples',type=int, default=1)
    parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])

    # Distributed training
    parser.add_argument('--distributed', action='store_true', help='Run distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')

    parser.add_argument("--resume", type=str, default=None, help='path to saved check point')
    parser.add_argument("--ckpt_to_load", type=str, nargs='?', default="", help='path to saved check point to load but not resume training from.')
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--save_freq_within_epoch", type=int, default=0, help="(>=0) Number of ITERATIONS(!) within an epoch in which to save model, calc metrics, visualize samples")
    parser.add_argument('--disable_viz', action='store_true', help="Disable viz")
    parser.add_argument("--plot_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--vis_n_images", type=int, default=100)
    parser.add_argument('--disable_cuda', action='store_true')

    parser.add_argument('--inference', type=eval, default=False, choices=[True, False])

    parser.add_argument('--disable_date', action='store_true')
    parser.add_argument('--copy_scripts', type=eval, default=True, choices=[True, False], help="Copy this and other scripts to save directory.")
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('-f', help="DUMMY arg for Jupyter")

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args(args=[])

    args.command = 'python ' + ' '.join(sys.argv)
    args.conv = True
    args.im_ch = 1 if args.data == 'mnist' else 3
    if args.inference:
        args.copy_scripts = False

    assert args.steer_b < args.time_length

    args.imagenet_classes = list(map(int, args.imagenet_classes.split(","))) if len(args.imagenet_classes) > 0 else []

    if args.data == 'mnist':
        args.alpha = 1e-6
    else:
        args.alpha = 0.05

    if not args.disable_date:
        args.save_path = os.path.join(os.path.dirname(args.save_path), f'{datetime.datetime.now():%Y%m%d_%H%M%S}_{os.path.basename(args.save_path)}')

    args.num_blocks = [int(args.num_blocks)] * args.max_scales if ',' not in args.num_blocks else list(map(int, args.num_blocks.split(",")))

    d, dl, st = args.dims.split(',')[0], len(args.dims.split(',')), args.strides.split(',')[0]
    args.save_path = f'{args.save_path}_M{args.mode[0]}_b{args.nbits}_sc{args.max_scales}_{args.scale}_d{d}_{dl}_st{st}_bl' + (f"{args.num_blocks}" if ',' not in args.num_blocks else "_".join(args.num_blocks.split(",")))
    args.save_path += f'_S{args.solver[0]+args.solver[-1]}_{args.optimizer}_ke{args.kinetic_energy}_jf{args.jacobian_norm2}_st{args.steer_b}_n{str(args.add_noise)[0]}_GN{args.max_grad_norm}'

    args.save_path += f'_nres{args.normal_resolution}'
    if args.std_scale:
        args.save_path += f"std"

    if args.joint:
        args.save_path += f'_j{str(args.joint)[0]}_e{args.num_epochs}_bs{args.batch_size}_lr{args.lr}'
        if args.test_batch_size == -1:
            args.test_batch_size = args.batch_size
    else:
        # epochs
        if args.epochs_per_scale is None:
            args.save_path += f'_j{str(args.joint)[0]}_ep{int(args.num_epochs / args.max_scales)}'
            args.epochs_per_scale = [int(args.num_epochs / args.max_scales)] * args.max_scales
        else:
            args.save_path += f'_j{str(args.joint)[0]}_es{"_".join(args.epochs_per_scale.split(","))}'
            args.epochs_per_scale = [int(args.epochs_per_scale)] * args.max_scales if ',' not in args.epochs_per_scale else list(map(int, args.epochs_per_scale.split(",")))
            assert len(args.epochs_per_scale) == args.max_scales, f"Specify 1 or max_scales # of epochs_per_scale! Given {args.epochs_per_scale}, max_scales {args.max_scales}"
        args.num_epochs = sum(args.epochs_per_scale)

        # batch size
        if args.batch_size_per_scale is None:
            args.save_path += f'_bs{args.batch_size}'
            args.batch_size_per_scale = [args.batch_size] * args.max_scales
        else:
            args.save_path += f'_bs{"_".join(args.batch_size_per_scale.split(","))}'
            args.batch_size_per_scale = [int(args.batch_size_per_scale)] * args.max_scales if ',' not in args.batch_size_per_scale else list(map(int, args.batch_size_per_scale.split(",")))
            assert len(args.batch_size_per_scale) == args.max_scales, f"Specify 1 or max_scales # of batch_size_per_scale! Given {args.batch_size_per_scale}, max_scales {args.max_scales}"

        if args.test_batch_size == -1:
            args.test_batch_size = min(args.batch_size_per_scale)

        # LR
        if args.lr_per_scale is None:
            args.save_path += f'_lr{args.lr}'
            args.lr_per_scale = [args.lr] * args.max_scales
        else:
            # args.save_path += f'_lr{"_".join(args.lr_per_scale.split(","))}'
            args.lr_per_scale = [float(args.lr_per_scale)] * args.max_scales if ',' not in args.lr_per_scale else list(map(float, args.lr_per_scale.split(",")))
            assert len(args.lr_per_scale) == args.max_scales, f"Specify 1 or max_scales # of lr_per_scale! Given {args.lr_per_scale}, max_scales {args.max_scales}"

    # ckpt_to_load
    if args.ckpt_to_load is not "" and args.ckpt_to_load is not None:
        args.resume = None

    return args


class MSFlow():

    def __init__(self, args=None, train_im_dataset=None):

        if args is None:
            self.args = get_args()
        else:
            self.args = args

        self.train_im_dataset = train_im_dataset

        torch.manual_seed(self.args.seed)

        # Get device
        self.args.device = "cuda:%d"%torch.cuda.current_device() if torch.cuda.is_available() and not args.disable_cuda else "cpu"
        self.device = torch.device(self.args.device)
        self.cuda = self.device != torch.device('cpu')
        self.cvt = lambda x: x.type(torch.float32).to(self.device, non_blocking=True)

        # Build model
        self.model = CNFMultiscale(**vars(args),
                                   regs=argparse.Namespace(kinetic_energy=args.kinetic_energy,
                                                           jacobian_norm2=args.jacobian_norm2))
        self.image_shapes = self.model.image_shapes
        self.input_shapes = self.model.input_shapes
        if self.args.mode == '1d' or self.args.mode == '2d' or 'wavelet' in self.args.mode:
            self.z_stds = self.model.z_stds
        self.num_scales = self.model.num_scales
        for cnf in self.model.scale_models:
            set_cnf_options(self.args, cnf)

        # if self.args.mode == 'wavelet':
        #     self.wavelet_shapes = self.model.wavelet_tx.wavelet_shapes

        # Distributed model
        if self.args.distributed:
            torch.cuda.set_device(self.args.local_rank)
            distributed.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url, world_size=dist_utils.env_world_size(), rank=dist_utils.env_rank())
            assert(dist_utils.env_world_size() == distributed.get_world_size())
            self.model = self.model.cuda()
            self.model = dist_utils.DDP(self.model,
                                   device_ids=[self.args.local_rank],
                                   output_device=self.args.local_rank)

        # Model to device, set to scale
        else:
            self.model = self.model.to(self.device)

        # Load (possibly partial) ckpt
        if self.args.ckpt_to_load:
            print(f"Loading weights from {self.args.ckpt_to_load}")
            assert os.path.exists(self.args.ckpt_to_load), f"ckpt_to_load does not exist! Given {self.args.ckpt_to_load}"
            ckpt = torch.load(self.args.ckpt_to_load, map_location=self.device)
            self.model.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            # If save_path exists, then resume from it
            if os.path.exists(self.args.save_path) and self.args.resume is None:
                self.args.resume = os.path.join(self.args.save_path, 'checkpoints', 'ckpt.pth')

        if not self.args.joint:
            # Turn off updates for parameters in other scale models
            if self.args.distributed:
                self.model.module.scale = self.args.scale
            else:
                self.model.scale = self.args.scale

        # Optimizer
        self.define_optimizer()

        # Meters
        self.init_meters()

        # Other variables
        if not self.args.resume:
            self.itr = 0
            self.begin_batch = 0
            self.train_time_total = 0.
            self.best_train_loss = float("inf")
            self.best_val_loss = float("inf")
            self.scale = self.args.scale
            self.begin_epoch = 1 if (self.scale == 0 or self.args.joint) else np.cumsum(self.args.epochs_per_scale[:self.scale])[-1] + 1

        # Restore parameters
        else:
            print(f"RESUMING {self.args.resume}")
            self.args.save_path = os.path.dirname(os.path.dirname(self.args.resume))
            checkpt = torch.load(self.args.resume, map_location=self.device)
            # Model
            self.model.load_state_dict(checkpt["state_dict"], strict=False)
            # self.load_my_state_dict(checkpt["state_dict"])
            # Optimizer
            if "optim_state_dict" in checkpt.keys():
                self.optimizer.load_state_dict(checkpt["optim_state_dict"])
                # Manually move optimizer state to device.
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = self.cvt(v)
            # Scale
            self.scale = checkpt['scale']
            if not self.args.joint:
                print(f"Setting to Scale {checkpt['scale']}")
                # Turn off updates for parameters in other scale models
                if self.args.distributed:
                    self.model.module.scale = checkpt["scale"]
                else:
                    self.model.scale = checkpt["scale"]
            # Fixed_z
            self.fixed_z = checkpt['fixed_z']
            self.fixed_strict_z = checkpt['fixed_strict_z']
            # Epoch
            try:
                self.begin_epoch = checkpt['epoch']
            except:
                self.begin_epoch = np.cumsum(self.args.epochs_per_scale[:self.scale])[-1] + 1
            # Logs
            chkdir = os.path.join(os.path.dirname(self.args.resume), "../")
            trdf = pd.read_csv(os.path.join(chkdir, 'train_log.csv'))
            try:
                self.itr = checkpt['itr'] + 1
            except:
                self.itr = trdf['itr'].to_numpy()[-1]
            try:
                self.begin_batch = checkpt['batch'] + 1
            except:
                self.begin_batch = trdf['batch'].to_numpy()[-1]
            try:
                self.train_time_total = checkpt['train_time']
            except:
                self.train_time_total = trdf['train_time'].to_numpy()[-1]
            tedf = pd.read_csv(os.path.join(chkdir, 'test_log.csv'))
            self.best_train_loss = float("inf")
            self.best_val_loss = tedf['val_loss'].min()
            # self.lr_meter.update(checkpt['lr_meter_val'], epoch=self.begin_epoch-1)
            loaded = self.load_meters()
            if not loaded:
                try:
                    self.lr_meter.update(checkpt['lr_meter_val'], epoch=self.begin_epoch-1)
                except:
                    self.lr_meter.update(self.args.lr, epoch=self.begin_epoch-1)
            # Print
            print(f"Scale {self.model.scale}, Epoch {self.begin_epoch}, Batch {self.begin_batch}, Itr {self.itr}, train time {self.train_time_total}, best val loss {self.best_val_loss}")

        # Only want master rank logging
        is_master = (not self.args.distributed) or (dist_utils.env_rank()==0)
        is_rank0 = self.args.local_rank == 0
        self.write_log = is_rank0 and is_master

        # Dirs, scripts
        if os.path.exists(self.args.save_path):
            # self.args.inference = True
            self.args.copy_scripts = False
        else:
            self.args.inference = False
            print(f"Making dir {self.args.save_path}")
            utils.makedirs(self.args.save_path)
            if args.copy_scripts: utils.copy_scripts(os.path.dirname(os.path.abspath(__file__)), self.args.save_path)
            utils.makedirs(os.path.join(self.args.save_path, "checkpoints"))
            utils.makedirs(os.path.join(self.args.save_path, "samples"))
            # utils.makedirs(os.path.join(self.args.save_path, "samples","temp0.9"))
            # utils.makedirs(os.path.join(self.args.save_path, "samples","temp0.8"))
            # utils.makedirs(os.path.join(self.args.save_path, "samples","temp0.7"))
            utils.makedirs(os.path.join(self.args.save_path, "plots"))

            # Args
            with open(os.path.join(self.args.save_path, 'args.yaml'), 'w') as f:
                yaml.dump(vars(self.args), f, default_flow_style=False)

        if self.write_log:
            self.init_logg()

    def find_moving_avg(self, vals, momentum=0.99):
        avg = vals[0]
        for val in vals[1:]:
            avg = avg * momentum + val * (1 - momentum)
        return avg

    def update_meter(self, meter, vals):
        meter.vals = vals
        meter.val = vals[-1]
        meter.avg = self.find_moving_avg(vals, meter.momentum)
        return meter

    # Meters
    def init_meters(self):
        self.lr_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.elapsed_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        # Train
        self.itr_time_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.train_time_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.loss_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.nll_loss_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.bpd_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.reg_loss_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.nfe_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.grad_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        # bpd
        self.bpd_mean_dict_meters = {}
        self.bpd_std_dict_meters = {}
        # logpz
        self.logpz_mean_dict_meters = {}
        self.logpz_std_dict_meters = {}
        # deltalogp
        self.deltalogp_mean_dict_meters = {}
        self.deltalogp_std_dict_meters = {}
        for sc in range(self.args.max_scales):
            # bpd
            self.bpd_mean_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            self.bpd_std_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            # logpz
            self.logpz_mean_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            self.logpz_std_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            # deltalogp
            self.deltalogp_mean_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            self.deltalogp_std_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
        # Val
        self.val_time_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.val_loss_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.val_bpd_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.val_nfe_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        # bpd
        self.val_bpd_mean_dict_meters = {}
        self.val_bpd_std_dict_meters = {}
        # logpz
        self.val_logpz_mean_dict_meters = {}
        self.val_logpz_std_dict_meters = {}
        # deltalogp
        self.val_deltalogp_mean_dict_meters = {}
        self.val_deltalogp_std_dict_meters = {}
        for sc in range(self.args.max_scales):
            # bpd
            self.val_bpd_mean_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            self.val_bpd_std_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            # logpz
            self.val_logpz_mean_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            self.val_logpz_std_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            # deltalogp
            self.val_deltalogp_mean_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            self.val_deltalogp_std_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
        # Noisy Val
        self.noisy_val_loss_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.noisy_val_bpd_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.noisy_val_nfe_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        # bpd
        self.noisy_val_bpd_mean_dict_meters = {}
        self.noisy_val_bpd_std_dict_meters = {}
        # logpz
        self.noisy_val_logpz_mean_dict_meters = {}
        self.noisy_val_logpz_std_dict_meters = {}
        # deltalogp
        self.noisy_val_deltalogp_mean_dict_meters = {}
        self.noisy_val_deltalogp_std_dict_meters = {}
        for sc in range(self.args.max_scales):
            # bpd
            self.noisy_val_bpd_mean_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            self.noisy_val_bpd_std_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            # logpz
            self.noisy_val_logpz_mean_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            self.noisy_val_logpz_std_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            # deltalogp
            self.noisy_val_deltalogp_mean_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
            self.noisy_val_deltalogp_std_dict_meters[sc] = utils.RunningAverageMeter(0.97, save_seq=True)
        # IS, FID
        self.isc_mean_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.isc_std_meter = utils.RunningAverageMeter(0.97, save_seq=True)
        self.fid_meter = utils.RunningAverageMeter(0.97, save_seq=True)

    def save_meters(self):
        meters_pkl = os.path.join(self.args.save_path, 'meters.pkl')
        with open(meters_pkl, "wb") as f:
            pickle.dump({
                'lr_meter': self.lr_meter,
                'elapsed_meter': self.elapsed_meter,
                'itr_time_meter': self.itr_time_meter,
                'train_time_meter': self.train_time_meter,
                'loss_meter': self.loss_meter,
                'nll_loss_meter': self.nll_loss_meter,
                'bpd_meter': self.bpd_meter,
                'reg_loss_meter': self.reg_loss_meter,
                'nfe_meter': self.nfe_meter,
                'grad_meter': self.grad_meter,
                'bpd_mean_dict_meters': self.bpd_mean_dict_meters,
                'bpd_std_dict_meters': self.bpd_std_dict_meters,
                'logpz_mean_dict_meters': self.logpz_mean_dict_meters,
                'logpz_std_dict_meters': self.logpz_std_dict_meters,
                'deltalogp_mean_dict_meters': self.deltalogp_mean_dict_meters,
                'deltalogp_std_dict_meters': self.deltalogp_std_dict_meters,
                'val_time_meter': self.val_time_meter,
                'val_loss_meter': self.val_loss_meter,
                'val_bpd_meter': self.val_bpd_meter,
                'val_nfe_meter': self.val_nfe_meter,
                'val_bpd_mean_dict_meters': self.val_bpd_mean_dict_meters,
                'val_bpd_std_dict_meters': self.val_bpd_std_dict_meters,
                'val_logpz_mean_dict_meters': self.val_logpz_mean_dict_meters,
                'val_logpz_std_dict_meters': self.val_logpz_std_dict_meters,
                'val_deltalogp_mean_dict_meters': self.val_deltalogp_mean_dict_meters,
                'val_deltalogp_std_dict_meters': self.val_deltalogp_std_dict_meters,
                'noisy_val_loss_meter': self.noisy_val_loss_meter,
                'noisy_val_bpd_meter': self.noisy_val_bpd_meter,
                'noisy_val_nfe_meter': self.noisy_val_nfe_meter,
                'noisy_val_bpd_mean_dict_meters': self.noisy_val_bpd_mean_dict_meters,
                'noisy_val_bpd_std_dict_meters': self.noisy_val_bpd_std_dict_meters,
                'noisy_val_logpz_mean_dict_meters': self.noisy_val_logpz_mean_dict_meters,
                'noisy_val_logpz_std_dict_meters': self.noisy_val_logpz_std_dict_meters,
                'noisy_val_deltalogp_mean_dict_meters': self.noisy_val_deltalogp_mean_dict_meters,
                'noisy_val_deltalogp_std_dict_meters': self.noisy_val_deltalogp_std_dict_meters,
                'isc_mean_meter': self.isc_mean_meter,
                'isc_std_meter': self.isc_std_meter,
                'fid_meter': self.fid_meter,
                },
                f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_meters(self):
        meters_pkl = os.path.join(self.args.save_path, 'meters.pkl')
        if not os.path.exists(meters_pkl):
            print(f"{meters_pkl} does not exist! Returning.")
            return False
        with open(meters_pkl, "rb") as f:
            a = pickle.load(f)
        # Load
        self.lr_meter = a['lr_meter']
        self.elapsed_meter = a['elapsed_meter']
        self.itr_time_meter = a['itr_time_meter']
        self.train_time_meter = a['train_time_meter']
        self.loss_meter = a['loss_meter']
        self.nll_loss_meter = a['nll_loss_meter']
        self.bpd_meter = a['bpd_meter']
        self.reg_loss_meter = a['reg_loss_meter']
        self.nfe_meter = a['nfe_meter']
        self.grad_meter = a['grad_meter']
        self.bpd_mean_dict_meters = a['bpd_mean_dict_meters']
        self.bpd_std_dict_meters = a['bpd_std_dict_meters']
        self.logpz_mean_dict_meters = a['logpz_mean_dict_meters']
        self.logpz_std_dict_meters = a['logpz_std_dict_meters']
        self.deltalogp_mean_dict_meters = a['deltalogp_mean_dict_meters']
        self.deltalogp_std_dict_meters = a['deltalogp_std_dict_meters']
        self.val_time_meter = a['val_time_meter']
        self.val_loss_meter = a['val_loss_meter']
        self.val_bpd_meter = a['val_bpd_meter']
        self.val_nfe_meter = a['val_nfe_meter']
        self.val_bpd_mean_dict_meters = a['val_bpd_mean_dict_meters']
        self.val_bpd_std_dict_meters = a['val_bpd_std_dict_meters']
        self.val_logpz_mean_dict_meters = a['val_logpz_mean_dict_meters']
        self.val_logpz_std_dict_meters = a['val_logpz_std_dict_meters']
        self.val_deltalogp_mean_dict_meters = a['val_deltalogp_mean_dict_meters']
        self.val_deltalogp_std_dict_meters = a['val_deltalogp_std_dict_meters']
        self.noisy_val_loss_meter = a['noisy_val_loss_meter']
        self.noisy_val_bpd_meter = a['noisy_val_bpd_meter']
        self.noisy_val_nfe_meter = a['noisy_val_nfe_meter']
        self.noisy_val_bpd_mean_dict_meters = a['noisy_val_bpd_mean_dict_meters']
        self.noisy_val_bpd_std_dict_meters = a['noisy_val_bpd_std_dict_meters']
        self.noisy_val_logpz_mean_dict_meters = a['noisy_val_logpz_mean_dict_meters']
        self.noisy_val_logpz_std_dict_meters = a['noisy_val_logpz_std_dict_meters']
        self.noisy_val_deltalogp_mean_dict_meters = a['noisy_val_deltalogp_mean_dict_meters']
        self.noisy_val_deltalogp_std_dict_meters = a['noisy_val_deltalogp_std_dict_meters']
        self.isc_mean_meter = a['isc_mean_meter']
        self.isc_std_meter = a['isc_std_meter']
        self.fid_meter = a['fid_meter']
        return True

    # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113
    def load_my_state_dict(self, state_dict):
        own_state = self.model.state_dict()
        for name, param in state_dict.items():
            print(name)
            if name not in own_state:
                print("continue")
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def define_optimizer(self):

        # Optimizer
        lr = self.args.lr if self.args.joint else self.args.lr_per_scale[self.model.module.scale if self.args.distributed else self.model.scale]
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=self.args.weight_decay, momentum=0.9, nesterov=False)

        # Scheduler
        if self.args.lr_scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.args.plateau_factor, patience=self.args.plateau_patience//self.args.val_freq, verbose=True, threshold=self.args.plateau_threshold, threshold_mode=self.args.plateau_threshold_mode)
        elif self.args.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.args.lr_step, self.args.lr_gamma, verbose=True)
        elif self.args.lr_scheduler == "multiplicative":
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: self.args.lr_gamma, verbose=True)

    def lr_warmup_factor(self):
        return min(float(self.itr + 1) / max(self.args.lr_warmup_iters, 1), 1.0)

    def update_lr(self, opt=None, final_lr=None, gamma=True):
        if opt is None:
            opt = self.optimizer
        if self.itr < self.args.lr_warmup_iters:
            if final_lr is None:
                final_lr = self.args.lr if self.args.joint else self.args.lr_per_scale[self.scale]
            lr = final_lr * self.lr_warmup_factor()
            for param_group in opt.param_groups:
                param_group["lr"] = lr
        elif gamma:
            if self.itr % len(self.train_loader) == 0:
                for param_group in opt.param_groups:
                    param_group["lr"] = param_group["lr"] * self.args.lr_gamma

    def update_scale(self, new_scale):

        self.save_model(os.path.join(args.save_path, "checkpoints", f"ckpt_scale{self.scale}.pth"))

        self.scale = new_scale

        if self.write_log:
            curr_time_str, elapsed = self.get_time()
            self.logger.info(f'\n{curr_time_str} | {elapsed} | SCALE UP: Setting to Scale {new_scale} : {self.image_shapes[new_scale]}\n')
            self.elapsed_meter.reset()
            self.itr_time_meter.reset()
            self.lr_meter.reset()
            # Train
            self.loss_meter.reset()
            self.nll_loss_meter.reset()
            self.bpd_meter.reset()
            self.reg_loss_meter.reset()
            self.nfe_meter.reset()
            self.grad_meter.reset()
            for sc in range(self.args.max_scales):
                # bpd
                self.bpd_mean_dict_meters[sc].reset()
                self.bpd_std_dict_meters[sc].reset()
                # logpz
                self.logpz_mean_dict_meters[sc].reset()
                self.logpz_std_dict_meters[sc].reset()
                # deltalogp
                self.deltalogp_mean_dict_meters[sc].reset()
                self.deltalogp_std_dict_meters[sc].reset()
            # Val
            self.val_time_meter.reset()
            self.val_loss_meter.reset()
            self.val_bpd_meter.reset()
            self.val_nfe_meter.reset()
            for sc in range(self.args.max_scales):
                # bpd
                self.val_bpd_mean_dict_meters[sc].reset()
                self.val_bpd_std_dict_meters[sc].reset()
                # logpz
                self.val_logpz_mean_dict_meters[sc].reset()
                self.val_logpz_std_dict_meters[sc].reset()
                # deltalogp
                self.val_deltalogp_mean_dict_meters[sc].reset()
                self.val_deltalogp_std_dict_meters[sc].reset()
            # Noisy Val
            self.noisy_val_loss_meter.reset()
            self.noisy_val_bpd_meter.reset()
            self.noisy_val_nfe_meter.reset()
            # bpd
            for sc in range(self.args.max_scales):
                self.noisy_val_bpd_mean_dict_meters[sc].reset()
                self.noisy_val_bpd_std_dict_meters[sc].reset()
            # logpz
            for sc in range(self.args.max_scales):
                self.noisy_val_logpz_mean_dict_meters[sc].reset()
                self.noisy_val_logpz_std_dict_meters[sc].reset()
            # deltalogp
            for sc in range(self.args.max_scales):
                self.noisy_val_deltalogp_mean_dict_meters[sc].reset()
                self.noisy_val_deltalogp_std_dict_meters[sc].reset()
            # IS, FID
            self.isc_mean_meter.reset()
            self.isc_std_meter.reset()
            self.fid_meter.reset()

        # Turn off updates for parameters in other scale models
        if self.args.distributed:
            self.model.module.scale = new_scale
        else:
            self.model.scale = new_scale

        # Loaders
        self.train_loader = self.train_loaders[new_scale]
        self.test_loader = self.test_loaders[new_scale]
        self.batches_in_epoch = len(self.train_loader)

        # Reset optimizer
        self.define_optimizer()

        # Fixed images for noise
        self.fixed_images_for_noise()

        # Reset itr
        self.itr = 0
        self.min_lr_counter = 0

        self.scale_change_epochs.append(self.epoch - 1)

    def save_model(self, save_path):
        if self.args.local_rank == 0:
            if self.write_log:
                curr_time_str, elapsed = self.get_time()
                self.logger.info(f"{curr_time_str} | {elapsed} | Saving model {save_path}")
            torch.save({
                "epoch": self.epoch,
                "batch": self.batch,
                "itr": self.itr,
                "scale": self.scale,
                "state_dict": self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                "lr_meter_val": self.lr_meter.val,
                "fixed_z": self.fixed_z,
                "fixed_strict_z": self.fixed_strict_z,
                "train_time": self.train_time_total
            }, save_path)

    def compute_loss(self, imgs, noisy=True):

        logpx, reg_states, bpd_dict, z_dict, logpz_dict, deltalogp_dict = self.model(imgs, noisy=noisy)  # run model forward

        if self.args.joint:
            dim = imgs.nelement()/len(imgs)
        else:
            dim = np.prod(self.image_shapes[self.model.module.scale if self.args.distributed else self.model.scale])
        # bpd = -(logpx/dim - np.log(2**self.args.nbits)) / np.log(2)
        bpd = logpx_to_bpd(logpx, dim, self.args.nbits)
        loss = bpd.mean()

        if torch.isnan(loss):
            if self.write_log: self.logger.info('ValueError: model returned nan during training')
            raise ValueError('model returned nan during training')
        elif torch.isinf(loss):
            if self.write_log: self.logger.info('ValueError: model returned inf during training')
            raise ValueError('model returned inf during training')

        reg_coeffs = self.model.module.regularization_coeffs if self.args.distributed else self.model.regularization_coeffs
        if reg_coeffs and len(reg_states):
            reg_loss = torch.stack([reg_state * coeff for reg_state, coeff in zip(reg_states, reg_coeffs)]).sum()
            loss = loss + reg_loss
        else:
            reg_loss = torch.tensor(0., device=self.device)

        return loss, bpd, reg_loss, reg_states, bpd_dict, logpz_dict, deltalogp_dict

    def fixed_images_for_noise(self):
        # Fixed x for z
        for (self.fixed_train_imgs, _) in self.train_loader:
            break
        for (self.fixed_val_imgs, _) in self.test_loader:
            break
        # Save train images
        nb = int(np.ceil(np.sqrt(float(self.fixed_train_imgs.size(0)))))
        fixed_train_imgs_resized = resize(self.fixed_train_imgs.float()/255, self.image_shapes[-1][-2:], interp)
        vutils.save_image(fixed_train_imgs_resized, os.path.join(self.args.save_path, "samples", f"noise_train_fixed_scale{self.scale}.png"), nrow=nb)
        # Save val images
        nb = int(np.ceil(np.sqrt(float(self.fixed_val_imgs.size(0)))))
        fixed_val_imgs_resized = resize(self.fixed_val_imgs.float()/255, self.image_shapes[-1][-2:], interp)
        vutils.save_image(fixed_val_imgs_resized, os.path.join(self.args.save_path, "samples", f"noise_val_fixed_scale{self.scale}.png"), nrow=nb)

    def _set_req_grad(self, module, value):
        for p in module.parameters():
            p.requires_grad = value

    def train(self, train_loaders, test_loaders, train_im_dataset=None):

        self.train_loaders = train_loaders
        self.test_loaders = test_loaders

        if self.args.joint:
            self.train_loader = self.train_loaders[0]
            self.test_loader = self.test_loaders[0]
        else:
            self.train_loader = self.train_loaders[self.args.scale]
            self.test_loader = self.test_loaders[self.args.scale]

        self.batches_in_epoch = len(self.train_loader)

        # Fixed images for noise
        self.fixed_images_for_noise()

        # Sync machines before training
        if self.args.distributed:
            if self.write_log: self.logger.info("Syncing machines before training")
            dist_utils.sum_tensor(torch.tensor([1.0]).float().cuda())

        if self.write_log:
            mem = torch.cuda.memory_allocated() / 10**9 if self.device != torch.device('cpu') else 0.0
            if self.write_log: self.logger.info(f"GPU Mem before train start: {mem:.3g} GB")
            self.start_time = time.time()

        self.min_lr_counter = 0
        self.lr_change_epochs = []
        self.scale_change_epochs = []
        self.lrs = []
        self.skip_epochs = 0

        if self.write_log:
            curr_time_str, elapsed = self.get_time()
            self.logger.info(f"EXPERIMENT {self.args.save_path}")
            self.logger.info(f'\n{curr_time_str} | {elapsed} | Starting at scale {self.scale if not self.args.joint else -1} : {self.image_shapes[self.scale if not self.args.joint else -1]}')

        for self.epoch in range(self.begin_epoch, self.args.num_epochs + 1):

            if self.epoch + self.skip_epochs > self.args.num_epochs:
                break

            # Check for new scale
            if not self.args.joint:
                new_scale = int(np.sum(self.epoch + self.skip_epochs > np.cumsum(self.args.epochs_per_scale)))
                if new_scale >= self.num_scales:
                    break
                if new_scale > self.scale:
                    self.update_scale(new_scale)

            self.model.train()

            for self.batch, (imgs, _) in enumerate(self.train_loader):

                if self.batch < self.begin_batch:
                    continue

                if self.write_log:
                    start = time.time()

                self.optimizer.zero_grad()

                self.update_lr()
                self.lr_meter.update(self.optimizer.param_groups[0]['lr'], self.epoch - 1 + (self.batch)/len(self.train_loader))

                # FFJORD Loss
                self.imgs = imgs.clone()
                loss, bpd, reg_loss, reg_states, bpd_dict, logpz_dict, deltalogp_dict = self.compute_loss(self.imgs, noisy=args.add_noise)
                loss.backward()

                mem = torch.cuda.memory_allocated() / 10**9 if self.device != torch.device('cpu') else 0.0

                # Optimize
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                # Only optimize if the grad_norm is less than 5*max_grad_norm
                if grad_norm < 5*self.args.max_grad_norm:
                    self.optimizer.step()
                    self.high_grad_norm = 0
                else:
                    self.high_grad_norm += 1

                # Accumulate from distributed training
                batch_size = self.imgs.size(0)
                nfe_opt = count_nfe(self.model)
                metrics = torch.tensor([1., mem, batch_size, loss.item() + loss_recon.item(), bpd.mean().item(), reg_loss.item(), nfe_opt, grad_norm, *reg_states]).float().to(self.device)
                # if not self.args.joint:
                self.rv = reg_states

                if self.args.distributed:
                    total_gpus, self.r_mem, batch_total, r_loss, r_bpd, r_reg_loss, r_nfe, r_grad_norm, *self.rv = dist_utils.sum_tensor(metrics).cpu().numpy()
                else:
                    total_gpus, self.r_mem, batch_total, r_loss, r_bpd, r_reg_loss, r_nfe, r_grad_norm, *self.rv = metrics.cpu().numpy()

                # Log
                if self.write_log:
                    itr_time = time.time() - start
                    self.train_time_total += itr_time
                    self.itr_time_meter.update(itr_time)
                    self.loss_meter.update(r_loss/total_gpus, self.epoch - 1 + (self.batch + 1)/len(self.train_loader))
                    self.bpd_meter.update(r_bpd/total_gpus)
                    self.reg_loss_meter.update(r_reg_loss/total_gpus)
                    self.nfe_meter.update(r_nfe/total_gpus)
                    self.grad_meter.update(r_grad_norm/total_gpus)
                    for sc in bpd_dict.keys():
                        self.bpd_mean_dict_meters[sc].update(bpd_dict[sc].mean().item(), self.epoch - 1 + (self.batch + 1)/len(self.train_loader))
                        self.bpd_std_dict_meters[sc].update(bpd_dict[sc].std().item())
                    for sc in logpz_dict.keys():
                        self.logpz_mean_dict_meters[sc].update(logpz_dict[sc].mean().item(), self.epoch - 1 + (self.batch + 1)/len(self.train_loader))
                        self.logpz_std_dict_meters[sc].update(logpz_dict[sc].std().item())
                    for sc in deltalogp_dict.keys():
                        self.deltalogp_mean_dict_meters[sc].update(deltalogp_dict[sc].mean().item(), self.epoch - 1 + (self.batch + 1)/len(self.train_loader))
                        self.deltalogp_std_dict_meters[sc].update(deltalogp_dict[sc].std().item())
                    self.logg(mode='train', total_gpus=total_gpus)

                del loss, bpd, reg_loss, reg_states, self.imgs

                self.itr += 1

                # Min lr counter
                if self.lr_meter.val <= self.args.min_lr:
                    self.min_lr_counter += 1
                # If min lr was for min_lr_max_iters epochs, break
                if self.min_lr_counter >= self.args.min_lr_max_iters:
                    if self.write_log:
                        curr_time_str, elapsed = self.get_time()
                        self.logger.info(f"{curr_time_str} | {elapsed} | min_lr <= {self.args.min_lr} lasted for {self.min_lr_counter} iterations, breaking!\n")
                    break

                # Save model within an epoch
                if self.args.save_freq_within_epoch > 0:
                    if self.itr % self.args.save_freq_within_epoch == 0:
                        # Time
                        if self.write_log:
                            self.train_time_meter.update(convert_time_stamp_to_hrs(str(datetime.timedelta(seconds=self.train_time_total))), self.epoch - 1 + (self.batch + 1)/len(self.train_loader))
                            curr_time = time.time()
                            elapsed = str(datetime.timedelta(seconds=(curr_time - self.start_time)))
                            self.elapsed_meter.update(convert_time_stamp_to_hrs(elapsed))
                        # Save model
                        if grad_norm < 5*self.args.max_grad_norm:
                            if self.write_log:
                                curr_time_str, elapsed = self.get_time()
                                self.logger.info(f"{curr_time_str} | {elapsed} | WITHIN EPOCH: Saving model ckpt.pth")
                            self.save_model(os.path.join(self.args.save_path, "checkpoints", "ckpt.pth"))
                        # Save best
                        loss = self.loss_meter.val
                        if loss < self.best_train_loss and self.args.local_rank==0: 
                            self.best_train_loss = loss
                            dest = os.path.join(self.args.save_path, "checkpoints", f"best_train_scale{self.scale}.pth")
                            if self.write_log:
                                curr_time_str, elapsed = self.get_time()
                                self.logger.info(f"{curr_time_str} | {elapsed} | Saving best model: {dest}")
                            shutil.copyfile(os.path.join(self.args.save_path, "checkpoints", "ckpt.pth"), dest)
                        # Visualize samples
                        if self.write_log and not self.args.disable_viz:
                            curr_time_str, elapsed = self.get_time()
                            self.logger.info(f"{curr_time_str} | {elapsed} | Scale {self.scale if not self.args.joint else -1} | Itr {self.itr:06d} | Epoch {self.epoch:04d} | Batch {self.batch}/{self.batches_in_epoch} | Visualizing samples...")
                            # Generate images
                            gen_imgs_scales, _, _ = self.model(self.fixed_z, reverse=True)
                            # Save gen images
                            nb = int(np.ceil(np.sqrt(float(self.fixed_z[0].size(0)))))
                            if not self.args.joint:
                                gen_imgs = gen_imgs_scales[self.scale].detach().cpu()
                                # gen_imgs = gen_imgs.reshape(-1, *self.image_shapes[self.scale])
                                gen_imgs = resize(gen_imgs, self.image_shapes[-1][-2:], interp)
                                vutils.save_image(gen_imgs, os.path.join(self.args.save_path, "samples",
                                    f"gen_scale{self.scale}_epoch{self.epoch - 1 + (self.batch + 1)/len(self.train_loader):09.04f}.png"), nrow=nb)
                            else:
                                for sc in sorted(gen_imgs_scales.keys()):
                                    gen_imgs = gen_imgs_scales[sc].detach().cpu()
                                    gen_imgs = resize(gen_imgs, self.image_shapes[-1][-2:], interp)
                                    vutils.save_image(gen_imgs, os.path.join(self.args.save_path, "samples",
                                        f"gen_scale{sc}_epoch{self.epoch - 1 + (self.batch + 1)/len(self.train_loader):09.04f}.png"), nrow=nb)
                            del gen_imgs_scales
                        # Plot graphs
                        if self.write_log:
                            self.save_meters()
                            curr_time_str, elapsed = self.get_time()
                            try:
                                plot_graphs_process.join()
                            except:
                                pass
                            self.logger.info(f"{curr_time_str} | {elapsed} | Scale {self.scale if not self.args.joint else -1} | Itr {self.itr:06d} | Epoch {self.epoch:04d} | Batch {self.batch}/{self.batches_in_epoch} | Plotting graphs...")
                            plot_graphs_process = Process(target=self.plot_graphs)
                            plot_graphs_process.start()
                            curr_time_str, elapsed = self.get_time()
                            self.logger.info(f"{curr_time_str} | {elapsed} | Plotting graphs DONE!\n")

                if self.high_grad_norm > self.args.grad_norm_patience:
                    break

            if self.high_grad_norm > self.args.grad_norm_patience:
                if self.args.joint:
                    if self.write_log:
                        self.logger.info(f"HIGH GRAD NORM for > patience {self.args.grad_norm_patience}!! ENDING!!\n")
                    break
                else:
                    if self.write_log:
                        self.logger.info(f"HIGH GRAD NORM for > patience {self.args.grad_norm_patience}!! SKIPPING SCALE!!\n")
                    self.skip_epochs += abs(self.epoch + self.skip_epochs - np.cumsum(self.args.epochs_per_scale)[self.scale])

            self.begin_batch = 0

            # Time
            if self.write_log:
                self.train_time_meter.update(convert_time_stamp_to_hrs(str(datetime.timedelta(seconds=self.train_time_total))), self.epoch)
                curr_time = time.time()
                elapsed = str(datetime.timedelta(seconds=(curr_time - self.start_time)))
                self.elapsed_meter.update(convert_time_stamp_to_hrs(elapsed))

            # Save
            if self.write_log:
                curr_time_str, elapsed = self.get_time()
                self.logger.info(f"{curr_time_str} | {elapsed} | AFTER EPOCH: Saving model ckpt.pth")
            self.save_model(os.path.join(self.args.save_path, "checkpoints", "ckpt.pth"))

            # Validate
            if self.epoch % self.args.val_freq == 0:

                if self.write_log:
                    curr_time_str, elapsed = self.get_time()
                    self.logger.info(f"{curr_time_str} | {elapsed} | Scale {self.scale if not self.args.joint else -1} | Epoch {self.epoch:04d} | Validating...")
                    start = time.time()

                val_metrics, val_bpd_mean_dict, val_bpd_std_dict, \
                    val_logpz_mean_dict, val_logpz_std_dict, \
                    val_deltalogp_mean_dict, val_deltalogp_std_dict, \
                    noisy_val_metrics, noisy_val_bpd_mean_dict, noisy_val_bpd_std_dict, \
                    noisy_val_logpz_mean_dict, noisy_val_logpz_std_dict, \
                    noisy_val_deltalogp_mean_dict, noisy_val_deltalogp_std_dict = self.validate(self.test_loader)

                # Accumulate from distributed training
                if self.args.distributed:
                    total_gpus, r_loss, r_bpd, r_nfe = dist_utils.sum_tensor(val_metrics).cpu().numpy()
                    noisy_total_gpus, noisy_r_loss, noisy_r_bpd, noisy_r_nfe = dist_utils.sum_tensor(noisy_val_metrics).cpu().numpy()
                else:
                    total_gpus, r_loss, r_bpd, r_nfe = val_metrics.cpu().numpy()
                    noisy_total_gpus, noisy_r_loss, noisy_r_bpd, noisy_r_nfe = noisy_val_metrics.cpu().numpy()

                # Log
                if self.write_log:
                    val_time = time.time() - start
                    self.val_time_meter.update(val_time/2)
                    self.val_loss_meter.update(r_loss/total_gpus, self.epoch)
                    self.val_bpd_meter.update(r_bpd/total_gpus)
                    self.val_nfe_meter.update(r_nfe/total_gpus)
                    # bpd
                    for sc in val_bpd_mean_dict.keys():
                        self.val_bpd_mean_dict_meters[sc].update(val_bpd_mean_dict[sc], self.epoch)
                        self.val_bpd_std_dict_meters[sc].update(val_bpd_std_dict[sc])
                    # logpz
                    for sc in val_logpz_mean_dict.keys():
                        self.val_logpz_mean_dict_meters[sc].update(val_logpz_mean_dict[sc], self.epoch)
                        self.val_logpz_std_dict_meters[sc].update(val_logpz_std_dict[sc])
                    # deltalogp
                    for sc in val_deltalogp_mean_dict.keys():
                        self.val_deltalogp_mean_dict_meters[sc].update(val_deltalogp_mean_dict[sc], self.epoch)
                        self.val_deltalogp_std_dict_meters[sc].update(val_deltalogp_std_dict[sc])
                    self.logg(mode='val', total_gpus=total_gpus)
                    # Noisy
                    self.noisy_val_loss_meter.update(noisy_r_loss/noisy_total_gpus)
                    self.noisy_val_bpd_meter.update(noisy_r_bpd/noisy_total_gpus)
                    self.noisy_val_nfe_meter.update(noisy_r_nfe/noisy_total_gpus)
                    # bpd
                    for sc in noisy_val_bpd_mean_dict.keys():
                        self.noisy_val_bpd_mean_dict_meters[sc].update(noisy_val_bpd_mean_dict[sc], self.epoch)
                        self.noisy_val_bpd_std_dict_meters[sc].update(noisy_val_bpd_std_dict[sc])
                    # logpz
                    for sc in noisy_val_logpz_mean_dict.keys():
                        self.noisy_val_logpz_mean_dict_meters[sc].update(noisy_val_logpz_mean_dict[sc], self.epoch)
                        self.noisy_val_logpz_std_dict_meters[sc].update(noisy_val_logpz_std_dict[sc])
                    # deltalogp
                    for sc in noisy_val_deltalogp_mean_dict.keys():
                        self.noisy_val_deltalogp_mean_dict_meters[sc].update(noisy_val_deltalogp_mean_dict[sc], self.epoch)
                        self.noisy_val_deltalogp_std_dict_meters[sc].update(noisy_val_deltalogp_std_dict[sc])
                    self.logg(mode='noisy_val', total_gpus=noisy_total_gpus)

                del val_metrics, noisy_val_metrics

                # Save best
                loss = self.val_loss_meter.val
                if loss < self.best_val_loss and self.args.local_rank==0: 
                    self.best_val_loss = loss
                    dest = os.path.join(self.args.save_path, "checkpoints", f"best_scale{self.scale}.pth")
                    shutil.copyfile(os.path.join(self.args.save_path, "checkpoints", "ckpt.pth"), dest)
                    curr_time_str, elapsed = self.get_time()
                    self.logger.info(f"{curr_time_str} | {elapsed} | Saving best val model: {dest}")

                # Schedule
                if self.itr > self.args.lr_warmup_iters:
                    if self.args.lr_scheduler == 'plateau':
                        self.scheduler.step(self.val_loss_meter.val)
                    else:
                        self.scheduler.step()

                # Record change in LR
                if self.optimizer.param_groups[0]["lr"] == self.args.plateau_factor * self.lr_meter.val:
                    self.lr_change_epochs.append(self.epoch)
                    if self.write_log: self.logger.info(f"Reduced LR: Epoch {self.epoch}, LR {self.optimizer.param_groups[0]['lr']}")

            # Visualize samples
            if self.write_log and self.epoch % self.args.plot_freq == 0 and not self.args.disable_viz:

                curr_time_str, elapsed = self.get_time()
                self.logger.info(f"{curr_time_str} | {elapsed} | Scale {self.scale if not self.args.joint else -1} | Epoch {self.epoch:04d} | Visualizing samples...")

                # Generate images
                with torch.no_grad():
                    gen_imgs_scales, _, _ = self.model(self.fixed_z, reverse=True)

                # Save gen images
                nb = int(np.ceil(np.sqrt(float(self.fixed_z[0].size(0)))))
                if not self.args.joint:
                    gen_imgs = gen_imgs_scales[self.scale].detach().cpu()
                    # gen_imgs = gen_imgs.reshape(-1, *self.image_shapes[self.scale])
                    gen_imgs = resize(gen_imgs, self.image_shapes[-1][-2:], interp)
                    vutils.save_image(gen_imgs, os.path.join(self.args.save_path, "samples",
                        f"gen_scale{self.scale}_epoch{self.epoch:09.04f}.png" if self.args.save_freq_within_epoch > 0 else f"gen_scale{self.scale}_epoch{self.epoch:04d}.png"), nrow=nb)
                else:
                    for sc in sorted(gen_imgs_scales.keys()):
                        gen_imgs = resize(gen_imgs_scales[sc], self.image_shapes[-1][-2:], interp)
                        vutils.save_image(gen_imgs, os.path.join(self.args.save_path, "samples",
                            f"gen_scale{self.scale}_epoch{self.epoch:09.04f}.png" if self.args.save_freq_within_epoch > 0 else f"gen_scale{sc}_epoch{self.epoch:04d}.png"), nrow=nb)
                del gen_imgs_scales

                # Generate images
                with torch.no_grad():
                    # TODO: visualize figures at multiple scales
                    gen_imgs_scales, _, _ = self.model(self.fixed_strict_z, reverse=True)
                    gen_imgs = gen_imgs_scales[self.scale].detach().cpu() if not self.args.joint else gen_imgs_scales[sorted(list(gen_imgs_scales.keys()))[-1]].detach().cpu()
                    del gen_imgs_scales

                # Save gen images
                # gen_imgs = gen_imgs.reshape(-1, *self.image_shapes[self.scale])
                gen_imgs = resize(gen_imgs, self.image_shapes[-1][-2:], interp)
                vutils.save_image(gen_imgs, os.path.join(self.args.save_path, "samples", f"gen_STRICT_scale{self.scale}.png"), nrow=8)

            # Plot graphs
            if self.write_log and self.epoch % self.args.plot_freq == 0:
                self.save_meters()
                curr_time_str, elapsed = self.get_time()
                try:
                    plot_graphs_process.join()
                except:
                    pass
                self.logger.info(f"{curr_time_str} | {elapsed} | Scale {self.scale if not self.args.joint else -1} | Epoch {self.epoch:04d} | Plotting graphs...")
                plot_graphs_process = Process(target=self.plot_graphs)
                plot_graphs_process.start()
                curr_time_str, elapsed = self.get_time()
                self.logger.info(f"{curr_time_str} | {elapsed} | Plotting graphs DONE!\n")

            # If min lr was for min_lr_max_iters epochs, skip to the next scale
            if self.min_lr_counter >= self.args.min_lr_max_iters:
                if self.args.joint:
                    break
                else:
                    self.skip_epochs += abs(self.epoch + self.skip_epochs - np.cumsum(self.args.epochs_per_scale)[self.scale])

    def validate(self, val_loader):

        self.model.eval()

        loss_means, bpd_means, nfes = [], [], []
        noisy_loss_means, noisy_bpd_means, noisy_nfes = [], [], []
        bpd_mean_dict, bpd_std_dict = {}, {}
        logpz_mean_dict, logpz_std_dict = {}, {}
        deltalogp_mean_dict, deltalogp_std_dict = {}, {}
        noisy_bpd_mean_dict, noisy_bpd_std_dict = {}, {}
        noisy_logpz_mean_dict, noisy_logpz_std_dict = {}, {}
        noisy_deltalogp_mean_dict, noisy_deltalogp_std_dict = {}, {}

        def add_to_dict(my_dict, key, val):
            if key in my_dict.keys():
                my_dict[key].append(val)
            else:
                my_dict[key] = [val]

        # with torch.no_grad():

        for batch, (imgs, _) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False, desc='Validating'):

            self.imgs = imgs.clone()

            # Not noisy
            loss, bpd, _, _, bpd_dict, logpz_dict, deltalogp_dict = self.compute_loss(self.imgs, noisy=False)
            del self.imgs

            loss_means.append(loss.item())
            bpd_means.append(bpd.mean().item())
            nfes.append(count_nfe(self.model))
            # bpd
            for sc in bpd_dict.keys():
                add_to_dict(bpd_mean_dict, sc, bpd_dict[sc].mean().item())
                add_to_dict(bpd_std_dict, sc, bpd_dict[sc].std().item())
            # logpz
            for sc in logpz_dict.keys():
                add_to_dict(logpz_mean_dict, sc, logpz_dict[sc].mean().item())
                add_to_dict(logpz_std_dict, sc, logpz_dict[sc].std().item())
            # deltalogp
            for sc in deltalogp_dict.keys():
                add_to_dict(deltalogp_mean_dict, sc, deltalogp_dict[sc].mean().item())
                add_to_dict(deltalogp_std_dict, sc, deltalogp_dict[sc].std().item())
            # del
            del loss, bpd, bpd_dict, logpz_dict, deltalogp_dict

            # Noisy
            self.imgs = imgs.clone()
            noisy_loss, noisy_bpd, _, _, noisy_bpd_dict, noisy_logpz_dict, noisy_deltalogp_dict = self.compute_loss(self.imgs, noisy=True)
            del self.imgs

            noisy_loss_means.append(noisy_loss.item())
            noisy_bpd_means.append(noisy_bpd.mean().item())
            noisy_nfes.append(count_nfe(self.model))
            # bpd
            for sc in noisy_bpd_dict.keys():
                add_to_dict(noisy_bpd_mean_dict, sc, noisy_bpd_dict[sc].mean().item())
                add_to_dict(noisy_bpd_std_dict, sc, noisy_bpd_dict[sc].std().item())
            # logpz
            for sc in noisy_logpz_dict.keys():
                add_to_dict(noisy_logpz_mean_dict, sc, noisy_logpz_dict[sc].mean().item())
                add_to_dict(noisy_logpz_std_dict, sc, noisy_logpz_dict[sc].std().item())
            # deltalogp
            for sc in noisy_deltalogp_dict.keys():
                add_to_dict(noisy_deltalogp_mean_dict, sc, noisy_deltalogp_dict[sc].mean().item())
                add_to_dict(noisy_deltalogp_std_dict, sc, noisy_deltalogp_dict[sc].std().item())
            # del
            del noisy_loss, noisy_bpd, noisy_bpd_dict, noisy_logpz_dict, noisy_deltalogp_dict

        loss_mean = np.mean(loss_means)
        bpd_mean = np.mean(bpd_means)
        nfe = np.mean(nfes)
        # bpd
        for sc in bpd_mean_dict.keys():
            bpd_mean_dict[sc] = np.mean(bpd_mean_dict[sc])
            bpd_std_dict[sc] = np.mean(bpd_std_dict[sc])
        # logpz
        for sc in logpz_mean_dict.keys():
            logpz_mean_dict[sc] = np.mean(logpz_mean_dict[sc])
            logpz_std_dict[sc] = np.mean(logpz_std_dict[sc])
        # deltalogp
        for sc in deltalogp_mean_dict.keys():
            deltalogp_mean_dict[sc] = np.mean(deltalogp_mean_dict[sc])
            deltalogp_std_dict[sc] = np.mean(deltalogp_std_dict[sc])

        noisy_loss_mean = np.mean(noisy_loss_means)
        noisy_bpd_mean = np.mean(noisy_bpd_means)
        noisy_nfe = np.mean(noisy_nfes)
        # bpd
        for sc in noisy_bpd_mean_dict.keys():
            noisy_bpd_mean_dict[sc] = np.mean(noisy_bpd_mean_dict[sc])
            noisy_bpd_std_dict[sc] = np.mean(noisy_bpd_std_dict[sc])
        # logpz
        for sc in noisy_logpz_mean_dict.keys():
            noisy_logpz_mean_dict[sc] = np.mean(noisy_logpz_mean_dict[sc])
            noisy_logpz_std_dict[sc] = np.mean(noisy_logpz_std_dict[sc])
        # deltalogp
        for sc in noisy_deltalogp_mean_dict.keys():
            noisy_deltalogp_mean_dict[sc] = np.mean(noisy_deltalogp_mean_dict[sc])
            noisy_deltalogp_std_dict[sc] = np.mean(noisy_deltalogp_std_dict[sc])

        metrics = torch.tensor([1., loss_mean, bpd_mean, nfe]).float().to(self.device)
        noisy_metrics = torch.tensor([1., noisy_loss_mean, noisy_bpd_mean, noisy_nfe]).float().to(self.device)

        return metrics, bpd_mean_dict, bpd_std_dict, \
                logpz_mean_dict, logpz_std_dict, \
                deltalogp_mean_dict, deltalogp_std_dict, \
                noisy_metrics, noisy_bpd_mean_dict, noisy_bpd_std_dict, \
                noisy_logpz_mean_dict, noisy_logpz_std_dict, \
                noisy_deltalogp_mean_dict, noisy_deltalogp_std_dict

    def savefig(self, path, bbox_inches='tight', pad_inches=0.1):
        try:
            plt.savefig(path, bbox_inches=bbox_inches, pad_inches=pad_inches)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print(sys.exc_info()[0])

    def plot_graphs(self):
        # Time
        plt.plot(self.train_time_meter.epochs, self.train_time_meter.vals, label='Train time')
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.xlabel("Epochs")
        plt.ylabel("Hrs")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'time_train.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # Elapsed
        plt.plot(self.train_time_meter.epochs, self.elapsed_meter.vals, label='Elapsed')
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.xlabel("Epochs")
        plt.ylabel("Hrs")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'time_elapsed.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # LR
        plt.plot(self.lr_meter.epochs, self.lr_meter.vals, color='red', label='lr')
        for e in self.lr_change_epochs:
            plt.axvline(e, linestyle='--', color='0.8')
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.xlabel("Epochs")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'lr.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # Train loss
        plt.plot(self.loss_meter.epochs, self.loss_meter.vals, color='C0', label="train loss")
        plt.plot(self.loss_meter.epochs, self.bpd_meter.vals, '--', color='C0', label="nll_loss (bpd)")
        plt.plot(self.loss_meter.epochs, self.reg_loss_meter.vals, '--', color='C4', label="reg loss")
        for e in self.lr_change_epochs:
            plt.axvline(e, linestyle='--', color='0.8')
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.xlabel("Epochs")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'train_loss.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # Train + Val losses
        plt.plot(self.loss_meter.epochs, self.loss_meter.vals, color='C0', alpha=0.4, label="train loss")
        plt.plot(self.val_loss_meter.epochs, self.val_loss_meter.vals, color='C1', alpha=0.7, label="val loss")
        plt.xlabel("Epochs")
        for e in self.lr_change_epochs:
            plt.axvline(e, linestyle='--', color='0.8')
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'losses.png'), bbox_inches='tight', pad_inches=0.1)
        plt.plot(self.val_loss_meter.epochs, self.noisy_val_loss_meter.vals, color='C2', alpha=0.7, label="noisy val loss")
        plt.legend()
        # plt.yscale("linear")
        self.savefig(os.path.join(self.args.save_path, 'plots', 'losses_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # Val BPD
        # VAL
        x, y = self.val_loss_meter.epochs, self.val_bpd_meter.vals
        plt.plot(x, y, color='C1', alpha=0.7, label="val bpd")
        try:
            plt.scatter(x[-1], y[-1], color='C1'); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
            min_index = y.index(min(y))
            if min_index != len(y) - 1:
                plt.scatter(x[min_index], y[min_index], color='C1'); plt.text(x[min_index], y[min_index], f"{y[min_index]:.02f}")
        except:
            pass
        plt.xlabel("Epochs")
        for e in self.lr_change_epochs:
            plt.axvline(e, linestyle='--', color='0.8')
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'bpd.png'), bbox_inches='tight', pad_inches=0.1)
        if (np.array(y) < 1e-6).sum() > 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")
        self.savefig(os.path.join(self.args.save_path, 'plots', 'bpd_logy.png'), bbox_inches='tight', pad_inches=0.1)
        # NOISY VAL
        x, y = self.val_loss_meter.epochs, self.noisy_val_bpd_meter.vals
        plt.plot(x, y, color='C2', alpha=0.7, label="noisy val bpd")
        try:
            plt.scatter(x[-1], y[-1], color='C2'); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
            min_index = y.index(min(y))
            if min_index != len(y) - 1:
                plt.scatter(x[min_index], y[min_index], color='C2'); plt.text(x[min_index], y[min_index], f"{y[min_index]:.02f}")
        except:
            pass
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'bpd_wnoisy_logy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.yscale("linear")
        self.savefig(os.path.join(self.args.save_path, 'plots', 'bpd_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # Train + Val BPD
        # TRAIN
        x, y = self.loss_meter.epochs, self.bpd_meter.vals
        plt.plot(x, y, color='C0', alpha=0.4, label="train bpd")
        try:
            plt.scatter(x[-1], y[-1], color='C0'); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
            min_index = y.index(min(y))
            if min_index != len(y) - 1:
                plt.scatter(x[min_index], y[min_index], color='b'); plt.text(x[min_index], y[min_index], f"{y[min_index]:.02f}")
        except:
            pass
        # VAL
        x, y = self.val_loss_meter.epochs, self.val_bpd_meter.vals
        plt.plot(x, y, color='C1', alpha=0.7, label="val bpd")
        try:
            plt.scatter(x[-1], y[-1], color='C1'); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
            min_index = y.index(min(y))
            if min_index != len(y) - 1:
                plt.scatter(x[min_index], y[min_index], color='C1'); plt.text(x[min_index], y[min_index], f"{y[min_index]:.02f}")
        except:
            pass
        plt.xlabel("Epochs")
        for e in self.lr_change_epochs:
            plt.axvline(e, linestyle='--', color='0.8')
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'bpd_all.png'), bbox_inches='tight', pad_inches=0.1)
        if (np.array(y) < 1e-6).sum() > 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")
        self.savefig(os.path.join(self.args.save_path, 'plots', 'bpd_all_logy.png'), bbox_inches='tight', pad_inches=0.1)
        # NOISY VAL
        x, y = self.val_loss_meter.epochs, self.noisy_val_bpd_meter.vals
        plt.plot(x, y, color='C2', alpha=0.7, label="noisy val bpd")
        try:
            plt.scatter(x[-1], y[-1], color='C2'); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
            min_index = y.index(min(y))
            if min_index != len(y) - 1:
                plt.scatter(x[min_index], y[min_index], color='C2'); plt.text(x[min_index], y[min_index], f"{y[min_index]:.02f}")
        except:
            pass
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'bpd_all_wnoisy_logy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.yscale("linear")
        self.savefig(os.path.join(self.args.save_path, 'plots', 'bpd_all_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # Train + Val NFE
        plt.plot(self.loss_meter.epochs, self.nfe_meter.vals, color='C0', alpha=0.7, label="train NFE")
        plt.plot(self.val_loss_meter.epochs, self.val_nfe_meter.vals, color='C1', alpha=0.7, label="val NFE")
        plt.plot(self.val_loss_meter.epochs, self.noisy_val_nfe_meter.vals, color='C2', alpha=0.7, label="noisy val NFE")
        plt.xlabel("Epochs")
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'nfe.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # Train grad
        plt.plot(self.loss_meter.epochs, self.grad_meter.vals, color='C0', alpha=0.7, label="train Grad Norm")
        plt.xlabel("Epochs")
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'grad.png'), bbox_inches='tight', pad_inches=0.1)
        plt.yscale("log")
        self.savefig(os.path.join(self.args.save_path, 'plots', 'grad_logy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # bpd_dict
        # sym = False
        # VAL
        for sc in self.val_bpd_mean_dict_meters.keys():
            # plt.errorbar(self.val_bpd_mean_dict_meters[sc].epochs, self.val_bpd_mean_dict_meters[sc].vals, yerr=self.val_bpd_std_dict_meters[sc].vals, alpha=0.5, label=f"val sc{sc}")
            x, y, err = self.val_bpd_mean_dict_meters[sc].epochs, np.array(self.val_bpd_mean_dict_meters[sc].vals), np.array(self.val_bpd_std_dict_meters[sc].vals)
            if len(x) > 0:
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.plot(x, y, alpha=0.5, label=f"val bpd sc{sc}", c=f"C{sc+1}")
                plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0, color=f"C{sc+1}")
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        plt.xlabel("Epochs")
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_bpd.png'), bbox_inches='tight', pad_inches=0.1)
        # Noisy VAL
        for sc in self.noisy_val_bpd_mean_dict_meters.keys():
            # plt.errorbar(self.noisy_val_bpd_mean_dict_meters[sc].epochs, self.noisy_val_bpd_mean_dict_meters[sc].vals, yerr=self.noisy_val_bpd_std_dict_meters[sc].vals, alpha=0.5, label=f"noisy val sc{sc}")
            x, y, err = self.noisy_val_bpd_mean_dict_meters[sc].epochs, np.array(self.noisy_val_bpd_mean_dict_meters[sc].vals), np.array(self.noisy_val_bpd_std_dict_meters[sc].vals)
            if len(x) > 0:
                plt.plot(x, y, alpha=0.5, label=f"noisy val bpd sc{sc}", c=f"C{sc+1}", linestyle="--")
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0, color=f"C{sc+1}")
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # plt.yscale("linear")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_bpd_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # bpd_dict
        # sym = False
        # Train
        x, y, err = [], [], []
        for sc in self.bpd_mean_dict_meters.keys():
            # plt.errorbar(self.bpd_mean_dict_meters[sc].epochs, self.bpd_mean_dict_meters[sc].vals, yerr=self.bpd_std_dict_meters[sc].vals, alpha=0.5, label=f"train sc{sc}")
            x += self.bpd_mean_dict_meters[sc].epochs
            y += self.bpd_mean_dict_meters[sc].vals
            err += self.bpd_std_dict_meters[sc].vals
        y, err = np.array(y), np.array(err)
        # if (y < 1e-6).sum() > 0:
        #     sym = True
        plt.plot(x, y, alpha=0.5, label=f"train bpd")
        plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
        if len(x) > 0:
            plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # VAL
        for sc in self.val_bpd_mean_dict_meters.keys():
            # plt.errorbar(self.val_bpd_mean_dict_meters[sc].epochs, self.val_bpd_mean_dict_meters[sc].vals, yerr=self.val_bpd_std_dict_meters[sc].vals, alpha=0.5, label=f"val sc{sc}")
            x, y, err = self.val_bpd_mean_dict_meters[sc].epochs, np.array(self.val_bpd_mean_dict_meters[sc].vals), np.array(self.val_bpd_std_dict_meters[sc].vals)
            if len(x) > 0:
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.plot(x, y, alpha=0.5, label=f"val bpd sc{sc}")
                plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.xlabel("Epochs")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_bpd_all.png'), bbox_inches='tight', pad_inches=0.1)
        # Noisy VAL
        for sc in self.noisy_val_bpd_mean_dict_meters.keys():
            # plt.errorbar(self.noisy_val_bpd_mean_dict_meters[sc].epochs, self.noisy_val_bpd_mean_dict_meters[sc].vals, yerr=self.noisy_val_bpd_std_dict_meters[sc].vals, alpha=0.5, label=f"noisy val sc{sc}")
            x, y, err = self.noisy_val_bpd_mean_dict_meters[sc].epochs, np.array(self.noisy_val_bpd_mean_dict_meters[sc].vals), np.array(self.noisy_val_bpd_std_dict_meters[sc].vals)
            if len(x) > 0:
                plt.plot(x, y, alpha=0.5, label=f"noisy val bpd sc{sc}")
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # plt.yscale("linear")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_bpd_all_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # VAL
        for sc in self.val_bpd_mean_dict_meters.keys():
            # plt.errorbar(self.val_bpd_mean_dict_meters[sc].epochs, self.val_bpd_mean_dict_meters[sc].vals, yerr=self.val_bpd_std_dict_meters[sc].vals, alpha=0.5, label=f"val sc{sc}")
            x, y, err = self.val_bpd_mean_dict_meters[sc].epochs, np.array(self.val_bpd_mean_dict_meters[sc].vals), np.array(self.val_bpd_std_dict_meters[sc].vals)
            if len(x) > 0:
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.plot(x, y, alpha=0.5, label=f"val bpd sc{sc}", c=f"C{sc+1}")
                # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0, color='C1')
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        plt.xlabel("Epochs")
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_bpd_wofill.png'), bbox_inches='tight', pad_inches=0.1)
        # Noisy VAL
        for sc in self.noisy_val_bpd_mean_dict_meters.keys():
            # plt.errorbar(self.noisy_val_bpd_mean_dict_meters[sc].epochs, self.noisy_val_bpd_mean_dict_meters[sc].vals, yerr=self.noisy_val_bpd_std_dict_meters[sc].vals, alpha=0.5, label=f"noisy val sc{sc}")
            x, y, err = self.noisy_val_bpd_mean_dict_meters[sc].epochs, np.array(self.noisy_val_bpd_mean_dict_meters[sc].vals), np.array(self.noisy_val_bpd_std_dict_meters[sc].vals)
            if len(x) > 0:
                plt.plot(x, y, alpha=0.5, label=f"noisy bpd val sc{sc}", c=f"C{sc+1}", linestyle='--')
                # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0, color='C2')
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # plt.yscale("linear")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_bpd_wofill_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # bpd_dict w/o fill_between
        # sym = False
        # Train
        x, y, err = [], [], []
        for sc in self.bpd_mean_dict_meters.keys():
            # plt.errorbar(self.bpd_mean_dict_meters[sc].epochs, self.bpd_mean_dict_meters[sc].vals, yerr=self.bpd_std_dict_meters[sc].vals, alpha=0.5, label=f"train sc{sc}")
            x += self.bpd_mean_dict_meters[sc].epochs
            y += self.bpd_mean_dict_meters[sc].vals
            err += self.bpd_std_dict_meters[sc].vals
        y, err = np.array(y), np.array(err)
        # if (y < 1e-6).sum() > 0:
        #     sym = True
        plt.plot(x, y, alpha=0.5, label=f"train bpd")
        # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
        if len(x) > 0:
            plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # VAL
        for sc in self.val_bpd_mean_dict_meters.keys():
            # plt.errorbar(self.val_bpd_mean_dict_meters[sc].epochs, self.val_bpd_mean_dict_meters[sc].vals, yerr=self.val_bpd_std_dict_meters[sc].vals, alpha=0.5, label=f"val sc{sc}")
            x, y, err = self.val_bpd_mean_dict_meters[sc].epochs, np.array(self.val_bpd_mean_dict_meters[sc].vals), np.array(self.val_bpd_std_dict_meters[sc].vals)
            if len(x) > 0:
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.plot(x, y, alpha=0.5, label=f"val bpd sc{sc}")
                # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        plt.xlabel("Epochs")
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_bpd_all_wofill.png'), bbox_inches='tight', pad_inches=0.1)
        # Noisy VAL
        for sc in self.noisy_val_bpd_mean_dict_meters.keys():
            # plt.errorbar(self.noisy_val_bpd_mean_dict_meters[sc].epochs, self.noisy_val_bpd_mean_dict_meters[sc].vals, yerr=self.noisy_val_bpd_std_dict_meters[sc].vals, alpha=0.5, label=f"noisy val sc{sc}")
            x, y, err = self.noisy_val_bpd_mean_dict_meters[sc].epochs, np.array(self.noisy_val_bpd_mean_dict_meters[sc].vals), np.array(self.noisy_val_bpd_std_dict_meters[sc].vals)
            if len(x) > 0:
                plt.plot(x, y, alpha=0.5, label=f"noisy bpd val sc{sc}")
                # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # plt.yscale("linear")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_bpd_all_wofill_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # logpz_dict
        # sym = False
        # VAL
        for sc in self.val_logpz_mean_dict_meters.keys():
            # plt.errorbar(self.val_logpz_mean_dict_meters[sc].epochs, self.val_logpz_mean_dict_meters[sc].vals, yerr=self.val_logpz_std_dict_meters[sc].vals, alpha=0.5, label=f"val sc{sc}")
            x, y, err = self.val_logpz_mean_dict_meters[sc].epochs, np.array(self.val_logpz_mean_dict_meters[sc].vals), np.array(self.val_logpz_std_dict_meters[sc].vals)
            if len(x) > 0:
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.plot(x, y, alpha=0.5, label=f"val logpz sc{sc}", c=f"C{sc+1}")
                plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0, color=f"C{sc+1}")
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        plt.xlabel("Epochs")
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_logpz.png'), bbox_inches='tight', pad_inches=0.1)
        # Noisy VAL
        for sc in self.noisy_val_logpz_mean_dict_meters.keys():
            # plt.errorbar(self.noisy_val_logpz_mean_dict_meters[sc].epochs, self.noisy_val_logpz_mean_dict_meters[sc].vals, yerr=self.noisy_val_logpz_std_dict_meters[sc].vals, alpha=0.5, label=f"noisy val sc{sc}")
            x, y, err = self.noisy_val_logpz_mean_dict_meters[sc].epochs, np.array(self.noisy_val_logpz_mean_dict_meters[sc].vals), np.array(self.noisy_val_logpz_std_dict_meters[sc].vals)
            if len(x) > 0:
                plt.plot(x, y, alpha=0.5, label=f"noisy val logpz sc{sc}", c=f"C{sc+1}", linestyle='--')
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0, color=f"C{sc+1}")
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # plt.yscale("linear")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_logpz_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # logpz_dict
        # sym = False
        # Train
        x, y, err = [], [], []
        for sc in self.logpz_mean_dict_meters.keys():
            # plt.errorbar(self.logpz_mean_dict_meters[sc].epochs, self.logpz_mean_dict_meters[sc].vals, yerr=self.logpz_std_dict_meters[sc].vals, alpha=0.5, label=f"train sc{sc}")
            x += self.logpz_mean_dict_meters[sc].epochs
            y += self.logpz_mean_dict_meters[sc].vals
            err += self.logpz_std_dict_meters[sc].vals
        y, err = np.array(y), np.array(err)
        # if (y < 1e-6).sum() > 0:
        #     sym = True
        plt.plot(x, y, alpha=0.5, label=f"train logpz")
        plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
        if len(x) > 0:
            plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # VAL
        for sc in self.val_logpz_mean_dict_meters.keys():
            # plt.errorbar(self.val_logpz_mean_dict_meters[sc].epochs, self.val_logpz_mean_dict_meters[sc].vals, yerr=self.val_logpz_std_dict_meters[sc].vals, alpha=0.5, label=f"val sc{sc}")
            x, y, err = self.val_logpz_mean_dict_meters[sc].epochs, np.array(self.val_logpz_mean_dict_meters[sc].vals), np.array(self.val_logpz_std_dict_meters[sc].vals)
            if len(x) > 0:
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.plot(x, y, alpha=0.5, label=f"val logpz sc{sc}")
                plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        plt.xlabel("Epochs")
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_logpz_all.png'), bbox_inches='tight', pad_inches=0.1)
        # Noisy VAL
        for sc in self.noisy_val_logpz_mean_dict_meters.keys():
            # plt.errorbar(self.noisy_val_logpz_mean_dict_meters[sc].epochs, self.noisy_val_logpz_mean_dict_meters[sc].vals, yerr=self.noisy_val_logpz_std_dict_meters[sc].vals, alpha=0.5, label=f"noisy val sc{sc}")
            x, y, err = self.noisy_val_logpz_mean_dict_meters[sc].epochs, np.array(self.noisy_val_logpz_mean_dict_meters[sc].vals), np.array(self.noisy_val_logpz_std_dict_meters[sc].vals)
            if len(x) > 0:
                plt.plot(x, y, alpha=0.5, label=f"noisy val logpz sc{sc}")
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # plt.yscale("linear")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_logpz_all_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # logpz_dict w/o fill_between
        # sym = False
        # VAL
        for sc in self.val_logpz_mean_dict_meters.keys():
            # plt.errorbar(self.val_logpz_mean_dict_meters[sc].epochs, self.val_logpz_mean_dict_meters[sc].vals, yerr=self.val_logpz_std_dict_meters[sc].vals, alpha=0.5, label=f"val sc{sc}")
            x, y, err = self.val_logpz_mean_dict_meters[sc].epochs, np.array(self.val_logpz_mean_dict_meters[sc].vals), np.array(self.val_logpz_std_dict_meters[sc].vals)
            if len(x) > 0:
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.plot(x, y, alpha=0.5, label=f"val logpz sc{sc}", c=f"C{sc+1}")
                # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0, color=f"C{sc+1}")
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        plt.xlabel("Epochs")
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_logpz_wofill.png'), bbox_inches='tight', pad_inches=0.1)
        # Noisy VAL
        for sc in self.noisy_val_logpz_mean_dict_meters.keys():
            # plt.errorbar(self.noisy_val_logpz_mean_dict_meters[sc].epochs, self.noisy_val_logpz_mean_dict_meters[sc].vals, yerr=self.noisy_val_logpz_std_dict_meters[sc].vals, alpha=0.5, label=f"noisy val sc{sc}")
            x, y, err = self.noisy_val_logpz_mean_dict_meters[sc].epochs, np.array(self.noisy_val_logpz_mean_dict_meters[sc].vals), np.array(self.noisy_val_logpz_std_dict_meters[sc].vals)
            if len(x) > 0:
                plt.plot(x, y, alpha=0.5, label=f"noisy logpz val sc{sc}", c=f"C{sc+1}", linestyle='--')
                # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0, color=f"C{sc+1}")
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # plt.yscale("linear")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_logpz_wofill_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # logpz_dict w/o fill_between
        # sym = False
        # Train
        x, y, err = [], [], []
        for sc in self.logpz_mean_dict_meters.keys():
            # plt.errorbar(self.logpz_mean_dict_meters[sc].epochs, self.logpz_mean_dict_meters[sc].vals, yerr=self.logpz_std_dict_meters[sc].vals, alpha=0.5, label=f"train sc{sc}")
            x += self.logpz_mean_dict_meters[sc].epochs
            y += self.logpz_mean_dict_meters[sc].vals
            err += self.logpz_std_dict_meters[sc].vals
        y, err = np.array(y), np.array(err)
        # if (y < 1e-6).sum() > 0:
        #     sym = True
        plt.plot(x, y, alpha=0.5, label=f"train logpz")
        # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
        if len(x) > 0:
            plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # VAL
        for sc in self.val_logpz_mean_dict_meters.keys():
            # plt.errorbar(self.val_logpz_mean_dict_meters[sc].epochs, self.val_logpz_mean_dict_meters[sc].vals, yerr=self.val_logpz_std_dict_meters[sc].vals, alpha=0.5, label=f"val sc{sc}")
            x, y, err = self.val_logpz_mean_dict_meters[sc].epochs, np.array(self.val_logpz_mean_dict_meters[sc].vals), np.array(self.val_logpz_std_dict_meters[sc].vals)
            if len(x) > 0:
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.plot(x, y, alpha=0.5, label=f"val logpz sc{sc}")
                # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        plt.xlabel("Epochs")
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_logpz_all_wofill.png'), bbox_inches='tight', pad_inches=0.1)
        # Noisy VAL
        for sc in self.noisy_val_logpz_mean_dict_meters.keys():
            # plt.errorbar(self.noisy_val_logpz_mean_dict_meters[sc].epochs, self.noisy_val_logpz_mean_dict_meters[sc].vals, yerr=self.noisy_val_logpz_std_dict_meters[sc].vals, alpha=0.5, label=f"noisy val sc{sc}")
            x, y, err = self.noisy_val_logpz_mean_dict_meters[sc].epochs, np.array(self.noisy_val_logpz_mean_dict_meters[sc].vals), np.array(self.noisy_val_logpz_std_dict_meters[sc].vals)
            if len(x) > 0:
                plt.plot(x, y, alpha=0.5, label=f"noisy logpz val sc{sc}")
                # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # plt.yscale("linear")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_logpz_all_wofill_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # deltdeltalogp_dict
        # sym = False
        # Train
        x, y, err = [], [], []
        for sc in self.deltalogp_mean_dict_meters.keys():
            # plt.errorbar(self.deltalogp_mean_dict_meters[sc].epochs, self.deltalogp_mean_dict_meters[sc].vals, yerr=self.deltalogp_std_dict_meters[sc].vals, alpha=0.5, label=f"train sc{sc}")
            x += self.deltalogp_mean_dict_meters[sc].epochs
            y += self.deltalogp_mean_dict_meters[sc].vals
            err += self.deltalogp_std_dict_meters[sc].vals
        y, err = np.array(y), np.array(err)
        # if (y < 1e-6).sum() > 0:
        #     sym = True
        plt.plot(x, y, alpha=0.5, label=f"train deltalogp")
        plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
        if len(x) > 0:
            plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # VAL
        for sc in self.val_deltalogp_mean_dict_meters.keys():
            # plt.errorbar(self.val_deltalogp_mean_dict_meters[sc].epochs, self.val_deltalogp_mean_dict_meters[sc].vals, yerr=self.val_deltalogp_std_dict_meters[sc].vals, alpha=0.5, label=f"val sc{sc}")
            x, y, err = self.val_deltalogp_mean_dict_meters[sc].epochs, np.array(self.val_deltalogp_mean_dict_meters[sc].vals), np.array(self.val_deltalogp_std_dict_meters[sc].vals)
            if len(x) > 0:
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.plot(x, y, alpha=0.5, label=f"val deltalogp sc{sc}")
                plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        plt.xlabel("Epochs")
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_deltalogp.png'), bbox_inches='tight', pad_inches=0.1)
        # Noisy VAL
        for sc in self.noisy_val_deltalogp_mean_dict_meters.keys():
            # plt.errorbar(self.noisy_val_deltalogp_mean_dict_meters[sc].epochs, self.noisy_val_deltalogp_mean_dict_meters[sc].vals, yerr=self.noisy_val_deltalogp_std_dict_meters[sc].vals, alpha=0.5, label=f"noisy val sc{sc}")
            x, y, err = self.noisy_val_deltalogp_mean_dict_meters[sc].epochs, np.array(self.noisy_val_deltalogp_mean_dict_meters[sc].vals), np.array(self.noisy_val_deltalogp_std_dict_meters[sc].vals)
            if len(x) > 0:
                plt.plot(x, y, alpha=0.5, label=f"noisy val deltalogp sc{sc}")
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # plt.yscale("linear")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_deltalogp_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
        # deltalogp_dict w/o fill_between
        # sym = False
        x, y, err = [], [], []
        for sc in self.deltalogp_mean_dict_meters.keys():
            # plt.errorbar(self.deltalogp_mean_dict_meters[sc].epochs, self.deltalogp_mean_dict_meters[sc].vals, yerr=self.deltalogp_std_dict_meters[sc].vals, alpha=0.5, label=f"train sc{sc}")
            x += self.deltalogp_mean_dict_meters[sc].epochs
            y += self.deltalogp_mean_dict_meters[sc].vals
            err += self.deltalogp_std_dict_meters[sc].vals
        y, err = np.array(y), np.array(err)
        if (y < 1e-6).sum() > 0:
            sym = True
        plt.plot(x, y, alpha=0.5, label=f"train deltalogp")
        # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
        if len(x) > 0:
            plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        for sc in self.val_deltalogp_mean_dict_meters.keys():
            # plt.errorbar(self.val_deltalogp_mean_dict_meters[sc].epochs, self.val_deltalogp_mean_dict_meters[sc].vals, yerr=self.val_deltalogp_std_dict_meters[sc].vals, alpha=0.5, label=f"val sc{sc}")
            x, y, err = self.val_deltalogp_mean_dict_meters[sc].epochs, np.array(self.val_deltalogp_mean_dict_meters[sc].vals), np.array(self.val_deltalogp_std_dict_meters[sc].vals)
            if len(x) > 0:
                if (y < 1e-6).sum() > 0:
                    sym = True
                plt.plot(x, y, alpha=0.5, label=f"val deltalogp sc{sc}")
                # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        plt.xlabel("Epochs")
        for e in self.scale_change_epochs:
            plt.axvline(e, color='k')
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_deltalogp_wofill.png'), bbox_inches='tight', pad_inches=0.1)
        for sc in self.noisy_val_deltalogp_mean_dict_meters.keys():
            # plt.errorbar(self.noisy_val_deltalogp_mean_dict_meters[sc].epochs, self.noisy_val_deltalogp_mean_dict_meters[sc].vals, yerr=self.noisy_val_deltalogp_std_dict_meters[sc].vals, alpha=0.5, label=f"noisy val sc{sc}")
            x, y, err = self.noisy_val_deltalogp_mean_dict_meters[sc].epochs, np.array(self.noisy_val_deltalogp_mean_dict_meters[sc].vals), np.array(self.noisy_val_deltalogp_std_dict_meters[sc].vals)
            if len(x) > 0:
                plt.plot(x, y, alpha=0.5, label=f"noisy deltalogp val sc{sc}")
                # plt.fill_between(x, y-err, y+err, alpha=0.2, linewidth=0)
                plt.scatter(x[-1], y[-1], color=plt.gca().lines[-1].get_color()); plt.text(x[-1], y[-1], f"{y[-1]:.02f}")
        # plt.yscale("linear")
        plt.legend()
        self.savefig(os.path.join(self.args.save_path, 'plots', 'scales_deltalogp_wofill_wnoisy.png'), bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()

    def init_logg(self):

        # Logger
        self.logger = utils.get_logger(logpath=os.path.join(self.args.save_path, 'logs'), saving=(not self.args.inference))
        self.logger.info(self.args)

        self.logger.info(self.model)
        self.logger.info(f"Number of parameters: {count_parameters(self.model)}")
        self.logger.info(f"Number of trainable parameters at current scale: {count_training_parameters(self.model)}")

        self.logger.info(f"Image shapes: {self.image_shapes}")
        self.logger.info(f"Input shapes: {self.input_shapes}")

        if self.args.distributed:
            self.logger.info('Distributed initializing process group')
            self.logger.info("Distributed: success (%d/%d)"%(self.args.local_rank, distributed.get_world_size()))

        self.train_log_csv = os.path.join(self.args.save_path,'train_log.csv')
        self.train_csv_columns = ['curr_time', 'elapsed', 'memory', 'scale', 'itr', 'epoch', 'batch',
                                  'train_time', 'train_time_avg', 'itr_time', 'itr_time_avg',
                                  'loss', 'loss_avg', 'bpd', 'bpd_avg', 'reg_loss', 'reg_loss_avg',
                                  'nfe', 'nfe_avg', 'grad_norm', 'grad_norm_avg']
        self.train_csv_columns = append_regularization_keys_header(self.train_csv_columns, self.model.module.regularization_fns if self.args.distributed else self.model.regularization_fns)
        self.test_log_csv = os.path.join(self.args.save_path,'test_log.csv')
        self.test_csv_columns = ['curr_time','elapsed', 'scale', 'itr','epoch',
                                 'val_time', 'val_time_avg',
                                 'val_loss', 'val_loss_avg', 'val_bpd', 'val_bpd_avg', 'val_reg_loss', 'val_reg_loss_avg',
                                 'val_nfe', 'val_nfe_avg']

        self.noisy_test_log_csv = os.path.join(self.args.save_path,'noisy_test_log.csv')
        self.noisy_test_csv_columns = ['curr_time','elapsed', 'scale', 'itr','epoch',
                                 'val_time', 'val_time_avg',
                                 'val_loss', 'val_loss_avg', 'val_bpd', 'val_bpd_avg', 'val_reg_loss', 'val_reg_loss_avg',
                                 'val_nfe', 'val_nfe_avg']

        if not self.args.resume:

            with open(self.train_log_csv,'w') as f:
                csvlogger = csv.DictWriter(f, self.train_csv_columns)
                csvlogger.writeheader()

            with open(self.test_log_csv,'w') as f:
                csvlogger = csv.DictWriter(f, self.test_csv_columns)
                csvlogger.writeheader()

            with open(self.noisy_test_log_csv,'w') as f:
                csvlogger = csv.DictWriter(f, self.noisy_test_csv_columns)
                csvlogger.writeheader()

            # For visualization
            generate_noise = self.model.module.generate_noise if self.args.distributed else self.model.generate_noise
            self.fixed_z = generate_noise(min(self.args.test_batch_size, self.args.vis_n_images))
            for sc in range(len(self.fixed_z)):
                nb = int(np.ceil(np.sqrt(float(self.fixed_z[sc].size(0)))))
                vutils.save_image(resize((self.fixed_z[sc]/2 + 0.5).clamp(0, 1)[:, :3], self.image_shapes[-1][-2:], interp),
                                  os.path.join(self.args.save_path, "samples", f"gen_fixed_scale{sc}.png"), nrow=nb)

            # Fixed strict z
            num = min(self.args.test_batch_size, self.args.vis_n_images)
            base = max(2, min(8, int(np.exp(1/(max(1, self.args.max_scales-1))*np.log(num)))))
            self.viz_base = base
            idx = np.array([[int(c) for c in f"{int(convert_base_from_10(n, base)):0{self.args.max_scales}d}"] for n in range(base**self.args.max_scales)])[:num]
            # t_per_scale = [torch.randn(len(np.unique(idx[:, sc])), *sh) * np.sqrt(1/2**(self.num_scales-1) if sc==0 else 1/2**(self.num_scales-sc)) for sc, sh in enumerate(self.input_shapes)]
            t_per_scale = [torch.randn(len(np.unique(idx[:, sc])), *sh) * (std or 1.0) for sc, (sh, std) in enumerate(zip(self.input_shapes, self.z_stds))]
            self.fixed_strict_z = [t_per_scale[sc][idx[:, sc]] for sc in range(self.args.max_scales)]

        else:
            self.logger.info(f"Resuming parameters from {self.args.resume}")

    def get_time(self):
        curr_time = time.time()
        curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
        elapsed = str(datetime.timedelta(seconds=(curr_time - self.start_time)))
        return curr_time_str, elapsed

    def logg(self, mode='train', total_gpus=1):

        assert mode in ['train', 'val', 'noisy_val'], f"mode can only be `train` or `val` or `noisy_val`! Given: {mode}"

        curr_time_str, elapsed = self.get_time()

        fmt = '{:.4f}'
        if mode == 'train':
            log_dict = {
                'curr_time': curr_time_str, 'elapsed': elapsed, 'memory': fmt.format(self.r_mem),
                'scale': self.scale if not self.args.joint else -1, 'itr': self.itr, 'epoch': self.epoch, 'batch': self.batch,
                'train_time': fmt.format(self.train_time_total), 'train_time_avg': fmt.format(self.train_time_total/(self.itr+1)),
                'itr_time': fmt.format(self.itr_time_meter.val), 'itr_time_avg': fmt.format(self.itr_time_meter.avg),
                'loss': fmt.format(self.loss_meter.val), 'loss_avg': fmt.format(self.loss_meter.avg),
                'bpd': fmt.format(self.bpd_meter.val), 'bpd_avg': fmt.format(self.bpd_meter.avg),
                'reg_loss': fmt.format(self.reg_loss_meter.val), 'reg_loss_avg': fmt.format(self.reg_loss_meter.avg),
                'nfe': self.nfe_meter.val, 'nfe_avg': self.nfe_meter.avg,
                'grad_norm': fmt.format(self.grad_meter.val), 'grad_norm_avg': fmt.format(self.grad_meter.avg)
            }
            reg_coeffs = self.model.module.regularization_coeffs if self.args.distributed else self.model.regularization_coeffs
            reg_fns = self.model.module.regularization_fns if self.args.distributed else self.model.regularization_fns

            if reg_coeffs:
                rv = tuple(v_/total_gpus for v_ in self.rv)
                log_dict = append_regularization_csv_dict(log_dict, reg_fns, rv)
            self.append_csv_log(self.train_log_csv, self.train_csv_columns, log_dict)

            if self.itr % self.args.log_freq == 0:
                log_message = (
                    f"{curr_time_str} | {elapsed} | {self.r_mem:.3g} GB | "
                    f"Scale {self.scale if not self.args.joint else -1} | Itr {self.itr:06d} | Epoch {self.epoch:04d} | Batch {self.batch}/{self.batches_in_epoch} | "
                    f"TrainTime {self.train_time_total/3600:.3e} ({self.train_time_total/(self.itr+1)/3600:.3e}) | "
                    f"TrainTime/Itr {self.itr_time_meter.val:.2f} ({self.itr_time_meter.avg:.2f}) | "
                    f"Loss {self.loss_meter.val:.2f} ({self.loss_meter.avg:.2f}) | "
                    f"BPD {self.bpd_meter.val:.2f} ({self.bpd_meter.avg:.2f}) | "
                    f"Reg_loss {self.reg_loss_meter.val:.2f} ({self.reg_loss_meter.avg:.2f}) | "
                    f"NFE {self.nfe_meter.val:.0f} ({self.nfe_meter.avg:.0f}) | "
                    f"Grad_Norm {self.grad_meter.val:.3e} ({self.grad_meter.avg:.3e})"
                )
                if reg_coeffs:
                    log_message = append_regularization_to_log(log_message, reg_fns, rv)
                self.logger.info(log_message)

        elif mode == 'val':

            fmt = '{:.4f}'
            log_dict = {
                'curr_time': curr_time_str, 'elapsed': elapsed,
                'scale': self.scale if not self.args.joint else -1, 'itr': self.itr, 'epoch': self.epoch,
                'val_time': fmt.format(self.val_time_meter.val), 'val_time_avg': fmt.format(self.val_time_meter.avg),
                'val_loss': fmt.format(self.val_loss_meter.val), 'val_loss_avg': fmt.format(self.val_loss_meter.avg),
                'val_bpd': fmt.format(self.val_bpd_meter.val), 'val_bpd_avg': fmt.format(self.val_bpd_meter.avg),
                'val_nfe': self.val_nfe_meter.val, 'val_nfe_avg': self.val_nfe_meter.avg,
            }
            self.append_csv_log(self.test_log_csv, self.test_csv_columns, log_dict)
            log_message = (
                f"{curr_time_str} | {elapsed} | "
                f"Scale {self.scale if not self.args.joint else -1} | Itr {self.itr:06d} | Epoch {self.epoch:04d} | "
                f"ValTime/Itr {self.val_time_meter.val:.2f} ({self.val_time_meter.avg:.2f}) | "
                f"Val Loss {self.val_loss_meter.val:.2f} ({self.val_loss_meter.avg:.2f}) | "
                f"Val BPD {self.val_bpd_meter.val:.2f} ({self.val_bpd_meter.avg:.2f}) | "
                f"Val NFE {self.val_nfe_meter.val:.0f} ({self.val_nfe_meter.avg:.0f})\n"
            )
            self.logger.info(log_message)

        elif mode == 'noisy_val':

            fmt = '{:.4f}'
            log_dict = {
                'curr_time': curr_time_str, 'elapsed': elapsed,
                'scale': self.scale if not self.args.joint else -1, 'itr': self.itr, 'epoch': self.epoch,
                'val_time': fmt.format(self.val_time_meter.val), 'val_time_avg': fmt.format(self.val_time_meter.avg),
                'val_loss': fmt.format(self.noisy_val_loss_meter.val), 'val_loss_avg': fmt.format(self.noisy_val_loss_meter.avg),
                'val_bpd': fmt.format(self.noisy_val_bpd_meter.val), 'val_bpd_avg': fmt.format(self.noisy_val_bpd_meter.avg),
                'val_nfe': self.noisy_val_nfe_meter.val, 'val_nfe_avg': self.noisy_val_nfe_meter.avg,
            }
            self.append_csv_log(self.noisy_test_log_csv, self.noisy_test_csv_columns, log_dict)
            log_message = (
                f"{curr_time_str} | {elapsed} | "
                f"Scale {self.scale if not self.args.joint else -1} | Itr {self.itr:06d} | Epoch {self.epoch:04d} | "
                f"noisy_ValTime/Itr {self.val_time_meter.val:.2f} ({self.val_time_meter.avg:.2f}) | "
                f"noisy_Val Loss {self.noisy_val_loss_meter.val:.2f} ({self.noisy_val_loss_meter.avg:.2f}) | "
                f"noisy_Val BPD {self.noisy_val_bpd_meter.val:.2f} ({self.noisy_val_bpd_meter.avg:.2f}) | "
                f"noisy_Val NFE {self.noisy_val_nfe_meter.val:.0f} ({self.noisy_val_nfe_meter.avg:.0f})\n"
            )
            self.logger.info(log_message)

    def append_csv_log(self, log_file, columns, log_dict):
        with open(log_file, 'a') as f:
            csvlogger = csv.DictWriter(f, columns)
            csvlogger.writerow(log_dict)


if __name__ == '__main__':

    # from train_cnf_multiscale import *
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        args = get_args()

        # Load dataset
        # (imgs, laps), each from smaller to larger scale
        from lib.dataloader import get_dataloaders

        if args.joint:
            bs = [args.batch_size]
        else:
            bs = args.batch_size_per_scale

        train_loaders, test_loaders, train_im_dataset = get_dataloaders(
            data=args.data, data_path=args.data_path, imagesize=args.im_size,
            batch_sizes=bs, nworkers=args.nworkers,
            ds_idx_mod=args.ds_idx_mod, ds_idx_skip=args.ds_idx_skip, ds_length=args.ds_length,
            test_ds_idx_mod=args.test_ds_idx_mod, test_ds_idx_skip=args.test_ds_idx_skip, test_ds_length=args.test_ds_length,
            distributed=args.distributed, imagenet_classes=args.imagenet_classes)

        # Model
        msflow = MSFlow(args, train_im_dataset)

        # Train
        msflow.train(train_loaders, test_loaders, train_im_dataset)
