import argparse
import glob
import imageio
import io
import math
import numpy as np
import os
import PIL.Image as Image
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import warnings
import yaml

import torch
import torchvision.datasets as dset

from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from lib.dataloader import get_dataloaders
from lib.gaussian_random_field import gaussian_random_field
from lib.multiscale import standard_normal_logprob
from lib.utils import logpx_to_bpd
from train_cnf_multiscale import MSFlow, get_args
from misc import standard_normal_logprob, logpz_to_z

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# LOAD MODEL FROM DIR
def load_model_spl(ckpt_path,
        sc_ckpt_paths=[], scales=[[]],
        model_type='ckpt', model_types=[]):
    mode = 'ckpt' if os.path.splitext(ckpt_path)[-1] == '.pth' else 'exp'
    exp_path = os.path.dirname(os.path.dirname(ckpt_path)) if mode == 'ckpt' else ckpt_path
    state_dict = yaml.load(open(os.path.join(exp_path, 'args.yaml'), 'r'), Loader=yaml.FullLoader)
    exp_args = get_args()
    for key in state_dict.keys():
        exp_args.__dict__[key] = state_dict[key]
    # Legacy issues
    exp_args.num_blocks = str(exp_args.num_blocks)
    exp_args.num_blocks = [int(exp_args.num_blocks)] * exp_args.max_scales if ',' not in exp_args.num_blocks else list(map(int, exp_args.num_blocks.split(",")))
    # Save other stuff
    exp_args.save_path = exp_path
    exp_args.lr_mode = exp_args.lr_scheduler
    exp_args.inference = True
    exp_args.copy_scripts = False
    if mode == 'exp':
        ckpts = [a for a in sorted(glob.glob(os.path.join(exp_path, 'checkpoints', '*.pth'))) if model_type in os.path.basename(a)]
        load_ckpt = ckpts[-1] if model_type == 'best' else ckpts[0]
    else:
        load_ckpt = ckpt_path
    exp_args.ckpt_to_load = None if len(sc_ckpt_paths) > 0 else load_ckpt
    exp_args.resume = None
    exp_args.max_scales = int(np.sum([len(a) for a in scales])) if len(sc_ckpt_paths) > 0 else exp_args.max_scales
    msflow = MSFlow(exp_args)
    if len(sc_ckpt_paths) > 0:
        new_state_dict = {}
        if mode == 'exp':
            for i, (sc_sc, sc_exp_path, sc_model_type) in enumerate(zip(scales, sc_ckpt_paths, model_types)):
                assert sc_model_type in ["best", "ckpt"]
                state_dict = torch.load([a for a in sorted(glob.glob(os.path.join(sc_exp_path, 'checkpoints', '*.pth'))) if sc_model_type in os.path.basename(a)][-1 if sc_model_type == 'best' else 0], map_location=device)['state_dict']
                for key in sorted(state_dict.keys()):
                    if i > 0 and key[:len('scale_models.')] == 'scale_models.':
                        sc_old = int(key[len('scale_models.')])
                        if sc_old in sc_sc:
                            new_sc = sc_sc.index(sc_old) + len(scales[i-1])
                            # print(f"\t {new_sc} <- {sc_old} {'scale_models.' + str(new_sc) + key[len('scale_models.')+1:]}")
                            new_state_dict['scale_models.' + str(new_sc) + key[len('scale_models.')+1:]] = state_dict[key]
                    else:
                        new_state_dict[key] = state_dict[key]
        else:
            for i, (sc_sc, sc_ckpt_path) in enumerate(zip(scales, sc_ckpt_paths)):
                state_dict = torch.load(sc_ckpt_path, map_location=device)['state_dict']
                for key in sorted(state_dict.keys()):
                    if i > 0 and key[:len('scale_models.')] == 'scale_models.':
                        sc_old = int(key[len('scale_models.')])
                        if sc_old in sc_sc:
                            new_sc = sc_sc.index(sc_old) + len(scales[i-1])
                            new_state_dict['scale_models.' + str(new_sc) + key[len('scale_models.')+1:]] = state_dict[key]
                    else:
                        new_state_dict[key] = state_dict[key]
        msflow.model.load_state_dict(new_state_dict, strict=True)
    msflow.model.eval()
    if len(sc_ckpt_paths) > 0:
        msflow.model.scale = int(exp_args.max_scales - 1)
    return msflow, exp_args


def fast_collate(batch, sz, shuffle=False, shuff_size=1, contrast=1.0):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), *sz), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        a = torch.from_numpy(np.array(nump_array))
        if shuffle:
            a = shuffle_imgs(a, shuff_size)
        if contrast != 1.0:
            a = (a * contrast).clamp_(0, 1)
        tensor[i] += a
    return tensor, targets


def my_dataloader(data, data_path, bs, imsize=32, train=False, shuffle_pixels=False, shuffle_size=1, contrast=1.0, subset_len=0, nworkers=8):
    if data == 'imagenet':
        train_dls, test_dls, train_im_ds = get_dataloaders(data='imagenet', data_path=data_path, imagesize=imsize,
                batch_sizes=[bs], ds_length=(subset_len if subset_len > 0 else None), test_ds_length=(subset_len  if subset_len > 0 else None))
        return train_dls[-1] if train else test_dls[-1]
    else:
        if data == 'cifar10':
            assert os.path.basename(data_path) == 'CIFAR10'
            dataset = dset.CIFAR10(root=data_path, train=train, transform=transforms.Resize(imsize), download=True)
        elif data == 'svhn':
            assert os.path.basename(data_path) == 'SVHN'
            dataset = dset.SVHN(root=data_path, split=("train" if train else "test"), transform=transforms.Resize(imsize), download=True)
        elif data == 'tinyimagenet':
            assert os.path.basename(data_path) == 'tiny-imagenet-200-test'
            dataset = dset.ImageFolder(root=data_path, transform=transforms.Resize(imsize))
        if subset_len > 0:
            dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
        dataloader = DataLoader(
            dataset=dataset, batch_size=bs, shuffle=False,
            num_workers=nworkers, pin_memory=True,
            collate_fn=lambda batch: fast_collate(batch, (3, imsize, imsize), shuffle=shuffle_pixels, shuff_size=shuffle_size, contrast=contrast)
        )
        return dataloader


def extend_in_dict(my_dict, key, val):
    if key in my_dict.keys():
        my_dict[key].append(val)
    else:
        my_dict[key] = [val]


def ood_stats(msflow, data_loader, noisy=False, desc=""):
    logpxs, noisy_logpxs = [], []
    bpds, logpzs, deltalogps = {}, {}, {}
    # noisy_bpds, noisy_logpzs, noisy_deltalogps = {}, {}, {}
    for batch, (imgs, _) in tqdm(enumerate(data_loader), total=len(data_loader), desc=desc):
        # imgs_in = imgs.clone()
        # Density
        logpx, _, bpd_dict, _, logpz_dict, deltalogp_dict = msflow.model(imgs, noisy=noisy)
        # Add data
        logpxs.extend(logpx.detach().cpu())
        for sc in bpd_dict.keys():
            extend_in_dict(bpds, sc, bpd_dict[sc])
            extend_in_dict(logpzs, sc, logpz_dict[sc])
            extend_in_dict(deltalogps, sc, deltalogp_dict[sc])
    # Concatenate
    for sc in sorted(bpds.keys()):
        bpds[sc], logpzs[sc], deltalogps[sc] = torch.cat(bpds[sc]).squeeze(), torch.cat(logpzs[sc]).squeeze(), torch.cat(deltalogps[sc]).squeeze()
    # Stack
    logpxs = torch.tensor(logpxs)
    bpds = torch.stack([i[1] for i in list(map(list, bpds.items()))])
    deltalogps = torch.stack([i[1] for i in list(map(list, deltalogps.items()))])
    logpzs = torch.stack([i[1] for i in list(map(list, logpzs.items()))])
    # Make logpx from logpz
    dims = torch.tensor(msflow.image_shapes).prod(dim=1, keepdims=True).float()
    logpx_full = torch.cumsum(deltalogps + logpzs, dim=0) - np.log(2**8)*dims
    return logpxs, bpds, logpx_full, deltalogps, logpzs

