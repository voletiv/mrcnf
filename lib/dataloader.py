import numpy as np
import os
import sys

from contextlib import redirect_stdout

import torch
import torchvision
import torchvision.datasets as dset

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from dist_utils import env_world_size, env_rank
from lib import layers
from lib.dataloaders.celeba import CelebAHQ
from lib.dataloaders.imagenet import ImageNetDataset


class TransformPILtoRGBTensor:
    def __call__(self, img):
        vassert(type(img) is Image.Image, 'Input is not a PIL.Image')
        width, height = img.size
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.convert(mode='RGB').tobytes())).view(height, width, 3)
        img = img.permute(2, 0, 1)
        return img


def get_dataloaders(data='mnist', data_path="./data/", imagesize=None,
                    batch_sizes=[200], nworkers=4,
                    ds_idx_mod=None, ds_idx_skip=0, ds_length=None,
                    test_ds_idx_mod=None, test_ds_idx_skip=0, test_ds_length=None,
                    distributed=False, imagenet_classes=[]):

    if data == "mnist":
        assert 'MNIST' in os.listdir(data_path)
        im_ch = 1
        im_size = 28 if imagesize is None else imagesize
        pad = (im_size-28)//2
        tf = transforms.Pad(pad)
        train_set = dset.MNIST(root=data_path, train=True, transform=tf, download=True)
        test_set = dset.MNIST(root=data_path, train=False, transform=tf, download=True)
        train_im_set = MNIST(image_only=True, root=data_path, train=True, transform=transforms.Compose([tf, TransformPILtoRGBTensor()]), download=True)

    if data == "fashion_mnist":
        assert 'FashionMNIST' in os.listdir(data_path)
        im_ch = 1
        im_size = 28 if imagesize is None else imagesize
        pad = (im_size-28)//2
        tf = transforms.Pad(pad)
        train_set = dset.FashionMNIST(root=data_path, train=True, transform=tf, download=True)
        test_set = dset.FashionMNIST(root=data_path, train=False, transform=tf, download=True)
        train_im_set = FashionMNIST(image_only=True, root=data_path, train=True, transform=transforms.Compose([tf, TransformPILtoRGBTensor()]), download=True)

    elif data == "cifar10":
        assert os.path.basename(data_path) == 'CIFAR10'
        im_ch = 3
        im_size = 32 if imagesize is None else imagesize
        train_set = dset.CIFAR10(
            root=data_path, train=True, transform=transforms.Compose([
                transforms.Resize(im_size),
                transforms.RandomHorizontalFlip(),
            ]), download=True
        )
        test_set = dset.CIFAR10(root=data_path, train=False, transform=transforms.Resize(im_size), download=True)
        train_im_set = CIFAR10_RGB(root=data_path, train=True, transform=transforms.Compose([transforms.Resize(im_size), TransformPILtoRGBTensor()]), download=True)

    elif data == "svhn":
        assert os.path.basename(data_path) == 'SVHN'
        im_ch = 3
        im_size = 32 if imagesize is None else imagesize
        tf = transforms.Resize(im_size)
        train_set = dset.SVHN(root=data_path, split="train", transform=tf, download=True)
        test_set = dset.SVHN(root=data_path, split="test", transform=tf, download=True)
        train_im_set = SVHN_RGB(root=data_path, split="train", transform=transforms.Compose([tf, TransformPILtoRGBTensor()]), download=True)

    elif data == 'celebahq':
        assert 'celebahq' in os.listdir(data_path)
        im_ch = 3
        im_size = 256 if imagesize is None else imagesize
        train_set = CelebAHQ(
            root=data_path, train=True, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(im_size),
                transforms.RandomHorizontalFlip(),
            ])
        )
        test_set = CelebAHQ(root=data_path, train=False, transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize(im_size)]))
        train_im_set = CelebAHQ_RGB(root=data_path, train=True, transform=transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(im_size),
                        TransformPILtoRGBTensor()]))

    elif data == 'lsun_church':
        im_ch = 3
        im_size = 64 if imagesize is None else imagesize
        train_set = dset.LSUN(
            data_path, ['church_outdoor_train'], transform=transforms.Compose([
                transforms.Resize(96),
                transforms.RandomCrop(im_size),
            ])
        )
        test_set = dset.LSUN(
            data_path, ['church_outdoor_val'], transform=transforms.Compose([
                transforms.Resize(96),
                transforms.RandomCrop(im_size),
            ])
        )
        train_im_set = LSUN_RGB(data_path, ['church_outdoor_train'], transform=transforms.Compose([
            transforms.Resize(96),
            transforms.RandomCrop(im_size),
            TransformPILtoRGBTensor()
            ])
        )

    elif data == 'imagenet':
        assert os.path.basename(data_path) == 'ilsvrc2012.hdf5'
        im_ch = 3
        im_size = 224 if imagesize is None else imagesize
        train_set = ImageNetDataset(data_path, "train", transform=transforms.Compose([
                transforms.RandomResizedCrop(224 if im_size <= 224 else 256),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(im_size)
                ]),
            classes=imagenet_classes,
        )
        test_set = ImageNetDataset(data_path, "val", transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224 if im_size <= 224 else 256),
                transforms.Resize(im_size)
                ]),
            classes=imagenet_classes,
        )
        train_im_set = ImageNet_RGB(data_path, "train", transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224 if im_size <= 224 else 256),
                transforms.Resize(im_size),
                TransformPILtoRGBTensor()
                ]),
            classes=imagenet_classes,
        )

    elif data == 'tinyimagenet_test':
        assert os.path.basename(data_path) == 'tiny-imagenet-200-test'
        im_ch = 3
        im_size = 224 if imagesize is None else imagesize
        train_set = dset.ImageFolder(root=data_path, transform=transforms.Resize(im_size))
        test_set = dset.ImageFolder(root=data_path, transform=transforms.Resize(im_size))
        train_im_set = ImageFolder_RGB(root=data_path, transform=transforms.Compose([transforms.Resize(im_size), TransformPILtoRGBTensor()]))

    elif data == 'imagenet64_cf':
        im_ch = 3
        im_size = 64
        train_set = Imagenet64_cf(train=True, root=data_path)
        test_set = Imagenet64_cf(train=False, root=data_path)
        train_im_set = None

    if ds_idx_mod is not None or ds_idx_skip > 0 or ds_length is not None:
        train_set = MyLengthDataset(train_set, ds_idx_mod, ds_idx_skip, ds_length)
        train_im_set = MyLengthDataset(train_im_set, ds_idx_mod, ds_idx_skip, ds_length)

    if test_ds_idx_mod is not None or test_ds_idx_skip > 0 or test_ds_length is not None:
        test_set = MyLengthDataset(test_set, test_ds_idx_mod, test_ds_idx_skip, test_ds_length)

    train_sampler = (DistributedSampler(train_set,
        num_replicas=env_world_size(), rank=env_rank()) if distributed
        else None)

    train_loaders = [DataLoader(
        dataset=train_set, batch_size=bs, shuffle=(train_sampler is None),
        num_workers=nworkers, pin_memory=True, sampler=train_sampler,
        collate_fn=lambda batch: fast_collate(batch, (im_ch, im_size, im_size))
    ) for bs in batch_sizes]

    test_sampler = (DistributedSampler(test_set,
        num_replicas=env_world_size(), rank=env_rank(), shuffle=False) if distributed
        else None)

    test_loaders = [DataLoader(
        dataset=test_set, batch_size=bs, shuffle=False,
        num_workers=nworkers, pin_memory=True, sampler=test_sampler,
        collate_fn=lambda batch: fast_collate(batch, (im_ch, im_size, im_size))
    ) for bs in batch_sizes]

    return train_loaders, test_loaders, train_im_set


class Imagenet64_cf(torchvision.datasets.ImageFolder):

    def __init__(self, train=True, transform=None, root='./data/'):
        self.train_loc = os.path.join(root, 'imagenet64/train/')
        self.test_loc = os.path.join(root, 'imagenet64/val/')
        return super().__init__(self.train_loc if train else self.test_loc, transform=transform)


class MyLengthDataset(Dataset):
    def __init__(self, ds, ds_idx_mod=None, ds_idx_skip=0, ds_length=None):
        super().__init__()
        self.ds = ds
        self.ds_idx_mod = ds_idx_mod
        self.ds_idx_skip = ds_idx_skip
        try:
            self.data = self.ds.data
        except AttributeError as err:
            print(err)
        if self.ds_idx_mod is not None:
            assert self.ds_idx_mod > 0, f"Modulo index must be a +ve number! Given {ds_idx_mod}"
        if self.ds_idx_skip is not None:
            assert self.ds_idx_skip >= 0, f"Skip index must be a non -ve number! Given {ds_idx_skip}"
        self.ds_length = ds_length
        if self.ds_length is not None:
            assert self.ds_length > 0, f"Length of dataset must be a +ve number! Given {ds_length}"
    def __len__(self):
        return self.ds_length if self.ds_length is not None else len(self.ds % self.ds_idx_mod if self.ds_idx_mod is not None else len(self))
    def __getitem__(self, idx):
        mod = self.ds_idx_mod if self.ds_idx_mod is not None else len(self)
        idx = idx % mod + self.ds_idx_skip
        return self.ds.__getitem__(idx)


def fast_collate(batch, sz):
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
        tensor[i] += torch.from_numpy(np.array(nump_array))
    return tensor, targets


class MNIST(dset.MNIST):
    def __init__(self, image_only, *args, **kwargs):
        self.image_only = image_only
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)
    def __getitem__(self, index):
        if self.image_only:
            img, target = super().__getitem__(index)
            return img
        else:
            super().__getitem__(index)


class FashionMNIST(dset.FashionMNIST):
    def __init__(self, image_only, *args, **kwargs):
        self.image_only = image_only
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)
    def __getitem__(self, index):
        if self.image_only:
            img, target = super().__getitem__(index)
            return img.repeat(3, 1, 1)
        else:
            super().__getitem__(index)


class CIFAR10_RGB(dset.CIFAR10):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


class SVHN_RGB(dset.SVHN):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


class CelebAHQ_RGB(CelebAHQ):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


class LSUN_RGB(dset.LSUN):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


class ImageNet_RGB(ImageNetDataset):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


class ImageFolder_RGB(dset.ImageFolder):
    def __init__(self, *args, **kwargs):
        with redirect_stdout(sys.stderr):
            super().__init__(*args, **kwargs)
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img
