import glob
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch

from numbers import Number
from PIL import Image


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def copy_scripts(src, dst):
    print("Copying scripts in", src, "to", dst)
    for file in glob.glob(os.path.join(src, '*.sh')) + \
            glob.glob(os.path.join(src, '*.py')) + \
            glob.glob(os.path.join(src, '*_means.pt')) + \
            glob.glob(os.path.join(src, '*.data')) + \
            glob.glob(os.path.join(src, '*.cfg')) + \
            glob.glob(os.path.join(src, '*.names')):
        shutil.copy(file, dst)
    for d in glob.glob(os.path.join(src, '*/')):
        if '__' not in os.path.basename(os.path.dirname(d)) and \
                '.' not in os.path.basename(os.path.dirname(d))[0] and \
                'ipynb' not in os.path.basename(os.path.dirname(d)) and \
                os.path.basename(os.path.dirname(d)) != 'data' and \
                os.path.basename(os.path.dirname(d)) != 'experiments':
            if os.path.abspath(d) in os.path.abspath(dst):
                continue
            print("Copying", d)
            # shutil.copytree(d, os.path.join(dst, d))
            new_dir = os.path.join(dst, os.path.basename(os.path.normpath(d)))
            os.makedirs(new_dir, exist_ok=True)
            copy_scripts(d, new_dir)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)

    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)

    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99, save_seq=False):
        self.momentum = momentum
        self.save_seq = save_seq
        if self.save_seq:
            self.epochs = []
            self.vals = []
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val, epoch=None):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        if self.save_seq:
            self.vals.append(val)
            if epoch is not None:
                self.epochs.append(epoch)


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)


def isnan(tensor):
    return (tensor != tensor)


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def logpx_to_bpd(logpx, dims, bits=8):
    return -(logpx/dims - np.log(2**bits)) / np.log(2)


def bpd_to_logpx(bpd, dims, bits=8):
    return (-bpd*np.log(2) + np.log(2**bits)) * dims


def logit_logpx_to_image_bpd(logpx, logit_x, bits=8, alpha=0.05):
    D = logit_x.nelement()/len(logit_x)
    return -logpx/D/np.log(2) + np.log2(1 - 2*alpha) + bits + (logit_x.sigmoid().log2() + (1 - logit_x.sigmoid()).log2()).reshape(len(logit_x), -1).sum(-1, keepdim=True)/D


def convert_base_from_10(x, b):
    """
    Converts given number x, from base 10 to base b
    x -- the number in base 10
    b -- base to convert
    """
    assert(x >= 0)
    assert(1< b < 37)
    if x == 0:
        return '0'
    r = ''
    import string
    while x > 0:
        r = string.printable[x % b] + r
        x //= b
    return r


def convert_time_stamp_to_hrs(time_day_hr):
    time_day_hr = time_day_hr.split(",")
    if len(time_day_hr) > 1:
        days = time_day_hr[0].split(" ")[0]
        time_hr = time_day_hr[1]
    else:
        days = 0
        time_hr = time_day_hr[0]
    # Hr
    hrs = time_hr.split(":")
    return float(days)*24 + float(hrs[0]) + float(hrs[1])/60 + float(hrs[2])/3600


def plot_time(exp_dir):
    # Read logs
    with open(os.path.join(exp_dir, 'logs')) as f:
        logs = f.readlines()
    # Take only val logs
    val_logs = [a for a in logs if "| ValTime/Itr" in a]
    # noisy_val_logs = [a for a in logs if "| noisy_ValTime/Itr" in a]
    times = []
    for log in val_logs:
        time_day_hr = log.split(' | ')[1]
        times.append(convert_time_stamp_to_hrs(time_day_hr))
    # Plot
    plt.plot(times)
    plt.xlabel("Epochs")
    plt.ylabel("Hours")
    plt.grid()
    plt.savefig(os.path.join(exp_dir, 'plots', 'time.png'), bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()


def build_gif(exp_dir, freq=2, fps=4):
    import imageio
    from cv2 import putText
    samples = sorted(glob.glob(os.path.join(exp_dir, 'samples', 'gen_scale*.png')))
    samples = ([samples[0]] + samples)[::freq]
    frames = []
    max_shape = imageio.imread(samples[-1]).shape[:2]
    for sample in samples:
        epoch = int(os.path.splitext(sample)[0].split('/')[-1].split('_epoch')[-1])
        frame = np.array(Image.fromarray(imageio.imread(sample)).resize(max_shape, 0)).astype('float')/255
        frame = np.pad(frame,
                       [((max_shape[0] - frame.shape[0])//2, max_shape[0] - (max_shape[0] - frame.shape[0])//2 - frame.shape[0]),
                        ((max_shape[1] - frame.shape[1])//2, max_shape[1] - (max_shape[1] - frame.shape[1])//2 - frame.shape[1]),
                        (0, 0)],
                       mode='constant', constant_values=1.0)
        frames.append((putText(np.concatenate((np.ones((40, frame.shape[1], frame.shape[2])), frame), axis=0), f"Epoch {epoch}", (8, 30), 0, 1, (0,0,0), 2)*255).astype('uint8'))
    # Save gif
    imageio.mimwrite(os.path.join(exp_dir, 'samples', 'samples.gif'), frames, fps=fps)


def make_image_shapes(max_scales, im_size, im_ch, factor=0.5, mode='image'):
    # Data shapes
    image_shapes = []
    if mode == '1d':
        image_shapes.append((im_ch, im_size, im_size))
        size_old = im_size
        l = 0
        while l < max_scales-1:
            if l % 2 == 0:
                size = int(round(size_old * factor))
                image_shapes.append((im_ch, size, size_old))
            else:
                image_shapes.append((im_ch, size, size))
                size_old = size
            l += 1
    else:
        for l in range(max_scales):
            size = int(round(im_size * factor**l))
            image_shapes.append((im_ch, size, size))
    image_shapes.reverse()
    return image_shapes


def build_gif_1d(exp_dir, freq=2, fps=4, max_res=32, epochs_per_scale=[100,100,100,100,100], save_dir=None):
    import imageio
    from cv2 import putText
    samples = sorted(glob.glob(os.path.join(exp_dir, 'samples', 'gen_scale*.png')))
    samples = ([samples[0]] + samples)[::freq]
    frames = []
    max_shape = imageio.imread(samples[-1]).shape[:2]
    cum_epochs = np.cumsum(epochs_per_scale)
    image_shapes = make_image_shapes(len(cum_epochs), max_res, 3, mode='1d')
    for sample in samples:
        epoch = int(os.path.splitext(sample)[0].split('/')[-1].split('_epoch')[-1])
        scale = np.nonzero(cum_epochs - epoch + 1 > 0)[0][0]
        shape = image_shapes[scale]
        frame = np.array(Image.fromarray(imageio.imread(sample)).resize(max_shape, 0)).astype('float')/255
        frame = np.pad(frame,
                       [((max_shape[0] - frame.shape[0])//2, max_shape[0] - (max_shape[0] - frame.shape[0])//2 - frame.shape[0]),
                        ((max_shape[1] - frame.shape[1])//2, max_shape[1] - (max_shape[1] - frame.shape[1])//2 - frame.shape[1]),
                        (0, 0)],
                       mode='constant', constant_values=1.0)
        frames.append((putText(np.concatenate((np.ones((40, frame.shape[1], frame.shape[2])), frame), axis=0), f"Epoch {epoch} {shape[1:]}", (8, 30), 0, 1, (0,0,0), 2)*255).astype('uint8'))
    # Save gif
    if save_dir is None:
        save_dir = exp_dir
    imageio.mimwrite(os.path.join(exp_dir, 'samples', 'samples.gif'), frames, fps=fps)
