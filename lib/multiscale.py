import argparse
import math
import numpy as np
import scipy.interpolate as interpolate
import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.layers as layers

from .regularization import create_regularization_fns
from .layers.elemwise import _logit as logit
from .layers.elemwise import _sigmoid as sigmoid
from .utils import logpx_to_bpd


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def normal_logprob(x, mu=0, sigma=1):
    if sigma is None:
        sigma = 1.0
    logZ = -math.log(sigma) -0.5 * math.log(2 * math.pi)
    return logZ - ((x - mu)/sigma).pow(2) / 2


def avg2d(x):
    bs, c, w, h = x.shape
    if x.shape[1:] == (3, 1, 1):
        return x.mean(1, keepdim=True)
    else:
        kernel = torch.tensor([[0.25, 0.25], [0.25, 0.25]]).unsqueeze(0).unsqueeze(0).expand(c, 1, 2, 2).to(x.device)
        return F.conv2d(x.float(), kernel, stride=2, groups=c)


def avg_2d_in_1d(x, ch='height'):
    assert ch in ['height', 'width']
    if x.shape[1:] == (3, 1, 1):
        return x.mean(1, keepdim=True)
    else:
        if ch == 'width':
            return (x[:, :, :, ::2] + x[:, :, :, 1::2])/2
        else:
            return (x[:, :, ::2, :] + x[:, :, 1::2, :])/2


class Downsample(nn.Module):
    def __init__(self, tau=0.5, iters=1):
        super().__init__()
        self.heat = Heat(tau, iters)
        #self.pick = Pick()
    def forward(self, X, sh):
        Y, _ = self.heat(X)
        out = F.interpolate(Y, size=sh, mode='nearest')
        return out


class Pyramid(nn.Module):
    def __init__(self, image_shapes, mode='image'):
        super().__init__()
        self.image_shapes = image_shapes
        self.mode = mode
    def forward(self, img):
        # img: [B, ch, height, width]
        imgs = []
        current = img.float()
        imgs.append(current)
        if self.mode == '1d':
            l = 0
            while l < len(self.image_shapes) - 1:
                if l % 2 == 0:
                    current = avg_2d_in_1d(current, ch='height')
                else:
                    current = avg_2d_in_1d(current, ch='width')
                imgs.append(current)
                l += 1
        else:
            for i in range(len(self.image_shapes)-1):
                current = avg2d(current)
                imgs.append(current)
        imgs.reverse()
        return imgs


def make_image_shapes(max_scales, im_size, im_ch, factor=0.5, mode='image'):
    # Data shapes
    image_shapes = []
    if mode == '1d':
        MAX = int(np.log2(im_size)*2 + 1)
        assert max_scales <= (MAX+1 if im_ch == 3 else MAX), f"max_scales cannot be greater than {MAX+1 if im_ch == 3 else MAX}, given {max_scales}"
        image_shapes.append((im_ch, im_size, im_size))
        size_old = im_size
        l = 0
        while l < MAX-1:
            if l % 2 == 0:
                size = int(round(size_old * factor))
                image_shapes.append((im_ch, size, size_old))
            else:
                image_shapes.append((im_ch, size, size))
                size_old = size
            l += 1
        if im_ch == 3:
            image_shapes.append((1, 1, 1))
    else:
        MAX = int(np.log2(im_size) + 1)
        assert max_scales <= (MAX+1 if im_ch == 3 else MAX), f"max_scales cannot be greater than {MAX+1 if im_ch == 3 else MAX}, given {max_scales}"
        for l in range(MAX):
            size = int(round(im_size * factor**l))
            image_shapes.append((im_ch, size, size))
        if im_ch == 3:
            image_shapes.append((1, 1, 1))
    image_shapes = image_shapes[:max_scales]
    image_shapes.reverse()
    return image_shapes


def std_for_shapes_1d(norm_res, input_shapes):
    # Actual norm_res (128) is double the default (64)! Because std formula has an erroneous "+ 1".
    # Retaining it for legacy.
    stds = []
    for shape in input_shapes:
        stds.append(np.sqrt(1/2**(2*np.log2(norm_res) - np.log2(shape[1]) - np.log2(shape[2]) + 1)))
    if input_shapes[-1][0] == 3 and input_shapes[0] == (1, 1, 1):
        stds[0], stds[1] = np.sqrt(1/3) * stds[0], np.sqrt(2/3) * stds[1]
    return stds


def std_for_shapes_2d(norm_res, input_shapes):
    stds = []
    for shape in input_shapes:
        stds.append(np.sqrt(3/4**(np.log2(norm_res) - np.log2(shape[1]))))
    stds[0] = stds[0]/np.sqrt(3)
    if input_shapes[-1][0] == 9 and input_shapes[0] == (1, 1, 1):
        stds[0], stds[1] = np.sqrt(1/3) * stds[0], np.sqrt(2/3) * stds[0]
    return stds


def combine1d(y, xbar):
    xa = xbar + y
    xb = xbar - y
    y_shape = list(y.shape)
    cat_dim = -1 if y_shape[-1] == y_shape[-2] else -2
    y_shape[cat_dim] = int(y_shape[cat_dim]*2)
    x = torch.cat((xa.unsqueeze(cat_dim), xb.unsqueeze(cat_dim)), dim=cat_dim).reshape(y_shape)
    return x


def combine1ch2ch(y1, y2, xbar):
    x1 = xbar + y1
    x2 = xbar - y1/2 + np.sqrt(3)/2*y2
    x3 = xbar - y1/2 - np.sqrt(3)/2*y2
    return torch.cat([x1, x2, x3], dim=1)


def combine2d(y1, y2, y3, xbar):
    # y1, y2, y3 = y[:, 0:xbar.shape[1]], y[:, xbar.shape[1]:2*xbar.shape[1]], y[:, 2*xbar.shape[1]:3*xbar.shape[1]]
    x1 = y1 + xbar
    x2 = - 1/3*y1 + 2*np.sqrt(2)/3*y2 + xbar
    x3 = - 1/3*y1 - np.sqrt(2)/3*y2 + np.sqrt(6)/3*y3 + xbar
    x4 = - 1/3*y1 - np.sqrt(2)/3*y2 - np.sqrt(6)/3*y3 + xbar
    x = torch.empty(*xbar.shape[:2], xbar.shape[2]*2, xbar.shape[3]*2).to(xbar)
    x[:, :, ::2, ::2] = x1
    x[:, :, ::2, 1::2] = x2
    x[:, :, 1::2, ::2] = x3
    x[:, :, 1::2, 1::2] = x4
    return x


def split2d(x):
    x1 = x[:, :, ::2, ::2]
    x2 = x[:, :, ::2, 1::2]
    x3 = x[:, :, 1::2, ::2]
    x4 = x[:, :, 1::2, 1::2]
    y1 = 3/4*x1 - 1/4*x2 - 1/4*x3 - 1/4*x4
    y2 = 2*np.sqrt(2)/4*x2 - np.sqrt(2)/4*(x3 + x4)
    y3 = np.sqrt(6)/4*(x3 - x4)
    return y1, y2, y3


def split2d_wavelet(x):
    x1 = x[:, :, ::2, ::2]
    x2 = x[:, :, ::2, 1::2]
    x3 = x[:, :, 1::2, ::2]
    x4 = x[:, :, 1::2, 1::2]
    y1 = 1/2*x1 + 1/2*x2 - 1/2*x3 - 1/2*x4
    y2 = 1/2*x1 - 1/2*x2 + 1/2*x3 - 1/2*x4
    y3 = 1/2*x1 - 1/2*x2 - 1/2*x3 + 1/2*x4
    xbar = 1/4*x1 + 1/4*x2 + 1/4*x3 + 1/4*x4
    return y1, y2, y3, xbar


def combine2d_wavelet(y1, y2, y3, xbar):
    # y1, y2, y3 = y[:, 0:xbar.shape[1]], y[:, xbar.shape[1]:2*xbar.shape[1]], y[:, 2*xbar.shape[1]:3*xbar.shape[1]]
    x1 = y1/2 + y2/2 + y3/2 + xbar
    x2 = y1/2 - y2/2 - y3/2 + xbar
    x3 = -y1/2 + y2/2 - y3/2 + xbar
    x4 = -y1/2 - y2/2 + y3/2 + xbar
    x = torch.empty(*xbar.shape[:2], xbar.shape[2]*2, xbar.shape[3]*2).to(xbar)
    x[:, :, ::2, ::2] = x1
    x[:, :, ::2, 1::2] = x2
    x[:, :, 1::2, ::2] = x3
    x[:, :, 1::2, 1::2] = x4
    return x



def split2d_mrcnf(x):
    c = math.pow(2, 2/3)
    x1 = x[:, :, ::2, ::2]
    x2 = x[:, :, ::2, 1::2]
    x3 = x[:, :, 1::2, ::2]
    x4 = x[:, :, 1::2, 1::2]
    y1 = 1/c*x1 + 1/c*x2 - 1/c*x3 - 1/c*x4
    y2 = 1/c*x1 - 1/c*x2 + 1/c*x3 - 1/c*x4
    y3 = 1/c*x1 - 1/c*x2 - 1/c*x3 + 1/c*x4
    xbar = 1/4*x1 + 1/4*x2 + 1/4*x3 + 1/4*x4
    return y1, y2, y3, xbar


def combine2d_mrcnf(y1, y2, y3, xbar):
    c = math.pow(2, 2/3)
    # y1, y2, y3 = y[:, 0:xbar.shape[1]], y[:, xbar.shape[1]:2*xbar.shape[1]], y[:, 2*xbar.shape[1]:3*xbar.shape[1]]
    x1 = c*y1/4 + c*y2/4 + c*y3/4 + xbar
    x2 = c*y1/4 - c*y2/4 - c*y3/4 + xbar
    x3 = -c*y1/4 + c*y2/4 - c*y3/4 + xbar
    x4 = -c*y1/4 - c*y2/4 + c*y3/4 + xbar
    x = torch.empty(*xbar.shape[:2], xbar.shape[2]*2, xbar.shape[3]*2).to(xbar)
    x[:, :, ::2, ::2] = x1
    x[:, :, ::2, 1::2] = x2
    x[:, :, 1::2, ::2] = x3
    x[:, :, 1::2, 1::2] = x4
    return x


class CNFMultiscale(nn.Module):

    def __init__(self, max_scales=2, factor=0.5, concat_input=True,
                 mode='image', std_scale=True, joint=False,
                 regs=argparse.Namespace(kinetic_energy=0.0, jacobian_norm2=0.0),
                 bn=False, im_ch=3, im_size=32, nbits=8,
                 dims="64,64,64", strides="1,1,1,1", num_blocks="2,2",
                 zero_last=True, conv=True, layer_type="concat", nonlinearity="softplus",
                 time_length=1.0, train_T=False, steer_b=0.0,
                 div_samples=1, divergence_fn="approximate",
                 logit=True, alpha=0.05, normal_resolution=64,
                 solver='bosh3',
                 disable_cuda=False,
                 **kwargs):
        super().__init__()
        self.max_scales = max_scales
        self.factor = factor
        self.concat_input = concat_input
        self.mode = mode
        assert self.mode in ['wavelet', 'mrcnf']
        self.std_scale = std_scale
        self.joint = joint
        self.regs = regs
        self.bn = bn
        self.im_ch, self.im_size, self.nbits = im_ch, im_size, nbits
        self.dims, self.strides, self.num_blocks = dims, strides, num_blocks
        self.zero_last, self.conv, self.layer_type, self.nonlinearity = zero_last, conv, layer_type, nonlinearity
        self.time_length, self.train_T, self.steer_b = time_length, train_T, steer_b
        self.div_samples, self.divergence_fn = div_samples, divergence_fn
        self.logit, self.alpha = logit, alpha
        self.normal_resolution = normal_resolution
        self.solver = solver
        self.disable_cuda = disable_cuda

        self._scale = -1
        self.device = torch.device("cuda:%d"%torch.cuda.current_device() if torch.cuda.is_available() and not disable_cuda else "cpu")
        self.cvt = lambda x: x.type(torch.float32).to(self.device, non_blocking=True)

        # Set image shapes
        self.image_shapes = make_image_shapes(max_scales=max_scales, im_size=im_size, im_ch=im_ch, mode=mode)
        self.num_scales = len(self.image_shapes)
        self.pyramid = Pyramid(image_shapes=self.image_shapes, mode=mode)

        MAX = int(np.log2(im_size) + 1)
        self.input_shapes = [self.image_shapes[-min(MAX, max_scales)]] + self.image_shapes[-min(MAX, max_scales):-1]
        self.input_shapes = [(sh[0] if i==0 else sh[0]*3, sh[1], sh[2]) for i, sh in enumerate(self.input_shapes)]
        self.ch1toch3 = False
        if max_scales == MAX+1 and im_ch == 3:
            self.ch1toch3 = True
            self.input_shapes = [(1, 1, 1), (2, 1, 1)] + self.input_shapes[1:]
        if self.mode == 'wavelet':
            self.z_stds = [np.sqrt(1/4**(np.log2(self.normal_resolution) - np.log2(sh[-1]))) for sh in self.image_shapes] if self.std_scale else [None] * self.num_scales
        elif self.mode == 'mrcnf':
            c = math.pow(2, 2/3)
            self.z_stds = [np.sqrt((1 if s == 0 else c)*1/4**(np.log2(self.normal_resolution) - np.log2(sh[-1]))) for s, sh in enumerate(self.image_shapes)] if self.std_scale else [None] * self.num_scales

        self.bns = None
        self.coarse_bns = None

        if self.concat_input:
            self.concat_shapes = [None] + self.image_shapes[:-1]
        else:
            self.concat_shapes = [None] * len(self.image_shapes)

        self.regularization_fns, self.regularization_coeffs = create_regularization_fns(self.regs)

        # Create models
        models = []
        first = True
        for input_sh, concat_sh, bl, std in zip(self.input_shapes, self.concat_shapes, self.num_blocks, self.z_stds):
            models.append(self.create_model(input_sh, concat_sh, bl, first=first, std=std))
            first = False

        self.scale_models = nn.ModuleList(models) # TODO: may be safer to use dict keyed by image size

    def create_model(self, input_shape, concat_shape=None, num_blocks=2, first=False, std=None):

        hidden_dims = tuple(map(int, self.dims.split(",")))
        strides = tuple(map(int, self.strides.split(",")))

        def build_cnf():
            diffeq = layers.ODEnet(
                hidden_dims=hidden_dims,
                input_shape=input_shape,
                concat_shape=concat_shape,
                strides=strides,
                zero_last_weight=self.zero_last,
                conv=self.conv,
                layer_type=self.layer_type,
                nonlinearity=self.nonlinearity,
            )
            odefunc = layers.ODEfunc(
                diffeq=diffeq,
                div_samples=self.div_samples,
                divergence_fn=self.divergence_fn,
            )
            cnf = layers.CNF(
                odefunc=odefunc,
                T=self.time_length,
                train_T=self.train_T,
                steer_b=self.steer_b,
                regularization_fns=self.regularization_fns,
                solver=self.solver,
            )
            return cnf

        chain = []
        if self.mode == 'wavelet':
            chain = [layers.LogitTransform(alpha=self.alpha)] if self.logit else [layers.ZeroMeanTransform()]
        elif self.mode == 'mrcnf' and first:
            chain = chain + [layers.LogitTransform(alpha=self.alpha)] if self.logit else [layers.ZeroMeanTransform()]

        chain = chain + [build_cnf() for _ in range(num_blocks)]

        if std is not None:
            chain = chain + [layers.AffineTransform(scale=std)]

        model = layers.SequentialFlow(chain)

        return model

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        assert type(val) == int, f"scale can only be set to an int, given: {type(val)}, {val}"
        assert val < self.num_scales
        self._scale = val
        if val < 0:
            print(f"ACTIVATING ALL {len(self.scale_models)} scale_models! (JOINT)")
            for sc in range(len(self.scale_models)):
                for p in self.scale_models[sc].parameters():
                    p.requires_grad_(True)
        else:
            for sc in range(len(self.scale_models)):
                if sc != val:
                    for p in self.scale_models[sc].parameters():
                        p.requires_grad_(False)
                else:
                    # Turn on learning
                    for p in self.scale_models[sc].parameters():
                        p.requires_grad_(True)

    def density(self, img, noisy=True):
        """Takes in a uint8 img with pixel range [0, 255]"""

        data_list = self.pyramid(img)

        z_dict, bpd_dict, logpz_dict, deltalogp_dict = {}, {}, {}, {}

        logpx = None
        reg_states = tuple([0.] * len(self.regularization_coeffs))

        for sc, (model, x) in enumerate(zip(self.scale_models, data_list)):

            if not self.joint and sc > self.scale:
                break

            x255 = x.clone()

            if not self.training or self.joint or sc == self.scale:

                x = self.cvt(x)
                if sc != 0:
                    coarser_up = self.cvt(coarser_up)

                # Add noise
                x = x // (2**(8-self.nbits)) if self.nbits < 8 else x
                noise = x.new().resize_as_(x).uniform_() if noisy else 0.5
                x = x.add_(noise).div_(2**self.nbits)

                # bsz, c, w, h  = x.shape
                # Make y
                if sc > 0:
                    if self.mode == 'wavelet':
                        y1, y2, y3, _ = split2d_wavelet(x)
                        # y1 in [-1, 1] -> [0, 1] => /2 + 0.5
                        y1 = y1/2 + 0.5
                        y2 = y2/2 + 0.5
                        y3 = y3/2 + 0.5
                        y = torch.cat([y1, y2, y3], dim=1).clamp_(0, 1)
                    elif self.mode == 'mrcnf':
                        y1, y2, y3, _ = split2d_mrcnf(x)
                        y = torch.cat([y1, y2, y3], dim=1)
                else:
                    y = x

                if sc > 0:
                    concat_var = coarser_up if self.concat_input else None

                # Forward through model
                if sc == 0:
                    z, deltalogp, reg_tup = model(y, reverse=False)
                else:
                    z, deltalogp, reg_tup = model(y, reverse=False, concat_var=concat_var)

                # LOGPROB
                logpz = normal_logprob(z, mu=0, sigma=self.z_stds[sc]).reshape(len(z), -1).sum(1, keepdim=True)

                z_dict[sc] = z.detach().cpu()
                logpz_dict[sc] = logpz.detach().cpu()
                deltalogp_dict[sc] = -deltalogp.detach().cpu()

                logpx_scale = logpz - deltalogp

                # Compensating logp for x->y tx, and y scaling
                if sc > 0 and self.mode == 'wavelet':
                    logpx_scale += np.prod(coarser_up.shape[-3:]) * np.log(1/2) + np.prod(coarser_up.shape[-3:]) * np.log(1/2 * 1/2 * 1/2)

                if not self.training:
                    logpx_scale = logpx_scale.detach()
                if logpx is None:
                    logpx = logpx_scale
                else:
                    if self.joint:
                        logpx += logpx_scale
                    else:
                        logpx = logpx.detach() + logpx_scale

                dims = np.prod(self.image_shapes[sc])
                bpd_dict[sc] = logpx_to_bpd(logpx.detach(), dims, self.nbits).cpu()

                # Regularization
                if not self.training:
                    reg_states = ()
                elif self.joint:
                    reg_states = tuple(r0 + rs.mean() for r0, rs in zip(reg_states, reg_tup)) if len(self.regularization_coeffs) else ()
                elif not self.joint and sc == self.scale:
                    reg_states = tuple(rs.mean() for rs in reg_tup) if len(self.regularization_coeffs) else ()
                else:
                    reg_states = ()

            # Make coarse_image for next scale
            # If training, only do this at scale just before current scale
            if (not self.training or self.joint or sc == self.scale-1) and (sc+1 < self.num_scales):
                noise = x255.new().resize_as_(x255).float().uniform_() if noisy else 0.5
                coarser_up = (x255.float()/256.0 + noise/float(2**self.nbits)).clamp_(0, 1)

        return logpx, reg_states, bpd_dict, z_dict, logpz_dict, deltalogp_dict

    def log_prob(self, img, return_dicts=True, noisy=True, at_sc=-1):
        """Takes in a uint8 img with pixel range [0, 255]"""

        data_list = self.pyramid(img)

        z_dict, bpd_dict, logpz_dict, deltalogp_dict = {}, {}, {}, {}

        logpx = None
        reg_states = tuple([0.] * len(self.regularization_coeffs))

        for sc, (model, x) in enumerate(zip(self.scale_models, data_list)):

            # if not self.joint and sc > self.scale:
            #     break

            # if self.mode != 'wavelet':
            x255 = x.clone()

            if at_sc == -1 or (at_sc > -1 and sc == at_sc):

                x = self.cvt(x)
                if sc != 0:
                    coarser_up = self.cvt(coarser_up)

                # # Init logp
                # deltalogp = torch.zeros(x.size(0), 1, device=x.device)

                # if self.mode != 'wavelet':
                # Add noise
                x = x // (2**(8-self.nbits)) if self.nbits < 8 else x
                noise = x.new().resize_as_(x).uniform_() if noisy else 0.5
                x = x.add_(noise).div_(2**self.nbits)

                # bsz, c, w, h  = x.shape
                # Make y
                if sc > 0:
                    if self.mode == 'wavelet':
                        y1, y2, y3, _ = split2d_wavelet(x)
                        # y1 in [-1, 1] -> [0, 1] => /2 + 0.5
                        y1 = y1/2 + 0.5
                        y2 = y2/2 + 0.5
                        y3 = y3/2 + 0.5
                        y = torch.cat([y1, y2, y3], dim=1).clamp_(0, 1)
                    elif self.mode == 'mrcnf':
                        y1, y2, y3, _ = split2d_mrcnf(x)
                        y = torch.cat([y1, y2, y3], dim=1)
                else:
                    y = x

                if sc > 0:
                    concat_var = coarser_up if self.concat_input else None

                # Forward through model
                if sc == 0:
                    z, deltalogp, _ = model(y, reverse=False)
                else:
                    z, deltalogp, _ = model(y, reverse=False, concat_var=concat_var)

                # LOGPROB
                logpz = normal_logprob(z, mu=0, sigma=self.z_stds[sc]).reshape(len(z), -1).sum(1, keepdim=True)

                if return_dicts:
                    z_dict[sc] = z.detach().cpu()
                    logpz_dict[sc] = logpz.detach().cpu()
                    deltalogp_dict[sc] = -deltalogp.detach().cpu()

                logpx_scale = logpz - deltalogp

                # Compensating logp for x->y tx, and y scaling
                if sc > 0 and self.mode == 'wavelet':
                    logpx_scale += np.prod(coarser_up.shape[-3:]) * np.log(1/2) + np.prod(coarser_up.shape[-3:]) * np.log(1/2 * 1/2 * 1/2)

                # if not self.training:
                if logpx is None:
                    logpx = logpx_scale.detach().cpu()
                else:
                    # if self.joint:
                    logpx += logpx_scale.detach().cpu()

                if return_dicts:
                    dims = np.prod(self.image_shapes[sc])
                    bpd_dict[sc] = logpx_to_bpd(logpx, dims, self.nbits).cpu()

            # Make coarse_image for next scale
            # If training, only do this at scale just before current scale
            # if (not self.training or self.joint or sc == self.scale-1) and (sc+1 < self.num_scales):
            if sc+1 < self.num_scales:
                noise = x255.new().resize_as_(x255).float().uniform_() if noisy else 0.5
                coarser_up = (x255.float()/256.0 + noise/float(2**self.nbits)).clamp_(0, 1)

        if return_dicts:
            return logpx, bpd_dict, z_dict, logpz_dict, deltalogp_dict
        else:
            return logpx

    def generate_noise(self, batch_size):
        noise = [torch.randn(batch_size, *sh) * (std or 1.0) for sh, std in zip(self.input_shapes, self.z_stds)]
        return noise

    def generate(self, noise_list, temp=1.0):
        # noise_list : [z_0, z_1, z_2, z_3] (from coarser to finer scales)

        x_dict = {}
        y_dict = {}

        for sc, (model, z) in enumerate(zip(self.scale_models, noise_list)):

            z = self.cvt(z*temp)

            if not self.joint and sc > self.scale:
                break

            if sc == 0:
                y, _, _ = model(z, reverse=True)
            else:
                concat_var = coarse_bn if self.bns is not None else coarser_up if self.concat_input else None

                y, _, _ = model(z, reverse=True, concat_var=concat_var)

            if self.bns is not None:
                mu = self.bns[sc].running_mean.reshape(1,-1, 1, 1)
                var = self.bns[sc].running_var
                eps = self.bns[sc].eps
                std = (var + eps).sqrt().reshape(1,-1, 1, 1)
                y = y*std + mu

            if sc == 0:
                x = y
            elif self.mode == 'wavelet':
                ch = coarser_up.shape[1]
                y11 = y[:, 0:ch]
                y22 = y[:, ch:2*ch]
                y33 = y[:, 2*ch:3*ch]
                # y1 in [-1, 1] -> [0, 1] => /2 + 0.5
                y1 = (y11 - 0.5)*2
                y2 = (y22 - 0.5)*2
                y3 = (y33 - 0.5)*2
                x = combine2d_wavelet(y1, y2, y3, coarser_up)
            elif self.mode == 'mrcnf':
                ch = coarser_up.shape[1]
                y1 = y[:, 0:ch]
                y2 = y[:, ch:2*ch]
                y3 = y[:, 2*ch:3*ch]
                x = combine2d_mrcnf(y1, y2, y3, coarser_up)

            if sc > 0:
                if self.mode == 'wavelet':
                    y_dict[sc] = [y11.detach().cpu(), y22.detach().cpu(), y33.detach().cpu()]
                    del y11, y22, y33
                else:
                    y_dict[sc] = y.detach().cpu()

            # Make coarser_up
            if sc+1 < self.max_scales:
                coarser_up = x.detach()

            # To compensate for addition of noise
            x = (x - 0.5/2**self.nbits).clamp_(0, 1)

            x_dict[sc] = x.detach().cpu()

        return x_dict, y_dict, x

    def forward(self, img, reverse=False, noisy=True, temp=1.0):
        if reverse:
            return self.generate(img, temp=temp)
        else:
            return self.density(img, noisy=noisy)
