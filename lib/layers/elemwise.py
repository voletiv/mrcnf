import math
import torch
import torch.nn as nn

_DEFAULT_ALPHA = 1e-6


class ZeroMeanTransform(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, logpx=None, reg_states=tuple(), reverse=False):
        logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        if reverse:
            x = x + .5
            return x, logpx, reg_states
        else:
            x = x - .5
            return x, logpx, reg_states


class AffineTransform(nn.Module):
    def __init__(self, scale=1.0, translation=0.0):
        nn.Module.__init__(self)
        self.scale = scale
        self.translation = translation
        assert self.scale != 0, "AffineTransform: scale cannot be 0"

    def forward(self, x, logpx=None, reg_states=tuple(), reverse=False):
        logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        if reverse:
            out = _affine(x, logpx, self.scale, self.translation, reverse=True)
            return out[0], out[1], reg_states
        else:
            out = _affine(x, logpx, self.scale, self.translation)
            return out[0], out[1], reg_states


def _affine(x, logpx=None, scale=1.0, translation=0.0, reverse=False):
    y = x*scale + translation if not reverse else (x - translation)/scale
    if logpx is None:
        return y
    if reverse:
        return y, logpx + x.nelement()/len(x)*math.log(scale)
    else:
        return y, logpx - x.nelement()/len(x)*math.log(scale)


class LogitTransform(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """
    def __init__(self, alpha=_DEFAULT_ALPHA):
        nn.Module.__init__(self)
        self.alpha = alpha
    def forward(self, x, logpx=None, reg_states=tuple(), reverse=False):
        logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        if reverse:
            out = _sigmoid(x, logpx, self.alpha)
            return out[0], out[1], reg_states
        else:
            out = _logit(x, logpx, self.alpha)
            return out[0], out[1], reg_states


class SigmoidTransform(nn.Module):
    """Reverse of LogitTransform."""
    def __init__(self, alpha=_DEFAULT_ALPHA):
        nn.Module.__init__(self)
        self.alpha = alpha
    def forward(self, x, logpx=None, reg_states=tuple(), reverse=False):
        if reverse:
            out = _logit(x, logpx, self.alpha)
            return out[0], out[1], reg_states
        else:
            out = _sigmoid(x, logpx, self.alpha)
            return out[0], out[1], reg_states


def _logit(x, logpx=None, alpha=_DEFAULT_ALPHA):
    # x in [0, 1]
    s = alpha + (1 - 2 * alpha) * x
    y = torch.log(s) - torch.log(1 - s)
    if logpx is None:
        return y
    return y, logpx - _logdetgrad(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)


def _sigmoid(y, logpy=None, alpha=_DEFAULT_ALPHA):
    # y in [-inf, inf]
    x = (torch.sigmoid(y) - alpha) / (1 - 2 * alpha)
    if logpy is None:
        return x
    return x, logpy + _logdetgrad(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)


def _logdetgrad(x, alpha):
    s = alpha + (1 - 2 * alpha) * x
    logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * alpha)
    return logdetgrad
