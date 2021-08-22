import torch.nn as nn
from .cnf import CNF


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layers_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layers_list)

    def forward(self, x, logpx=None, reg_states=tuple(), reverse=False, inds=None, concat_var=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        # if concat_var is not None:
        #     concat_var -= 0.5
        for i in inds:
            if type(self.chain[i]) == CNF:
                x, logpx, reg_states = self.chain[i](x, logpx, reg_states, reverse=reverse, concat_var=concat_var)
            else:
                x, logpx, reg_states = self.chain[i](x, logpx, reg_states, reverse=reverse)

        return x, logpx, reg_states
