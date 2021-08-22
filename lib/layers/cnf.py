import torch
import torch.nn as nn

from .torchdiffeq import odeint_adjoint as odeint
# from .torchdiffeq import odeint

from .wrappers.cnf_regularization import RegularizedODEfunc

__all__ = ["CNF"]


class ConcatODEfunc(nn.Module):
    def __init__(self, odefunc, concat_var):
        super().__init__()
        self.odefunc = odefunc
        self.concat_var = concat_var

    def forward(self, *states):
        return self.odefunc(*states, concat_var=self.concat_var)


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, steer_b=0.0,
                 regularization_fns=None, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        self.register_buffer("steer_b", torch.tensor(steer_b))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        self.test_solver_options = {}

    def forward(self, z, logpz=None, reg_states=tuple(), integration_times=None, reverse=False, concat_var=None):

        if not len(reg_states)==self.nreg and self.training:
            reg_states = tuple(torch.zeros(z.size(0)).to(z) for i in range(self.nreg))

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            end_time = self.sqrt_end_time * self.sqrt_end_time
            if self.steer_b > 0.0:
                end_time = end_time + (torch.rand(1)[0] - 0.5)/0.5*self.steer_b
            integration_times = torch.tensor([0.0, end_time]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()
        func = ConcatODEfunc(self.odefunc, concat_var)

        if self.training:
            state_t = odeint(
                func,
                (z, _logpz) + reg_states,
                integration_times.to(z),
                atol=[self.atol, self.atol] + [1e20] * len(reg_states) if self.solver in ['dopri5', 'bosh3'] else self.atol,
                rtol=[self.rtol, self.rtol] + [1e20] * len(reg_states) if self.solver in ['dopri5', 'bosh3'] else self.rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                func,
                (z, _logpz),
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
                options=self.test_solver_options,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        reg_states = state_t[2:]

        return z_t, logpz_t, reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
