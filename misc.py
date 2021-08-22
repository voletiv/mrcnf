import six
import math

import lib.layers.wrappers.cnf_regularization as reg_lib
import lib.layers as layers
from lib.layers.odefunc import divergence_bf, divergence_approx


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def logpz_to_z(logpz):
    logZ = -0.5 * math.log(2 * math.pi)
    return (2*(logZ - logpz)).pow(0.5)


def set_step_size(step_size, model):

    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver_options['step_size'] = step_size

    model.apply(_set)


def set_cnf_options(args, model):

    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver = args.solver
            module.atol = args.atol
            module.rtol = args.rtol
            if args.step_size is not None:
                module.solver_options['step_size'] = args.step_size
            if args.first_step is not None:
                module.solver_options['first_step'] = args.first_step

            # If using fixed-grid adams, restrict order to not be too high.
            if args.solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.solver
            module.test_atol = args.test_atol if args.test_atol else args.atol
            module.test_rtol = args.test_rtol if args.test_rtol else args.rtol
            if args.test_step_size is not None:
                module.test_solver_options['step_size'] = args.test_step_size
            if args.test_first_step is not None:
                module.test_solver_options['first_step'] = args.test_first_step

    model.apply(_set)


def override_divergence_fn(model, divergence_fn):

    def _set(module):
        if isinstance(module, layers.ODEfunc):
            if divergence_fn == "brute_force":
                module.divergence_fn = divergence_bf
            elif divergence_fn == "approximate":
                module.divergence_fn = divergence_approx

    model.apply(_set)


def count_nfe(model):

    class AccNumEvals(object):

        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, layers.ODEfunc):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals


def count_training_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_total_time(model):

    class Accumulator(object):

        def __init__(self):
            self.total_time = 0

        def __call__(self, module):
            if isinstance(module, layers.CNF):
                self.total_time = self.total_time + module.sqrt_end_time * module.sqrt_end_time

    accumulator = Accumulator()
    model.apply(accumulator)
    return accumulator.total_time


# https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961
def calc_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)
