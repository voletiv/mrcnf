import six

import lib.layers as layers
import lib.layers.wrappers.cnf_regularization as reg_lib

REGULARIZATION_FNS = {
    "kinetic_energy": reg_lib.quadratic_cost,
    "jacobian_norm2": reg_lib.jacobian_frobenius_regularization_fn,
}
abbreviations = {"kinetic_energy":"KE", "jacobian_norm2":"JF"}

INV_REGULARIZATION_FNS = {v: k for k, v in six.iteritems(REGULARIZATION_FNS)}


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []
    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if arg_key in args:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(eval("args." + arg_key))
    regularization_fns = tuple(regularization_fns)
    regularization_coeffs = tuple(regularization_coeffs)
    return regularization_fns, regularization_coeffs


def get_regularization(model, regularization_coeffs):
    if len(regularization_coeffs) == 0:
        return None

    acc_reg_states = tuple([0.] * len(regularization_coeffs))

    for module in model.modules():
        if isinstance(module, layers.CNF):
            reg = module.get_regularization_states()
            acc_reg_states = tuple(acc_reg_states[i] + reg[i] for i in range(len(reg)))

    return acc_reg_states


def append_regularization_to_log(log_message, regularization_fns, reg_states):
    for i, reg_fn in enumerate(regularization_fns):
        log_message = log_message + " | " + abbreviations[INV_REGULARIZATION_FNS[reg_fn]] + ": {:.2e}".format(reg_states[i].item())
    return log_message


def append_regularization_keys_header(header, regularization_fns):
    for reg_fn in regularization_fns:
        header.append(INV_REGULARIZATION_FNS[reg_fn])
    return header


def append_regularization_csv_dict(d, regularization_fns, reg_states):
    for i, reg_fn in enumerate(regularization_fns):
        d[INV_REGULARIZATION_FNS[reg_fn]] = '{:.4f}'.format(reg_states[i].item())
    return d
