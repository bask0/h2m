"""
Data reshaping etc.
"""

import torch
import numpy as np
from typing import Iterable, Dict, Any, List


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Detach and convert pytorch tensors to numpy.

    Parameters
    ----------
    x
        A tensor or a dict of tensors to detach and convert to numpy.

    Returns
    ----------
    detached: numpy array

    """
    if isinstance(x, dict):
        r = {}
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                if v.device.type == 'cuda':
                    r.update({k: v.detach().cpu().numpy()})
                else:
                    r.update({k: v.detach().numpy()})
            else:
                r.update({k: v})
        return r
    else:
        if x.device.type == 'cuda':
            return x.detach().cpu().numpy()
        else:
            return x.detach().numpy()


def params_from_optimizer(
        optimizer: torch.optim.Optimizer,
        param_names: Iterable[str] = ['optim']) -> Dict[str, Any]:
    """Create a dict of optimizer parameters.

    Parameters
    ----------
    ompimizer
        An optimizer to get parameters from.
    name
        This is added in front of parameter, e.g. parameter 'lr' becomes
        'optim_lr'. If there are multiple parameter groups, the names
        correspond to these groups.

    Returns
    ----------
    param_dict
        A dict of parameters and their values.

    """

    param_groups = optimizer.state_dict()['param_groups']
    if len(param_groups) != len(param_names):
        raise ValueError('The number of parameter groups in ``optimizer`` '
                         '({}) must match the number of ``param_names`` ({}).'.format(
                             len(param_groups), len(param_names)))

    optim_params = {}
    for i, params in enumerate(param_groups):
        params.pop('params')
        for k, v in params.items():
            optim_params.update({param_names[i] + '_' + k: v})
    return optim_params


def batch_to_device(
        batch: Dict[str, torch.Tensor],
        device: str,
        test_features_nan: List[str] = []) -> Dict[str, torch.Tensor]:
    """Move batch of type ``dict`` to a device.

    Parameters
    ----------
    batch
        The batch containing the feature and the target data.
    device
        The device to move tensors to.
    test_features_nan (default: True)
        If True, features will be checked for NaN and error is raised if NaN found.

    Returns
    ----------
    data: Batch on device ``device``.

    """

    if device == 'cuda':
        data = {}
        for k, v in batch.items():
            if (k in test_features_nan):
                if torch.isnan(v).any():
                    raise ValueError(
                        'NaN in {}, training stopped.'.format(k))
            data.update({k: v.to(device, non_blocking=True)})
        return data
    elif device == 'cpu':
        return batch
    else:
        raise ValueError(f'Device {device} not understood, should be one of "cpu", "cuda".')


def standardize(
        x: torch.Tensor,
        stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Standardize a tensor using precalculated global mean and std.

    Parameters
    ----------
    x
        A tensor to be standardized.
    stats
        A dict with keys ``mean`` and ``std``.

    Returns
    ----------
    Standardized x.
    """

    x_scaled = (x - stats['mean']) / stats['std']
    return x_scaled


def unstandardize(
        x: torch.Tensor,
        stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Reverse-standardize a tensor using precalculated global mean and std.

    Parameters
    ----------
    x
        A tensor to be reverse-standardized.
    stats
        A dict with keys ``mean`` and ``std``.

    Returns
    ----------
    Reverse-standardized x.

    """
    x_scaled = x * stats['std'] + stats['mean']
    return x_scaled


def tensordict2items(d: Dict, to_cpu: bool = True) -> Dict:
    """Convert tensors in dict to a standard Python number.

    If a tensor with more than one element is present, an error will be raised.

    Parameters
    ----------
    d
        A dict with some single valued torch Tensors.
    to_cpu
        Wheter to move tensor to cpu.

    Returns
    ----------
    A dict of standard python types.

    """

    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            if to_cpu:
                v = v.cpu()
            d[k] = v.item()

    return d
