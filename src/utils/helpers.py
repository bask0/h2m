"""
Helpers that don't fit anywhere else.
"""

import torch
import numpy as np
import os


def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


def get_max_concurrent(ngpu, verbose=True):
    maxgpu = torch.cuda.device_count()
    maxcpu = os.cpu_count()

    max_concurrent = int(np.floor(maxgpu / ngpu))

    if verbose:
        print(
            f'\nStarting training\nAvailable resources: {maxgpu} GPUs  {maxcpu} CPUs\n'
            f'Max concurrent runs: {max_concurrent}.'
        )

    return max_concurrent


def trial_str_creator(trial):
    return f'trial_{trial.trial_id}'
