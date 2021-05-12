
import numpy as np
from typing import Callable


def exp_decay_with_warmup(warmup: int, gamma: float, min_val: float = 1e-8) -> Callable:
    """Returns exponential decay with warmup function.

    The function increases linearly from ``min_val`` to 1.0 until ``step`` is equal
    to warmup. For a ``step`` larger than ``warmup``, the function decays with a
    given ``gamma`` (last_val * gamma).

    Parameters
    ----------
    warmup
        The number of steps until which the function increases linearly.
    gamma
        The parameter of decay in (0, 1). Large numbers for slow decay.
    min_val
        The minimum lr factor that is used for the 0-th step, a small number > 0.

    Returns
    ----------
    A function taking the current step as single argument.

    """

    def f(x):
        return min_val + x * (1.0 - min_val) / warmup if x < warmup else gamma ** (x - warmup)

    return f


def cos_decay_with_warmup(warmup: int, T: int, start_val: float = 1e-8) -> Callable:
    """Returns cosine decay with warmup function.

    The function increases linearly from ``min_val`` to 1.0 until ``step`` is equal
    to warmup. For a ``step`` larger than ``warmup``, the function is a cosine decay
    (which is flatter that the exponential decay at beginning) and converges to 0
    asymptotically.

    Parameters
    ----------
    warmup : int
        The number of steps until which the function increases linearly from
        `min_val` to 1.
    T : int
        The number of steps after the warmup the function is (alsmost) 0. Note that the
        function converges to 0 asymptotically, it will never reach 0.
    start_val : float
        The minimum lr factor that is used for the 0-th step, a small number > 0.

    Returns
    ----------
    A function taking the current step as single argument.

    """

    def f(x):
        if x < 0:
            raise ValueError(f'``x`` must be ≥0 but is {x}.')
        elif x < warmup:
            return start_val + x * (1.0 - start_val) / warmup
        elif x < warmup + T:
            return 1 / 2 * (1 + np.cos((x - warmup) * np.pi / T))
        else:
            wt = warmup + T
            print(1/np.log2(x - wt + 3))
            return f(wt - 1/np.log2(x - wt + 3))

    return f
