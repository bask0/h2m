"""
Wrappers to execute functions in parallel.

The wrappers all work on input / output file basis, where each call handles the list
of files consecutively:
- 1st from input -> 1st output
- 2nd from input -> 2nd output
- etc.

The wrappers also handle single file calls where input and output are strings.

"""

import ray
from typing import Iterable, Dict, Callable, List
from datetime import datetime


def parcall(
        fun: Callable,
        iter_kwargs: List[Dict[str, Iterable]],
        num_cpus: int = 1,
        verbose: bool = True,
        dry_run: bool = False,
        ray_init_kwargs={
            'ignore_reinit_error': True
        },
        init_ray=True,
        **kwargs):
    """Run function in parallel on dict of of arguments.

    Notes
    ----------
    Consider a function ``f(a, b, c)``; we want to run the function with varying
    arguments a and b, while c remains constant:
    - f(a=1, b=2, c=10)
    - f(a=3, b=4, c=10)

    Now, run this function in parallel:
    iter_kwargs = [
        {'a': 1, 'b': 2},
        {'a': 3, 'b': 4}
    ]
    parcall(f, iter_kwargs, num_cpus=2, c=10)

    Parameters
    ----------
    fun
        Function to be executed in parallel. Must take arguments must take the
        input file path as first argument and if ``out_files`` is not ``None``
        the output file path as second argument, You can pass function arguments
        via ``fun_kwargs``.
    iter_kwargs
        A list of dicts, each dict being a parameter combination. All keys must
        match a keyword in ``fun``. See ``Notes``.
    num_cpus
        Number of CPUs to use.
    verbose
        If `True` (default), some information is printed.
    dry_run
        If `True`, some information is printed. Default is `False`.
    ray_init_kwargs
        Keayword arguments passed to ``ray.init``. Default is
        {'node_ip_address': '0.0.0.0'}.
    init_ray
        If `True` (defaut), ray is initialized, else not.
    kwargs:
        Keyword arguments passed to ``fun``. Use this for arguments that remain
        constant over the different calls. See ``Notes``.

    """

    if not (isinstance(iter_kwargs, tuple) or isinstance(iter_kwargs, list)):
        raise ValueError('argument `iter_kwargs` must be of type `tuple` or `list`.')
    for a in iter_kwargs:
        if not isinstance(a, dict):
            raise ValueError('one or more elements of `iter_kwargs` are not of '
                            'type `dict`.')

    @ray.remote
    def remote_fun(kwargs):
        return fun(**kwargs)

    if verbose:
        tic = datetime.now()
        print(f'{tic.strftime("%Y-%m-%d %H:%M:%S")} - Parallel execution of '
              f'function `{fun.__name__}` using {num_cpus} CPUs with {len(iter_kwargs)} total runs.')

    try:
        if dry_run:
            print('\033[92mDry run, function calls:\033[0m')
            for i, a in enumerate(iter_kwargs):
                s = ', '.join([f'{k}={v}' for k, v in {**a, **kwargs}.items()])
                print(f'\033[92m{i:5}: {fun.__name__}({s})\033[0m')
        else:
            if init_ray:
                ray.init(num_cpus=num_cpus, **ray_init_kwargs)
            results = ray.get(
                [remote_fun.remote({**a, **kwargs}) for a in iter_kwargs]
            )
            return results

    finally:
        if not dry_run:
            ray.shutdown()

        if verbose:
            toc = datetime.now()
            elapsed = toc - tic
            elapsed_per_el = elapsed / len(iter_kwargs)

            print(f'{toc.strftime("%Y-%m-%d %H:%M:%S")} - Done, elapsed time: {ptime(elapsed)} ({ptime(elapsed_per_el)} per call).')


def ptime(t):
    mins = t.seconds // 60
    secs = t.seconds - mins * 60
    return f'{t.seconds // 60} m {secs} s'
