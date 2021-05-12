"""
NCO wrappers.
"""

import os


def ncremap(i: str, o: str, d: str = None, unpack: bool = False, no_area: bool=True, **kwargs):
    """Wrapper to NCOs ncremap.

    Example
    ----------
        >> remap(
            i='/input_05x05.nc',
            o='/output_10x10.nc'
            d='/reference_10_10.nc',
            D=1)

    Parameters
    ----------
    i
        Input file.
    o
        Output file.
    d
        Data file to infer destination grid from (optional).
    unpack
        If true, dataset will be decompressed (as compression may cause
        problems with ncremap).
    no_area
        If True (default), no area variable will be added.
    kwargs
        Keword arguments passed to ncremap (e.g. dbg_lvl or D).

    """

    if unpack:
        tmp_file = os.path.join(
            os.path.dirname(o), 'UNPACKED' + os.path.basename(o))

        unpacking_str = f'ncpdq -U {i} {tmp_file}'
        print(f'\n\nUnpacking...\n')
        os.system(unpacking_str)

        i = tmp_file

    c_str = f'ncremap -i {i} -o {o}'
    if d:
        c_str += f' -d {d}'
    if no_area:
        c_str += ' --no_cll_msr'
    for k, v in kwargs.items():
        c_str += ' '
        c_str += f'{"-" if len(k)==1 else "--"}{k} {v}'

    print(f'\n\nExecuting command\n{"-"*30}\n{c_str}\n{"-"*30}\n\n')
    os.system(c_str)

    try:
        os.system('rm ncremap_*')
    except Exception as e:
        pass

