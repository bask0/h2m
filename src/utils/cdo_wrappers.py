"""
Cdo wrappers.

https://code.mpimet.mpg.de/projects/cdo/embedded/cdo.pdf
"""
from typing import Union, List
import os
import sys
from shutil import rmtree
import ray


def generate_lonlat(path: str, nlat: int, nlon: int) -> None:
    """Write gridspec file.

    Parameters
    ----------
    path
        Path to save gridspec file.
    nlat
        Number of latitude cells.
    nlon
        Number of longitude cells.
    """

    lat_res = 180 / nlat
    lon_res = 360 / nlon

    spec = (
        'gridtype = lonlat\n'
        f'xsize = {nlon}\n'
        f'ysize = {nlat}\n'
        f'xfirst = -{180-lon_res/2}\n'
        f'xinc = {lat_res}\n'
        f'yfirst = -{90-lat_res/2}\n'
        f'yinc = {lon_res}'
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='w') as f:
        f.write(spec)


def cdo_remap(
        in_files: Union[str, List[str]],
        out_files: Union[str, List[str]],
        nlat_target: int,
        nlon_target: int,
        remap_alg: str) -> None:
    """Wrapper to CDOs remap tools.

    ONLY FOR GLOBAL DATASETS!

    Parameters
    ----------
    in_files
        Input file(s). Must be on same grid, remapping will only be
        calculated once.
    out_files
        Output file(s), same structure as input.
    nlat_target
        Number of latitude cells.
    nlon_target
        Number of longitude cells.
    remap_alg: Remapping algorithm (see CDO doc), one of:
        bil 	Generate bilinear interpolation weights
        bic 	Generate bicubic interpolation weights
        dis 	Generate distance-weighted average remap weights
        nn 	    Generate nearest neighbor remap weights
        con 	Generate 1st order conservative remap weights
        con2 	Generate 2nd order conservative remap weights
        laf 	Generate largest area fraction remap weights

    """

    algos = (
        'bil',
        'bic',
        'dis',
        'nn',
        'con',
        'con2',
        'laf',
    )

    if remap_alg not in algos:
        raise ValueError(
            f'Remapping algorithm ``{remap_alg}`` not implemented, use '
            f'one of {algos}.')

    remap_alg = 'gen' + remap_alg

    if isinstance(in_files, str):
        in_files = [in_files]
    if isinstance(out_files, str):
        out_files = [out_files]

    if len(in_files) != len(out_files):
        raise ValueError(
            f'The number of ``in_files`` ({len(in_files)}) not equal to the '
            f'number of ``out_files`` ({len(out_files)}).'
        )

    tmp_dir = os.path.join(os.path.dirname(out_files[0]), 'tmp123o4u12sdfasae3')
    gridspec_path = os.path.join(tmp_dir, 'gridspec')
    weights_path = os.path.join(tmp_dir, 'remapweights.nc')

    try:
        generate_lonlat(gridspec_path, nlat_target, nlon_target)

        call = f'cdo {remap_alg},{gridspec_path} {in_files[0]} {weights_path}'
        print(call)
        os.system(call)

        for f_in, f_out in zip(in_files, out_files):
            call = f'cdo remap,{gridspec_path},{weights_path} {f_in} {f_out}'
            print(call)
            os.system(call)

        rmtree(tmp_dir)
    except Exception as e:
        if os.path.isdir(tmp_dir):
            rmtree(tmp_dir)
            for f_out in out_files:
                if os.path.isfile(f_out):
                    os.remove(f_out)
        if os.path.isdir(tmp_dir):
            rmtree(tmp_dir)
        raise e


def cdo_gridbox(
        in_files: Union[str, List[str]],
        out_files: Union[str, List[str]],
        nlat: int,
        nlon: int,
        remap_alg: str) -> None:
    """Wrapper to CDOs gridbox tools (see GRIDBOXSTAT).

    ONLY FOR GLOBAL DATASETS!

    Parameters
    ----------
    in_files
        Input file(s).
    out_files
        Output file(s), same structure as input.
    nlat_target
        Number of latitude cells.
    nlon_target
        Number of longitude cells.
    remap_alg: Remapping algorithm (see CDO doc), one of:
        min     Minimum value of the selected grid boxes.
        max     Maximum value of the selected grid boxes.
        range   Range(max-min value) of the selected grid boxes.
        sum     Sum of the selected grid boxes.
        mean    Mean of the selected grid boxes.
        avg     Average of the selected grid boxes.
        std     Standard deviation of the selected grid boxes. Normalize by n.
        std1    Standard deviation of the selected grid boxes. Normalize by(n-1).
        var     Variance of the selected grid boxes. Normalize by n.
        var1    Variance of the selected grid boxes. Normalize by(n-1).
    file_rename_fun
        An optional file renaming function to rename source -> target files. The function must
        take one argument, a string, and return a string. The default (None) does not change the
        file name. E.g.
        >>> cdo_remap(..., file_rename_fun=lambda x: x.replace('.nc', 'remapped.nc')).

    """

    algos = (
        'min',
        'max',
        'range',
        'sum',
        'mean',
        'avg',
        'std',
        'std1',
        'var',
        'var1'
    )

    if remap_alg not in algos:
        raise ValueError(
            f'Remapping algorithm ``{remap_alg}`` not implemented, use '
            f'one of {algos}.')

    remap_alg = 'gridbox' + remap_alg

    if isinstance(in_files, str):
        in_files = [in_files]
    if isinstance(out_files, str):
        out_files = [out_files]

    if len(in_files) != len(out_files):
        raise ValueError(
            f'The number of ``in_files`` ({len(in_files)}) not equal to the '
            f'number of ``out_files`` ({len(out_files)}).'
        )

    try:
        for f_in, f_out in zip(in_files, out_files):
            call = f'cdo {remap_alg},{nlon},{nlat} {f_in} {f_out}'
            print(call)
            os.system(call)
    except Exception as e:
        for f_out in out_files:
            if os.path.isfile(f_out):
                os.remove(f_out)
        raise e


def call_on_files(
        in_files: Union[str, List[str]],
        out_files: Union[str, List[str]],
        command: str,
        nworkers: int = 1,
        dryrun: bool = False) -> None:
    """System call of ``command`` for each ``in_file`` ``out_file`` combination.

    The function supports arbitrary calls (not only cdo, also gdal, nco or what ever).
    The function must accept `in_file out_file` as last argument, though.

    Example
    ----------
        Convert multiple files from .tif to netcdf. The following
        command does not actually convert the files but prints the
        commands as ``dryrun=True``.
        >>> call_on_files(
                in_files=['/in1.tif', '/in2.tif'],
                out_files=['/out1.nc', '/out2'],
                command='gdal_translate -of NETCDF',
                dryrun=True)
            gdal_translate -of NETCDF /in1.tif /out1.nc
            gdal_translate -of NETCDF /in2.tif /out2.nc

    Parameters
    ----------
    in_files
        Input file(s). Must be on same grid, remapping will only be
        calculated once.
    out_files
        Output file(s), same structure as input.
    command
        An arbitraty command to a function that takes 'in_file our_file' as
        last arguments.
    nworkers (default: 1)
        Number of workers ≥ 1.
    dryrun (default: False)
        If True, the command will pe printed but nut executed.

    """

    if isinstance(in_files, str):
        in_files = [in_files]
    if isinstance(out_files, str):
        out_files = [out_files]

    if len(in_files) != len(out_files):
        raise ValueError(
            f'The number of ``in_files`` ({len(in_files)}) not equal to the '
            f'number of ``out_files`` ({len(out_files)}).'
        )

    @ray.remote
    def call_single(command):
        print(f'System command:\n{"-" * 50}\n{command}\n{"-" * 50}')
        if not dryrun:
            os.system(command)
        return 1

    commands = [
        f'{command.strip()} {f_in} {f_out}' for f_in, f_out in zip(in_files, out_files)
    ]

    try:
        if nworkers == 1:
            for command in commands:
                print(f'System command:\n{"-" * 50}\n{command}\n{"-" * 50}')
                if not dryrun:
                    os.system(command)
        elif nworkers > 1:
            ray.init(num_cpus=nworkers)
            results = ray.get(
                [call_single.remote(command) for command in commands]
            )
            if len(results) != len(commands):
                raise AssertionError('Something went wrong.')
        else:
            raise ValueError('Arg ``nworkers`` must be an integer > 0.')

    except Exception as e:
        for f_out in out_files:
            if os.path.isfile(f_out):
                os.remove(f_out)
        raise e
