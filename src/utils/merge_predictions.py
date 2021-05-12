"""
Merge model predictions into zarr files.
"""

import argparse
from typing import Dict, Any, Iterable
import os
import glob
import numpy as np
import pickle
import xarray as xr
import time
import pandas as pd
from warnings import warn

from dataset import Bucket
from utils.pyutils import print_progress
from utils.experimentmanager import ExpManager

np.seterr(divide='ignore', invalid='ignore')

TASKS = ['tws', 'q', 'et', 'swe']
TASKS_MOTHLY = ['tws', 'q', 'et']
LATENT_VARS = ['winp', 'qf', 'qb', 'smrec',
               'gwrec', 'gw', 'ef', 'cwd', 'cwd_overflow',
               'qf_frac', 'smrec_frac', 'gwrec_frac',
               'sacc', 'smelt', 'snowc']
# LATENT_VARS = ['sacc', 'smelt', 'sfrac']
LATENT_STATIC_VARS = ['qb_frac', 'sscale']
SPINUP_LAST_DATE = '2008-12-31'


def parse_args() -> Dict[str, Any]:
    """Parse arguments.

    Returns
    --------
    Dict of arguments.

    """
    parser = argparse.ArgumentParser(
        description=(
            'Merge predictions from pickle files to netcdf.'
        ))

    parser.add_argument(
        '-p',
        '--path',
        type=str,
        help='path to experiment manager file',
        required=True
    )

    parser.add_argument(
        '-f',
        '--fold',
        type=int,
        help='fold ID (integer)',
        required=True
    )

    parser.add_argument(
        '-o',
        '--offset',
        nargs='+',
        type=int,
        help='offset (two integers)',
        required=True
    )

    parser.add_argument(
        '--is_spinup',
        action='store_true',
        help='if `true`, the data is the spinup periode'
    )

    parser.add_argument(
        '--is_all',
        action='store_true',
        help='if `true`, the data is the ful time-series / spatial predictions'
    )

    args = parser.parse_args()

    return args


def merge_predictions(
        path: str,
        fold: int,
        offset: Iterable[int],
        is_spinup: bool = False,
        is_all: bool = False):

    em = ExpManager.load_from_path(path)
    em.set_fold_paths(fold, offset)

    experiment = em.experiment
    name = em.name
    if is_spinup:
        path = em.predictions_spinup_dir
    else:
        if is_all:
            path = em.predictions_all_dir
        else:
            path = em.predictions_dir

    config = em.config

    tasks = config['tasks']

    print(f'Merging {"spinup " if is_spinup else ""}predictions...\n')

    pck_files = glob.glob(path + '/pred_*')
    if len(pck_files) == 0:
        if is_spinup:
            warn(f'No predictions found in dir `{path}`.')
            return
        else:
            raise ValueError(f'No predictions found in dir `{path}`.')

    bucket = Bucket(config['dataset'],
                    sample_formatter_path=config['dataconfig'])

    # We need to set space_id and rep_years as required by Bucket, but has no impact.
    bucket.set_space_ids({
        'train': [0],
        'valid': [0],
        'test': [0],
        'all': [0]})
    bucket.set_rep_years(0, ref_var='tair')

    sampler = bucket.get_sampler('all' if is_all else 'test')

    attrs_dict = {
        'experiment': experiment,
        'name': name,
        'description': f'Model prdictions from: {experiment}/{name}.',
        'author': 'Basil Kraft [bkraft@bgc-jena.mpg.de]',
        'created': time.strftime("%Y-%m-%d %H:%M")
    }

    print('  reading predictions...\n')
    predictions = []
    for pck_file_i, pck_file in enumerate(pck_files):
        with open(pck_file, 'rb') as f:
            predictions.append(pickle.load(f))

    agg_tasks = set(TASKS_MOTHLY).intersection(tasks)

    if not is_spinup:
        for task_i, task in enumerate(agg_tasks):
            print_progress(task_i, n_total=len(agg_tasks), prefix=f'  merging aggregated tasks...')

            ds = empty_like(sampler, task, attrs_dict)

            # dimensions: time x lat x lon
            arr = np.empty_like(ds.data)
            arr[:] = 0

            count = arr[:1, :, :].astype(np.int)

            # Iterate each predictions file.
            for pred_i, pred in enumerate(predictions):
                # Iterate prediction element (is a list of batches).
                for p_i, p in enumerate(pred['pred']):
                    for i, (lat, lon) in enumerate(zip(p['lat'], p['lon'])):
                        arr[:, lat, lon] += p[task + '_agg'][i, ..., 0]
                        count[:, lat, lon] += 1

            arr /= count
            mask = np.broadcast_to(count, arr.shape)
            arr[mask == 0] = np.nan

            ds.data[:] = arr

            file_path = os.path.join(path, task + '_monthly_pred.nc')
            if os.path.isfile(file_path):
                os.remove(file_path)

            ds.to_netcdf(file_path)

    p = predictions[0]['pred'][0]
    available_vars = list(p.keys())
    seq_len = p[TASKS[0]].shape[1]

    daily_paths = []
    daily_names = []
    for task_i, task in enumerate(tasks + LATENT_VARS):
        print_progress(task_i, n_total=len(tasks + LATENT_VARS),
                       prefix=f'  merging daily tasks...     ')

        if task not in available_vars:
            continue

        # The dataset 'swe' serves as reference for all variables as the aggregated
        # ones do not have daily original data.

        ds = empty_like(sampler, 'swe', attrs_dict)

        if is_spinup:
            # seq_len = config['num_spinup_years'] * 365
            ds = ds.isel(time=slice(0, seq_len))
            spinup_time = pd.date_range(
                pd.to_datetime(SPINUP_LAST_DATE) - pd.Timedelta(1824, unit='d'),
                pd.to_datetime(SPINUP_LAST_DATE))
            ds = ds.assign_coords(time=spinup_time)

        # dimensions: time x lat x lon
        arr = np.empty_like(ds.data)
        arr[:] = 0

        count = arr[:1, :, :].astype(np.int)

        # Iterate each predictions file.
        for pred_i, pred in enumerate(predictions):

            daily_bin = pred['daily_bin']

            # Iterate prediction element (is a list of batches).
            for p_i, p in enumerate(pred['pred']):
                for i, (lat, lon) in enumerate(zip(p['lat'], p['lon'])):
                    arr[:, lat, lon] += \
                        p[task][i, daily_bin[0]:daily_bin[1], ..., 0]
                    count[:, lat, lon] += 1

        arr /= count
        mask = np.broadcast_to(count, arr.shape)
        arr[mask == 0] = np.nan

        ds.data[:] = arr

        ds.data.attrs.update({'long_name': task + ' (mod)'})

        file_path = os.path.join(path, task + '_daily_pred.nc')
        if os.path.isfile(file_path):
            os.remove(file_path)
        ds.to_netcdf(file_path)

        daily_paths.append(file_path)
        daily_names.append(task)

    data_to_stack = []
    for pth, name in zip(daily_paths, daily_names):
        data_to_stack.append(
            xr.open_dataset(pth).rename(data=name)
        )

    print('  merging daily data to single file...\n')

    daily_data = xr.merge(data_to_stack)
    daily_data.attrs = {
        **attrs_dict,
        'description': 'Daily predictions.'}

    if not is_spinup:

        mask = daily_data[TASKS[0]].isel(time=0).notnull()

        rn = xr.open_zarr('/scratch/hydrodl/data/bucket.zarr/rn').data.sel(
            time=slice(daily_data.time[0], daily_data.time[-1]))
        rn = rn.where(mask)
        daily_data['rn'] = rn
        daily_data.rn.attrs.update({'long_name': 'Rn (obs)'})
        daily_data.rn.attrs.update({'units': 'W m-2'})

        prec = xr.open_zarr('/scratch/hydrodl/data/bucket.zarr/prec').data.sel(
            time=slice(daily_data.time[0], daily_data.time[-1]))
        prec = prec.where(mask)
        daily_data['prec'] = prec
        daily_data.prec.attrs.update({'long_name': 'Prec (obs)'})
        daily_data.prec.attrs.update({'units': 'mm'})

        tair = xr.open_zarr('/scratch/hydrodl/data/bucket.zarr/tair').data.sel(
            time=slice(daily_data.time[0], daily_data.time[-1]))
        tair = tair.where(mask)
        daily_data['tair'] = tair
        daily_data.tair.attrs.update({'long_name': 'Tair (obs)'})
        daily_data.tair.attrs.update({'units': 'K'})

    daily_data.to_netcdf(os.path.join(path, 'daily_pred.nc'))

    if not is_spinup:
        # Static encodings.
        n_enc = p['static_enc'].shape[1]

        ds = xr.Dataset({
            'data': xr.DataArray(
                coords=[np.arange(n_enc), ds.lat, ds.lon],
                dims=['d', 'lat', 'lon'])},
                attrs=attrs_dict)

        ds.attrs = {
            **attrs_dict,
            'long_name': 'static enc',
            'units': '-'
        }

        ds.data.attrs = {
            'long_name': 'static enc',
            'units': '-'
        }

        # dimensions: lat x lon x n_enc
        arr = np.zeros_like(ds.data)
        arr[:] = np.nan

        # Iterate each predictions file.
        for pred_i, pred in enumerate(predictions):

            # Iterate prediction element (is a list of batches).
            for p_i, p in enumerate(pred['pred']):

                for i, (lat, lon) in enumerate(zip(p['lat'], p['lon'])):
                    arr[:, lat, lon] = \
                        p['static_enc'][i, :]

        ds.data[:] = arr

        file_path = os.path.join(path, 'static_enc.nc')
        if os.path.isfile(file_path):
            os.remove(file_path)

        ds.to_netcdf(file_path)

        # Get static variables.
        print('  merging static variables')

        for var in LATENT_STATIC_VARS:
            ds = xr.Dataset({
                'data': xr.DataArray(
                    coords=[ds.lat, ds.lon],
                    dims=['lat', 'lon'])},
                attrs=attrs_dict)
            ds.attrs = {
                **attrs_dict,
                'long_name': var,
                'units': '-'
            }

            ds.data.attrs = {
                'long_name': var,
                'units': '-'
            }

            # dimensions: lat x lon x n_enc
            arr = np.zeros_like(ds.data)
            arr[:] = np.nan

            # Iterate each predictions file.
            for pred_i, pred in enumerate(predictions):

                # Iterate prediction element (is a list of batches).
                for p_i, p in enumerate(pred['pred']):

                    for i, (lat, lon) in enumerate(zip(p['lat'], p['lon'])):
                        if p[var].ndim == 1:
                            arr[lat, lon] = p[var]
                        elif p[var].ndim == 2:
                            arr[lat, lon] = p[var][i]

            ds.data[:] = arr

            file_path = os.path.join(path, f'{var}.nc')
            if os.path.isfile(file_path):
                os.remove(file_path)

            ds.to_netcdf(file_path)

        aoi_mask = ds.notnull()
        m = aoi_mask.where(aoi_mask == 1, drop=True)

        lat_min = m.lat.min()
        lat_max = m.lat.max()
        lon_min = m.lon.min()
        lon_max = m.lon.max()

    # Copy original data.
    if not is_spinup:
        print('  copying original data...\n')
        for task in TASKS:
            ds = xr.open_zarr(sampler.path, group=task).isel(time=slice(*sampler.get_slice(task))).load()

            # mask (in case AOI was restricted).
            ds = ds.where(
                (ds.lat >= lat_min) &
                (ds.lat <= lat_max) &
                (ds.lon >= lon_min) &
                (ds.lon <= lon_max))

            ds.attrs = {
                **attrs_dict,
                'units': sampler.get_unit(task)
            }

            ds.data.attrs = {
                **attrs_dict,
                'long_name': task + ' (obs)',
                'units': sampler.get_unit(task)
            }

            name_suffix = 'monthly' if task in TASKS_MOTHLY else 'daily'
            orig_path = os.path.join(
                path, task + f'_{name_suffix}_obs.nc')

            pred_path = os.path.join(
                path, task + f'_{name_suffix}_pred.nc')

            if os.path.isfile(orig_path):
                os.remove(orig_path)

            ds.to_netcdf(orig_path)

    # Also copy mask.
    ds = xr.open_zarr(sampler.path, group='mask')

    # Fix https://github.com/pydata/xarray/issues/2278
    del ds['data'].encoding['chunks']

    ds = ds.chunk({'lat': 30, 'lon': 30})
    ds.attrs = {
        'description': 'Cross validation mask; 0=Ignored, 1-5=Spatial folds.'}

    file_path = os.path.join(path, 'mask.nc')
    if os.path.isfile(file_path):
        os.remove(file_path)
    ds.to_netcdf(file_path)

    if not is_spinup:
        print(f'  aggregate daily tasks to monthly\n')
        # Aggregate tasks with no monthly original data to monthly.
        daily_tasks = set(tasks).difference(TASKS_MOTHLY)

        for task in daily_tasks:
            pred_path = os.path.join(path, task + '_daily_pred.nc')
            obs_path = os.path.join(path, task + '_daily_obs.nc')

            pred_monthly_path = os.path.join(path, task + '_monthly_pred.nc')
            obs_monthly_path = os.path.join(path, task + '_monthly_obs.nc')

            pred = xr.open_dataset(pred_path)
            obs = xr.open_dataset(obs_path)

            pred_m = pred.resample(time='m', keep_attrs=True, skipna=True, label='left').mean()
            obs_m = obs.resample(time='m', keep_attrs=True, skipna=True, label='left').mean()

            # Resampling assigns left date, we want middle of month.
            pred_m = pred_m.assign_coords(time=pred_m.time[:] + pd.Timedelta('16 days'))
            obs_m = obs_m.assign_coords(time=obs_m.time[:] + pd.Timedelta('16 days'))

            pred_m.to_netcdf(pred_monthly_path)
            obs_m.to_netcdf(obs_monthly_path)

    if not is_spinup:
        print('  masking and merging monthly data...\n')
        for task in tasks:
            path_pred = os.path.join(path, task + '_monthly_pred.nc')
            path_obs = os.path.join(path, task + '_monthly_obs.nc')

            pred = xr.open_dataset(path_pred)
            obs = xr.open_dataset(path_obs)

            pred.attrs.update({**attrs_dict})
            pred.data.attrs.update({**attrs_dict, 'units': sampler.get_unit(task)})

            obs.attrs.update({**attrs_dict})
            obs.data.attrs.update({**attrs_dict, 'units': sampler.get_unit(task)})

            mask = pred.notnull() & obs.notnull()

            pred_masked = pred.where(mask)
            obs_masked = obs.where(mask)

            pred.attrs.update({'description': 'monthly model predictions'})
            pred.data.attrs.update({'long_name': task + ' (mod)'})
            obs.attrs.update({'description': 'monthly observation'})
            obs.data.attrs.update({'long_name': task + ' (obs)'})
            pred_masked.attrs.update(
                {'description': 'monthly model predictions; pixels where `pred_raw` & `obs_raw` are not missing.'})
            pred_masked.data.attrs.update({'long_name': task + ' (mod)'})
            obs_masked.attrs.update(
                {'description': 'monthly observation; pixels where `pred_raw` & `obs_raw` are not missing'})
            obs_masked.data.attrs.update({'long_name': task + ' (obs)'})

            pred = pred.rename(data='mod_raw')
            obs = obs.rename(data='obs_raw')
            pred_masked = pred_masked.rename(data='mod')
            obs_masked = obs_masked.rename(data='obs')

            merged = xr.merge([
                pred,
                obs,
                pred_masked,
                obs_masked
            ], compat='override')

            merged.attrs.update(attrs_dict)

            file_masked_path = os.path.join(
                path, task + '.nc')
            merged.to_netcdf(file_masked_path)

            os.remove(path_pred)
            os.remove(path_obs)

    print('  cleaning up...\n')
    for pck_file in pck_files:
        os.remove(pck_file)

    for f in daily_paths:
        os.remove(f)

    print('Done\n')


def empty_like(
        sampler: Bucket,
        var: str,
        attrs_dict: Dict = {}) -> xr.Dataset:

    """Create an empty netcdf file to store predictions.

    Parameters
    ----------
    sapler
        Data bucket.
    variables
        Variables to create in zarr file.
    attrs_dict
        Attributes added to the zarr file.

    Returns
    ----------
    An xr.Dataset same as the given `var`.

    """

    ds = xr.open_zarr(sampler.path, group=var).isel(time=slice(*sampler.get_slice(var))).load()

    ds.attrs = {
        **attrs_dict,
        'units': sampler.get_unit(var)
    }

    ds.data.attrs = {
        **attrs_dict,
        'long_name': var,
        'units': sampler.get_unit(var)
    }

    return ds


if __name__ == '__main__':

    args = parse_args()

    merge_predictions(
        args.path, args.fold, args.offset, args.is_spinup, args.is_all
    )
