import xarray as xr
import numpy as np
import os

from utils.metrics import xr_quantile
from utils.parallel import parcall


def add_msc_and_ano(ds, timescale='month', variables=['mod', 'obs'], label_months=True, add_yano=True):
    if variables == 'all':
        variables = list(ds.data_vars)
    for v in variables:
        msc = ds[v].groupby('time.' + timescale).mean('time',
                                                      keep_attrs=True).compute()
        ano = (ds[v].groupby('time.' + timescale) - msc).compute()
        if timescale == 'month':
            if label_months:
                msc['month'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                                'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            else:
                msc['month'] = np.arange(1, 13)
        ds[v + '_msc'] = msc
        ds[v + '_ano'] = ano
        if add_yano:
            yearly = ds[v].groupby('time.year').mean(
                'time', keep_attrs=True).compute()
            yearly -= yearly.mean('year', keep_attrs=True).compute()
            ds[v + '_yano'] = yearly
        ds[v + '_ano'].attrs = ds[v].attrs
        ds[v + '_msc'].attrs = ds[v].attrs
        ds[v + '_yano'].attrs = ds[v].attrs
    return ds


def subset_region(ds, r, region=['Subtropic', 'TemperateNH', 'Tropic', 'ColdNH', 'SH']):
    """Subset dataset bt a regional mask.

    Sets all values outside regions to np.nan.

    Parameters
    ----------
    ds: xarray.DataArray or xarray.Dataset
        The dataset to subset.
    regions : str or list of str
        The region name to select. If `region` is a list, an xarray.Dataset with
        a new dimension corresponding to the region names will be returned. If
        region is a string, an xarray.Dataset is returned.
    r : xr.Dataset
        The regions.

    Returns
    -------
    xarray.Dataset or xarray.DataArray

    """
    return ds.where(r[region], np.nan)


def coords2samp(path, cr, write_file=True, data_vars=None, **kwargs):
    ds = xr.open_dataset(path, **kwargs)
    if data_vars:
        ds = ds[data_vars]
    mask_path = os.path.join(os.path.dirname(path), 'cv_masks.nc').replace(
        'predictions_spinup', 'predictions')
    m = xr.open_dataset(mask_path).test
    ds['regions'] = cr.regions * m
    ds = ds.stack(sample=('lat', 'lon')).reset_index('sample')
    ds = ds.where(ds.regions > 0, drop=True)
    ds['sample'] = np.arange(len(ds.sample))
    if write_file:
        p = path.replace('.nc', 'regions.nc')
        if os.path.exists(p):
            os.remove(p)
        ds.to_netcdf(p)
    else:
        return ds


class PathIter(object):
    def __init__(
            self,
            pred_path,
            offsets=[[1, 0], [0, 1], [1, 1]],
            folds=range(5)):
        self.pred_path = pred_path
        self.offsets = offsets
        self.folds = folds

    def iter_offsets_star(self, filename):
        for offset in self.offsets:
            path = f'{self.pred_path}pred_{offset[0]}{offset[1]}_*/{filename}'
            yield path

    def iter_offsets(self, filename):
        for offset in self.offsets:
            folds = []
            for fold in self.folds:
                path = f'{self.pred_path}pred_{offset[0]}{offset[1]}_{fold}/{filename}'
                folds.append(path)
            yield folds

    def iter_paths(self, filename):
        for offset in self.offsets:
            for fold in self.folds:
                path = f'{self.pred_path}pred_{offset[0]}{offset[1]}_{fold}/{filename}'
                yield path


def preproc(ds):

    offset_lookup = {
        '01': 0,
        '10': 1,
        '11': 2
    }

    filepath = ds.encoding['source']

    offset_fold = filepath.split('pred_')[1]
    offset = offset_lookup[offset_fold[:2]]
    fold = int(offset_fold[3])

    ds = ds.expand_dims(dim={'fold': [fold]})
    ds = ds.expand_dims(dim={'offset': [offset]})

    return ds


def preproc_aggr(ds):
    regions_lookup = {
        'SH': 5,
        'Tropic': 3,
        'Subtropic': 1,
        'TemperateNH': 2,
        'ColdNH': 4
    }
    data_units = {
        'swe': 'mm',
        'et': 'mm d-1',
        'q': 'mm d-1',
        'tws': 'mm',
        'winp': 'mm d-1',
        'qf': 'mm d-1',
        'qb': 'mm d-1',
        'smrec': 'mm d-1',
        'gwrec': 'mm d-1',
        'gw': 'mm',
        'cwd': 'mm',
        'cwd_overflow': 'mm d-1',
        'qf_frac': '-',
        'smrec_frac': '-',
        'gwrec_frac': '-',
        'smelt': 'mm d-1',
        'sacc': 'mm d-1',
        'sfrac': '-',
        'ef': '-'
    }

    ds = preproc(ds)

    ds_new_list = []
    regions_list = []
    for r, r_i in regions_lookup.items():
        regions_list.append(r)
        ds_r = ds.where(ds.regions == r_i)
        ds_new = ds_r.mean('sample', keep_attrs=True)
        for var in data_units.keys():
            ds_new[var + '_q'] = xr_quantile(ds_r[var],
                                             [0.2, 0.5, 0.8], dim='sample')
        for k, v in data_units.items():
            ds_new[k].attrs['units']: v
            ds_new[k + '_q'].attrs['units']: v
        ds_new_list.append(ds_new)
    ds_combined = xr.combine_nested(ds_new_list, concat_dim='region')
    ds_combined['region'] = regions_list
    return ds_combined


def postprocess(
        offsets=[[1, 0], [0, 1], [1, 1]],
        folds=range(5),
        is_all=False):

    print('Processing predictions...')

    cr = xr.open_dataset(
        '/scratch/hydrodl/data/processed/1d/static/clusterRegions.nc')
    # regions = ['Subtropic', 'TemperateNH', 'Tropic', 'ColdNH', 'SH']

    if is_all:
        pred_path = '/scratch/hydrodl/experiments/hybrid/all_vars_task_weighting/cv/predictions/'
    else:
        pred_path = '/scratch/hydrodl/experiments/hybrid/all_vars_task_weighting/cv/predictions_all/'

    paths = []
    i = 0
    for offset_i, offset in enumerate(offsets):
        for fold in folds:
            path = f'{pred_path}pred_{offset[0]}{offset[1]}_{fold}/daily_pred.nc'
            paths.append({'path': path})
            i += 1

    print('  processing folds')

    pathiter = PathIter(pred_path, offsets, folds)
    parcall(coords2samp, paths, cr=cr, num_cpus=6,
            write_file=True, chunks={'time': 400})

    print('  combining folds')

    p = f'{pred_path}daily_metrics_combined.nc'
    if os.path.exists(p):
        os.remove(p)

    ds = [xr.open_mfdataset(p, combine='by_coords', preprocess=preproc_aggr, chunks={
                            'time': 400}) for p in pathiter.iter_offsets('daily_predregions.nc')]

    xr.combine_by_coords(ds).to_netcdf(p)

    if not is_all:
        print('Processing spinup...')

        pred_path = '/scratch/hydrodl/experiments/hybrid/all_vars_task_weighting/cv/predictions_spinup/'

        paths = []
        i = 0
        for offset_i, offset in enumerate(offsets):
            for fold in folds:
                path = f'{pred_path}pred_{offset[0]}{offset[1]}_{fold}/daily_pred.nc'
                paths.append({'path': path})
                i += 1

        print('  processing folds')

        pathiter = PathIter(pred_path, offsets, folds)
        parcall(coords2samp, paths, cr=cr, num_cpus=6,
                write_file=True, chunks={'time': 400})
