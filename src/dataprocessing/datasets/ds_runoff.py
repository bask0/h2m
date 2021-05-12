"""
Preprocess runoff dataset.

GRUN_v1:
https://www.research-collection.ethz.ch/handle/20.500.11850/324386

In:
Spatial:  0.5 deg
Temporal: monthly

Out:
Spatial:  1 deg
Temporal: monthly

Steps:
1) Harmonize
2) Select years
2) Regrid(mean of 2x2 grid)

"""

import os
import xarray as xr
import logging
import pandas as pd
import numpy as np

from utils.pyutils import exit_if_exists, rm_existing
from utils.cdo_wrappers import cdo_gridbox
from dataprocessing.plotting import plot_var
from dataprocessing.datasets.config import \
    years_targets, \
    dir_target, \
    dir_ppl, \
    overwrite

logging.info('Processing dataset: runoff')
file_in = os.path.join(
    dir_ppl, 'hydrodl/data/raw/GRUN_v1_GSWP3_WGS84_05_1902_2014.nc'
)
file_out = os.path.join(
    dir_target, 'processed/1d/monthly/q.nc'
)
file_tmp = file_out.replace('.nc', '_tmp.nc')

exit_if_exists(file_out, overwrite)
os.makedirs(os.path.dirname(file_out), exist_ok=True)

cdo_gridbox(
    in_files=file_in,
    out_files=file_tmp,
    nlat=2,
    nlon=2,
    remap_alg='mean'
)

Q = xr.open_dataset(file_tmp)
Q = Q.sel(time=slice(
    '{:d}-01-01'.format(years_targets[0]), '{:d}-12-31'.format(years_targets[-1])))

month_bounds = np.concatenate((
    pd.date_range(
        start='{:d}-01-01'.format(years_targets[0]),
        end='{:d}-12-31'.format(
            years_targets[-1]), freq='MS').values.reshape(-1, 1),
    pd.date_range(
        start='{:d}-01-01'.format(years_targets[0]),
        end='{:d}-12-31'.format(
            years_targets[-1]), freq='M').values.reshape(-1, 1)), axis=1)
month_bounds = xr.DataArray(
    month_bounds, coords=[Q.time, xr.IndexVariable('bounds', [0, 1])])

Q['time_bnds'] = month_bounds
Q['time'] = Q.time + pd.Timedelta(15, 'D')
Q.to_netcdf(file_out)

rm_existing(file_tmp)

plot_path = __file__.replace('.py', '.jpg')
plot_var(path=file_out, plot_path=plot_path)

logging.info('Done processing dataset: runoff')
