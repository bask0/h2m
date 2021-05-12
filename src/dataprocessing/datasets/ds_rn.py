"""
Preprocess net radiadion dataset.

CERES/Ed4A:
https://icdc.cen.uni-hamburg.de/1/daten/atmosphere/ceres-radiation.html

In:
Spatial:  1 deg
Temporal: daily

Out:
Spatial:  1 deg
Temporal: daily

Steps:
1) Stack selected years to single datasets.
2) Harmonize.

"""

import os
import xarray as xr
import logging

from utils.pyutils import exit_if_exists
from dataprocessing.plotting import plot_var
from dataprocessing.datasets.config import \
    years_features, \
    dir_source, \
    dir_target, \
    overwrite

logging.info('Processing dataset: rn')
files_in = [os.path.join(
    dir_source,
    '1d00_daily/CERES/Ed4A/Data/Rn/Rn.360.180.{:03d}.{:4d}.nc'.format(366 if y % 4 == 0 else 365, y))
    for y in years_features
]

file_out = os.path.join(
    dir_target, 'processed/1d/daily/rn.nc'
)

exit_if_exists(file_out, overwrite)
os.makedirs(os.path.dirname(file_out), exist_ok=True)

data = xr.open_mfdataset(files_in)
data = data.rename({'latitude': 'lat', 'longitude': 'lon'})
data.to_netcdf(file_out)

plot_path = __file__.replace('.py', '.jpg')
plot_var(path=file_out, plot_path=plot_path)

logging.info('Done processing dataset: rn')
