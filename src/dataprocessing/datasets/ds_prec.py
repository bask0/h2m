"""
Preprocess precipitation dataset.

GPCP_v1_2:
https://climatedataguide.ucar.edu/climate-data/gpcp-daily-global-precipitation-climatology-project

In:
Spatial:  1 deg
Temporal: daily

Out:
Spatial:  1 deg
Temporal: daily

Steps:
1) Stack selected years to single datasets.

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

logging.info('Processing dataset: prec')

files_in = [os.path.join(
    dir_source,
    '1d00_daily/GPCP/v1_2/Data/v6/Precip.GPCP_v1_2.{:4d}.nc'.format(y))
    for y in years_features
]
file_out = os.path.join(
    dir_target, 'processed/1d/daily/prec.nc'
)

exit_if_exists(file_out, overwrite)
os.makedirs(os.path.dirname(file_out), exist_ok=True)

data = xr.open_mfdataset(files_in)
data.to_netcdf(file_out)

plot_path = __file__.replace('.py', '.jpg')
plot_var(path=file_out, plot_path=plot_path)

logging.info('Done processing dataset: prec')
