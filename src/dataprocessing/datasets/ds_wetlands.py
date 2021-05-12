"""
Preprocess wetlands dataset.

TOOTCHI
https://doi.org/10.1594/PANGAEA.892657

In:
Spatial:  0.066 deg

Out:
Spatial:  0.033 deg

Steps:
1) Harmonize

"""

import os
import xarray as xr
import logging

from utils.pyutils import rm_existing
from dataprocessing.plotting import plot_var
from dataprocessing.datasets.config import \
    dir_source, \
    dir_target

logging.info('Processing dataset: globland')

file_in = os.path.join(
    dir_source, '0d06_static/wetlands/Tootchi_2019/Data/CW_TCI.fractions.10800.5400.nc'
)
file_out = os.path.join(
    dir_target, 'processed/0d033/static/wetlands.nc'
)

rm_existing(file_out)

os.makedirs(os.path.dirname(file_out), exist_ok=True)

ds = xr.open_dataset(file_in).rename({'latitude': 'lat', 'longitude': 'lon'})

ds_out = xr.Dataset()
ds_out['data'] = ds.to_array().rename({'variable': 'var'})
ds_out.attrs = ds.attrs

# Fill missing values with zeros.
ds_out = ds_out.where(ds_out.sel(var='none').notnull(), 0.0)

ds_out.to_netcdf(file_out)

plot_path = __file__.replace('.py', '.jpg')
plot_var(path=file_out, plot_path=plot_path)

logging.info('Done processing dataset: wetlands')
