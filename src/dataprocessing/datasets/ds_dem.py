"""
Preprocess dem dataset.

GTOPO
https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-global-30-arc-second-elevation-gtopo30?qt-science_center_objects=0#qt-science_center_objects

In:
Spatial:  0.0083 deg

Out:
Spatial:  0.033 deg

Steps:
1) Harmonize
2) Regrid

"""

import os
import xarray as xr
import logging
import numpy as np

from utils.pyutils import exit_if_exists, rm_existing
from utils.cdo_wrappers import cdo_gridbox
from dataprocessing.plotting import plot_var
from dataprocessing.datasets.config import \
    dir_source, \
    dir_target, \
    overwrite

logging.info('Processing dataset: dem')

file_in = os.path.join(
    dir_source, '0d0083_static/GTOPO30/not_defined/Data/GTOPO30.43200.21600.nc'
)
file_out = os.path.join(
    dir_target, 'processed/0d033/static/dem.nc'
)
file_tmp = file_out.replace('.nc', '_tmp.nc')

exit_if_exists(file_out, overwrite)
os.makedirs(os.path.dirname(file_out), exist_ok=True)

cdo_gridbox(
    in_files=file_in,
    out_files=file_tmp,
    nlat=4,
    nlon=4,
    remap_alg='mean')

dem = xr.open_dataset(file_tmp)

# Coordinates are dropped by CDO (WHY??), add theme back.
dem = dem.rename({'GTOPO30': 'data', 'y': 'lat', 'x': 'lon'})
lat = np.convolve(
    np.arange(90, -90 - 1e-8, -1/30),
    np.ones(2) / 2,
    mode='valid')
lon = np.convolve(
    np.arange(-180, 180 + 1e-8, 1/30),
    np.ones(2) / 2,
    mode='valid')

dem['lat'] = lat
dem['lon'] = lon

dem['data'] = dem.data.expand_dims('var', 0)
dem['var'] = ['dem']

dem = dem.where(~dem.data.isnull(), 0)
dem.to_netcdf(file_out)

rm_existing(file_tmp)

plot_path = __file__.replace('.py', '.jpg')
plot_var(path=file_out, plot_path=plot_path)

logging.info('Done processing dataset: dem')
