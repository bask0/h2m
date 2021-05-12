"""
Preprocess mcd (modis land cover) dataset.

MODIS land cover fractions
https://lpdaac.usgs.gov/product_search/?collections=Combined+MODIS&collections=Terra+MODIS&collections=Aqua+MODIS&view=list

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
from utils.cdo_wrappers import cdo_remap
from dataprocessing.plotting import plot_var
from dataprocessing.datasets.config import \
    dir_source, \
    dir_target, \
    overwrite

logging.info('Processing dataset: mcd')

file_in = os.path.join(
    dir_source, '0d0083_static/MCD12Q1/V005/Data/v005_2/MCD12Q1plusC4_fraction.GLOBAL01KM.2001001.LC.01KM.nc'
)
file_out = os.path.join(
    dir_target, 'processed/0d033/static/mcd.nc'
)
file_tmp = file_out.replace('.nc', '_tmp.nc')

exit_if_exists(file_out, overwrite)
os.makedirs(os.path.dirname(file_out), exist_ok=True)


ds = xr.open_dataset(file_in)
ds = ds.rename({
    'MCD12Q1plusC4_fraction': 'data',
    'longitude': 'lon',
    'latitude': 'lat'})

lat_attrs = dict(
    long_name='Latitude',
    standard_name='latitude',
    units='degrees_north',
    axis='Y',
    valid_min=-90.0,
    valid_max=90.0
)
lon_attrs = dict(
    long_name='Longitude',
    standard_name='longitude',
    units='degrees_east',
    axis='X',
    modulo=360.0,
    topology='circular',
    valid_min=-180.0,
    valid_max=180.0,
)
ds.lat.attrs.update(lat_attrs)
ds.lon.attrs.update(lon_attrs)

ds.attrs['classes'] = np.array([
    l[1][1:].decode('utf-8').replace(' ', '_').lower()
    for l in ds.Legend.values
    ])[:-2]

ds = ds.drop('Legend')
ds['data'] = ds.data.expand_dims('var', 0)

ds = ds.where(~ds.data.isnull(), 0)

ds.to_netcdf(file_tmp)
ds.close()

cdo_remap(
    in_files=file_tmp,
    out_files=file_out,
    nlat_target=180*30,
    nlon_target=360*30,
    remap_alg='laf')

rm_existing(file_tmp)

plot_path = __file__.replace('.py', '.jpg')
plot_var(path=file_out, plot_path=plot_path)

logging.info('Done processing dataset: mcd')
