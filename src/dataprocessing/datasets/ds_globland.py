"""
Preprocess globland (land cover fractions from 30m lancover classification data).

globland30
http://www.globallandcover.com/GLC30Download/index.aspx

In:
Spatial:  0.0083 deg

Out:
Mask: Spatial: 0.5 deg
Fractions: Spatial: 0.033 deg

Steps:
1) Create mask.
2) Recalculate masked fractions for 1 deg.

"""

import os
import xarray as xr
import logging
import numpy as np

from utils.pyutils import rm_existing
from utils.cdo_wrappers import cdo_gridbox
from dataprocessing.plotting import plot_var
from dataprocessing.datasets.config import \
    dir_source, \
    dir_target

logging.info('Processing dataset: globland')

file_in = os.path.join(
    dir_source, '0d0083_static/globland30/v1/Data/globland30.fractional.v2.43200.21600.nc'
)
file_in_mask = os.path.join(
    dir_source, '0d50_static/globland30/v1/Data/globland30.fractional.v2.720.360.nc'
)
file_out = os.path.join(
    dir_target, 'processed/0d033/static/globland.nc'
)
file_out_mask = os.path.join(
    dir_target, 'processed/1d/static/globland_mask.nc'
)
file_tmp = file_out.replace('.nc', '_tmp.nc')
file_tmp_mask = file_out.replace('.nc', '_tmp.nc')

rm_existing(file_out)
rm_existing(file_out_mask)


os.makedirs(os.path.dirname(file_out), exist_ok=True)
os.makedirs(os.path.dirname(file_out_mask), exist_ok=True)

cdo_gridbox(
    in_files=file_in,
    out_files=file_tmp,
    nlat=4,
    nlon=4,
    remap_alg='mean'
)

ds_orig = xr.open_dataset(file_tmp)

ds = xr.Dataset()
ds['data'] = ds_orig.to_array().rename({'variable': 'var'})

# Oceans are NaN, fill Water_bodies with 100, other classes with 0.
fill_mask = ds.sel(var='Water_bodies').data.notnull()
ds = ds.where(ds.notnull(), 0.0)
ds_water_bodies = ds.sel(var='Water_bodies').data.where(fill_mask, 100)
ds.data.loc['Water_bodies', :, :] = ds_water_bodies

ds.to_netcdf(file_out)

# Mask.
ds = xr.open_dataset(file_out).coarsen({'lat': 30, 'lon': 30}).mean()
m = (ds.sel(var='Water_bodies').data < 50) & (ds.sel(var='Permanent_Snow_Ice').data < 90) & (
    ds.sel(var='Artificial_Surfaces').data < 90) & (ds.sel(var='Bareland').data < 90)
mask = xr.Dataset()
mask['data'] = m

# Coordinates slightly off after coarsening, round to 2 decimals.
mask.coords['lat'] = np.round(mask.coords['lat'], 2)
mask.coords['lon'] = np.round(mask.coords['lon'], 2)

mask.to_netcdf(file_out_mask)

rm_existing(file_tmp)
rm_existing(file_tmp_mask)

plot_path = __file__.replace('.py', '.jpg')
plot_var(path=file_out, plot_path=plot_path)

plot_path = __file__.replace('.py', '_mask.jpg')
plot_var(path=file_out_mask, plot_path=plot_path, is_2d=True)

logging.info('Done processing dataset: globland')
