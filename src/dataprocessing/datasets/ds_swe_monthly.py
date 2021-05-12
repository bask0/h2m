# -*- coding: utf-8 -*-
"""
Preprocess snow water equivalent dataset.

CRUNCEP/v8/:
http://www.globsnow.info/

In:
Spatial:  0.25 deg
Temporal: daily

Out:
Spatial:  1 deg
Temporal: monthly

Steps:
1) Harmonize
2) Gapfilling: Fill SWE values with 0 where 24 days of data are missing (SWE
   has missing values where no snow, and entire Southern Hemisphere) and the
   mean over the same window iw below 10 in the snow cover fraction dataset.
   This is a very conservative gapfilling for pixel-time-steps where we are
   very confident that no snow is present.

"""

import os
import xarray as xr
import logging
import numpy as np
import pandas as pd

from utils.pyutils import exit_if_exists, rm_existing
from utils.cdo_wrappers import cdo_gridbox
from dataprocessing.plotting import plot_var
from dataprocessing.datasets.config import \
    years_targets, \
    dir_source, \
    dir_target, \
    overwrite

logging.info('Processing dataset: swe')
files_in = [os.path.join(
    dir_source,
    '0d25_daily/Globsnow_SWE/v2/Data/SWE.1440.720.{:4d}.nc'.format(y))
    for y in years_targets
]
file_out = os.path.join(
    dir_target, 'processed/1d/monthly/swe.nc'
)
file_tmp0 = file_out.replace('.nc', '_tmp0.nc')
file_tmp1 = file_out.replace('.nc', '_tmp1.nc')

files_scf_in = [os.path.join(
    dir_source,
    '1d00_8daily/MODIS/MOD10C2.006/Data/Eight_Day_CMG_Snow_Cover/Eight_Day_CMG_Snow_Cover.360.180.{:4d}.nc'.format(y))
    for y in years_targets
]
files_scf_out = os.path.join(
    dir_target, 'processed/1d/8daily/swe.nc'
)

exit_if_exists(file_out, overwrite)
os.makedirs(os.path.dirname(file_out), exist_ok=True)
os.makedirs(os.path.dirname(files_scf_out), exist_ok=True)

scf = xr.open_mfdataset(files_scf_in)
scf = scf.rename({
    'Eight_Day_CMG_Snow_Cover': 'val',
    'latitude': 'lat',
    'longitude': 'lon'})
scf.to_netcdf(files_scf_out)

# Drop SWE_var as not needed, stack datasets.
# Set -1 and -2 to nan as these are masked / non-land values.
swe_stack = []
for p in files_in:
    d = xr.open_dataset(p).drop('SWE_var').rename({'SWE': 'val'})
    d = d.where(d.val != -2, np.nan)
    d = d.where(d.val != -1, np.nan)
    swe_stack.append(d)

# Fix missing attributes for lat and lon (because of cdo).
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
swe = xr.concat(swe_stack, dim='time')
swe.lat.attrs.update(lat_attrs)
swe.lon.attrs.update(lon_attrs)
swe.to_netcdf(file_tmp0)
swe.close()

# Remap to 1° resolution.
cdo_gridbox(
    in_files=file_tmp0,
    out_files=file_tmp1,
    nlat=4,
    nlon=4,
    remap_alg='mean')

# Find pixel-time-steps with where the mean in a rolling window of 3
# observations (24 days) is below a threshold.
scf = xr.open_dataset(files_scf_out)
scf_threshold = 10
scf_red = scf.rolling(time=3, min_periods=3, center=True).mean()
# Put last observarion back as this is dropped above.
fill_slice = scf_red.isel(time=0)
fill_slice['time'] = [np.datetime64('{}-12-31'.format(years_targets[-1]))]
scf_red = scf_red.merge(fill_slice)
scf_red_mask = scf_red < scf_threshold

# Resample to daily resolution. This will take the nearest neighbor in the time
# dimension, but only if in range of 4 days, such that missing values persist.
scf_red_mask_1d = scf_red_mask.resample(time='1D').nearest(
    tolerance='4D').sel(time=slice('{}-01-01'.format(years_targets[0]), None))

# Get mask per pixel-time-step where in a window of 24 days window all pixels are
# missing. We only fill those values.
swe = xr.open_dataset(file_tmp1)
swe_missing = swe.isnull()
swe_num_missing = swe_missing.rolling(time=24).sum().sel(
    time=slice('{}-01-01'.format(years_targets[0]), None))
swe_num_missing_mask = swe_num_missing == 24

# Fill gaps with 0.
scf_red_mask_1d['time'] = swe_num_missing_mask.time
fill_mask = scf_red_mask_1d.val * swe_num_missing_mask.val.astype(np.float)

swe_timesubs = swe.sel(time=slice('{}-01-01'.format(years_targets[0]), None))
swe_gapfiled = swe_timesubs.where(1-fill_mask, 0)
scf_land_mask = scf.val.notnull().any('time')
swe_gapfiled = swe_gapfiled.where(scf_land_mask, np.nan)

# to monthly.
swe_gapfiled = swe_gapfiled.resample(time='MS', keep_attrs=True, skipna=True).mean()
month_bounds = np.concatenate((
    pd.date_range(
        start='{:d}-01-01'.format(years_targets[0]),
        end='{:d}-12-31'.format(
            years_targets[-1]), freq='MS').values.reshape(-1, 1),
    pd.date_range(
        start='{:d}-01-01'.format(years_targets[0]),
        end='{:d}-12-31'.format(years_targets[-1]), freq='M').values.reshape(-1, 1)), axis=1)

month_bounds = xr.DataArray(
    month_bounds, coords=[swe_gapfiled.time, xr.IndexVariable('bounds', [0, 1])])
swe_gapfiled['time_bnds'] = month_bounds
swe_gapfiled['time'] = swe_gapfiled.time + pd.Timedelta(15, 'D')

swe_gapfiled.to_netcdf(file_out)

rm_existing(file_tmp0)
rm_existing(file_tmp1)
rm_existing(files_scf_out)

plot_path = __file__.replace('.py', '.jpg')
plot_var(path=file_out, plot_path=plot_path)

logging.info('Done processing dataset: swe')
