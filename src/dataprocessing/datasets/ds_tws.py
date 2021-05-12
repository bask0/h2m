import os
import xarray as xr
import numpy as np
import logging

from utils.pyutils import exit_if_exists, rm_existing
from utils.cdo_wrappers import cdo_gridbox
from dataprocessing.plotting import plot_var
from dataprocessing.datasets.config import \
    years_targets, \
    dir_source, \
    dir_target, \
    tws_norm_period, \
    overwrite


logging.info('Processing dataset: tws')
logging.warning(
    f'TWS is normalized by the period {tws_norm_period[0]} to {tws_norm_period[1]}.'
    'You may want to change this to match a different test period.')

file_in = os.path.join(
    dir_source,
    '0d50_monthly/GRACE/mascons/RL06/Data/lwe_thickness.CRI.GRCTellus.JPL.200204_201706.GLO.RL06M_1.MSCNv01CRIv01.nc')
file_in_scale = '/workspace/bkraft/data/CLM4.SCALE_FACTOR.JPL.MSCNv01CRIv01.nc'
file_in_landmask = '/workspace/bkraft/data/LAND_MASK.CRIv01.nc'
file_in_human_impact = '/workspace/bkraft/data/noDirectHuman.tif'
file_out = os.path.join(
    dir_target,
    'processed/1d/monthly/tws_outlier_rm.nc'
)

file_tmp1 = file_out.replace('.nc', '_tmp1.nc')
file_tmp2 = file_out.replace('.nc', '_tmp2.nc')

exit_if_exists(file_out, overwrite)
os.makedirs(os.path.dirname(file_out), exist_ok=True)

TWS = xr.open_dataset(file_in)
sf = xr.open_dataset(file_in_scale)
sf.coords['lon'] = (sf.coords['lon'] + 180) % 360 - 180

# Somehow, CDO remapping breaks the dataset's time_bnds dimension. Fix: copy it from original dataset.
time_bnds = TWS.time_bounds

# Apply scale factor (https://grace.jpl.nasa.gov/data/get-data/jpl_global_mascons/)
# and multiply by 10 (cm -> mm).
TWS['lwe_thickness'] = TWS['lwe_thickness'] * sf['scale_factor'] * 10

TWS.to_netcdf(file_tmp1)

cdo_gridbox(
    in_files=file_tmp1,
    out_files=file_tmp2,
    nlat=2,
    nlon=2,
    remap_alg='mean')

TWS = xr.open_dataset(file_tmp2)

TWS = TWS.sel(time=slice(
    '{:d}-01-01'.format(years_targets[0]), '{:d}-12-31'.format(years_targets[-1])))
TWS = TWS - \
    TWS.sel(time=slice(tws_norm_period[0], tws_norm_period[1])).mean('time')

# Apply human impact mask.
hi = xr.open_rasterio(file_in_human_impact).rename(
    {'x': 'lon', 'y': 'lat'}).isel(band=0).drop('band') == -3.4028234663852886e+38
TWS = TWS.where(hi)

TWS['time_bnds'] = time_bnds

# Remove outliers (< -500 | > 500)
TWS['lwe_thickness'] = TWS.lwe_thickness.where(
    TWS.lwe_thickness > -500, np.nan)
TWS['lwe_thickness'] = TWS.lwe_thickness.where(
    TWS.lwe_thickness < 500, np.nan)

TWS.to_netcdf(file_out)

rm_existing(file_tmp1)
rm_existing(file_tmp2)

plot_path = __file__.replace('.py', '.jpg')

plot_var(
    path=file_out,
    plot_path=plot_path,
    pvar='lwe_thickness',
    time='2010-05-16')

logging.info('Done processing dataset: tws')
