"""
Preprocess et dataset.

FLUXCOM

In:
Spatial:  0.5 deg
Temporal: daily

Out:
Spatial:  1 deg
Temporal: daily

Steps:
1) Harmonize
2) Missing 29. Feb in leap years, repeat 28. Feb
3) Regrid(mean of 2 x2 grid)
4) Stack years to single file

"""

import os
import xarray as xr
import logging
import pandas as pd
import numpy as np

from utils.pyutils import exit_if_exists, rm_existing
from dataprocessing.plotting import plot_var
from dataprocessing.datasets.config import \
    years_targets, \
    dir_bgi, \
    dir_target, \
    overwrite

logging.info('Processing dataset: et daily')
files_in = [os.path.join(
    dir_bgi,
    'work_3/FluxcomDataStructure/EnergyFluxes/RS_METEO/ensemble/era5/daily/'
    'LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-era5.720_360.daily.{:4d}.nc'.format(y))
    for y in years_targets
]
file_out = os.path.join(
    dir_target, 'processed/1d/daily/et.nc'
)
file_tmp = file_out.replace('.nc', '_tmp.nc')

exit_if_exists(file_out, overwrite)
os.makedirs(os.path.dirname(file_out), exist_ok=True)


def preproc_fun(ds):
    year = ds.encoding['source'][-7:-3]
    dates = pd.date_range(f'{year}-01-01', f'{year}-12-31')
    ds['time'] = dates
    return ds


ET = xr.open_mfdataset(files_in, combine='by_coords',
                       decode_times=False, preprocess=preproc_fun)

ET_new = xr.Dataset(
    coords={'lat': ET.lat.data, 'lon': ET.lon.data, 'time': ET.time.data})
# ET_new['ET'] = xr.DataArray(
#     ET.LE.values/2.45, coords=[ET.time.data, ET.lat.data, ET.lon.data], dims=['time', 'lat', 'lon'])

ET_new['ET'] = ET.LE/2.45
ET_new['ET_mad'] = ET.LE_mad / 2.45

ET_new = ET_new.coarsen(lon=2, lat=2).mean()

ET_new.to_netcdf(file_out)

plot_path = __file__.replace('.py', '.jpg')
plot_var(path=file_out, plot_path=plot_path, pvar='ET')

logging.info('Done processing dataset: et')
