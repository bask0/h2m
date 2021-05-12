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

logging.info('Processing dataset: et')
files_in = [os.path.join(
    dir_bgi,
    'work_3/FluxcomDataStructure/EnergyFluxes/RS/ensemble/720_360/monthly/'
    'LE.RS.EBC-ALL.MLM-ALL.METEO-NONE.720_360.monthly.{:4d}.nc'.format(y))
    for y in years_targets
]
file_out = os.path.join(
    dir_target, 'processed/1d/monthly/et.nc'
)
file_tmp = file_out.replace('.nc', '_tmp.nc')

exit_if_exists(file_out, overwrite)
os.makedirs(os.path.dirname(file_out), exist_ok=True)

ET = xr.open_mfdataset(files_in)

ET_new = xr.Dataset(
    coords={'lat': ET.lat.data, 'lon': ET.lon.data, 'time': ET.time.data})
# ET_new['ET'] = xr.DataArray(
#     ET.LE.values/2.45, coords=[ET.time.data, ET.lat.data, ET.lon.data], dims=['time', 'lat', 'lon'])

ET_new['ET'] = ET.LE/2.45
ET_new['ET_mad'] = ET.LE_mad / 2.45

month_bounds = np.concatenate((
    pd.date_range(
        start='{:d}-01-01'.format(years_targets[0]),
        end='{:d}-12-31'.format(
            years_targets[-1]), freq='MS').values.reshape(-1, 1),
    pd.date_range(
        start='{:d}-01-01'.format(years_targets[0]),
        end='{:d}-12-31'.format(years_targets[-1]), freq='M').values.reshape(-1, 1)), axis=1)
month_bounds = xr.DataArray(
    month_bounds, coords=[ET_new.time, xr.IndexVariable('bounds', [0, 1])])
ET_new['time_bnds'] = month_bounds

ET_new = ET_new.coarsen(lon=2, lat=2).mean()

ET_new.to_netcdf(file_out)

plot_path = __file__.replace('.py', '.jpg')
plot_var(path=file_out, plot_path=plot_path, pvar='ET')

logging.info('Done processing dataset: et')
