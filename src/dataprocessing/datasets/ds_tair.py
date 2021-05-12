"""
Preprocess air temperature dataset.

CRUNCEP/v8/:
https://rda.ucar.edu/datasets/ds314.3/

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

from utils.pyutils import exit_if_exists, rm_existing
from utils.cdo_wrappers import cdo_gridbox
from dataprocessing.plotting import plot_var
from dataprocessing.datasets.config import \
    years_features, \
    dir_source, \
    dir_target, \
    overwrite

logging.info('Processing dataset: tair')
files_in = [os.path.join(
    dir_source,
    '0d50_daily/CRUNCEP/v8/Data/tair/tair.CRUNCEP.v8.720.360.{:04d}.nc'.format(y))
    for y in years_features
]
file_out = os.path.join(
    dir_target, 'processed/1d/daily/tair.nc'
)
files_tmp0 = [os.path.join(
    os.path.dirname(file_out),
    os.path.basename(f).replace('.nc', '_tmp0.nc')) for f in files_in
]
files_tmp1 = [os.path.join(
    os.path.dirname(file_out),
    os.path.basename(f).replace('.nc', '_tmp1.nc')) for f in files_in
]

exit_if_exists(file_out, overwrite)
os.makedirs(os.path.dirname(file_out), exist_ok=True)

cdo_gridbox(
    in_files=files_in,
    out_files=files_tmp0,
    nlat=2,
    nlon=2,
    remap_alg='mean'
)

for i, o, year in zip(files_tmp0, files_tmp1, years_features):
    if os.path.exists(o):
        os.remove(o)
    T = xr.open_dataset(i)
    if year % 4 == 0:
        # Leap year Feb.29. missing, repeat Feb. 28th where needed
        fill_date = '{:d}-02-29'.format(year) + pd.to_datetime(
            str(T.time[0].data)).strftime('T%H:%M:%S.%f')
        interp = T.interp(time=[fill_date])
        T = T.merge(interp)
        T.to_netcdf(o)
    else:
        T.to_netcdf(o)

data = xr.open_mfdataset(files_tmp1)

data.to_netcdf(file_out)

rm_existing(files_tmp0)
rm_existing(files_tmp1)

plot_path = __file__.replace('.py', '.jpg')
plot_var(path=file_out, plot_path=plot_path)

logging.info('Done processing dataset: tair')
