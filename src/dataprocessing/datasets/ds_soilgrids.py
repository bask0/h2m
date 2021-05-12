"""
Preprocess soilgrids dataset.

GTOPO
https://soilgrids.org/#!/?layer=ORCDRC_M_sl2_250m&vector=1

In:
Spatial:  0.0083 deg

Out:
Spatial:  0.033 deg

Steps:
1) Harmonize
2) Regrid

"""

import os
import glob
import xarray as xr
import logging
import numpy as np

from utils.pyutils import exit_if_exists, rm_existing
from utils.parallel import parcall
from dataprocessing.plotting import plot_var
from dataprocessing.datasets.config import \
    dir_target, \
    overwrite

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

logging.info('Processing dataset: soilgrids')

dl_dir = os.path.join(dir_target, 'raw', 'soilgrids_1km')
varaible_file = os.path.join(os.path.dirname(
    __file__), 'ds_soilgrids_variables.txt')

with open(varaible_file) as f:
    var = f.read().split(',')


def dl_file(v):
    if os.path.isfile(os.path.join(dl_dir, v + '.tif')):
        print('File exists: ', v)
    else:
        print('Downloading file: ', v)
        os.system(
            f'wget -q -P {dl_dir} {"https://files.isric.org/soilgrids/data/aggregated/1km/" + v + ".tif"}')
        print('Done downloading file: ', v)


parcall(iterable=dict(v=var), fun=dl_file, num_cpus=8)

files_in = glob.glob(
    os.path.join(
        dl_dir, '*.tif'
    )
)
file_out = os.path.join(
    dir_target, 'processed/0d033/static/soilgrids.nc'
)
files_tmp = [os.path.join(
    os.path.dirname(file_out),
    os.path.basename(f).replace('.tif', '_tmp0.tif')) for f in files_in
]

exit_if_exists(file_out, overwrite)

# Scale factor relative to 1 deg resolution.
scale_factor = 120
lat = np.linspace(90-1/scale_factor/2, -90+1/scale_factor/2, num=180*scale_factor)


def harmonize(file_in, file_out):
    print('Processing ', os.path.basename(file_in))
    name = os.path.basename(file_in).split('.')[0]

    d = xr.open_rasterio(file_in).isel(band=0).drop('band')
    nodatavals = d.attrs['nodatavals']
    d = d.astype('float32')
    d.values[np.isin(d, nodatavals)] = np.nan
    d = d.rename({'x': 'lon', 'y': 'lat'})
    d = d.interp(coords={'lat': lat}, method='nearest',
                 kwargs={'fill_value': np.nan})
    d = d.coarsen({'lat': 4, 'lon': 4}).mean()
    d.name = name
    d = d.where(d.notnull(), 0)

    d.to_netcdf(file_out)


parcall(
    iterable=dict(file_in=files_in, file_out=files_tmp),
    fun=harmonize,
    num_cpus=8)


d = xr.open_mfdataset(files_tmp)

# Average over all soil depth layers, THIS PART IS OPTIONAL

if True:
    def sum_arrays(ds, names):
        return ds[names].to_array().mean('variable')

    d_agg = xr.Dataset()
    d_agg['BDTICM'] = d['BDTICM_M_1km_ll']
    d_agg['BBLDFI_mean'] = sum_arrays(
        d,
        ['BLDFIE_M_sl3_1km_ll',
        'BLDFIE_M_sl7_1km_ll',
        'BLDFIE_M_sl6_1km_ll',
        'BLDFIE_M_sl2_1km_ll',
        'BLDFIE_M_sl1_1km_ll',
        'BLDFIE_M_sl5_1km_ll',
        'BLDFIE_M_sl4_1km_ll'])
    d_agg['CLYPPT_mean'] = sum_arrays(
        d,
        ['CLYPPT_M_sl4_1km_ll',
        'CLYPPT_M_sl1_1km_ll',
        'CLYPPT_M_sl5_1km_ll',
        'CLYPPT_M_sl6_1km_ll',
        'CLYPPT_M_sl2_1km_ll',
        'CLYPPT_M_sl3_1km_ll',
        'CLYPPT_M_sl7_1km_ll'])
    d_agg['CRFVOL_mean'] = sum_arrays(
        d,
        ['CRFVOL_M_sl2_1km_ll',
        'CRFVOL_M_sl6_1km_ll',
        'CRFVOL_M_sl7_1km_ll',
        'CRFVOL_M_sl3_1km_ll',
        'CRFVOL_M_sl4_1km_ll',
        'CRFVOL_M_sl5_1km_ll',
        'CRFVOL_M_sl1_1km_ll'])
    d_agg['SLTPPT_mean'] = sum_arrays(
        d,
        ['SLTPPT_M_sl4_1km_ll',
        'SLTPPT_M_sl5_1km_ll',
        'SLTPPT_M_sl1_1km_ll',
        'SLTPPT_M_sl2_1km_ll',
        'SLTPPT_M_sl6_1km_ll',
        'SLTPPT_M_sl7_1km_ll',
        'SLTPPT_M_sl3_1km_ll'])
    d_agg['SNDPPT_mean'] = sum_arrays(
        d,
        ['SNDPPT_M_sl4_1km_ll',
        'SNDPPT_M_sl5_1km_ll',
        'SNDPPT_M_sl1_1km_ll',
        'SNDPPT_M_sl2_1km_ll',
        'SNDPPT_M_sl6_1km_ll',
        'SNDPPT_M_sl7_1km_ll',
        'SNDPPT_M_sl3_1km_ll'])
    d = d_agg

ds = xr.Dataset()
ds['data'] = d.to_array().rename({'variable': 'var'})
ds.to_netcdf(file_out)

rm_existing(files_tmp)

plot_path = __file__.replace('.py', '.jpg')
plot_var(path=file_out, plot_path=plot_path)

logging.info('Done processing dataset: mcd')
