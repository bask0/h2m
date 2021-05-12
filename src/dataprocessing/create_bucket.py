import numpy as np
import time
import datetime
import os
import xarray as xr

from dataset import SpatialData, Bucket
from utils.parallel import parcall

'''
--------------------------------------------------------------------
Create data bucket.

First, make sure that the datasets are up to date, the files to
create them are in the directory src/dataprocessing/datasets.
Then, create a new bucket with this script.

Configuration:
'''

bucket_path = '/scratch/hydrodl/data/bucket.zarr/'
config_path = '/workspace/hydrodl/src/data_config.json'
data_path = '/scratch/hydrodl/data/'
globland_mask = '/scratch/hydrodl/data/processed/1d/static/globland_mask.nc'
overwrite = True
nlat = 180
nlon = 360
chunk_size = 5
max_perc_missing_targets = 50

'''
--------------------------------------------------------------------
'''

tic = time.time()

print(
    '\n-----------------------------------------------------------------------\n'
    f'{datetime.datetime.now()}\n'
    'Create data bucket (this may take a while) at:\n'
    f'{os.path.abspath(bucket_path)}'
    '\n-----------------------------------------------------------------------\n'
)

print('  Defining datasets...\n')

et = SpatialData(
    path=data_path + 'processed/1d/monthly/et.nc',
    varname='ET',
    varname_new='et',
    hastimebin=True,
    timebinname='time_bnds',
    istarget=True,
    unit='mm d-1')
# et_daily = SpatialData(
#     path=data_path + 'processed/1d/daily/et.nc',
#     varname='ET',
#     varname_new='et_daily',
#     hastimebin=False,
#     istarget=True,
#     unit='mm d-1')
swe = SpatialData(
    path=data_path + 'processed/1d/daily/swe.nc',
    varname='val',
    varname_new='swe',
    istarget=True,
    unit='mm')
# swe = SpatialData(
#     path=data_path + 'processed/1d/monthly/swe.nc',
#     varname='val',
#     varname_new='swe',
#     hastimebin=True,
#     timebinname='time_bnds',
#     istarget=True,
#     unit='mm')
tws = SpatialData(
    path=data_path + 'processed/1d/monthly/tws_outlier_rm.nc',
    varname='lwe_thickness',
    varname_new='tws',
    hastimebin=True,
    timebinname='time_bnds',
    istarget=True,
    unit='mm')
q = SpatialData(
    path=data_path + 'processed/1d/monthly/q.nc',
    varname='Runoff',
    varname_new='q',
    hastimebin=True,
    timebinname='time_bnds',
    istarget=True,
    unit='mm d-1')

tair = SpatialData(
    path=data_path + 'processed/1d/daily/tair.nc',
    varname='tair',
    varname_new='tair',
    unit='K')
rn = SpatialData(
    path=data_path + 'processed/1d/daily/rn.nc',
    varname='Rn',
    varname_new='rn',
    unit='W m-2')
prec = SpatialData(
    path=data_path + 'processed/1d/daily/prec.nc',
    varname='Precip',
    varname_new='prec',
    unit='mm d-1')

# soilgrids = SpatialData(
#     path=data_path + 'processed/0d033/static/soilgrids.nc',
#     varname='data',
#     varname_new='soilgrids',
#     hastime=False,
#     unit='-')
# mcd = SpatialData(
#     path=data_path + 'processed/0d033/static/mcd.nc',
#     varname='data',
#     varname_new='mcd',
#     hastime=False,
#     unit='-')
# globland = SpatialData(
#     path=data_path + 'processed/0d033/static/globland.nc',
#     varname='data',
#     varname_new='globland',
#     hastime=False,
#     unit='% cover')
# dem = SpatialData(
#     path=data_path + 'processed/0d033/static/dem.nc',
#     varname='data',
#     varname_new='dem',
#     hastime=False,
#     unit='m a.s.l.')
static = SpatialData(
    path=data_path + 'processed/1d/static/static_enc.nc',
    varname='data',
    varname_new='static',
    hastime=False,
    unit='-')


def get_mask(ds):
    if ds.hastime:
        ntime = len(ds.var.time)
        perc_missing = ds.var.isnull().sum('time') / ntime * 100
        if ds.istarget:
            mask = perc_missing <= max_perc_missing_targets
        else:
            mask = perc_missing == 0
        return mask


def check_no_missing(ds):
    if ds.var.isnull().any():
        raise ValueError(
            f'Variables "{ds.varname_new}" has at least one NaN.')


check_missing = [dict(
    ds=static,
)]
parcall(iter_kwargs=check_missing, fun=check_no_missing, num_cpus=3)

#vars = dict(
#    ds=[et, swe, tws, q, tair, rn, prec],
#)
vars = [dict(
    ds=var,
) for var in [et, swe, tws, q, tair, rn, prec]]
masks = parcall(iter_kwargs=vars, fun=get_mask, num_cpus=4)

# Add mask for pixels where no TWS variation is present.
tws_mask = (tws.var.min('time') - tws.var.max('time')) != 0
masks.append(tws_mask)

# Add globland mask.
if not os.path.exists(globland_mask):
    raise ValueError(
        f'Mask file missing: {globland_mask}.\n',
        'Create the mask by calling `dataprocessing.ds_globland.py`. This mask is required.'
    )
globland_mask = xr.open_dataset(globland_mask).data
masks.append(globland_mask)

mask = xr.concat(masks, dim='ds').all('ds')
mask.name = 'data'


def stripes_like(xr_array, num_stripes):
    """Creates diagonally striped xr.DataArray, with repeated stribes
    from 1-``num_stripes``."""
    x = np.zeros_like(xr_array.data, dtype=int)
    i, j = np.indices(xr_array.data.shape)
    for s in range(num_stripes):
        x[(i+s) % num_stripes == j % num_stripes] = s + 1
    striped = xr.zeros_like(xr_array, dtype=int)
    striped.data = x
    return striped


def blocks_like(xr_array, num_sets, block_size):
    """Creates randomly blocked xr.DataArray with given block_size.
    from 1-``num_stripes``."""

    nlat = len(xr_array.lat)
    nlon = len(xr_array.lon)

    nlat_blocks = np.ceil(nlat / block_size).astype(np.int)
    nlon_blocks = np.ceil(nlon / block_size).astype(np.int)

    # Do lon increasing block.
    a = np.arange(nlon_blocks)
    a = np.tile(a, (nlat_blocks, 1))

    # Do lat increasing block.
    b = (np.arange(0, nlat_blocks) * nlon_blocks).reshape(-1, 1)
    b = b.repeat(nlon_blocks, axis=1)

    # Combine lona & lat, repeat to create blocks.
    b = (a + b).repeat(block_size, axis=0).repeat(block_size, axis=1)

    # Increment blocks to cv sets.
    b_unique = np.unique(b)
    blocks = np.zeros_like(b)
    np.random.shuffle(b_unique)
    sets = np.array_split(b_unique, num_sets)

    for i, set in enumerate(sets):
        blocks[np.isin(b, set)] = i + 1

    # Make xr.Dataset.
    m = xr.Dataset({'mask': xr.DataArray(blocks, coords=[
                   xr_array.lat, xr_array.lon], dims=['lat', 'lon'])})

    m = m.where(xr_array, 0)

    return m


mask = blocks_like(mask, 32, 3)

mask = mask.sortby(mask.lat, ascending=False)

mask_path = data_path + 'processed/1d/static/mask.nc'
if os.path.exists(mask_path):
    os.remove(mask_path)
mask.to_netcdf(mask_path)

print('  Defining mask...\n')
mask = SpatialData(
    path=mask_path,
    varname='mask',
    varname_new='land_mask',
    hastime=False,
    dtype=np.int8)

print('  Creating bucket...\n')
bucket = Bucket(
    path=bucket_path,
    nlat=nlat,
    nlon=nlon,
    mask=mask,
    read_only=False,
    overwrite=overwrite,
    chunk_size=chunk_size,
    sample_formatter_path=config_path)


print('  Writing datasets...')
print('   > Static variables')
# print('     > soilgrids')
# bucket.add(soilgrids)
# print('     > land cover fractions')
# bucket.add(mcd)
# print('     > globland')
# bucket.add(globland)
# print('     > digital elevation model')
# bucket.add(dem)
print('     > dem, soilgrids, globland')
bucket.add(static)
print('   > Dynamic variables')
print('     > air temperature')
bucket.add(tair)
print('     > net radiation')
bucket.add(rn)
print('     > precipitation')
bucket.add(prec)
print('     > evapotranspiration')
bucket.add(et)
print('     > snow water equivalent')
bucket.add(swe)
# print('     > snow water equivalent (daily)')
# bucket.add(swe_daily)
print('     > total water storage')
bucket.add(tws)
print('     > runoff')
bucket.add(q)

toc = time.time()

print(
    '\n-----------------------------------------------------------------------\n'
    f'{datetime.datetime.now()}\n'
    'Data bucket created at: \n'
    f'{os.path.abspath(bucket_path)}\n\n'
    f'Elapsed time: {int((toc-tic)/60)} min\n\n'
    f'{bucket}\n'
    '\n-----------------------------------------------------------------------\n'
)
