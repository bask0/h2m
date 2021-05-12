# Reshape static datasets from var x lat x lon to lat_index, lon_index, var, lat, lon.

import os
import shutil
import numpy as np
import xarray as xr
from numcodecs import Blosc

from dataset import SpatialData

encoding = {
    'lat': {'compressor': None},
    'lon': {'compressor': None},
    'var': {'compressor': None},
    'lat_index': {'compressor': None},
    'lon_index': {'compressor': None},
    'data': {'dtype': np.float32, 'compressor': Blosc(clevel=1)}
}
chunks = {
    'lat': 10,
    'lon': 10,
    'var': -1,
    'lat_index': -1,
    'lon_index': -1,
}

print('Reshaping static data...')

print('  dem...')

dem_path = '/scratch/hydrodl/data/processed/0d033/static/dem.nc'
dem_out_path = '/scratch/hydrodl/data/processed/1d/static/dem_30_30.zarr'

if os.path.exists(dem_out_path):
    shutil.rmtree(dem_out_path)

dem = SpatialData(dem_path, varname='data', hastime=False)
dem = dem.get_writeable(scale_factor=30)
dem.chunk(chunks).to_zarr(
    dem_out_path, encoding=encoding)

print('  globland...')

globland_path = '/scratch/hydrodl/data/processed/0d033/static/globland.nc'
globland_out_path = '/scratch/hydrodl/data/processed/1d/static/globland_30_30.zarr'

if os.path.exists(globland_out_path):
    shutil.rmtree(globland_out_path)

globland = SpatialData(globland_path, varname='data', hastime=False)
globland = globland.get_writeable(scale_factor=30)
globland.chunk(chunks).to_zarr(
    globland_out_path, encoding=encoding)

print('  soilgrids...')

soilgrids_path = '/scratch/hydrodl/data/processed/0d033/static/soilgrids.nc'
soilgrids_out_path = '/scratch/hydrodl/data/processed/1d/static/soilgrids_30_30.zarr'

if os.path.exists(soilgrids_out_path):
    shutil.rmtree(soilgrids_out_path)

soilgrids = SpatialData(soilgrids_path, varname='data', hastime=False)
soilgrids = soilgrids.get_writeable(scale_factor=30)
soilgrids.chunk(chunks).to_zarr(
    soilgrids_out_path, encoding=encoding)

print('  wetlands...')

wetlands_path = '/scratch/hydrodl/data/processed/0d033/static/wetlands.nc'
wetlands_out_path = '/scratch/hydrodl/data/processed/1d/static/wetlands_30_30.zarr'

if os.path.exists(wetlands_out_path):
    shutil.rmtree(wetlands_out_path)

wetlands = SpatialData(wetlands_path, varname='data', hastime=False)
wetlands = wetlands.get_writeable(scale_factor=30)
wetlands.chunk(chunks).to_zarr(
    wetlands_out_path, encoding=encoding)

print('\n  merging datasets...')

encoding = {
    'lat': {'compressor': None},
    'lon': {'compressor': None},
    'd': {'compressor': None},
    'data': {'dtype': np.float32, 'compressor': Blosc(clevel=1)}
}
chunks = {
    'lat': 10,
    'lon': 10,
    'd': -1
}

static_out_path = '/scratch/hydrodl/data/processed/1d/static/static_30_30.zarr'

if os.path.exists(static_out_path):
    shutil.rmtree(static_out_path)

globland = xr.open_zarr(globland_out_path)
dem = xr.open_zarr(dem_out_path)
soilgrids = xr.open_zarr(soilgrids_out_path)
wetlands = xr.open_zarr(wetlands_out_path)

combined = xr.concat((globland, soilgrids, dem, wetlands), dim='var').rename(
    {'var': 'd'}).chunk(chunks)
combined.to_zarr(static_out_path, encoding=encoding)


print('Done!')
