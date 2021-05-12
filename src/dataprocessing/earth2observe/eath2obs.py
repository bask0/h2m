
import glob
import shutil
import os
import xarray as xr
import pandas as pd

root_dir = '/Net/Groups/BGI/work_2/EartH2O/'
target_dir = '/scratch/bkraft/hydrodl/data/EartH2O/'

models = ['csiro', 'jrc', 'metfr', 'uu']
variables = ['GroundMoist', 'SWE', 'SurfStor', 'RootMoist', 'TotMoist', 'CanopInt']

for model in models:
    model_root_dir = root_dir + model
    model_target_dir = target_dir + model
    os.makedirs(model_target_dir, exist_ok=True)
    ds_stack = []
    target_files = []
    for variable in variables:
        print(f'{model_root_dir}/*mon_{variable}*.nc')
        file = glob.glob(f'{model_root_dir}/*mon_{variable}*.nc')
        
        if len(file) == 0:
            print(f'{model}: {variable} not available.')
            continue
        elif len(file) > 1:
            print(f'{model}: {variable} request returned more than on file.')
            continue
        else:
            source_file = file[0]
            target_file = os.path.join(model_target_dir, os.path.basename(source_file))
            target_files.append(target_file)
            print(f'shutil.copy({source_file}, {target_file})')
            shutil.copy(source_file, target_file)

            ds = xr.open_dataset(target_file).sel(time=slice('2002', '2012'))

            for data_var in ds.data_vars:
                ds = ds.rename({data_var: data_var.lower()})

            if 'latitude' in ds:
                ds = ds.rename({'latitude': 'lat'})
            if 'longitude' in ds:
                ds = ds.rename({'longitude': 'lon'})

            ds = ds.coarsen({'lat': 2, 'lon': 2}).mean()
            ds = ds.sortby('lat', ascending=False)
            ds['time'] = pd.date_range(start='2002-01-01', end='2012-12-31', freq='MS') + pd.Timedelta(14, 'D')
            ds_stack.append(ds)
            
    ds_combined = xr.combine_by_coords(ds_stack)
    mask = xr.open_dataset('/scratch/bkraft/hydrodl/data/EartH2O/mask.nc').data
    ds_combined = ds_combined.where(mask)
    ds_combined.to_netcdf(os.path.join(model_target_dir, f'{model}_water_comp_2002_2012.nc'))

    for file in target_files:
        os.remove(file)