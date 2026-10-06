from utc_src import config
from utc_src.process_data import derive_indices, aggregate_to_seasons
from utc_src.process_data import canopy_percent_from_lidar_landcover
from utc_src.spatial_sample import make_block_map, spatial_sample
import numpy as np
import os
import geopandas as gpd
import xarray as xr
import rioxarray
import shutil
from dask.distributed import Client as da_client
from dask.distributed import LocalCluster
import pandas as pd
import warnings
warnings.filterwarnings('ignore', message='Sending large graph', category=UserWarning)

##### set input #######
output_label = 'nyc21'
year = 2021

# if True, this will:
    # 1. derive input features from satellite data
    # 2. create a tree canopy layer from lidar landcover data
    # 3. create train/test datasets to train and validate a new model
# If False, input features will be derived for use with an existing model. 
train_new_model = True

######################
def main():

    cluster = LocalCluster(
            n_workers=4,              
            threads_per_worker=1,     
            memory_limit='6GB',       
        )
    dask_client = da_client(cluster)
    print(f'Open Dashboard: {dask_client.dashboard_link}')

    ref_path = config.DATA_DIR / output_label / f'{output_label}_daily_raw_aligned.zarr'

    ref_ds = xr.open_zarr(ref_path)
    ref = ref_ds['reflectance']

    ds_time = ref_ds.sizes['time']

    print(f'Calculating spectral indices for {ds_time} dates')
    derive_indices(ref=ref,output_label=output_label)

    # remove raw_aligned file
    ref_ds.close()
    shutil.rmtree(ref_path)
    print('Removed bands-only file')

    print('Calculating seasonal aggregated values')
    aggregate_to_seasons(output_label=output_label)

    print(f'Done. Features saved to {config.DATA_DIR} / {output_label} / {output_label}_seasonalstats.zarr')

    if not train_new_model:
        return
   
    ### make 10m tree canopy layer if it does not exist 
    if not os.path.exists(config.DATA_DIR / output_label / f'{output_label}_tree_canopy_lidar.tif'):
        if year == 2018:
            landcover_path = config.LIDAR_LC_2017
        elif year == 2021:
            landcover_path = config.LIDAR_LC_2021
        else:
            print(f'No lidar-based landcover data available for {year}')

        out_tif_path = config.DATA_DIR / output_label / f'{output_label}_tree_canopy_lidar.tif'
        ref_data_path = config.DATA_DIR / output_label / f'{output_label}_seasonalstats.zarr'

        print('Deriving 10 meter resolution tree canopy raster from 15 cm resolution lidar-based landcover')
        canopy_percent_from_lidar_landcover(landcover_path=landcover_path,out_tif_path=out_tif_path,ref_data_path=ref_data_path)
    else:
        print(f'Lidar-based tree canopy raster available for ground truth: {config.DATA_DIR} / {output_label} / {output_label}_tree_canopy_lidar.tif')

    ### sample data if train/test dfs don't exist
    if not os.path.exists(config.DATA_DIR /output_label/f'{output_label}_test_df_clf.parquet'):

        block_size = 100   # size of grid for train/test blocks
        pixels_per_block = 100   # number of pixels to sample from each block
        test_size = 0.3   ## proportion of blocks to be held out for testing

        lidar_raster = rioxarray.open_rasterio(config.DATA_DIR / output_label / f'{output_label}_tree_canopy_lidar.tif').squeeze('band',drop=True)
        features = xr.open_zarr(config.DATA_DIR/output_label/f'{output_label}_seasonalstats.zarr')
        features = features['features']

        assert features.sizes['x'] == lidar_raster.shape[1], 'lidar and feature rasters have different extents'
        assert features.sizes['y'] == lidar_raster.shape[0], 'lidar and feature rasters have different extents'
        assert all(features.y.values[:3] == lidar_raster.y.values[:3]), 'lidar and feature raster grids not aligned'
        assert all(features.x.values[:3] == lidar_raster.x.values[:3]), 'lidar and feature raster grids not aligned'

        # define inputs
        n_bands = features.sizes["band"]
        ny      = features.sizes["y"]
        nx      = features.sizes["x"]
        band_names = [f"band_{b}" for b in features.coords["band"].values]

        bands_flat = features.data.reshape(n_bands, ny * nx)
        canopy_flat = lidar_raster.values.ravel()
        canopy_2d = lidar_raster.values

        valid_2d = np.isfinite(lidar_raster.values)
        
        if not os.path.exists(config.DATA_DIR / 'train_test_blocks.npz'):
            block_map = make_block_map(valid_2d=valid_2d,block_size=block_size,test_size=test_size,ny=ny,nx=nx)
            np.savez(config.DATA_DIR / 'train_test_blocks.npz',block_map)
        else:
            block_map = np.load(config.DATA_DIR / 'train_test_blocks.npz')
            block_map = block_map['arr_0']

        print('sampling train/test data for classifier')
        spatial_sample(valid_2d,block_size,block_map,ny,nx,bands_flat,canopy_flat,band_names,pixels_per_block,output_label=output_label,model='clf',random_state=42)

        print('sampling train/test data for regressor')
        spatial_sample(valid_2d,block_size,block_map,ny,nx,bands_flat,canopy_flat,band_names,pixels_per_block,output_label=output_label,model='rgr',random_state=0)
    else:
        print(f'Train/test datasets for {output_label} exist')





if __name__=='__main__':
    main()