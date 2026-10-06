from utc_src import config
from utc_src.process_data import build_daily_timeseries_incremental, coregister_images
import geopandas as gpd
import xarray as xr
import rioxarray
import shutil
from dask.distributed import Client as da_client
from dask.distributed import LocalCluster
import pandas as pd


 
###### set inputs ###################################

output_label = 'nyc21'
year = 2021

##################################################

def main():
    boundary = gpd.read_file(config.BOUNDARY)


    if boundary.crs is None:
        raise ValueError('Boundary has no CRS defined')
    elif boundary.crs.to_epsg() != config.EPSG:
        print(f'Transforming boundary CRS from {boundary.crs.to_epsg()} to {config.EPSG}')
        boundary = boundary.to_crs(config.EPSG)

    cluster = LocalCluster(
        n_workers=4,              
        threads_per_worker=1,     
        memory_limit='6GB',       
    )
    dask_client = da_client(cluster)
    print(f'Open Dashboard: {dask_client.dashboard_link}')


    ## download march - november timeseries of all available data acquisitions.
    ## data is accessed and written by month. the process resumes automatically if API connection fails.
    ## can take 80 - 90 minutes to complete. open the dask dashboard to track progress.

    raw_path = config.DATA_DIR / output_label / f'{output_label}_daily_raw.zarr'

    def months_done():
        if not raw_path.exists():         
            return set()
        ds = xr.open_zarr(raw_path)
        try:
            return set(pd.to_datetime(ds.time.values).month)
        finally:
            ds.close() 

    target = set(range(3, 12)) # we want all months march - november 
    MAX_PASSES = 12  ## will only make 12 attempts maximum

    for _ in range(MAX_PASSES):
        build_daily_timeseries_incremental(boundary, year=year,
                                        output_label=output_label, dask_client=dask_client)
        done = months_done()
        if target.issubset(done):
            break
   
   
    # coregister all time steps (only runs if all months successfully downloaded)
    # the median NIR image is used as the reference image for alignment
    if len(months_done())==9:
        raw_ds = xr.open_zarr(raw_path)
        ref = raw_ds['reflectance']
        ref = ref.rio.write_crs(config.EPSG).rio.set_spatial_dims(x_dim='x',y_dim='y').rio.write_coordinate_system()

        out_path = config.DATA_DIR / output_label/ f'{output_label}_daily_raw_aligned.zarr'
        print('Aligning images')
        coregister_images(ref,downscale=4,zarr_path=out_path)

        ## remove nonaligned file once aligned is complete
        raw_ds.close()
        shutil.rmtree(raw_path)
        print(f'Done. Aligned Sentinel-2 data saved to {out_path}')
    else:
        print(f'Exited process without aligning. Downloaded months: {sorted(months_done())}')

if __name__=='__main__':
    main()