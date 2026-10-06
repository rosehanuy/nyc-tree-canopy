
import planetary_computer
from pystac_client import Client
import stackstac
import xarray as xr
import rioxarray as rio
import numpy as np
import pandas as pd
import geopandas as gpd
import os
from tqdm import tqdm
from itertools import product
import zarr
import numpy as np
from shapely.geometry import shape, box
from rasterio.enums import Resampling
import rasterio
from rasterio.warp import reproject, transform_bounds
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
import calendar
import pandas as pd
import shutil
import time
import cv2
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from pystac_client.stac_api_io import StacApiIO
from utc_src import config
from osgeo import gdal
import warnings


warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)

####### download data from planetary computer ##############

def _mask_and_scale(data,scl,month_items):
    stack = data.drop_attrs().reset_coords(drop=True).astype('float32')
    stack  = stack.assign_coords(time=stack['time'].dt.floor('D')) # convert timestamp to just the day

    stack_scl = scl.drop_attrs().reset_coords(drop=True).astype('uint8')
    stack_scl = stack_scl.assign_coords(time=stack_scl['time'].dt.floor('D')).squeeze('band')

    scl_values = [0,1,3, 8, 9, 10, 11]   # no water mask (6) and no dark pixels mask (2)

    mask = ~stack_scl.isin(scl_values)
    stack_masked = stack.where(mask)

    # apply offset if baseline >= 4.0
    baseline_lookup = {i.id: float(i.properties['s2:processing_baseline']) for i in month_items}
    baselines = xr.DataArray([baseline_lookup[item_id] for item_id in data['id'].values],dims=('time',),coords={'time':stack.time})
    offset_mask = (baselines >= 4.0).astype('float32')

    scaled = (stack_masked - (offset_mask * 1000)) / 10000  
    # https://clearsky.vision/knowledge/sentinel2-scaling-harmonization

    # merge images from different tiles taken on same day
    scaled = scaled.groupby('time').median(dim='time',skipna=True) 
    scaled.name = 'reflectance'

    ## drop scenes that are over 80% cloudy 
    observed = stack_scl != 0  ## pixels that are part of the tile
    cloudy   = stack_scl.isin([1, 3, 8, 9, 10, 11])

    obs_px   = observed.sum(('y', 'x'))              
    cloud_px = (cloudy & observed).sum(('y', 'x')) ## which pixels are cloudy and also part of the tile

    obs_day   = obs_px.groupby('time').sum()          # combine the tiles by day
    cloud_day = cloud_px.groupby('time').sum()
    cloud_frac_day = cloud_day / obs_day              # cloud/observed, over the assembled tiles
    keep_days = (cloud_frac_day <= 0.80).values

    scaled = scaled.isel(time=keep_days)

    return scaled

def _get_zarr_length(ref_zarr_path):
        if ref_zarr_path.exists():
            g = zarr.open_group(str(ref_zarr_path),mode='r')
            return int(g['time'].shape[0])
        else:
            return 0
   

### loop over months to avoid stale tokens

def build_daily_timeseries_incremental(boundary, year,dask_client,output_label):

    assets=['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
    epsg = boundary.crs.to_epsg()

    bbox_4326 = tuple(boundary.to_crs(4326).total_bounds)
    bbox_utm = tuple(boundary.total_bounds)

    retry = Retry(total=5, backoff_factor=2, status_forcelist=[408, 429, 500, 502, 503, 504], allowed_methods=None)

    stac_api_io = StacApiIO(max_retries=retry)

    ## define output path 
    os.makedirs(config.DATA_DIR / output_label,exist_ok=True)
    ref_zarr_path = config.DATA_DIR / output_label / f'{output_label}_daily_raw.zarr'

    done_months = set()
    if ref_zarr_path.exists():
        existing = xr.open_zarr(ref_zarr_path)
        done_months = set(pd.to_datetime(existing.time.values).month)

    ## access one month at a time
    for month in range(3,12):
        if month in done_months:
            print(f'{calendar.month_name[month]} already present, skipping')
            continue

        # determine write mode based on whether zarr store already exists
        mode = 'w' if not ref_zarr_path.exists() else 'a'
        append_dim = 'time' if mode == 'a' else None
        # update time dimension length
        zarr_length = _get_zarr_length(ref_zarr_path)

        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            stac_io=stac_api_io,
            modifier=planetary_computer.sign_inplace,
        )
        last = calendar.monthrange(year, month)[1] # get last day of month (30 or 31)
        items = catalog.search(
            bbox=bbox_4326,
            collections=["sentinel-2-l2a"],
            datetime=f"{year}-{month:02d}-01/{year}-{month:02}-{last:02d}T23:59:59Z" 
        ).item_collection()

        items = [i for i in items if i.properties["eo:cloud_cover"] < 80] # filter out tiles with over 80% cloud coverage

        print(f'{calendar.month_name[month]} {year}: number of images found: {len(items)}')

        # put details into a dataframe
        records = []
        for i in items:
            records.append({
                'id': i.id,
                'datetime': i.datetime,
                'tile' : i.properties.get('s2:mgrs_tile'),
                'baseline': float(i.properties.get('s2:processing_baseline'))
            })

        df = pd.DataFrame(records)
        df['datetime'] = df['datetime'].dt.floor('D')

        # if two versions of image exist, select the one with the higher processing baseline
        selected_records = df.sort_values('baseline', ascending=False).drop_duplicates(['datetime','tile'], keep='first')
        selected_ids = set(selected_records['id'])
        selected_items = [i for i in items if i.id in selected_ids]

        month_items = [planetary_computer.sign(i) for i in selected_items]
        
     
        max_retries = 3
        for attempt in range(1,6):
        #create xarray
            try:
                stack_ref = stackstac.stack(
                    month_items,
                    epsg=epsg,
                    resolution=10,
                    bounds=bbox_utm,
                    assets=assets,
                    resampling=Resampling.bilinear)
                
                
                stack_scl = stackstac.stack(
                    month_items,
                    epsg=epsg,
                    resolution=10,
                    bounds=bbox_utm,
                    assets=['SCL'],
                    fill_value=0,
                    resampling=Resampling.nearest) # scl band is categorical
                
                scaled_10m = _mask_and_scale(data=stack_ref,scl=stack_scl,month_items=month_items)

                ## ensure uniform time chunk sizes
                scaled_10m.chunk({'time':1,'band':-1,'y':1024,'x':1024}).to_zarr(ref_zarr_path,mode=mode, append_dim=append_dim)

                print(f'{calendar.month_name[month]} successfully written')
                break  ## exit retry loop
                
            ### catch server side errors e.g. http response code 503, 403
            except Exception as e:
                print(f'attempt {attempt} failed: {e}')
                
                ## if this was first month delete zarr store entirely
                if zarr_length == 0:
                    if ref_zarr_path.exists():
                        shutil.rmtree(ref_zarr_path)
                        print('deleted zarr')
                ## otherwise truncate to before failed month attempt
                else:
                    store = zarr.open(ref_zarr_path,mode='r+')

                    store['reflectance'].resize((zarr_length,*store['reflectance'].shape[1:]))
                    store['time'].resize((zarr_length,))
                    zarr.consolidate_metadata(ref_zarr_path) 
                    print(f'truncated zarr to length {zarr_length}')
                
                if attempt < (max_retries+1):
                    dask_client.restart()  ## restart workers in case that was cause of error
                    wait = 30 * attempt
                    print(f'waiting {wait}s before retrying {calendar.month_name[month]}')
                    time.sleep(wait)
                    # reset zarr variables
                    mode = 'w' if not ref_zarr_path.exists() else 'a'
                    append_dim = 'time' if mode == 'a' else None
                    zarr_length = _get_zarr_length(ref_zarr_path)
                    
                else:
                    print(f'{calendar.month_name[month]} failed after {max_retries} tries')
                    print(f'Restarting process to resume writing {calendar.month_name[month]}')
                    return

                      
def _get_gradient(im) :
        # Calculate the x and y gradients using Sobel operator
        grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
        grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
        # Combine the two gradients
        grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)

        return grad

def coregister_images(data_10m,downscale,zarr_path):

    ## downsize image and do a first pass warp, then a warm-started warp at full resolution

    warp_mode = cv2.MOTION_TRANSLATION
    #criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
    criteria_small= (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
    criteria_full = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

    ## reference image for alignment is the median NIR 
    nir_all = data_10m.sel(band='B08').persist()
    reference_image = nir_all.median(dim='time', skipna=True).to_numpy()
    # replace nans with zeroes
    reference_image = np.where(np.isfinite(reference_image), reference_image, 0)
    ref_gradient_full = _get_gradient(reference_image)
    # downsize before calculating gradient
    ref_small = cv2.resize(reference_image, None, fx=1/downscale, fy=1/downscale, interpolation=cv2.INTER_AREA)
    ref_gradient_small = _get_gradient(ref_small)

    time = data_10m.shape[0]
    bands_10 = data_10m.shape[1]
    h10, w10 = data_10m.shape[2], data_10m.shape[3]

    #aligned_10m = np.full((time, bands_10, h10, w10), np.nan, dtype=np.float32)
    #success = np.zeros(time, dtype=bool)

    mode = 'w' 
    append_dim = None

    qc = []
    for i in tqdm(range(time)):
        aligned = np.full((1, bands_10, h10, w10), np.nan, dtype=np.float32)
        # single time slice, convert to numpy
        nir = nir_all.isel(time=i).to_numpy()
        # replace nans with zeroes, get full scale gradient
        nir_full = np.where(np.isfinite(nir), nir, 0).astype(np.float32)
        nir_gradient_full = _get_gradient(nir_full)
        ## downsize and get coarse gradient
        nir_small = cv2.resize(nir_full, None, fx=1/downscale, fy=1/downscale, interpolation=cv2.INTER_AREA)
        nir_gradient_small = _get_gradient(nir_small)

        # get valid masks for full size and coarse versions
        valid_full = np.isfinite(nir).astype('uint8')
        valid_small = cv2.resize(valid_full, None, fx=1/downscale, fy=1/downscale, interpolation=cv2.INTER_NEAREST)

        #nir_gradient = get_gradient(nir_small)
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            ## coarse version first
            _, warp_matrix = cv2.findTransformECC(
                templateImage=ref_gradient_small, inputImage=nir_gradient_small, warpMatrix=warp_matrix, motionType=warp_mode, criteria=criteria_small, inputMask=valid_small)
            # rescale warp matrix
            warp_matrix[:, 2] *= downscale

            # now full scale version
            _, warp_matrix = cv2.findTransformECC(
                templateImage=ref_gradient_full, inputImage=nir_gradient_full, warpMatrix=warp_matrix, motionType=warp_mode, criteria=criteria_full, inputMask=valid_full)

        except cv2.error:
            print(f'Failed to converge at timestep {i}, skipping')
            continue

        ## revert suspiciously large warps
        if abs(warp_matrix[0,2]) > 5 or abs(warp_matrix[1,2]) > 5: 
            warp_matrix = np.eye(2,3,dtype=np.float32)
            print(f'warp too large, reverting')
            qc.append(data_10m.time.values[i])
        
        # apply to 10m stack
        im10 = data_10m.isel(time=i).to_numpy()
        #valid_10m = np.isfinite(im10[6]).astype('uint8')
        valid_10m_warped = cv2.warpAffine(valid_full, warp_matrix, (w10, h10),
                                           flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype('bool')
        for j in range(bands_10):
            w = cv2.warpAffine(im10[j], warp_matrix, (w10, h10),
                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            w[~valid_10m_warped] = np.nan
            aligned[0, j] = w

        #success[i] = True

    # rebuild DataArrays with matched time coords
        aligned_time = xr.DataArray(aligned,
                                    coords={'time': [data_10m.time.values[i]],
                                            'band': data_10m.band,
                                            'y': data_10m.y, 'x': data_10m.x},
                                    dims=['time', 'band', 'y', 'x'],name='reflectance')

        aligned_time.to_zarr(zarr_path, mode=mode,append_dim=append_dim)
        
        mode = 'a'
        append_dim = 'time'
    

  ##################### derive features ##################3

def safe_div(num, den):
    # below the floor the denominator is noise/invalid and result is NaN.
    # prevents absurdly high index values due to tiny denominator
    floor=1e-3
    return num / den.where(den > floor)

def derive_indices(ref,output_label):

    ## put bands and time together; make spatial chunks smaller
    ref = ref.chunk(chunks={'time': -1, 'band': -1, 'y': 256, 'x': 256})
   
    blue  = ref.sel(band="B02")
    green = ref.sel(band="B03")
    red   = ref.sel(band="B04")
    re1   = ref.sel(band="B05")
    re2   = ref.sel(band="B06")
    re3   = ref.sel(band="B07")
    nir   = ref.sel(band="B08")
    nir_a = ref.sel(band="B8A")
    sw1   = ref.sel(band="B11")
    sw2   = ref.sel(band="B12")

    ds = xr.Dataset()
    
    # moisture
    ds["s2wi"]  = safe_div(re1 - sw2, re1 + sw2)
    ds["ndmi"]  = safe_div(nir - sw1, nir + sw1)
 
    ## vegetation/chlorophyll
    ds["ndre2"] = safe_div(nir_a - re2, nir_a + re2)
    ds["grvi"]  = safe_div(green - red, green + red)
    ds["psri"]  = safe_div(red - blue, re2)
    ds["tdvi"]  = 1.5 * ((nir - red) / (nir**2 + red + 0.5) ** 0.5)
    
    # bare/built up
    ds["nbr"]   = safe_div(nir - sw2, nir + sw2)
    ds["bsi"]   = safe_div((sw1 + red) - (nir + blue), (sw1 + red) + (nir + blue))

    ## clip to physical range
    nd = ["s2wi","ndmi","ndre2","nbr","grvi","bsi"]
    for k in nd:
        ds[k] = ds[k].clip(-1, 1)
  
    all_indices = ds.to_array(dim='band').transpose('time','band', 'y', 'x')

    time_chunk = -1  ### keep all time steps in the same chunk for later time dimension aggregations
    band_chunk = 1   ### put bands in separate chunks because they are only used one at a time later.
    spatial_chunk = 256

    all_data = xr.concat([ref,all_indices],dim='band').astype('float32').chunk({'time':time_chunk,'band':band_chunk,'y':spatial_chunk,'x':spatial_chunk})
    all_data.name = 'variables'
    
    encoding = {'variables':{'compressor': zarr.Blosc(cname='zstd',clevel=3,shuffle=2)}}
    all_data.to_zarr(config.DATA_DIR / output_label / f'{output_label}_daily_bandsandindices.zarr',mode='w',encoding=encoding)



def aggregate_to_seasons(output_label):
    

    out_path = config.DATA_DIR / output_label / f'{output_label}_seasonalstats.zarr'
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    all_data = xr.open_zarr(config.DATA_DIR / output_label / f'{output_label}_daily_bandsandindices.zarr')['variables']
    #all_data = all_data.chunk({'time':-1,'band':-1,'y':256,'x':256})
    band_chunk = all_data.sizes['band']
    spatial_chunk = all_data.chunksizes['y'][0]

    ## set vars for first zarr write
    encoding = {'features':{'chunks':(band_chunk,spatial_chunk,spatial_chunk),
                                            'compressor': zarr.Blosc(cname='zstd',clevel=3,shuffle=2)}}
    mode = 'w'
    append_dim = None

    ### clip to city boundary
    boundary = gpd.read_file(config.DATA_DIR / 'nyc_boundary.gpkg')
    
    all_data = all_data.rio.write_crs(config.EPSG).rio.set_spatial_dims(x_dim='x',y_dim='y').rio.write_coordinate_system()
    all_data = all_data.rio.clip(boundary.geometry)

    windows = {"season": (3, 11),"spring": (3, 5),"summer": (6, 8),"fall": (9, 11)}
    for wname, (m1,m2) in windows.items():
        w = 'growing season' if wname == 'season' else wname
        print(f'Aggregating data to {w} ({calendar.month_name[m1]} to {calendar.month_name[m2]})')
        season = all_data.sel(time=all_data.time.dt.month.isin(range(m1,m2+1)))

        # aggregations

        q = season.quantile([0.25, 0.5, 0.75], dim="time", skipna=True)
        p25 = q.sel(quantile=0.25).drop_vars("quantile")
        p50 = q.sel(quantile=0.5).drop_vars('quantile')
        p75 = q.sel(quantile=0.75).drop_vars("quantile")
        iqr = p75 - p25

        stats = {'median':p50,
                'iqr':iqr}
        
        ## add new band coordinate to each stat array
        segments = []
        for st in stats:
            a = stats[st]
            new_band = (a['band'].astype('str') + f'_{wname}_{st}').astype(object)
            a = a.assign_coords({'band':new_band})
            segments.append(a)

        all_segs = xr.concat(segments,dim='band')

        all_segs = all_segs.to_dataset(name='features').astype('float32')
        all_segs = all_segs.chunk({'band':band_chunk,'y':spatial_chunk,'x':spatial_chunk})

        all_segs.to_zarr(out_path,mode=mode,encoding=encoding,append_dim = append_dim)

        ## reset vars for zarr appends
        mode = 'a'
        append_dim = 'band'
        encoding = None

    ## get delta of medians for march/april - may/june and august/sept - october/nov
    month_deltas = {'spring_delta':(6,4),'fall_delta':(9,11)}
    for dname,(start,end) in month_deltas.items():
        print(f'Calculating deltas for {dname.split('_')[0]} ({calendar.month_name[start]},{calendar.month_name[start-1]} median - {calendar.month_name[end]},{calendar.month_name[end-1]} median)')
        m_1 = all_data.sel(time=all_data.time.dt.month.isin([start,start-1])).median(dim='time',skipna=True)
        m_2 = all_data.sel(time=all_data.time.dt.month.isin([end,end-1])).median(dim='time',skipna=True)

        delta = m_1 - m_2

        new_bands = (delta['band'].astype('str') + f'_{dname}').astype(object)
        delta = delta.assign_coords({'band':new_bands})

        delta = delta.to_dataset(name='features').astype('float32')
        delta = delta.chunk({'band':band_chunk,'y':spatial_chunk,'x':spatial_chunk})

        delta.to_zarr(out_path,mode=mode,encoding=encoding,append_dim = append_dim)

    
########### make tree canopy layer from lidar data #################3

def canopy_percent_from_lidar_landcover(
    landcover_path,
    out_tif_path,
    ref_data_path,
    canopy_class_values=(1,),
    block=512,
    out_dtype="float32",
):

    t = xr.open_zarr(ref_data_path)
    ref_raster = t.rio.write_crs(config.EPSG).rio.set_spatial_dims(x_dim='x',y_dim='y').rio.write_coordinate_system()
    dst_crs = ref_raster.rio.crs
    dst_transform = ref_raster.rio.transform()
    dst_height, dst_width = ref_raster.sizes['y'], ref_raster.sizes['x']
    
    with rasterio.open(landcover_path) as src:
        profile = {
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "count": 1,
            "dtype": out_dtype,
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": np.nan,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": min(block, dst_width),
            "blockysize": min(block, dst_height),
        }

        src_nodata = src.nodata

        with rasterio.open(out_tif_path, "w", **profile) as dst:
            blocks = list(product(range(0, dst_height, block),range(0, dst_width, block )))
            for row0, col0 in tqdm(blocks, desc='windows'):
                    h = min(block, dst_height - row0)
                    w = min(block, dst_width - col0)
                    win = Window(col0, row0, w, h)
                    win_tr = window_transform(win, dst_transform)

                    # Bounds of this destination window (in dst_crs)
                    dst_bounds = rasterio.windows.bounds(win, dst_transform)

                    # Transform those bounds into source CRS, read only what we need
                    src_bounds = transform_bounds(dst_crs, src.crs, *dst_bounds, densify_pts=21)
                    src_win = rasterio.windows.from_bounds(*src_bounds, transform=src.transform)
                    src_win = src_win.round_offsets().round_lengths()

                    src_data = src.read(1, window=src_win, boundless=True)

                    # Binary canopy mask (0/1) with nodata -> nan
                    canopy = np.isin(src_data,canopy_class_values).astype(np.float32)
                    if src_nodata is not None:
                        canopy = np.where(src_data == src_nodata, np.nan, canopy)

                    frac = np.full((h, w), np.nan, dtype=np.float32)

                    reproject(
                        source=canopy,
                        destination=frac,
                        src_transform=rasterio.windows.transform(src_win, src.transform),
                        src_crs=src.crs,
                        src_nodata=np.nan,
                        dst_transform=win_tr,
                        dst_crs=dst_crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.average,  
                    )

                    dst.write((frac * 100.0).astype(out_dtype), 1, window=win)

    print(f'Tree canopy lidar raster written to {out_tif_path}')

    