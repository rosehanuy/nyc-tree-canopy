from pathlib import Path

ROOT = Path(__file__).parents[1].resolve()

EPSG = 26918

DATA_DIR = ROOT / 'data'
MODEL_OUTPUTS = ROOT / 'model_outputs'
MODELS = ROOT / 'models'
RESAMP = ROOT / 'model_outputs' / 'resamp'
FIGURES = ROOT / 'figs'

BOUNDARY = DATA_DIR / 'nyc_boundary.gpkg'
LIDAR_LC_2017 = DATA_DIR / 'lidar' / 'Land_Cover' / 'NYC_2017_LiDAR_LandCover.img'
LIDAR_LC_2021 = DATA_DIR / 'lidar' / 'landcover_nyc_2021_6in.tif'
CHM = DATA_DIR / 'lidar' / 'uchmbt_raster_1p64ft_mosaic.tif'

