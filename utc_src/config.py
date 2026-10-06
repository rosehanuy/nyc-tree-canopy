from pathlib import Path

ROOT = Path(__file__).parents[1].resolve()

EPSG = 26918

DATA_DIR = ROOT / 'data'
MODEL_OUTPUTS = ROOT / 'model_outputs'
MODELS = ROOT / 'models'
RESAMP = ROOT / 'model_outputs' / 'resamp'

BOUNDARY = DATA_DIR / 'nyc_boundary.gpkg'
LIDAR_LC_2017 = DATA_DIR / 'lidar_landcover' / 'Land_Cover' / 'NYC_2017_LiDAR_LandCover.img'
LIDAR_LC_2021 = DATA_DIR / 'lidar_landcover' / 'landcover_nyc_2021_6in.tif'

