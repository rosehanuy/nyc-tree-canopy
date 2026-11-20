from utc_src import config
from utc_src import residual_hotspot as rh
import pandas as pd
import os

### characterize the landcovers found in high-high and low-low residual clusters

## dict of model labels and filenames
models = {'Growing_Season': 'growing_season', 'Three Seasons':'three_seasons','Two Seasons':'summer_fall'}

template_path = config.DATA_DIR / 'nyc21' / 'nyc21_tree_canopy_lidar.tif' # defines the 10 m grid
fractions_10m = config.DATA_DIR / 'nyc21' / 'nyc21_landcover_height_fractions_10m.tif'

def main():
    if not os.path(fractions_10m).exists():
        landcover_path = config.LIDAR_LC_2021
        chm_path = config.CHM
        if not os.path(landcover_path).exists():
            print(f'Lidar landcover raster not found at this location: {landcover_path}')
        elif not os.path(chm_path).exists():
            print(f'CHM raster not found at this location: {chm_path}') 
        print('Generating 10m landcover rasters from 15cm Lidar landcover and 0.5m CHM')
        rh.landcover_height_fractions(landcover_path=landcover_path,chm_path=chm_path,template_path=template_path,out_path=fractions_10m,chm_units='ft')
    else:
        print('10m landcover rasters exist; deriving landcover makeup of clustered residuals')
        for model in models.values():
            fractions = rh.cell_fractions(fractions_10m_path=fractions_10m,model=model,cell_size=100)
            cluster_fracs = rh.compare_clusters(model=model,fractions=fractions)
            cluster_fracs.to_csv(config.MODEL_OUTPUTS / model / f'cluster_fractions_{model}.csv')


        ### plot results ######

        ## read in cluster fraction tables
        tables1 = {l:pd.read_csv(config.MODEL_OUTPUTS/ m / f'cluster_fractions_{m}.csv') for l,m in models.items()}

        ## select relevant classes only
        classes = ['bare_soil','lawn','shrub','tree_2.5_5m','tree_5m_plus','building_under_15m','building_15m_plus']
        tables = {k:t.loc[t['class'].isin(classes)].copy() for k,t in tables1.items()}

        rh.plot_comparison_models(tables, figure_filename='cluster_landcover_models',classes=classes)

if __name__ == '__main__':
    main()