from utc_src import config
from utc_src import residual_hotspot as rh
import os

### calculate local Moran's I for 100 m residuals. output significantly significant clusters as csv and tif.
# each output cell value is one of {0: 'Not significant', 1: 'High-High', 2: 'Low-High', 3: 'Low-Low', 4: 'High-Low'}

features = "summer_fall"
out_path = config.MODEL_OUTPUTS / 'residuals' / features
cell_size = 100

os.makedirs(out_path,exist_ok=True)

def main():
    pred_path = config.MODEL_OUTPUTS / features / f'nyc21_canopy_from_nyc21_{features}_OOF_model.tif'
    lidar_path = config.DATA_DIR / 'nyc21' / f'nyc21_tree_canopy_lidar.tif'

    resid, lidar = rh.load_residuals(pred_path, lidar_path)

    ## aggregate to chosen scale
    cell_resid = rh.aggregate(resid,cell_size=cell_size)
    cell_lidar = rh.aggregate(lidar,cell_size=cell_size).where(cell_resid.notnull())

    # adjust residuals by subtracting known canopy bias and save results
    cell_adjust, bins = rh.canopy_adjust(cell_resid, cell_lidar) 
    bins.to_csv(out_path / f'{features}_canopy_bias_by_bin_{cell_size}m.csv', index=False)
    cell_adjust.rio.to_raster(out_path / f'{features}_resid_adj_{cell_size}m.tif', compress='lzw')

    distance_band = 1.5 * cell_size
    permutations = 9999

    # calculate local morans I within specified distance band and save results
    df, glob = rh.local_statistics(cell_adjust,neighbour_m=distance_band,permutations=permutations,alpha=0.05)
    print(f"Global Moran's I at {cell_size}: {glob.I:.3f} (pseudo p = {glob.p_sim:.4f})")
    df.to_csv(out_path / f'{features}_local_stats_{cell_size}m.csv', index=False)

    ## convert dataframe to raster and save; 
    rh.to_raster(df, cell_adjust, 'lisa_cluster', out_path / f'{features}_lisa_clusters_{cell_size}m.tif', dtype='uint8', nodata=255)
    
    summ = rh.cluster_summary(df)
    summ.to_csv(out_path / f'{features}_lisa_summary_{cell_size}m.csv')

if __name__ == '__main__':
    main()