from utc_src.chm_height_fractions import chm_height_fractions, check_output, load_pixels, summarize, plot_shrub_fraction_error
from utc_src.process_data import canopy_percent_from_lidar_landcover
from utc_src import config
import os

### use the 0.5 m resolution CHM available for 2021 to create four 10 meter rasters:
#### fraction grass/herbaceous per pixel (< 0.5 meters)
#### fraction shrub per pixel (0.5 - 2.5 meters)
#### fraction short trees per pixel (2.5 - 5 meters)
#### fraction tall trees per pixel (> 5 meters)

#### then, assess model performance along gradient of mostly-lawn to mostly-shrub pixels (all with < 5% tree canopy)


gs_path = config.DATA_DIR / 'nyc21' / 'nyc21_grass_shrub_lidar.tif'
chm_path = config.CHM
template = config.DATA_DIR / 'nyc21' / 'nyc21_tree_canopy_lidar.tif'   # 10 m grid to align to
out_path = config.DATA_DIR / 'nyc21' / 'nyc21_height_fractions.tif'

PREDICTIONS = {     # the tree canopy estimates to evaluate
    'Spring + Summer + Fall': config.MODEL_OUTPUTS /'nyc21_canopy_from_nyc21_all_seasons_OOF_model.tif',
    'Growing season': config.MODEL_OUTPUTS /'nyc21_canopy_from_nyc21_median_season_only_OOF_model.tif',
}

## save path for the plot
OUT_FIG = config.FIGURES / 'height_band_errors.png'
# save path for the residual df
OUT_CSV = config.models /'nyc21'/'height_band_errors.csv'

def main():
    ## create grass/shrub raster if it doesn't exist
    if not os.path.isfile(gs_path):
        print("Deriving 10m grass/shrub percent raster from 15cm Lidar landcover")
        landcover_path = config.LIDAR_LC_2021
        ref_data_path = config.DATA_DIR / 'nyc21' / f'nyc21_seasonalstats.zarr'
        canopy_percent_from_lidar_landcover(landcover_path=landcover_path,ref_data_path=ref_data_path,out_tif_path=gs_path,canopy_class_values=(2,))


    if not os.path.isfile(out_path):
        print('Deriving canopy height fractions')
        chm_height_fractions(
            chm_path=chm_path,
            template_path=template,
            grass_pct_path=gs_path,
            out_path=out_path,
            chm_units='ft'         
        )
        check_output(out_path, template)

    d = load_pixels()
    print(f'{len(d):,} tree-free vegetated pixels (tree fraction < {MAX_TREE:g}) in {d["block"].nunique():,} blocks')
    t = summarize(d)
    t.to_csv(OUT_CSV, index=False)

    plot_shrub_fraction_error(t)



if __name__ == '__main__':
    main()
