from utc_src import config
from utc_src.model import predict_tree_canopy
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

#### set inputs #####
#output_label = 'nyc18'     # {output_label}_seasonal_stats.zarr 
model_label = 'nyc21'      # model training year
model_feature_set = 'summer_fall'   ### training feature set
out_of_fold = True

def main():

    for output_label in ['nyc21']:
        predict_tree_canopy(output_label=output_label,model_label=model_label,model_feature_set=model_feature_set,out_of_fold=out_of_fold)
        print(f'completed {output_label}')

if __name__ == '__main__':
    main()