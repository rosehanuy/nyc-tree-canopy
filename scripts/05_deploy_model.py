from utc_src import config
from utc_src.model import predict_tree_canopy

#### set inputs #####
output_label = 'nyc21'     # {output_label}_seasonal_stats.zarr 
model_label = 'nyc21'      
model_feature_set = 'medians_only'
out_of_fold = True

def main():
    predict_tree_canopy(output_label=output_label,model_label=model_label,model_feature_set=model_feature_set,out_of_fold=out_of_fold)

if __name__ == '__main__':
    main()