from utc_src import config
from utc_src.evaluate import resample_and_evaluate, resample_only
import os

output_label = 'nyc18'
model_label = 'nyc21'
OOF = True
train_features = 'medians_only'

## list scales you want to aggregate the 10m results to
scales = [50,100,150,200,300,400,500,1000,1500,2000]

def main():
    if os.path.isfile(config.DATA_DIR / output_label / f'{output_label}_tree_canopy_lidar.tif'):
        print('Reference tree canopy lidar data exists. Aggregating and evaluating results.')
        resample_and_evaluate(output_label=output_label,model_label=model_label,OOF=OOF,train_features=train_features,scales=scales)
    else:
        print('No reference tree canopy lidar data exists. Aggregating results.')
        resample_only(output_label=output_label,model_label=model_label,OOF=OOF,train_features=train_features,scales=scales)

if __name__ == '__main__':
    main()