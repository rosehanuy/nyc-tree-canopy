from utc_src import config
from utc_src.model import train_full_dataset_models, train_oof_models
import pandas as pd



#### set inputs ########
output_label= 'nyc18'
feature_set_name = 'medians_only'

#  The out_of_fold option is set to avoid data leakage when evaluating model performance. Set out_of_fold to True to train a series of models that hold out a different portion of the training data each time. This series of out_of_fold (OOF) models can then be deployed to make citywide predictions for the same year as the training data without data leakage.

# If you are not concerned with model evaluation on same-year data (e.g. you are going to make citywide predictions for different years than the training data), out_of_fold can be set set to False. In this case, only one model version will be trained using all the available train/test data.

out_of_fold = True
k_folds = 5    ## number of out-of-fold models to train. only used if out_of_fold = True.
###########################
def main():
    ### read in all train/test data sets
    train_clf = pd.read_parquet(config.DATA_DIR / output_label / f'{output_label}_train_df_clf.parquet')
    test_clf = pd.read_parquet(config.DATA_DIR / output_label / f'{output_label}_test_df_clf.parquet')
    train_rgr = pd.read_parquet(config.DATA_DIR / output_label / f'{output_label}_train_df_rgr.parquet')
    test_rgr = pd.read_parquet(config.DATA_DIR / output_label / f'{output_label}_test_df_rgr.parquet')

    #### combine all available train/test data 
    all_training_data = pd.concat([train_clf,test_clf,train_rgr,test_rgr])

    #### define feature sets
    band_cols = [c for c in train_clf.columns if c.startswith('band')]
    feature_sets = {'all_vars': band_cols,
                        'medians_only' :[c for c in band_cols if c.endswith('median')],
                        'deltas_only':[c for c in band_cols if c.endswith('delta')],
                        'median_season_only':[c for c in band_cols if c.endswith('season_median')]}


    features = feature_sets[feature_set_name]

    if not out_of_fold:
        train_full_dataset_models(all_training_data=all_training_data,output_label=output_label,feature_cols=features,feature_set_name=feature_set_name)
    else:
        train_oof_models(train_df=all_training_data,output_label=output_label,feature_cols=features,feature_set_name=feature_set_name,k=k_folds)


if __name__ == '__main__':
    main()

