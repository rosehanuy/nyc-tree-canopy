from utc_src import config
from utc_src.model import train_full_dataset_models, train_oof_models
import pandas as pd



#### set inputs ########
output_label= 'nyc21'
feature_set_name = 'summer_fall'

#  The out_of_fold option is set to avoid data leakage when evaluating model performance. Set out_of_fold to True to train a series of models that hold out a different portion of the training data each time. This series of out_of_fold (OOF) models can then be deployed to make citywide predictions on just the portion of data that was held out from training.

# If you are not concerned with model evaluation, out_of_fold can be set set to False. In this case, only one model version will be trained using all the available train/test data.

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
    band_cols = [c for c in all_training_data.columns if c.startswith('band')]
    
    medians = [c for c in band_cols if c.endswith('median') and not c.endswith('season_median')]
    seasons = ['spring', 'summer', 'fall']
    season_sets = {s: [c for c in medians if c.endswith(f'{s}_median')] for s in seasons}

    feature_sets = {
           'spring': season_sets['spring'],
           'summer': season_sets['summer'],
           'fall':   season_sets['fall'],
           'spring_summer': season_sets['spring'] + season_sets['summer'],
           'spring_fall':   season_sets['spring'] + season_sets['fall'],
           'summer_fall':   season_sets['summer'] + season_sets['fall'],
           'three_seasons':   medians, # spring summer fall. previously called all_seasons
           'all_vars': band_cols,
           'all_seasons' :[c for c in band_cols if c.endswith('median')], # growing season, spring, summer, fall. previously called medians_only
           'deltas_only':[c for c in band_cols if c.endswith('delta')],
           'medians_deltas': [c for c in band_cols if c.endswith(('median','delta'))],
           'iqr_only':       [c for c in band_cols if c.endswith('iqr')],
           'growing_season':[c for c in band_cols if c.endswith('season_median')]} # previously called median_season_only


    features = feature_sets[feature_set_name]

    if not out_of_fold:
        train_full_dataset_models(all_training_data=all_training_data,output_label=output_label,feature_cols=features,feature_set_name=feature_set_name)
    else:
        train_oof_models(train_df=all_training_data,output_label=output_label,feature_cols=features,feature_set_name=feature_set_name,k=k_folds)


if __name__ == '__main__':
    main()

