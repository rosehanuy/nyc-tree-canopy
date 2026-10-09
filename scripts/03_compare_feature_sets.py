
from utc_src import config
from utc_src.model import compare_clf_models, compare_rgr_models
from utc_src.viz import plot_feature_comparison,  plot_shap_feature_importance
from utc_src.evaluate import evaluate_feature_sets, compile_differences
import pandas as pd

### set inputs
output_label_1= 'nyc18'
year_1 = 2018
output_label_2 = 'nyc21'
year_2 = 2021


shap_n = 200  ### size of sample to derive shap values

def main():
    ## read in classifier train/test datasets
    train_clf_1 = pd.read_parquet(config.DATA_DIR / output_label_1 / f'{output_label_1}_train_df_clf.parquet')
    test_clf_1 = pd.read_parquet(config.DATA_DIR/ output_label_1 / f'{output_label_1}_test_df_clf.parquet')

    train_clf_2 = pd.read_parquet(config.DATA_DIR/ output_label_2 / f'{output_label_2}_train_df_clf.parquet')
    test_clf_2 = pd.read_parquet(config.DATA_DIR/ output_label_2 / f'{output_label_2}_test_df_clf.parquet')

    ## read in regressor train/test datasets
    train_rgr_1 = pd.read_parquet(config.DATA_DIR / output_label_1 / f'{output_label_1}_train_df_rgr.parquet')
    test_rgr_1 = pd.read_parquet(config.DATA_DIR/ output_label_1 / f'{output_label_1}_test_df_rgr.parquet')

    train_rgr_2 = pd.read_parquet(config.DATA_DIR/ output_label_2 / f'{output_label_2}_train_df_rgr.parquet')
    test_rgr_2 = pd.read_parquet(config.DATA_DIR/ output_label_2 / f'{output_label_2}_test_df_rgr.parquet')


    ## define feature sets we want to compare
    band_cols = [c for c in train_clf_1.columns if c.startswith('band')]

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
        'all_seasons':   medians,
        'all_vars': band_cols,
        'medians_only' :[c for c in band_cols if c.endswith('median')],
        'deltas_only':[c for c in band_cols if c.endswith('delta')],
        'medians_deltas': [c for c in band_cols if c.endswith(('median','delta'))],
        'iqr_only':       [c for c in band_cols if c.endswith('iqr')],
        'median_season_only':[c for c in band_cols if c.endswith('season_median')]}

    # train model with each training year/feature set combination
    #calculate relevant metric (auc, rmse, r2) for same year and other year with bootstrapped confidence interval
    print(f'Training classifiers for {output_label_1}')
    compare_clf_models(feature_sets=feature_sets,output_label=output_label_1,train_df=train_clf_1,test_same_year=test_clf_1,test_other_year=test_clf_2,shap_n=shap_n)

    print(f'Training classifiers for {output_label_2}')
    compare_clf_models(feature_sets=feature_sets,output_label=output_label_2,train_df=train_clf_2,test_same_year=test_clf_2,test_other_year=test_clf_1,shap_n=shap_n)

    print(f'Training regressors for {output_label_1}')
    compare_rgr_models(feature_sets=feature_sets,output_label=output_label_1,train_df=train_rgr_1,test_same_year=test_rgr_1,test_other_year=test_rgr_2,shap_n=shap_n)

    print(f'Training regressors for {output_label_2}')
    compare_rgr_models(feature_sets=feature_sets,output_label=output_label_2,train_df=train_rgr_2,test_same_year=test_rgr_2,test_other_year=test_rgr_1,shap_n=shap_n)

    for label in (output_label_1, output_label_2):
        for model_type in ('clf', 'rgr'):
            print(f'Bootstrapping {model_type} results for {label}')
            evaluate_feature_sets(label, model_type, n_boot=1000)

    plot_shap_feature_importance(output_label=output_label_1,feature_sets=feature_sets)
    plot_shap_feature_importance(output_label=output_label_2,feature_sets=feature_sets)

    season_groups = ['spring', 'summer', 'fall', 'spring_summer', 'spring_fall', 'summer_fall','all_seasons','median_season_only']
    types   = ['all_vars', 'medians_only','deltas_only', 'iqr_only', 'medians_deltas']

    diffs = compile_differences([output_label_1, output_label_2],
                               {'all_seasons': season_groups, 'medians_only': types})
    diffs.to_csv(config.MODELS / f'feature_set_differences_{output_label_1}_{output_label_2}.csv', index=False)

    ### add feature comparison bar plot
    plot_feature_comparison(output_label_1=output_label_1,output_label_2=output_label_2,year_1=year_1,year_2=year_2,features_to_plot=season_groups,fname='seasongroups')

    plot_feature_comparison(output_label_1=output_label_1,output_label_2=output_label_2,year_1=year_1,year_2=year_2,features_to_plot=types,fname='typegroups')

    print(f'Done.')     

if __name__=='__main__':
    main()