
from utc_src import config
from utc_src.model import compare_clf_models, compare_rgr_models, plot_shap_feature_importance
import pandas as pd

### set inputs
output_label_1= 'nyc18'
output_label_2 = 'nyc21'

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
    feature_sets = {'all_vars': band_cols,
                    'medians_only' :[c for c in band_cols if c.endswith('median')],
                    'deltas_only':[c for c in band_cols if c.endswith('delta')],
                    'median_season_only':[c for c in band_cols if c.endswith('season_median')]}

    print(f'Training classifiers for {output_label_1}')
    compare_clf_models(feature_sets=feature_sets,output_label=output_label_1,train_df=train_clf_1,test_same_year=test_clf_1,test_other_year=test_clf_2)

    print(f'Training classifiers for {output_label_2}')
    compare_clf_models(feature_sets=feature_sets,output_label=output_label_2,train_df=train_clf_2,test_same_year=test_clf_2,test_other_year=test_clf_1)

    print(f'Training regressors for {output_label_1}')
    compare_rgr_models(feature_sets=feature_sets,output_label=output_label_1,train_df=train_rgr_1,test_same_year=test_rgr_1,test_other_year=test_rgr_2)

    print(f'Training regressors for {output_label_2}')
    compare_rgr_models(feature_sets=feature_sets,output_label=output_label_2,train_df=train_rgr_2,test_same_year=test_rgr_2,test_other_year=test_rgr_1)

    plot_shap_feature_importance(output_label=output_label_1,feature_sets=feature_sets)
    plot_shap_feature_importance(output_label=output_label_2,feature_sets=feature_sets)

    print(f'Done. Results saved to:\n{config.MODELS} / {output_label_1}\n{config.MODELS} / {output_label_2}')     

if __name__=='__main__':
    main()