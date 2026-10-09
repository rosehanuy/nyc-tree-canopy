
from utc_src import config
from utc_src.model import compare_clf_models, compare_rgr_models
from utc_src.viz import plot_feature_comparison,  plot_shap_feature_importance
import pandas as pd

### set inputs
output_label_1= 'nyc18'
year_1 = 2018
output_label_2 = 'nyc21'
year_2 = 2021

n_seeds = 10  ### number of model iterations to evaluate
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
    feature_sets = {'all_vars': band_cols,
                    'medians_only' :[c for c in band_cols if c.endswith('median')],
                    'deltas_only':[c for c in band_cols if c.endswith('delta')],
                    'median_season_only':[c for c in band_cols if c.endswith('season_median')]}

    # ## train ten iterations on each feature set and take the mean and std of performance metrics (AUC for classifiers, R2 and RMSE for regressors)
    # print(f'Training classifiers for {output_label_1}')
    compare_clf_models(feature_sets=feature_sets,output_label=output_label_1,train_df=train_clf_1,test_same_year=test_clf_1,test_other_year=test_clf_2,n_seeds=n_seeds,shap_n=shap_n)

    print(f'Training classifiers for {output_label_2}')
    compare_clf_models(feature_sets=feature_sets,output_label=output_label_2,train_df=train_clf_2,test_same_year=test_clf_2,test_other_year=test_clf_1,n_seeds=n_seeds,shap_n=shap_n)

    print(f'Training regressors for {output_label_1}')
    compare_rgr_models(feature_sets=feature_sets,output_label=output_label_1,train_df=train_rgr_1,test_same_year=test_rgr_1,test_other_year=test_rgr_2,n_seeds=n_seeds,shap_n=shap_n)

    print(f'Training regressors for {output_label_2}')
    compare_rgr_models(feature_sets=feature_sets,output_label=output_label_2,train_df=train_rgr_2,test_same_year=test_rgr_2,test_other_year=test_rgr_1,n_seeds=n_seeds,shap_n=shap_n)

    plot_shap_feature_importance(output_label=output_label_1,feature_sets=feature_sets)
    plot_shap_feature_importance(output_label=output_label_2,feature_sets=feature_sets)

    ### add feature comparison bar plot
    plot_feature_comparison(output_label_1=output_label_1,output_label_2=output_label_2,year_1=year_1,year_2=year_2)

    print(f'Done. Results saved to:\n{config.MODELS} / {output_label_1}\n{config.MODELS} / {output_label_2}\n{config.FIGURES}')     

if __name__=='__main__':
    main()