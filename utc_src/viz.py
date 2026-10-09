import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from utc_src import config
import os


def read_in_feature_comp_results(output_label):
    c = pd.read_csv(config.MODELS / output_label / f'{output_label}_clf_results_means.csv')
    c = c.rename(columns={'sameyear_auc_mean':'sameyear','otheryear_auc_mean':'otheryear','sameyear_auc_std':'sameyear_std','otheryear_auc_std':'otheryear_std'})
    r = pd.read_csv(config.MODELS / output_label / f'{output_label}_rgr_results_means.csv')
    r = r.rename(columns={'sameyear_r2_mean':'sameyear','otheryear_r2_mean':'otheryear','sameyear_r2_std':'sameyear_std','otheryear_r2_std':'otheryear_std'})

    order = ['all_vars', 'medians_only', 'deltas_only', 'median_season_only']
    c = c.set_index('feature_set').loc[order].reset_index()
    r = r.set_index('feature_set').loc[order].reset_index()
    return c, r

def plot_feature_comparison(output_label_1, output_label_2, year_1, year_2):
    c_1, r_1 = read_in_feature_comp_results(output_label_1)
    c_2, r_2 = read_in_feature_comp_results(output_label_2)

    colors_1 = ('#007979','#24B1B1')
    colors_2 = ('#5FACD3','#97DDE9')

    c_label = 'Mean AUC'
    r_label = 'Mean R²'

    c_ylim = (0.85, 0.93)
    r_ylim = (0.60, 0.80)

    result_dfs = [(c_1, f'{year_1}-trained Classifiers',c_ylim, colors_1,c_label), (c_2, f'{year_2}-trained Classifiers',c_ylim,colors_2,c_label),(r_1,f'{year_1}-trained Regressors',r_ylim,colors_1,r_label), (r_2, f'{year_2}-trained Regressors',r_ylim,colors_2,r_label)]

    feature_sets = ['All\nVariables','Medians','Deltas','Growing Season\nMedians']

    fig, axes = plt.subplots(2,2,figsize=(9,7))
    axes = axes.ravel()

    x, w = np.arange(len(feature_sets)), 0.38

    for i, (data,title,ylim,colors,label) in enumerate(result_dfs):
        ax = axes[i]
        
        same = data['sameyear']
        same_err = data['sameyear_std']
        cross =data['otheryear']
        cross_err = data['otheryear_std']
        ax.grid(axis='y', alpha=0.3)
        ax.bar(x - w/2, same, w, color=colors[0],label=f'Same-year predictions')
        ax.errorbar(x - w/2, same, yerr=same_err, fmt='none', ecolor='0.3', elinewidth=1,capsize=4,capthick=1)
        ax.bar(x + w/2, cross, w, color=colors[1], label=f'Cross-year predictions')
        ax.errorbar(x + w/2, cross, yerr=cross_err, fmt='none', ecolor='0.3', elinewidth=1,capsize=4,capthick=1)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        if i > 1:
            ax.set_xlabel('Feature Set')
        ax.set_ylabel(label)
        

    custom_handles1 = [
    Patch(color=colors_1[0], label=f'{year_1}-trained Same-year Predictions'),
    Patch(color=colors_1[1], label=f'{year_1}-trained Cross-year Predictions')]
    labels1 = [h.get_label() for h in custom_handles1]

    custom_handles2 = [
    Patch(color=colors_2[0], label=f'{year_2}-trained Same-year Predictions'),
    Patch(color=colors_2[1], label=f'{year_2}-trained Cross-year Predictions')]
    labels2 = [h.get_label() for h in custom_handles2]

    fig.legend(custom_handles1, labels1, loc='upper center', ncol=1, frameon=False, bbox_to_anchor=(0.28, 1.08))
    fig.legend(custom_handles2, labels2, loc='upper center', ncol=1, frameon=False, bbox_to_anchor=(0.77, 1.08))
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(feature_sets, fontsize=9, ha='center')
    plt.tight_layout()
    
    os.makedirs(config.FIGURES, exist_ok=True)
    plt.savefig(config.FIGURES / f'feature_set_comparison_{output_label_1}_{output_label_2}.png')
    plt.close()

def plot_shap_feature_importance(output_label,feature_sets):
    os.makedirs(config.MODELS / output_label/ 'shap_plots',exist_ok=True)
    for features in feature_sets.keys():
        shap_df_rgr = pd.read_csv(config.MODELS / output_label / f'{output_label}_rgr_shap_{features}_mean.csv')
        shap_df_clf = pd.read_csv(config.MODELS/ output_label / f'{output_label}_clf_shap_{features}_mean.csv')

        top30 = shap_df_rgr.head(30)
        plt.figure(figsize=(5, 7))
        plt.barh(top30['feature'][::-1], top30['mean_abs_shap'][::-1])
        plt.xlabel('Mean absolute SHAP value')
        plt.title('Feature importance (SHAP) - Regressor')
        plt.tight_layout()

        plt.savefig(config.MODELS / output_label/ 'shap_plots' / f'{output_label}_rgr_shap_{features}.png')
        plt.close()

        top30 = shap_df_clf.head(30)
        plt.figure(figsize=(5, 7))
        plt.barh(top30['feature'][::-1], top30['mean_abs_shap'][::-1])
        plt.xlabel('Mean absolute SHAP value')
        plt.title('Feature importance (SHAP) - Classifier')
        plt.tight_layout()
    
        plt.savefig(config.MODELS / output_label/ 'shap_plots' / f'{output_label}_clf_shap_{features}.png')
        plt.close()