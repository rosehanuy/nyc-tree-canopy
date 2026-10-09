import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from utc_src import config
import os


# display names for x tick labels
LABELS = {
    'spring': 'Spring', 'summer': 'Summer', 'fall': 'Fall',
    'spring_summer': 'Spring +\nSummer', 'spring_fall': 'Spring +\nFall', 'summer_fall': 'Summer +\nFall',
    'all_seasons': 'All Three\nSeasons',
    'all_vars': 'All\nVariables', 'medians_only': 'Medians', 'deltas_only': 'Deltas',
    'iqr_only': 'IQR', 'medians_deltas': 'Medians\n + Deltas', 'median_season_only': 'Growing\nSeason\nMedians',
}


def read_in_results(output_label, features_to_plot):
    out = []
    for model_type, metric in (('clf', 'auc'), ('rgr', 'r2')):
        df = pd.read_csv(config.MODELS / output_label / f'{output_label}_{model_type}_results_ci.csv')
        # enforces rows in the same order as the tick labels
        df = df.set_index('feature_set').loc[features_to_plot].reset_index()
        out.append(df.rename(columns=lambda c: c.replace(f'_{metric}', '')))   # sameyear_auc_low -> sameyear_low
    return out  # c, r


def get_err(est, high, low):
    return np.array([est - low, high - est])


def shared_ylim(dfs, pad=0.15):
    """y-limits covering every CI in dfs, padded by `pad` x the data range on each side."""
    lo = min(df[['sameyear_low', 'otheryear_low']].min().min() for df in dfs)
    hi = max(df[['sameyear_high', 'otheryear_high']].max().max() for df in dfs)
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def plot_feature_comparison(output_label_1, output_label_2, year_1, year_2, features_to_plot, fname):
    c_1, r_1 = read_in_results(output_label_1, features_to_plot)
    c_2, r_2 = read_in_results(output_label_2, features_to_plot)

    colors_1 = ('#007979', '#24B1B1')
    colors_2 = ('#5FACD3', '#97DDE9')

    # one y-range per row (model type), shared by both training years so the panels compare directly
    c_ylim = shared_ylim([c_1, c_2])
    r_ylim = shared_ylim([r_1, r_2])

    result_dfs = [(c_1, f'{year_1}-trained Classifiers', c_ylim, colors_1, 'AUC'),
                  (c_2, f'{year_2}-trained Classifiers', c_ylim, colors_2, 'AUC'),
                  (r_1, f'{year_1}-trained Regressors', r_ylim, colors_1, 'R²'),
                  (r_2, f'{year_2}-trained Regressors', r_ylim, colors_2, 'R²')]

    n = len(features_to_plot)
    fig, axes = plt.subplots(2, 2, figsize=(max(9, 1.6 * n + 2), 7))   # widen with more feature sets
    axes = axes.ravel()
    x, w = np.arange(n), 0.38
    tick_labels = [LABELS.get(f, f) for f in features_to_plot]

    for i, (data, title, ylim, colors, label) in enumerate(result_dfs):
        ax = axes[i]
        ax.grid(axis='y', alpha=0.3)
        ax.bar(x - w/2, data['sameyear'], w, color=colors[0])
        ax.errorbar(x - w/2, data['sameyear'], yerr=get_err(data['sameyear'], data['sameyear_high'], data['sameyear_low']),
                    fmt='none', ecolor='0.3', elinewidth=1, capsize=4, capthick=1)
        ax.bar(x + w/2, data['otheryear'], w, color=colors[1])
        ax.errorbar(x + w/2, data['otheryear'], yerr=get_err(data['otheryear'], data['otheryear_high'], data['otheryear_low']),
                    fmt='none', ecolor='0.3', elinewidth=1, capsize=4, capthick=1)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=9)
        if i > 1:
            ax.set_xlabel('Feature Set')

    handles1 = [Patch(color=colors_1[0], label=f'{year_1}-trained, same-year test'),
                Patch(color=colors_1[1], label=f'{year_1}-trained, cross-year test')]
    handles2 = [Patch(color=colors_2[0], label=f'{year_2}-trained, same-year test'),
                Patch(color=colors_2[1], label=f'{year_2}-trained, cross-year test')]
    fig.legend(handles=handles1, loc='upper center', frameon=False, bbox_to_anchor=(0.28, 1.08))
    fig.legend(handles=handles2, loc='upper center', frameon=False, bbox_to_anchor=(0.77, 1.08))
    plt.tight_layout()

    config.FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES / f'feature_comparison_{fname}_{output_label_1}_{output_label_2}.png', bbox_inches='tight', dpi=300)  
   
    plt.close()


def plot_shap_feature_importance(output_label,feature_sets):
    os.makedirs(config.MODELS / output_label/ 'shap_plots',exist_ok=True)
    for features in feature_sets.keys():
        shap_df_rgr = pd.read_csv(config.MODELS / output_label / 'shap' / f'{output_label}_rgr_shap_{features}_mean.csv')
        shap_df_clf = pd.read_csv(config.MODELS/ output_label / 'shap' / f'{output_label}_clf_shap_{features}_mean.csv')

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