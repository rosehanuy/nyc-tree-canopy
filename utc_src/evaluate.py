"""Paired spatial block bootstrap for comparing feature sets.

Workflow
--------
1. compare_clf_models / compare_rgr_models train one model per feature set and
   save test-set predictions to  models/<label>/<label>_<clf|rgr>_preds_<sameyear|otheryear>.parquet
   (columns: block_id, y_true, then one prediction column per feature set).
2. evaluate_feature_sets() bootstraps those predictions and writes
   models/<label>/<label>_<clf|rgr>_results_ci.csv  (point estimate + 95% CI per feature set)
   models/<label>/<label>_<clf|rgr>_<metric>_<sameyear|otheryear>_boot.parquet  (raw bootstrap scores)
3. paired_differences() gives the CI of the difference between each feature set and a reference.

"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error

from utc_src import config

TEST_SETS = ('sameyear', 'otheryear')
METRICS_BY_MODEL = {'clf': ('auc',), 'rgr': ('r2', 'rmse')}


def _rmse(y_true, y_pred, sample_weight=None):
    return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))


METRIC_FUNCS = {'auc': roc_auc_score, 'r2': r2_score, 'rmse': _rmse}


def block_ids(df):
    """Spatial block identifier for each row of a train/test dataframe."""
    return df['block_row'].astype(str) + '_' + df['block_col'].astype(str)


def prediction_frame(test_df, target_col):
    """Start a predictions table for one test set; feature-set columns are added later."""
    return pd.DataFrame({'block_id': block_ids(test_df).to_numpy(),
                         'y_true': test_df[target_col].to_numpy()})


def _feature_set_cols(preds):
    return [c for c in preds.columns if c not in ('block_id', 'y_true')]


def bootstrap_scores(preds, metric, n_boot=1000, seed=0):
    """Score every feature set on the same n_boot block resamples.

    Returns a DataFrame of shape (n_boot, n_feature_sets).
    """
    codes, uniq = pd.factorize(preds['block_id'])
    n_blocks = len(uniq)
    rng = np.random.default_rng(seed)
    # counts[i, b] = how many times block b is drawn in resample i (drawing n_blocks with replacement)
    counts = rng.multinomial(n_blocks, np.full(n_blocks, 1 / n_blocks), size=n_boot)

    fn = METRIC_FUNCS[metric]
    y = preds['y_true'].to_numpy()
    sets = _feature_set_cols(preds)
    p = {s: preds[s].to_numpy() for s in sets}

    out = np.empty((n_boot, len(sets)))
    for i in range(n_boot):
        w = counts[i, codes]
        keep = w > 0
        yk, wk = y[keep], w[keep]
        for j, s in enumerate(sets):
            out[i, j] = fn(yk, p[s][keep], sample_weight=wk)
    return pd.DataFrame(out, columns=sets)


def point_estimates(preds, metric):
    """Metric on the full (un-resampled) test set for each feature set."""
    fn = METRIC_FUNCS[metric]
    y = preds['y_true'].to_numpy()
    return pd.Series({s: fn(y, preds[s].to_numpy()) for s in _feature_set_cols(preds)})


def paired_differences(preds, boot, metric, reference, alpha=0.05):
    """Difference (feature set - reference) with a paired bootstrap CI.

    p_not_better = share of resamples where the feature set did not beat the reference
    (for RMSE, lower is better, so 'beat' means a lower value).
    """
    est = point_estimates(preds, metric)
    diffs = boot.sub(boot[reference], axis=0).drop(columns=reference)
    sign = -1 if metric == 'rmse' else 1
    return pd.DataFrame({
        'feature_set': diffs.columns,
        'reference': reference,
        'diff': (est[diffs.columns] - est[reference]).to_numpy(),
        'ci_low': diffs.quantile(alpha / 2).to_numpy(),
        'ci_high': diffs.quantile(1 - alpha / 2).to_numpy(),
        'p_not_better': (sign * diffs <= 0).mean().to_numpy(),
    })


def evaluate_feature_sets(output_label, model_type, n_boot=1000, seed=0, alpha=0.05):
    """Bootstrap saved predictions for one training year and model type ('clf' or 'rgr').

    Writes <label>_<model_type>_results_ci.csv with one row per feature set and columns
    like  sameyear_auc, sameyear_auc_low, sameyear_auc_high, otheryear_auc, ...
    """
    out_dir = config.MODELS / output_label
    summary = None
    for which in TEST_SETS:
        preds = pd.read_parquet(out_dir / f'{output_label}_{model_type}_preds_{which}.parquet')
        for metric in METRICS_BY_MODEL[model_type]:
            # same seed for every metric/feature set -> all comparisons are paired
            boot = bootstrap_scores(preds, metric, n_boot=n_boot, seed=seed)
            boot.to_parquet(out_dir / f'{output_label}_{model_type}_{metric}_{which}_boot.parquet', index=False)

            col = f'{which}_{metric}'
            part = pd.DataFrame({
                'feature_set': boot.columns,
                col: point_estimates(preds, metric)[boot.columns].to_numpy(),
                f'{col}_low': boot.quantile(alpha / 2).to_numpy(),
                f'{col}_high': boot.quantile(1 - alpha / 2).to_numpy(),
            })
            summary = part if summary is None else summary.merge(part, on='feature_set')

    summary.to_csv(out_dir / f'{output_label}_{model_type}_results_ci.csv', index=False)
    return summary


def load_boot(output_label, model_type, metric, which):
    """Load saved predictions + bootstrap scores, e.g. for paired_differences()."""
    out_dir = config.MODELS / output_label
    preds = pd.read_parquet(out_dir / f'{output_label}_{model_type}_preds_{which}.parquet')
    boot = pd.read_parquet(out_dir / f'{output_label}_{model_type}_{metric}_{which}_boot.parquet')
    return preds, boot

def compile_differences(output_labels, comparisons, alpha=0.05):
    """Run paired_differences for every training year, model type and test set.
 
    comparisons: {reference: [feature sets to compare against it]}, e.g.
        {'all_seasons':  ['spring', 'summer', 'fall', 'spring_summer', 'spring_fall', 'summer_fall'],
         'medians_only': ['all_vars', 'deltas_only', 'iqr_only', 'medians_deltas', 'median_season_only']}
 
    Returns one tidy table (also the basis for a supplementary table) with columns:
    train_label, model_type, metric, test_set, feature_set, reference, diff, ci_low, ci_high,
    p_not_better, ci_excludes_zero.
    Requires evaluate_feature_sets() to have been run first.
    """
    rows = []
    for label in output_labels:
        for model_type, metrics in METRICS_BY_MODEL.items():
            for metric in metrics:
                for which in TEST_SETS:
                    preds, boot = load_boot(label, model_type, metric, which)
                    for reference, targets in comparisons.items():
                        d = paired_differences(preds, boot, metric, reference, alpha=alpha)
                        d = d[d['feature_set'].isin(targets)]
                        d.insert(0, 'test_set', which)
                        d.insert(0, 'metric', metric)
                        d.insert(0, 'model_type', model_type)
                        d.insert(0, 'train_label', label)
                        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out['ci_excludes_zero'] = (out['ci_low'] > 0) | (out['ci_high'] < 0)
    return out