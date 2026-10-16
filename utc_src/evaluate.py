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

######### resample tree canopy estimates and evaluate against lidar ########################

import os
import numpy as np
import pandas as pd
import rioxarray
from affine import Affine
from sklearn.metrics import mean_squared_error, r2_score

from utc_src import config


def _metrics(lidar, pred):
    """Accuracy metrics for paired arrays (NaNs dropped)."""
    ok = np.isfinite(lidar) & np.isfinite(pred)
    l, p = lidar[ok], pred[ok]
    err = p - l
    return {'n_cells': int(ok.sum()),
            'bias': float(err.mean()),
            'mae': float(np.abs(err).mean()),
            'rmse': float(np.sqrt(mean_squared_error(l, p))),
            'r2': float(r2_score(l, p))}


def _coarse_coords(src, factor, nx, ny):
    """Correct cell-centre coordinates and transform for a grid `factor` times coarser than `src`,
    anchored at src's top-left corner (padded edge cells get full-size coordinates too)."""
    tf = src.rio.transform()
    new_tf = tf * Affine.scale(factor)
    x = new_tf.c + (np.arange(nx) + 0.5) * new_tf.a
    y = new_tf.f + (np.arange(ny) + 0.5) * new_tf.e
    return x, y, new_tf


def resample_and_evaluate(output_label, model_label, OOF, train_features, scales,
                          min_coverage=0.75, write_scales=(30, 100, 500, 1000)):
    """Aggregate 10 m predictions and lidar canopy to coarser square grids and compute accuracy.

    Coarse cell value = mean over the 10 m pixels that are valid in BOTH rasters.
    Cells with fewer than `min_coverage` x (pixels per cell) valid pixels are excluded.
    Also reports rmse_indep: the RMSE expected at each scale if 10 m errors were spatially
    independent, sqrt(bias^2 + sigma^2 * mean(1/n_valid)), for the RMSE-vs-scale reference curve.
    """
    t = '_OOF_' if OOF else '_'
    name = f'{output_label}_canopy_from_{model_label}_{train_features}{t}model'
    # masked=True turns each file's nodata value into NaN (otherwise e.g. -9999 counts as data)
    pred = rioxarray.open_rasterio(config.MODEL_OUTPUTS / f'{name}.tif', masked=True).squeeze('band', drop=True)
    lidar = rioxarray.open_rasterio(config.DATA_DIR / output_label / f'{output_label}_tree_canopy_lidar.tif',
                                    masked=True).squeeze('band', drop=True)

    # same grid, not just the same shape
    assert pred.shape == lidar.shape, 'prediction and lidar rasters differ in shape'
    assert pred.rio.transform().almost_equals(lidar.rio.transform()), 'prediction and lidar grids are not aligned'
    assert pred.rio.crs == lidar.rio.crs, 'prediction and lidar CRS differ'
    for nm, r in (('prediction', pred), ('lidar', lidar)):          # both should be 0-100
        mx = float(r.max())
        assert mx > 1.5, f'{nm} raster max is {mx:.2f}; looks like 0-1 rather than 0-100'
    os.makedirs(config.RESAMP, exist_ok=True)

    valid = lidar.notnull() & pred.notnull()
    p10 = pred.where(valid).values
    l10 = lidar.where(valid).values

    rows = [{'res': 10, **_metrics(l10.ravel(), p10.ravel()), 'mean_valid_px': 1.0}]
    bias10, rmse10 = rows[0]['bias'], rows[0]['rmse']
    sigma2 = rmse10 ** 2 - bias10 ** 2                     # variance of 10 m errors
    rows[0]['rmse_indep'] = rmse10

    for res in scales:
        print(f'{res} meters')
        assert res % 10 == 0, f'scale {res} m is not a multiple of 10 m'
        factor = res // 10
        coarsen = dict(x=factor, y=factor, boundary='pad')
        lidar_sum = lidar.where(valid).fillna(0).coarsen(**coarsen).sum()
        pred_sum = pred.where(valid).fillna(0).coarsen(**coarsen).sum()
        n_valid = valid.astype('float32').coarsen(**coarsen).sum()

        keep = n_valid >= min_coverage * factor ** 2
        pred_c = (pred_sum / n_valid).where(keep).astype('float32')
        lidar_c = (lidar_sum / n_valid).where(keep).astype('float32')

        # metrics
        pc, lc, nv = pred_c.values.ravel(), lidar_c.values.ravel(), n_valid.where(keep).values.ravel()
        m = _metrics(lc, pc)
        ok = np.isfinite(pc) & np.isfinite(lc)
        mean_inv_n = float(np.mean(1.0 / nv[ok]))
        rows.append({'res': res, **m, 'mean_valid_px': float(np.mean(nv[ok])),
                     'rmse_indep': float(np.sqrt(bias10 ** 2 + sigma2 * mean_inv_n))})

        # write rasters with correct georeferencing
        if res in write_scales:
            # rebuild georeference from source data rather than using the coords created by coarsen()
            x, y, new_tf = _coarse_coords(pred, factor, pred_c.sizes['x'], pred_c.sizes['y'])
            for arr, fname in ((pred_c, f'resamp_{res}_{name}.tif'),
                               (lidar_c, f'resamp_{res}_{output_label}_tree_canopy_lidar.tif')):
                arr = (arr.assign_coords(x=x, y=y)
                          .rio.write_crs(pred.rio.crs)
                          .rio.write_transform(new_tf)
                          .rio.write_nodata(np.nan, encoded=False))
                arr.rio.to_raster(config.RESAMP / fname, compress='lzw')

    out = pd.DataFrame(rows)
    out.to_csv(config.RESAMP / f'{name}.csv', index=False)

    pred.close()
    lidar.close()
   
def resample_only(output_label, model_label, OOF, train_features, scales,
                          min_coverage=0.75):
    """Aggregate 10 m predictions to coarser square grids.

    Coarse cell value = mean over the 10 m pixels that are valid.
    Cells with fewer than `min_coverage` x (pixels per cell) valid pixels are excluded.
    """
    t = '_OOF_' if OOF else '_'
    name = f'{output_label}_canopy_from_{model_label}_{train_features}{t}model'
    # masked=True turns each file's nodata value into NaN (otherwise e.g. -9999 counts as data)
    pred = rioxarray.open_rasterio(config.MODEL_OUTPUTS / f'{name}.tif', masked=True).squeeze('band', drop=True)
    
    os.makedirs(config.RESAMP, exist_ok=True)

    valid = pred.notnull()
  
    for res in scales:
        print(f'{res} meters')
        assert res % 10 == 0, f'scale {res} m is not a multiple of 10 m'
        factor = res // 10
        coarsen = dict(x=factor, y=factor, boundary='pad')
        pred_sum = pred.where(valid).fillna(0).coarsen(**coarsen).sum()
        n_valid = valid.astype('float32').coarsen(**coarsen).sum()

        keep = n_valid >= min_coverage * factor ** 2
        pred_c = (pred_sum / n_valid).where(keep).astype('float32')

        # write rasters with correct georeferencing
        
        x, y, new_tf = _coarse_coords(pred, factor, pred_c.sizes['x'], pred_c.sizes['y'])
      
        arr = pred_c.assign_coords(x=x, y=y).rio.write_crs(pred.rio.crs).rio.write_transform(new_tf).rio.write_nodata(np.nan, encoded=False)
        arr.rio.to_raster(config.RESAMP / f'resamp_{res}_{name}.tif', compress='lzw')

    pred.close()

   