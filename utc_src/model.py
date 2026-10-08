import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                              mean_squared_error, r2_score)
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import os

from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.model_selection import cross_val_score
import json

import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
import xarray as xr
import rasterio


from utc_src import config


def train_clf_models(output_label, train_df, test_same_year, test_other_year,
                     feature_set, n_seeds, shap_n):
    import shap
    band_cols = [c for c in train_df.columns if c.startswith('band')]
    feature_sets = {
        'all_vars':            band_cols,
        'medians_only':        [c for c in band_cols if c.endswith('median')],
        'deltas_only':         [c for c in band_cols if c.endswith('delta')],
        'median_season_only':  [c for c in band_cols if c.endswith('season_median')],
    }
    feature_cols = feature_sets[feature_set]

    X_train = train_df[feature_cols]
    y_train = train_df['canopy_binary']
    X_test_same,  y_test_same  = test_same_year[feature_cols],  test_same_year['canopy_binary']
    X_test_other, y_test_other = test_other_year[feature_cols], test_other_year['canopy_binary']

    # take sample for shap once so it's the same for every iteration
    X_small = X_test_same.sample(min(shap_n, len(X_test_same)), random_state=50)

    same_aucs, other_aucs, shap_cols = [], [], []
    
    for seed in range(n_seeds):
        print(seed)
        clf = RandomForestClassifier(n_jobs=-1, random_state=seed)
        clf.fit(X_train, y_train)

        same_aucs.append(roc_auc_score(y_test_same,  clf.predict_proba(X_test_same)[:, 1]))
        other_aucs.append(roc_auc_score(y_test_other, clf.predict_proba(X_test_other)[:, 1]))

        vals = shap.TreeExplainer(clf)(X_small).values[:, :, 1]
        shap_cols.append(np.abs(vals).mean(axis=0))   # (n_features,) per seed
        last_clf = clf

    same_aucs, other_aucs = np.array(same_aucs), np.array(other_aucs)
    shap_mat = np.vstack(shap_cols)                   # (n_seeds, n_features)

   
    shap_df = (pd.DataFrame({
        'feature': X_small.columns,
        'mean_abs_shap': shap_mat.mean(axis=0),
        'std_abs_shap':  shap_mat.std(axis=0),
    }).sort_values('mean_abs_shap', ascending=False))
    shap_df.to_csv(config.MODELS / output_label / f'{output_label}_clf_shap_{feature_set}_mean.csv', index=False)

    return {
        'train_year':       output_label,
        'feature_set':      feature_set,
        'n_seeds':          n_seeds,
        'sameyear_auc_mean':  float(same_aucs.mean()),
        'sameyear_auc_std':   float(same_aucs.std()),
        'otheryear_auc_mean': float(other_aucs.mean()),
        'otheryear_auc_std':  float(other_aucs.std()),
    }

def compare_clf_models(feature_sets,output_label,train_df,test_same_year, test_other_year,n_seeds,shap_n):
    all_results = []
    for feature_set, feature_cols in feature_sets.items():
        print(feature_set)
        results = train_clf_models(output_label=output_label,train_df=train_df,test_same_year=test_same_year,test_other_year=test_other_year,feature_set=feature_set,feature_cols=feature_cols,n_seeds=n_seeds,shap_n=shap_n)
        all_results.append(results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(config.MODELS/ output_label / f'{output_label}_clf_results.csv')


def train_rgr_models(output_label,train_df,test_same_year, test_other_year, feature_set,n_seeds,shap_n):
    import shap

    band_cols = [c for c in train_df.columns if c.startswith('band')]
    
    feature_sets = {'all_vars': band_cols,
    'medians_only' :[c for c in band_cols if c.endswith('median')],
    'deltas_only':[c for c in band_cols if c.endswith('delta')],
    'median_season_only':[c for c in band_cols if c.endswith('season_median')]}

    feature_cols = feature_sets[feature_set]

    X_train = train_df.loc[:,feature_cols]
    y_train = train_df['canopy_pct']

    X_test_sameyear = test_same_year.loc[:,feature_cols]
    y_test_sameyear = test_same_year['canopy_pct']

    X_test_otheryear = test_other_year.loc[:,feature_cols]
    y_test_otheryear = test_other_year['canopy_pct']

    X_small = X_test_sameyear.sample(min(shap_n, len(X_test_sameyear)), random_state=50)

    same_r2, same_rmse, other_r2, other_rmse, shap_cols = [], [], [], [], []

    for seed in range(n_seeds):
        print(seed)
        rgr = RandomForestRegressor(n_estimators=100,
        min_samples_leaf=5,
        max_depth=20,
        n_jobs=-1,
        random_state=seed)

        rgr.fit(X_train,y_train)

        y_pred_sameyear = rgr.predict(X_test_sameyear)
        y_pred_otheryear = rgr.predict(X_test_otheryear)
   
        sameyear_rmse = np.sqrt(mean_squared_error(y_test_sameyear, y_pred_sameyear))
        otheryear_rmse = np.sqrt(mean_squared_error(y_test_otheryear, y_pred_otheryear))
        sameyear_r2 = r2_score(y_test_sameyear, y_pred_sameyear)
        otheryear_r2 = r2_score(y_test_otheryear, y_pred_otheryear)

        same_r2.append(sameyear_r2)
        other_r2.append(otheryear_r2)
        same_rmse.append(sameyear_rmse)
        other_rmse.append(otheryear_rmse)

        if seed < 5:
            explainer = shap.TreeExplainer(rgr)
        
            shap_values = explainer(X_small, check_additivity=False)
        
            vals = shap_values.values
            mean_abs_shap = np.abs(vals).mean(axis=0)

            shap_cols.append(mean_abs_shap)
        
         
    same_r2, other_r2 = np.array(same_r2), np.array(other_r2)
    same_rmse, other_rmse = np.array(same_rmse), np.array(other_rmse)
    shap_mat = np.vstack(shap_cols) 

    result_dict = {'train_year': output_label,
                    'feature_set':feature_set,
                   'sameyear_rmse_mean': float(same_rmse.mean()),
                   'sameyear_rmse_std': float(same_rmse.std()),
                   'otheryear_rmse_mean': float(other_rmse.mean()),
                   'otheryear_rmse_std': float(other_rmse.std()),
                   'sameyear_r2_mean': float(same_r2.mean()),
                   'sameyear_r2_std': float(same_r2.std()),
                   'otheryear_r2_mean': float(other_r2.mean()),
                   'otheryear_r2_std': float(other_r2.std())}
        
    shap_df = (pd.DataFrame({
        'feature': X_small.columns,
        'mean_abs_shap': shap_mat.mean(axis=0),
        'std_abs_shap':  shap_mat.std(axis=0),
    }).sort_values('mean_abs_shap', ascending=False))

    shap_df.to_csv(config.MODELS/ output_label/ f'{output_label}_rgr_shap_{feature_set}_mean.csv')
   

    return result_dict



def compare_rgr_models(feature_sets,output_label,train_df,test_same_year, test_other_year,n_seeds,shap_n):
    all_results = []
    for feature_set, feature_cols in feature_sets.items():
        print(feature_set)
        results = train_rgr_models(output_label=output_label,train_df=train_df,test_same_year=test_same_year,test_other_year=test_other_year,feature_set=feature_set,feature_cols=feature_cols,n_seeds=n_seeds,shap_n=shap_n)
        all_results.append(results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(config.MODELS/ output_label / f'{output_label}_rgr_results.csv')






def train_full_dataset_models(all_training_data,output_label,feature_cols,feature_set_name):

    os.makedirs(config.MODELS / output_label,exist_ok=True)
    
    X = all_training_data[feature_cols]
    y_binary = all_training_data['canopy_binary']
    y_pct = all_training_data['canopy_pct']

    print('Training classifier')
    clf = RandomForestClassifier(n_jobs=-1,random_state=42)

    clf.fit(X,y_binary)

    print('Training regressor')
    rgr = RandomForestRegressor( n_estimators=100,
        min_samples_leaf=5,
        max_depth=20,
        n_jobs=-1,
        random_state=42)

    rgr.fit(X,y_pct)

    joblib.dump({"clf_model":clf,"rgr_model":rgr,"input_features":feature_cols}, config.MODELS / output_label / f'{output_label}_{feature_set_name}_models.joblib')
    print(f'Done. Trained models saved to : {config.MODELS} / {output_label} / {output_label}_{feature_set_name}_models.joblib')


def train_oof_models(train_df, output_label,feature_cols, feature_set_name, k, seed=42):
    os.makedirs(config.MODELS / output_label,exist_ok=True)
    # add block id col to train df
    train_df['block_id'] = train_df['block_row'].astype(str) + '_' + train_df['block_col'].astype(str)
    # assign each spatial block to a fold 
    blocks = train_df['block_id'].unique()
    rng = np.random.default_rng(seed)
    fold_of_block = {b: i % k for i, b in enumerate(rng.permutation(blocks))}
    train_df = train_df.assign(fold=train_df['block_id'].map(fold_of_block))

    # make fold array: assign pixels to folds based on fold_of_block dictionary
    # read in existing block map for size refernce
    block_map = np.load(config.DATA_DIR / 'train_test_blocks.npz')
    block_map = block_map['arr_0']

    block_size = 100
    ny, nx = block_map.shape
    n_blocks_y = ny // block_size   # drop partial edge tiles
    n_blocks_x = nx // block_size

    fold_array = np.full((ny, nx), 7, dtype=np.uint16)
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            f = fold_of_block.get(f'{by}_{bx}')
            if f is None:
                continue
            y0, y1 = by * block_size, (by + 1) * block_size
            x0, x1 = bx * block_size, (bx + 1) * block_size
            fold_array[y0:y1, x0:x1] = f

    base_clf = RandomForestClassifier(n_jobs=-1,random_state=42)
    base_rgr = RandomForestRegressor( n_estimators=100,   
                                            min_samples_leaf=5,
                                            max_depth=20,
                                            n_jobs=-1,
                                            random_state=42)

    print(f'Training {k} OOF models')
    models = {}
    for f in range(k):
        print(f'Classifier {f+1}')
        tr = train_df[train_df.fold != f]  # leave current fold out of training
        clf = clone(base_clf).fit(tr[feature_cols], tr['canopy_binary'])
        print(f'Regressor {f+1}')
        rgr = clone(base_rgr).fit(tr[feature_cols], tr['canopy_pct'])
        models[f] = (clf, rgr)

    joblib.dump({'models':models,'fold':fold_of_block,'fold_array':fold_array,"input_features":feature_cols},config.MODELS / output_label / f'{output_label}_{feature_set_name}_OOF_models.joblib' )
    print(f'Done. OOF models saved to {config.MODELS} / {output_label} / {output_label}_{feature_set_name}_OOF_models.joblib')

############# predict using pre-trained model ##############################


def predict_block(block,clf,rgr):

    n_band, bh, bw = block.shape
    pixels = block.reshape(n_band,-1).T # (pixels, bands)

    valid = np.all(np.isfinite(pixels),axis=1)
    #tc_binary = np.full(bh * bw, np.nan)
    tc_percent = np.full(bh * bw, np.nan)

    if valid.any():
        clf_preds = clf.predict(pixels[valid])
        #clf_preds = (clf_probs >= clf_thresh).astype(int)
        #tc_binary[valid] = clf_preds

        tc_percent[valid] = 0  ## set to zero not nan

        canopy_mask = valid.copy()
        canopy_mask[valid] = (clf_preds == 1)

        if canopy_mask.any():
            tc_percent[canopy_mask] = rgr.predict(pixels[canopy_mask])
    
    return tc_percent.reshape(bh,bw)

def predict_block_oof(chunk, fold_chunk, models):
    out = np.full(chunk.shape[1:], np.nan, dtype=np.float32)
    for f in np.unique(fold_chunk):
        if f == 7 :
            continue                # excluded pixels
        else:
            clf, rgr = models[int(f)]           # select the model that held this fold out of training
        m = (fold_chunk == f)
        if not m.any():
            continue
        # predict only this fold's pixels within the chunk
        sub = np.where(m[None, :, :], chunk, np.nan)
        pred = predict_block(sub, clf, rgr)      # same function as non-oof preds
        out[m] = pred[m]
    return out




def predict_tree_canopy(output_label,model_label, model_feature_set,out_of_fold):

    input_data = xr.open_zarr(config.DATA_DIR / output_label / f'{output_label}_seasonalstats.zarr')['features']
    input_data = input_data.assign_coords(band=[f"band_{b}" for b in input_data.coords["band"].values])

    if not out_of_fold:
        model_items = joblib.load(config.MODELS / model_label / f'{model_label}_{model_feature_set}_models.joblib')
        clf = model_items['clf_model']
        rgr = model_items['rgr_model']
        features = model_items['input_features']
    else:
        oof_items = joblib.load(config.MODELS / model_label / f'{model_label}_{model_feature_set}_OOF_models.joblib')
        oof_models = oof_items['models']
        fold_raster = xr.DataArray(oof_items['fold_array'],dims=['y','x'],coords={'y':input_data.y,'x':input_data.x})
        features = oof_items['input_features']

    
    input_data = input_data.sel(band=features)
    input_data = input_data.rio.write_crs(26918).rio.set_spatial_dims( x_dim="x",y_dim="y").rio.write_coordinate_system()

    height, width = input_data.shape[1], input_data.shape[2]
    transform = input_data.rio.transform()
    crs = input_data.rio.crs
    chunk_y = 256
    chunk_x = 256

    dest_name = f'{output_label}_canopy_from_{model_label}_{model_feature_set}_model.tif' if not out_of_fold else f'{output_label}_canopy_from_{model_label}_{model_feature_set}_OOF_model.tif'

    with rasterio.open(
        config.MODEL_OUTPUTS / dest_name,
        'w', driver='GTiff', height=height, width=width,
        count=1, dtype=np.float32, crs=crs,
        transform=transform, compress='lzw'
    ) as dst:
        for y_start in range(0, height, chunk_y):
            for x_start in range(0, width, chunk_x):
                y_end = min(y_start + chunk_y, height)
                x_end = min(x_start + chunk_x, width)

                chunk = input_data.isel(
                    y=slice(y_start, y_end),
                    x=slice(x_start, x_end)
                ).compute().values

                if not out_of_fold:
                    result = predict_block(
                        chunk, clf=clf,rgr=rgr
                    )
                else:
                    fold_chunk = fold_raster.isel(y=slice(y_start, y_end), x=slice(x_start, x_end)).values
                    result = predict_block_oof(chunk, fold_chunk, oof_models)

                window = rasterio.windows.Window(
                    x_start, y_start,
                    x_end - x_start, y_end - y_start
                )
                dst.write(result.astype(np.float32), 1, window=window)
                
    print("Done")