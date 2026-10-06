import numpy as np
import pandas as pd
from utc_src import config


def get_pixels(index_list,bands_flat,canopy_flat,band_names):
    all_flat  = np.concatenate([t[0] for t in index_list])
    block_rows = np.concatenate([np.full(len(t[0]), t[1]) for t in index_list])
    block_cols = np.concatenate([np.full(len(t[0]), t[2]) for t in index_list])

    sort_order = np.argsort(all_flat)
    all_flat   = all_flat[sort_order]
    block_rows = block_rows[sort_order]
    block_cols = block_cols[sort_order]

    band_vals   = bands_flat[:, all_flat].compute()   # (n_bands, N)
    canopy_vals = canopy_flat[all_flat]    # (N,)


    # filter out nans/infs
    valid = np.all(np.isfinite(band_vals), axis=0) & np.isfinite(canopy_vals)
    band_vals   = band_vals[:, valid]
    canopy_vals = canopy_vals[valid]
    block_rows  = block_rows[valid]
    block_cols  = block_cols[valid]

    df = pd.DataFrame(band_vals.T, columns=band_names)
    df["canopy_pct"]    = canopy_vals
    df["canopy_binary"] = (canopy_vals > 0).astype(int)
    df["block_row"]     = block_rows
    df["block_col"]     = block_cols
    return df

## get pixel indices within each block
def sample_indices_nonstratified(pixels_per_block,valid_2d,block_size,block_map,ny,nx,random_state):
    train_indices = []
    test_indices = []

    rng = np.random.default_rng(random_state)

    n_blocks_y = ny // block_size   # drop partial edge tiles
    n_blocks_x = nx // block_size
   
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0, y1 = by * block_size, (by + 1) * block_size
            x0, x1 = bx * block_size, (bx + 1) * block_size

            label = block_map[y0, x0]   # 0=train, 1=test, 255=excl
            if label == 255:
                continue

            block_valid_yx = np.argwhere(valid_2d[y0:y1, x0:x1]) # get valid pixels for this block
            if len(block_valid_yx) == 0:
                continue

            # sample given number of pixels
            n_sample = min(pixels_per_block, len(block_valid_yx))
            chosen   = rng.choice(len(block_valid_yx), size=n_sample, replace=False)
            yx       = block_valid_yx[chosen]

            # flat_idx = row * array_width + col
            flat_idx = (y0 + yx[:, 0]) * nx + (x0 + yx[:, 1])

            if label == 0:
                train_indices.append((flat_idx, by, bx))
            else:
                test_indices.append((flat_idx, by, bx))
    
    return train_indices, test_indices

def make_block_map(valid_2d, block_size,test_size,ny,nx):

    ## find blocks that are mostly valid pixels
    n_blocks_y = ny // block_size   # drop partial edge tiles
    n_blocks_x = nx // block_size
    n_blocks   = n_blocks_y * n_blocks_x
    #print(f'number of blocks: {n_blocks}')
   
    usable = np.zeros((n_blocks_y, n_blocks_x), dtype=bool)
    min_pixels = int(block_size * block_size * 0.10)

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0, y1 = by * block_size, (by + 1) * block_size
            x0, x1 = bx * block_size, (bx + 1) * block_size
            
            n_valid = valid_2d[y0:y1, x0:x1].sum()
            usable[by, bx] = n_valid >= min_pixels
               

    usable_indices = list(zip(*np.where(usable)))
    n_usable = len(usable_indices)
    n_test   = max(1, int(round(n_usable * test_size)))
    n_train  = n_usable - n_test

    labels = np.array([0] * n_train + [1] * n_test, dtype=np.uint8)
    rng = np.random.default_rng(42)
    rng.shuffle(labels)

    # assign labels to blocks
    block_map = np.full((ny, nx), 255, dtype=np.uint8)
    for i, (by, bx) in enumerate(usable_indices):
        y0, y1 = by * block_size, (by + 1) * block_size
        x0, x1 = bx * block_size, (bx + 1) * block_size
        block_map[y0:y1, x0:x1] = labels[i]
    
    return block_map

def spatial_sample(valid_2d,block_size,block_map,ny,nx,bands_flat,canopy_flat,band_names,pixels_per_block,output_label,model,random_state):

    train_indices, test_indices = sample_indices_nonstratified(pixels_per_block,block_size=block_size,block_map=block_map,valid_2d=valid_2d,ny=ny,nx=nx,random_state=random_state)
    train_df = get_pixels(train_indices,bands_flat=bands_flat,canopy_flat=canopy_flat,band_names=band_names)
    test_df = get_pixels(test_indices,bands_flat=bands_flat,canopy_flat=canopy_flat,band_names=band_names)

    test_df.to_parquet(config.DATA_DIR / output_label / f'{output_label}_test_df_{model}.parquet',compression='zstd')
    train_df.to_parquet(config.DATA_DIR / output_label / f'{output_label}_train_df_{model}.parquet',compression='zstd')

    print(f'{model} train pixels: {train_df.shape[0]}\n{model} test pixels: {test_df.shape[0]}')