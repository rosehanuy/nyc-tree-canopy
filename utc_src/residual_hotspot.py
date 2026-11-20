import numpy as np
import pandas as pd
import rioxarray
import matplotlib.pyplot as plt
from affine import Affine
from libpysal.weights import DistanceBand
import esda

import os



def load_residuals(pred_path, lidar_path):
    """10 m residual (pred - lidar) and lidar canopy, both NaN wherever either raster is invalid."""
    pred = rioxarray.open_rasterio(pred_path, masked=True).squeeze('band', drop=True)
    lidar = rioxarray.open_rasterio(lidar_path, masked=True).squeeze('band', drop=True)
    assert pred.shape == lidar.shape and pred.rio.transform().almost_equals(lidar.rio.transform()), \
        'prediction and lidar rasters are not on the same grid'
    valid = pred.notnull() & lidar.notnull()
    resid = (pred - lidar).where(valid)
    return resid, lidar.where(valid)


def canopy_bins(canopy, width=5):
    """Bin index per pixel: 0 = exactly 0%, last = exactly 100%, otherwise `width`-point bins."""
    n_mid = int(np.ceil(100 / width)) ## there are 20 5-point bins between 0 and 100
    b = 1 + np.clip(np.floor(canopy / width), 0, n_mid - 1) # makes bins 1 - 20
    b = np.where(canopy <= 0, 0, b)  # 0% canopy gets bin 0
    b = np.where(canopy >= 100, n_mid + 1, b)   # 100% canopy gets bin 21
    return np.where(np.isfinite(canopy), b, -1).astype(int)  # 2D array where each pixel is the bin number

## must run after aggregation. running before adds artifical spatial structure not present in residuals.
def canopy_adjust(resid, lidar, width=5):
    """Subtract the citywide mean residual of each canopy bin. Returns (adjusted DataArray, bin df)."""
    r, b = resid.values, canopy_bins(lidar.values, width)
    ok = b >= 0  # exclude bin -1 that contains invalid pixels
    ## get mean residual, min canopy, max canopy and n pixels per  bin 
    table = (pd.DataFrame({'bin': b[ok], 'resid': r[ok], 'canopy': lidar.values[ok]})
               .groupby('bin').agg(canopy_lo=('canopy', 'min'), canopy_hi=('canopy', 'max'),
                                   n=('resid', 'size'), mean_resid=('resid', 'mean'))
               .reset_index())
    lookup = np.full(b.max() + 2, np.nan)
    lookup[table['bin'].to_numpy()] = table['mean_resid'].to_numpy() # 1D array mapping bin numbers to mean residuals
    adj = np.where(ok, r - lookup[np.where(ok, b, 0)], np.nan) # substract mean residual according to bin number
    return resid.copy(data=adj.astype('float32')), table ## wrap output in same data array as input residuals


def correlogram(arr, distances, res=10.0, n_pairs=300_000, seed=42):
    """Autocorrelation of a 2-D array at each distance, from random pixel pairs.

    For each distance d: draw n_pairs valid pixels, step d metres in a random direction, keep pairs where
    both pixels are valid, and compute mean((a - mu)(b - mu)) / var. This is the Moran's I-type
    autocorrelation for that distance class (1 = identical, 0 = unrelated).
    """
    rng = np.random.default_rng(seed)
    ok = np.isfinite(arr)
    rows, cols = np.nonzero(ok)
    vals = arr[ok].astype('float64')
    mu, var = vals.mean(), vals.var()
    H, W = arr.shape
    out = []
    for d in distances:
        i = rng.integers(len(vals), size=n_pairs)
        th = rng.uniform(0, 2 * np.pi, n_pairs)
        dr = np.rint(d / res * np.sin(th)).astype(int)
        dc = np.rint(d / res * np.cos(th)).astype(int)
        keep = (dr != 0) | (dc != 0)
        r2, c2 = rows[i] + dr, cols[i] + dc
        keep &= (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        a = vals[i[keep]]
        b = arr[r2[keep], c2[keep]].astype('float64')
        good = np.isfinite(b)
        a, b = a[good], b[good]
        dist = np.hypot(dr[keep][good], dc[keep][good]) * res
        out.append({'distance_m': d, 'mean_pair_distance_m': dist.mean(), 'n_pairs': len(a),
                    'autocorrelation': np.mean((a - mu) * (b - mu)) / var,
                    'covariance': np.mean((a - mu) * (b - mu))}) 
    return pd.DataFrame(out)

def plot_correlogram(corrs):
    """corrs: {label: correlogram DataFrame}."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    styles = [dict(color='black', ls='-'), dict(color='gray', ls='--')]
    
    ax.plot(corrs['mean_pair_distance_m'], corrs['autocorrelation'], marker='o', ms=4, lw=1.5, color='black',ls='-')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xscale('log')
    ticks = [10, 30, 100, 300, 1000, 3000]
    ax.set_xticks(ticks); ax.set_xticklabels([f'{t:,}' for t in ticks]); ax.minorticks_off()
    ax.set_xlabel('Distance between pixels (m)')
    ax.set_ylabel('Autocorrelation of 10 m residuals')
    ax.grid(axis='y', alpha=0.3)
    # for s in ('top', 'right'):
    #     ax.spines[s].set_visible(False)
    plt.tight_layout()
    #plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()

def _coarse_coords(src, factor, nx, ny):
    tf = src.rio.transform()
    new_tf = tf * Affine.scale(factor)
    x = new_tf.c + (np.arange(nx) + 0.5) * new_tf.a
    y = new_tf.f + (np.arange(ny) + 0.5) * new_tf.e
    return x, y, new_tf


def aggregate(da, cell_size, min_coverage=0.75, res=10):
    """Mean of valid 10 m pixels per coarse cell; cells below min_coverage set to NaN."""
    
    f = int(cell_size // res)
    kw = dict(x=f, y=f, boundary='pad')
    s = da.fillna(0).coarsen(**kw).sum()
    n = da.notnull().astype('float32').coarsen(**kw).sum()
    out = (s / n).where(n >= min_coverage * f * f).astype('float32')
    x, y, tf = _coarse_coords(da, f, out.sizes['x'], out.sizes['y'])
    return (out.assign_coords(x=x, y=y).rio.write_crs(da.rio.crs)
               .rio.write_transform(tf).rio.write_nodata(np.nan, encoded=False))


## control family-wise error rate
def _bh_significant(p, alpha=0.05):
    """True where p is significant at false-discovery rate alpha."""
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    passed = np.nonzero(ranked <= alpha)[0]
    sig = np.zeros(len(p), bool)
    if passed.size:
        sig[order[:passed.max() + 1]] = True
    return sig

CLUSTER_NAMES = {0: 'Not significant', 1: 'High-High', 2: 'Low-High', 3: 'Low-Low', 4: 'High-Low'}
def local_statistics(cells, neighbour_m, permutations, alpha, seed=42):
    """LISA for every valid coarse cell. Returns (per-cell DataFrame, global Moran's I object)."""
    
    df = cells.to_series().dropna().rename('resid_adj').reset_index()

    def weights():
        return DistanceBand(df[['x', 'y']].to_numpy(), threshold=neighbour_m, binary=True, silence_warnings=True)
    w = weights()
    if w.islands:                                   # cells with no valid neighbours cannot be tested
        df = df.drop(index=w.islands).reset_index(drop=True)
        w = weights()
    w_bin = weights()                               # binary copy for Gi* (which adds the cell itself)
    w.transform = 'r'                               # row-standardised for Moran's I
    y = df['resid_adj'].to_numpy('float64')

    glob = esda.Moran(y, w, permutations=permutations)
    lisa = esda.Moran_Local(y, w, permutations=permutations, seed=seed, n_jobs=-1,keep_simulations=False)
    #gi = esda.G_Local(y, w_bin, transform='B', permutations=permutations, star=True, seed=seed,keep_simulations=False,n_jobs=1)

    df['n_neighbours'] = [w.cardinalities[i] for i in range(len(df))]
    df['lisa_I'] = lisa.Is
    df['lisa_quadrant'] = lisa.q                      # 1 HH, 2 LH, 3 LL, 4 HL (relative to the citywide mean)
    df['lisa_p'] = lisa.p_sim
    df['lisa_sig_fdr'] = _bh_significant(lisa.p_sim, alpha)
    df['lisa_cluster'] = np.where(df['lisa_sig_fdr'], df['lisa_quadrant'], 0)
    df['lisa_label'] = df['lisa_cluster'].map(CLUSTER_NAMES)

    # df['gi_z'] = gi.Zs                                # > 0: neighbourhood of over-prediction; < 0: under-prediction
    # df['gi_p'] = gi.p_sim
    # df['gi_sig_fdr'] = _bh_significant(gi.p_sim, alpha)
    # df['gi_spot'] = np.select([df['gi_sig_fdr'] & (df['gi_z'] > 0), df['gi_sig_fdr'] & (df['gi_z'] < 0)],
    #                           ['Hot (over-prediction)', 'Cold (under-prediction)'], 'Not significant')
    return df, glob


def to_raster(df, template, column, path, dtype='float32', nodata=np.nan):
    """Write one per-cell column back onto the coarse grid."""
    arr = np.full(template.shape, nodata, dtype=dtype)
    r = np.searchsorted(-template['y'].values, -df['y'].values)     # y decreases down the rows
    c = np.searchsorted(template['x'].values, df['x'].values)
    arr[r, c] = df[column].to_numpy(dtype)
    template.copy(data=arr).rio.write_nodata(nodata, encoded=False).rio.to_raster(path, compress='lzw')


def cluster_summary(df):
    """Cells per LISA cluster type, with their mean adjusted residual (the effect size)."""
    return (df.groupby('lisa_label')
              .agg(n_cells=('resid_adj', 'size'), mean_resid_adj=('resid_adj', 'mean'))
              .assign(share_of_cells=lambda t: t['n_cells'] / t['n_cells'].sum())
              .reindex([v for v in CLUSTER_NAMES.values()]).dropna(how='all'))


### part 2 ###############
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.transform import array_bounds
import pyproj
from utc_src import config
from affine import Affine


M_TO_FT = 3.28084
INF = np.inf

# (name, land-cover code, min height m, max height m): a 0.5 m cell belongs to a class if its land-cover
# code matches and min <= height < max. Classes with (-INF, INF) do not use the CHM.
CLASSES = [
    ('tree_under_2.5m',     1, -INF, 2.5),    # tree canopy in land cover but low in the CHM (product mismatch)
    ('tree_2.5_5m',         1,  2.5, 5.0),    # short trees
    ('tree_5m_plus',        1,  5.0, INF),    # tall trees
    ('lawn',                2, -INF, 0.5),
    ('shrub',               2,  0.5, 2.5),
    ('grass_shrub_2.5m_plus', 2, 2.5, INF),   # mismatch (e.g. overhanging canopy); usually small
    ('bare_soil',           3, -INF, INF),
    ('water',               4, -INF, INF),
    ('building_under_15m',  5, -INF, 15.0),   # up to ~4 storeys
    ('building_15m_plus',   5, 15.0, INF),    # tall enough to cast long shadows
    ('road',                6, -INF, INF),
    ('other_impervious',    7, -INF, INF),
    ('railroad',            8, -INF, INF),
]
NAMES = [c[0] for c in CLASSES]


def _horizontal_crs(crs):
    """Horizontal part of a compound (horizontal + vertical) CRS; other CRSs unchanged."""
    try:
        import pyproj
        pc = pyproj.CRS.from_wkt(crs.to_wkt())
        if pc.is_compound:
            return CRS.from_wkt(pc.sub_crs_list[0].to_wkt())
    except ImportError:
        pass
    return crs

def landcover_height_fractions(landcover_path, chm_path, template_path, out_path,
                               chm_units='ft', tile=128, lc_crs=None, chm_crs=None, classes=CLASSES):
    """Write a GeoTIFF on the 10 m template grid: one band per class (fraction 0-1) + a coverage band
    (share of the 10 m pixel with a valid class). Fractions sum to 1 where coverage > 0.

    chm_units: vertical units of the CHM ('ft' or 'm'); any scale/offset in the file is applied.
    tile:      tile size in 10 m pixels (128 -> 1.28 km tiles; the 15 cm land cover of one tile is ~73M cells).
    """
    to_m = 1 / M_TO_FT if chm_units == 'ft' else 1.0
    with rasterio.open(template_path) as tmpl, rasterio.open(chm_path) as chm, rasterio.open(landcover_path) as lc:
        chm_crs = CRS.from_user_input(chm_crs) if chm_crs else _horizontal_crs(chm.crs)
        lc_crs = CRS.from_user_input(lc_crs) if lc_crs else _horizontal_crs(lc.crs)
        scale, offset = (chm.scales[0] or 1.0), (chm.offsets[0] or 0.0)
        lc_nodata = lc.nodata if lc.nodata is not None else 0
        print(f'CHM crs={chm_crs.to_string()} scale={scale} offset={offset} units={chm_units}; '
              f'land cover crs={lc_crs.to_string()} nodata={lc_nodata}')

        profile = tmpl.profile.copy()
        profile.update(count=len(classes) + 1, dtype='float32', nodata=np.nan, compress='lzw',
                       tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(out_path, 'w', **profile) as dst:
            for i, name in enumerate([c[0] for c in classes] + ['coverage'], start=1):
                dst.set_band_description(i, name)

            n_tiles = int(np.ceil(tmpl.height / tile) * np.ceil(tmpl.width / tile))
            k = 0
            for row0 in range(0, tmpl.height, tile):
                for col0 in range(0, tmpl.width, tile):
                    k += 1
                    win = Window(col0, row0, min(tile, tmpl.width - col0), min(tile, tmpl.height - row0))
                    tf, shape = tmpl.window_transform(win), (int(win.height), int(win.width))
                    out = np.full((len(classes) + 1,) + shape, np.nan, dtype='float32')

                    # find the 0.5 m CHM cells covering this tile (+ small buffer)

                    ### bounding box of window in template crs
                    b = array_bounds(shape[0], shape[1], tf) 
                    if chm_crs != tmpl.crs: ## convert bounding box to chm crs if necessary
                        b = transform_bounds(tmpl.crs, chm_crs, *b, densify_pts=21)
                    buf = 2 * max(abs(chm.res[0]), abs(chm.res[1])) ## add buffer

                    # get row/columns from bounding box coordinates, use chm transform
                    cwin = from_bounds(b[0] - buf, b[1] - buf, b[2] + buf, b[3] + buf, chm.transform) 
                    # round to avoid fractions of pixels
                    cwin = cwin.round_offsets().round_lengths()
                    # if window is fully outside bounds of chm, continue without writing
                    if (cwin.col_off >= chm.width or cwin.row_off >= chm.height or
                            cwin.col_off + cwin.width <= 0 or cwin.row_off + cwin.height <= 0):
                        dst.write(out, window=win); continue
                    ## trim the window if it partially falls outside bounds of chm
                    cwin = cwin.intersection(Window(0, 0, chm.width, chm.height))
                    # get the transform for this window specifically (needed later for reprojection)
                    ctf = chm.window_transform(cwin)
                    ## read pixels in this window, mask no data values
                    h = chm.read(1, window=cwin, masked=True).astype('float32').filled(np.nan)
                    ## apply scale and offset, set small negatives to 0, convert ft to meters
                    h = np.maximum(h * scale + offset, 0) * to_m               

                    # get land cover category that mostly covers each 0.5 m CHM cell in this window
                    ## bounding coordinates of this window in chm crs
                    cb = array_bounds(h.shape[0], h.shape[1], ctf)
                    ## transform to landcover crs if necessary
                    if lc_crs != chm_crs:
                        cb = transform_bounds(chm_crs, lc_crs, *cb, densify_pts=21)
                    lbuf = 4 * max(abs(lc.res[0]), abs(lc.res[1]))
                    ## get row/cols from bounding coordinates
                    lwin = from_bounds(cb[0] - lbuf, cb[1] - lbuf, cb[2] + lbuf, cb[3] + lbuf, lc.transform)
                    ## round and get intersection of this window and the landcover raster
                    lwin = lwin.round_offsets().round_lengths().intersection(Window(0, 0, lc.width, lc.height))
                    ## read pixels in this window
                    lc_src = lc.read(1, window=lwin)
                    lc05 = np.zeros(h.shape, dtype=lc_src.dtype)
                    ## reproject landcover using mode resampling; each 0.5 m pixel is assigned the most common landcover category present
                    reproject(lc_src, lc05, src_transform=lc.window_transform(lwin), src_crs=lc_crs,
                              src_nodata=lc_nodata, dst_transform=ctf, dst_crs=chm_crs, dst_nodata=0,
                              resampling=Resampling.mode)

                    # combine landcover class and height to assign final classes to 0.5 m pixels. 
                    # cells without a land-cover code, or without a height when the class needs one, are invalid
                    cls = np.full(h.shape, -1, dtype=np.int16)
                    for j, (_, code, hmin, hmax) in enumerate(classes):
                        uses_h = np.isfinite(hmin) or np.isfinite(hmax)
                        m = lc05 == code
                        if uses_h:
                            m &= np.isfinite(h) & (h >= hmin) & (h < hmax)
                        cls[m] = j
                    valid = cls >= 0

                    def to_10m(arr):
                        d = np.full(shape, np.nan, dtype='float32')
                        reproject(arr, d, src_transform=ctf, src_crs=chm_crs, src_nodata=np.nan,
                                  dst_transform=tf, dst_crs=tmpl.crs, dst_nodata=np.nan,
                                  resampling=Resampling.average)
                        return d
                    # reproject 0.5 m to 10m using average resampling
                    for j in range(len(classes)):
                        out[j] = to_10m(np.where(valid, (cls == j).astype('float32'), np.nan))
                    # coverage: share of the 10 m pixel that has a valid class (NaN source cells count as 0)
                    out[-1] = to_10m(valid.astype('float32'))
                    dst.write(out, window=win)
                    if k % 50 == 0 or k == n_tiles:
                        print(f'  tile {k}/{n_tiles}')
    print(f'Wrote {out_path}')



def _coarse_coords(src, factor, nx, ny):
    tf = src.rio.transform()
    new_tf = tf * Affine.scale(factor)
    x = new_tf.c + (np.arange(nx) + 0.5) * new_tf.a
    y = new_tf.f + (np.arange(ny) + 0.5) * new_tf.e
    return x, y, new_tf

def load_valid(pred_path, lidar_path):
    """NaN wherever either raster is invalid."""
    pred = rioxarray.open_rasterio(pred_path, masked=True).squeeze('band', drop=True)
    lidar = rioxarray.open_rasterio(lidar_path, masked=True).squeeze('band', drop=True)
    assert pred.shape == lidar.shape and pred.rio.transform().almost_equals(lidar.rio.transform()), \
        'prediction and lidar rasters are not on the same grid'
    valid = pred.notnull() & lidar.notnull()
    
    return valid

def cell_fractions(fractions_10m_path, model, cell_size=100, min_coverage=0.75, res=10):
    """Mean class fractions per coarse cell, over the 10 m pixels that are valid for the LISA analysis
    (`valid`, e.g. resid.notnull()) AND have land-cover data. Returns a DataFrame indexed by (y, x)."""
    pred_path = config.MODEL_OUTPUTS / model / f'nyc21_canopy_from_nyc21_{model}_OOF_model.tif'
    lidar_path = config.DATA_DIR / 'nyc21' / 'nyc21_tree_canopy_lidar.tif'
    valid = load_valid(pred_path,lidar_path)

    fr = rioxarray.open_rasterio(fractions_10m_path, masked=True)
    names = list(fr.attrs.get('long_name', NAMES + ['coverage']))
    fr = fr.assign_coords(band=names)
    assert fr.rio.transform().almost_equals(valid.rio.transform()) and fr.shape[1:] == valid.shape, \
        'fractions raster is not on the model grid'
    ok = valid & fr.sel(band=names[0]).notnull()
    f = int(cell_size // res)
    kw = dict(x=f, y=f, boundary='pad')
    n = ok.astype('float32').coarsen(**kw).sum()
    keep = n >= min_coverage * f * f
    x, y, _ = _coarse_coords(valid, f, n.sizes['x'], n.sizes['y'])
    cols = {}
    for nm in names[:-1]:
        s = fr.sel(band=nm).where(ok).fillna(0).coarsen(**kw).sum()
        cols[nm] = (s / n).where(keep).assign_coords(x=x, y=y).to_series()
    return pd.DataFrame(cols).dropna(how='all')

def _canopy_bin(canopy, width=10):
    return np.clip(np.floor(canopy / width), 0, int(100 / width) - 1).astype(int)


def compare_clusters(model, fractions, cluster_col='lisa_label',
                     clusters=('High-High', 'Low-Low'), reference='Not significant',
                     canopy_bin_width=10, block_cells=10, n_boot=1000, seed=0):
    """Mean class fractions per cluster type vs. canopy-matched non-significant cells.

    local_stats: DataFrame with x, y and cluster_col (e.g. the *_local_stats_100m.csv from residual_hotspots)
    fractions:   output of cell_fractions() (index y, x)
    cell_canopy: mean lidar canopy % per 100 m cell (DataArray on the same grid)
    Reference = non-significant cells, reweighted so their canopy distribution (bins of canopy_bin_width)
    matches the cluster's. CIs from resampling 1 km blocks (block_cells x block_cells cells).
    Returns a long table: cluster, class, cluster_mean, reference_mean, diff, ci_low, ci_high (in % of area).
    """
    cell_canopy = rioxarray.open_rasterio(config.MODEL_OUTPUTS / 'resamp' / 'resamp_100_nyc21_tree_canopy_lidar.tif')
    local_stats = pd.read_csv(config.MODEL_OUTPUTS / 'residuals' / model / f'{model}_local_stats_100m.csv')
    d = local_stats[['x', 'y', cluster_col]].merge(fractions.reset_index(), on=['x', 'y'], how='inner')
    can = cell_canopy.to_series().rename('canopy').reset_index()
    d = d.merge(can, on=['x', 'y'], how='inner').dropna()
    classes = [c for c in fractions.columns]
    F = d[classes].to_numpy() * 100                           # percent of cell area
    cbin = _canopy_bin(d['canopy'].to_numpy(), canopy_bin_width)
    nb = cbin.max() + 1

    # 1 km blocks from cell position
    xs, ys = np.sort(d['x'].unique()), np.sort(d['y'].unique())[::-1]
    step = np.median(np.diff(xs))
    col = np.rint((d['x'] - xs.min()) / step).astype(int)
    row = np.rint((ys.max() - d['y']) / step).astype(int)
    blocks, block_idx = np.unique((row // block_cells) * 100000 + col // block_cells, return_inverse=True)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=n_boot)

    ref = (d[cluster_col] == reference).to_numpy()

    def matched(w, grp):
        """Weighted cluster mean, canopy-matched reference mean."""
        wc, wr = w * grp, w * ref
        ref_sum = np.zeros((nb, F.shape[1])); np.add.at(ref_sum, cbin, F * wr[:, None])
        ref_w = np.bincount(cbin, weights=wr, minlength=nb)
        grp_w = np.bincount(cbin, weights=wc, minlength=nb)
        usable = (ref_w > 0) & (grp_w > 0)                     # canopy bins present in both groups
        share = np.where(usable, grp_w, 0) / max(grp_w[usable].sum(), 1e-12)
        ref_mean = (share[usable, None] * ref_sum[usable] / ref_w[usable, None]).sum(0)
        in_usable = usable[cbin] & grp
        cl_mean = (F[in_usable] * w[in_usable, None]).sum(0) / max(w[in_usable].sum(), 1e-12)
        return cl_mean, ref_mean

    rows = []
    for cl in clusters:
        grp = (d[cluster_col] == cl).to_numpy()
        if not grp.any():
            continue
        cm, rm = matched(np.ones(len(d)), grp)
        boots = np.array([np.subtract(*matched(W[i][block_idx].astype(float), grp)) for i in range(n_boot)])
        lo, hi = np.nanquantile(boots, [0.025, 0.975], axis=0)
        for j, c in enumerate(classes):
            rows.append({'cluster': cl, 'class': c, 'n_cells': int(grp.sum()),
                         'cluster_mean': cm[j], 'reference_mean': rm[j], 'diff': cm[j] - rm[j],
                         'ci_low': lo[j], 'ci_high': hi[j]})
    return pd.DataFrame(rows)


from matplotlib.lines import Line2D

LABELS = {'tree_under_2.5m': 'Tree canopy < 2.5 m (mismatch)', 'tree_2.5_5m': 'Short Trees (2.5–5 m)',
          'tree_5m_plus': 'Tall Trees (≥ 5 m)', 'lawn': 'Lawn (< 0.5 m)', 'shrub': 'Shrubs (0.5–2.5 m)',
          'grass_shrub_2.5m_plus': 'Grass/shrub ≥ 2.5 m (mismatch)', 'bare_soil': 'Bare soil', 'water': 'Water',
          'building_under_15m': 'Short Buildings (< 15 m)', 'building_15m_plus': 'Tall Buildings (≥ 15 m)', 'road': 'Roads',
          'other_impervious': 'Other impervious', 'railroad': 'Railroads'}

CLUSTER_TEXT = {'High-High': 'Clustered over-prediction',
                'Low-Low': 'Clustered under-prediction'}

c1 = '#285A48'
c2 = '#8BBB92'
c3 = '#3B9797'
MODEL_COLORS = {'Three Seasons Model': c1, 'Two Seasons Model': c3,
                'Growing Season Model': c2}        
MODEL_MARKERS = ['o', 's', 'D']


def plot_comparison_models(tables, figure_filename=None, title=None, classes=None):
    """tables: {model label: compare_clusters() output}. One panel per cluster type; within each panel,
    one row per land-cover class with a dot + 95% CI per model (offset vertically)."""
    models = list(tables)
    first = tables[models[0]]
    clusters = list(dict.fromkeys(first['cluster']))
    classes = classes or list(dict.fromkeys(first['class']))
    k = len(models)
    step = 0.22                                           # vertical spacing between models within a class row
    offsets = (np.arange(k) - (k - 1) / 2) * step

    fig, axes = plt.subplots(1, len(clusters), figsize=(4.6 * len(clusters), 0.42 * len(classes) + 1.6),
                             sharey=True, sharex=True, squeeze=False)
    ypos = np.arange(len(classes))[::-1]
    for ax, cl in zip(axes[0], clusters):
        ax.axvline(0, color='0.5', lw=0.8)
        for y in ypos[1::2]:                              # light bands to separate class rows
            ax.axhspan(y - 0.5, y + 0.5, color='0.95', zorder=0, lw=0)
        ns = []
        for i, m in enumerate(models):
            t = tables[m]
            t = t[t.cluster == cl].set_index('class').reindex(classes)
            ns.append(f"{int(t['n_cells'].dropna().iloc[0]):,}")
            ax.errorbar(t['diff'], ypos - offsets[i],
                        xerr=[t['diff'] - t['ci_low'], t['ci_high'] - t['diff']],
                        fmt=MODEL_MARKERS[i % len(MODEL_MARKERS)], ms=4.5,
                        color=MODEL_COLORS.get(m, f'C{i}'), elinewidth=1.3, capsize=0)
        #ax.set_title(f"{CLUSTER_TEXT.get(cl, cl)}\n{' / '.join(ns)}", fontsize=9, loc='left')
        ax.set_title(f"{CLUSTER_TEXT.get(cl, cl)}", fontsize=9, loc='left')
        #ax.set_xlabel('Difference in share of cell area (percentage points)\n'
                      #'vs. non-clustered cells with similar canopy', fontsize=8)
        ax.set_xlabel('Difference in Cell Coverage (percentage points)\nclustered cells vs. non-clustered cells', fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        ax.set_ylim(-0.5, len(classes) - 0.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0, 0].set_yticks(ypos)
    axes[0, 0].set_yticklabels([LABELS.get(c, c) for c in classes], fontsize=8)
    axes[0, 0].tick_params(axis='y', length=0)

    handles = [Line2D([], [], color=MODEL_COLORS.get(m, f'C{i}'), marker=MODEL_MARKERS[i % len(MODEL_MARKERS)],
                      ms=5, lw=1.3, label=m) for i, m in enumerate(models)]
    fig.legend(handles=handles, loc='upper center', ncol=k, frameon=False, fontsize=9,
               bbox_to_anchor=(0.55, 1.0 if not title else 0.98))
    if title:
        fig.suptitle(title, fontsize=10, y=1.06)
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    
    plt.savefig(config.FIGURES / f'{figure_filename}.png', dpi=300, bbox_inches='tight')

    print(f'figure saved to {config.FIGURES} / {figure_filename}.png')
   


  