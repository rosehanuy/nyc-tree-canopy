"""Derive 10 m vegetation-structure rasters from a high-resolution canopy height model (CHM).

For every 10 m pixel on the model's feature grid, computes the fraction of CHM cells in each
height band:

    lawn        < 0.5 m      mown grass (allows for CHM noise near zero)
    shrub       0.5-2.5 m    the vegetation most likely to be confused with trees
    short_tree  2.5-5 m      the trees most likely to be missed
    tall_tree   >= 5 m       unambiguous canopy

Pixels that are not fully vegetated (tree canopy % + grass/shrub % below `min_veg_pct`) are
set to NaN, which masks out impervious surface, water and bare soil.

Output: one GeoTIFF on the 10 m template grid with 5 bands:
    1 lawn, 2 shrub, 3 short_tree, 4 tall_tree   (fractions 0-1; sum to 1 where valid)
    5 chm_coverage                               (fraction of the 10 m pixel with valid CHM data)

The CHM is read and processed in tiles, so the full 0.5 m city-wide raster never has to fit
in memory. It may be in a different CRS from the template; each tile is reprojected onto the
10 m grid with average resampling (the average of a 0/1 indicator is the fraction).
"""
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.transform import array_bounds
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
import pandas as pd
from utc_src import config

BANDS = ('lawn', 'shrub', 'short_tree', 'tall_tree')
M_TO_FT = 3.28084


def _horizontal_crs(crs):
    """Return the horizontal part of a compound (horizontal + vertical) CRS; other CRSs unchanged. This catches cases when lidar data includes the vertical crs
    """
    try:
        import pyproj
        pc = pyproj.CRS.from_wkt(crs.to_wkt())
        if pc.is_compound:
            return CRS.from_wkt(pc.sub_crs_list[0].to_wkt())
    except ImportError:
        pass
    return crs


def _to_pct(a):
    """Return a percentage array (0-100) whether the input is stored as 0-1 or 0-100."""
    finite = a[np.isfinite(a)]
    return a * 100 if finite.size and np.nanmax(finite) <= 1.0 else a


def chm_height_fractions(chm_path, template_path, grass_pct_path, out_path,
                         breaks_m=(0.5, 2.5, 5.0), chm_units='m', min_veg_pct=99.0,
                         min_coverage=0.9, tile=256, chm_crs=None):
    """
    chm_path:        0.5 m CHM GeoTIFF (heights above ground). Any scale/offset stored in the file's
                     metadata (e.g. integer heights x 100) is applied automatically.
    template_path:   10 m raster defining the output grid AND giving tree canopy % (e.g. nyc21_tree_canopy_lidar.tif)
    grass_pct_path:  10 m grass/shrub % raster on the same grid as the template
    out_path:        output GeoTIFF
    breaks_m:        height-band boundaries in metres
    chm_units:       'm' or 'ft', the vertical units of the CHM (NYC lidar products are often in feet)
    min_veg_pct:     keep pixels with canopy % + grass/shrub % >= this (99 allows for rounding)
    min_coverage:    also mask pixels where less than this fraction has valid CHM data
    chm_crs:         optional override for the CHM's horizontal CRS (e.g. 'EPSG:6539'); by default it is read
                     from the file, dropping any vertical component
    tile:            tile size in 10 m pixels (256 -> ~5 km tiles, ~26M CHM cells per tile at 0.5 m)
    """
    breaks = np.array(breaks_m) * (M_TO_FT if chm_units == 'ft' else 1.0)

    with rasterio.open(template_path) as tmpl, rasterio.open(grass_pct_path) as grass_src, \
         rasterio.open(chm_path) as chm:

        if (grass_src.transform != tmpl.transform) or (grass_src.shape != tmpl.shape):
            raise ValueError('grass/shrub % raster must be on the same 10 m grid as the template')

        # Stored values may be scaled integers (e.g. height x 100). QGIS applies the scale/offset
        # metadata automatically; rasterio does not, so apply it here.
        chm_crs = CRS.from_user_input(chm_crs) if chm_crs else _horizontal_crs(chm.crs)
        if chm_crs != chm.crs:
            print(f'CHM has a compound CRS; using its horizontal part: {chm_crs.to_string()}')
        scale, offset = (chm.scales[0] or 1.0), (chm.offsets[0] or 0.0)
        print(f'CHM: dtype={chm.dtypes[0]}, scale={scale}, offset={offset}, units={chm_units}, '
              f'height breaks={np.round(breaks, 2).tolist()} {chm_units}')

        profile = tmpl.profile.copy()
        profile.update(count=len(BANDS) + 1, dtype='float32', nodata=np.nan,
                       compress='lzw', tiled=True, blockxsize=256, blockysize=256)

        with rasterio.open(out_path, 'w', **profile) as dst:
            for i, name in enumerate(BANDS + ('chm_coverage',), start=1):
                dst.set_band_description(i, name)

            for row0 in range(0, tmpl.height, tile):
                for col0 in range(0, tmpl.width, tile):
                    win = Window(col0, row0, min(tile, tmpl.width - col0), min(tile, tmpl.height - row0))
                    tf = tmpl.window_transform(win)
                    shape = (int(win.height), int(win.width))

                    # --- vegetation mask from the 10 m land-cover rasters
                    canopy = _to_pct(tmpl.read(1, window=win, masked=True).filled(np.nan).astype('float32'))
                    grass = _to_pct(grass_src.read(1, window=win, masked=True).filled(np.nan).astype('float32'))
                    veg_ok = (canopy + grass) >= min_veg_pct

                    out = np.full((len(BANDS) + 1,) + shape, np.nan, dtype='float32')
                    if not veg_ok.any():
                        dst.write(out, window=win)
                        continue

                    # --- read the CHM cells covering this tile (with a small buffer)
                    b = array_bounds(shape[0], shape[1], tf)            # (west, south, east, north)
                    if chm_crs != tmpl.crs:
                        b = transform_bounds(tmpl.crs, chm_crs, *b, densify_pts=21)
                    if not np.all(np.isfinite(b)):
                        raise ValueError(
                            f'Could not convert tile bounds (row {row0}, col {col0}) from {tmpl.crs} to the CHM CRS '
                            f'{chm_crs.to_string()}: got {b}. Check the CHM CRS, or pass chm_crs="EPSG:xxxx" explicitly.')
                    buf = 2 * max(abs(chm.res[0]), abs(chm.res[1]))
                    cwin = from_bounds(b[0] - buf, b[1] - buf, b[2] + buf, b[3] + buf, chm.transform)
                    cwin = cwin.round_offsets().round_lengths()
                    if not _overlaps(cwin, chm):            # tile lies outside the CHM extent
                        dst.write(out, window=win)
                        continue
                    cwin = cwin.intersection(Window(0, 0, chm.width, chm.height))

                    h = chm.read(1, window=cwin, masked=True).astype('float32').filled(np.nan)
                    h = h * scale + offset                                   # stored integers -> heights
                    ctf = chm.window_transform(cwin)
                    valid = np.isfinite(h)
                    h = np.where(valid, np.maximum(h, 0), np.nan)        # negative heights -> 0
                    band_idx = np.digitize(h, breaks)                    # 0..3 (NaN -> 4, masked below)

                    def to_10m(arr):
                        dest = np.full(shape, np.nan, dtype='float32')
                        reproject(arr, dest, src_transform=ctf, src_crs=chm_crs, src_nodata=np.nan,
                                  dst_transform=tf, dst_crs=tmpl.crs, dst_nodata=np.nan,
                                  resampling=Resampling.average)
                        return dest

                    for k in range(len(BANDS)):
                        ind = np.where(valid, (band_idx == k).astype('float32'), np.nan)
                        out[k] = to_10m(ind)
                    out[-1] = to_10m(valid.astype('float32'))           # share of the pixel with CHM data

                    keep = veg_ok & (out[-1] >= min_coverage)
                    out[:, ~keep] = np.nan
                    dst.write(out, window=win)

    print(f'Wrote {out_path}')


def _overlaps(win, src):
    return not (win.col_off >= src.width or win.row_off >= src.height or
                win.col_off + win.width <= 0 or win.row_off + win.height <= 0)


def check_output(out_path, template_path, sample=200_000, seed=0):
    """Sanity checks: fractions sum to 1, and CHM tree fraction (>= 2.5 m) tracks the lidar canopy %."""
    with rasterio.open(out_path) as src, rasterio.open(template_path) as tmpl:
        f = src.read().astype('float64')
        canopy = _to_pct(tmpl.read(1, masked=True).filled(np.nan).astype('float64'))
    ok = np.isfinite(f[0])
    print(f'fully vegetated pixels kept: {ok.sum():,}')
    if not ok.any():
        return
    s = f[:4, ok].sum(axis=0)
    print(f'band fractions sum to 1: min={s.min():.3f}, max={s.max():.3f}')
    tree = (f[2, ok] + f[3, ok]) * 100
    c = canopy[ok]
    idx = np.random.default_rng(seed).choice(ok.sum(), size=min(sample, ok.sum()), replace=False)
    r = np.corrcoef(tree[idx], c[idx])[0, 1]
    print(f'CHM tree % (>= 2.5 m) vs lidar canopy %: r = {r:.3f}, mean diff = {np.mean(tree[idx] - c[idx]):+.1f} pts')
    print('mean band fractions:', {b: round(float(np.nanmean(f[k, ok])), 3) for k, b in enumerate(BANDS)})
    # If r is low or the mean difference is large, check: CHM units (m vs ft), CRS/alignment,
    # and whether the CHM and land cover come from the same lidar acquisition.

########### evaluate model performance in different vegetation landscapes #############3

# paths
HEIGHT_FRACTIONS = config.DATA_DIR / 'nyc21' / 'nyc21_height_fractions.tif'            # from chm_height_fractions.py
LIDAR_CANOPY = config.DATA_DIR / 'nyc21' / 'nyc21_tree_canopy_lidar.tif'                # reference canopy %
PREDICTIONS = {                                                        # label -> OOF prediction raster
    'Spring + Summer + Fall': config.MODEL_OUTPUTS /'nyc21_canopy_from_nyc21_all_seasons_OOF_model.tif',
    'Growing season': config.MODEL_OUTPUTS /'nyc21_canopy_from_nyc21_median_season_only_OOF_model.tif',
}

OUT_FIG = config.FIGURES / 'height_band_errors.png'
OUT_CSV = config.models /'nyc21'/'height_band_errors.csv'

MAX_TREE = 0.05          # maximum amount of tree canopy allowed in pixels
BIN_EDGES = np.array([0, 0.01, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60])
X_MAX = 0.60
NEIGHBOR_PX = 5          # 5 x 5 pixels = 50 m window for the neighbourhood canopy
BLOCK_PX = 100           # 1 km blocks, matching the train/test split
MIN_PIXELS, MIN_BLOCKS = 200, 10
N_BOOT, SEED = 1000, 0
MODEL_STYLE = [dict(color='#7A4FA3', ls='-', marker='o'), dict(color='#D07A2A', ls='--', marker='s')]

def _read(path, band=None):
    with rasterio.open(path) as src:
        a = (src.read(masked=True) if band is None else src.read(band, masked=True))
        return a.astype('float32').filled(np.nan), src.transform, src.shape
 
 
def _to_pct(a):
    finite = a[np.isfinite(a)]
    return a * 100 if finite.size and finite.max() <= 1.0 else a
 
 
def _neighbour_mean(a, size):
    """Mean of the size x size window around each pixel, excluding the pixel itself (NaNs ignored)."""
    valid = np.isfinite(a)
    filled = np.where(valid, a, 0).astype('float64')
    n = size * size
    s = uniform_filter(filled, size, mode='constant') * n - filled
    c = uniform_filter(valid.astype('float64'), size, mode='constant') * n - valid
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(c > 0, s / c, np.nan)
 
 
def load_pixels():
    f, transform, shape = _read(HEIGHT_FRACTIONS)                 # lawn, shrub, short, tall, coverage
    lidar, t2, s2 = _read(LIDAR_CANOPY, 1)
    if (t2, s2) != (transform, shape):
        raise ValueError(f'{LIDAR_CANOPY} is not on the same grid as {HEIGHT_FRACTIONS}')
    lidar = _to_pct(lidar)
    neighbour = _neighbour_mean(lidar, NEIGHBOR_PX)
    preds = {}
    for name, path in PREDICTIONS.items():
        p, t3, s3 = _read(path, 1)
        if (t3, s3) != (transform, shape):
            raise ValueError(f'{path} is not on the same grid as {HEIGHT_FRACTIONS}')
        preds[name] = _to_pct(p)
 
    tree = f[2] + f[3]
    ok = (np.isfinite(f[0]) & (tree < MAX_TREE) & np.isfinite(lidar)
          & np.all([np.isfinite(p) for p in preds.values()], axis=0))
    rows, cols = np.nonzero(ok)
    d = pd.DataFrame({'x': f[1][ok] / (f[0][ok] + f[1][ok]), 'lidar': lidar[ok], 'neighbour': neighbour[ok],
                      'block': (rows // BLOCK_PX) * (shape[1] // BLOCK_PX + 1) + cols // BLOCK_PX})
    for name, p in preds.items():
        d[name] = p[ok] 
    return d[d['x'] <= X_MAX]
 
 
def block_bootstrap_bins(values, bins, blocks, n_bins):
    """Per-bin mean with a 95% CI from resampling blocks (same resamples for every variable)."""
    codes, uniq = pd.factorize(blocks)
    nb = len(uniq)
    S = np.zeros((nb, n_bins)); C = np.zeros((nb, n_bins))
    np.add.at(S, (codes, bins), values)
    np.add.at(C, (codes, bins), 1)
    W = np.random.default_rng(SEED).multinomial(nb, np.full(nb, 1 / nb), size=N_BOOT)
    with np.errstate(invalid='ignore', divide='ignore'):
        boot = (W @ S) / (W @ C)
        mean = S.sum(0) / C.sum(0)
    lo, hi = np.nanquantile(boot, [0.025, 0.975], axis=0)
    return mean, lo, hi, C.sum(0), (C > 0).sum(0)
 
 
def summarize(d):
    m1, m2 = PREDICTIONS
    n_bins = len(BIN_EDGES) - 1
    bins = np.clip(np.digitize(d['x'], BIN_EDGES[1:-1], right=True), 0, n_bins - 1)
    blocks = d['block'].to_numpy()
    stat = lambda v: block_bootstrap_bins(np.asarray(v, dtype=float), bins, blocks, n_bins)
 
    #res = {m: stat(d[m]) for m in (m1, m2)}
    res = {m: stat(d[m] - d['lidar']) for m in (m1, m2)} ### get residual
    calls = {m: stat((d[m] > 0) * 100) for m in (m1, m2)}
    diff = stat((d[m2] - d['lidar']).abs() - (d[m1] - d['lidar']).abs())
    lid, nbr = stat(d['lidar']), stat(d['neighbour'])
 
    n, nblk = res[m1][3], res[m1][4]
    rows = []
    for k in range(n_bins):
        r = {'bin': f'{BIN_EDGES[k]*100:g}–{BIN_EDGES[k+1]*100:g}%', 'bin_lo': BIN_EDGES[k], 'bin_hi': BIN_EDGES[k + 1],
             'x': (BIN_EDGES[k] + BIN_EDGES[k + 1]) / 2, 'n_pixels': int(n[k]), 'n_blocks': int(nblk[k]),
             'plotted': bool(n[k] >= MIN_PIXELS and nblk[k] >= MIN_BLOCKS),
             'lidar_canopy': lid[0][k], f'neighbour_canopy_{NEIGHBOR_PX * 10}m': nbr[0][k]}
        for m in (m1, m2):
            r[f'{m} | mean'], r[f'{m} | lo'], r[f'{m} | hi'] = (res[m][i][k] for i in range(3))
            r[f'{m} | % called canopy'] = calls[m][0][k]
        r['diff | mean'], r['diff | lo'], r['diff | hi'] = (diff[i][k] for i in range(3))
            
        rows.append(r)
    return pd.DataFrame(rows)
 
def plot_shrub_fraction_error(t, fname=None):
 
    m1, m2 = PREDICTIONS
    t = t[t.plotted].reset_index(drop=True)
    n_bins = len(t)
    x = np.arange(n_bins)
    w = 0.38
 
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (m, st) in enumerate(zip((m1, m2), MODEL_STYLE)):
        mean = t[f'{m} | mean']
        err = np.vstack([mean - t[f'{m} | lo'], t[f'{m} | hi'] - mean])
        ax.bar(x + (i - 0.5) * w, mean, w, color=st['color'], edgecolor='white', lw=1, label=f'{m} Model')
        ax.errorbar(x + (i - 0.5) * w, mean, yerr=err, fmt='none', ecolor='0.25', elinewidth=1, capsize=3)
 
    labels = [f"{lo*100:g}–{hi*100:g}%\nn = {n:,}" for lo, hi, n in zip(t['bin_lo'], t['bin_hi'], t['n_pixels'])]
    ax.set_title('Overprediction of Tree Canopy in Shrub/Lawn Pixels')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.6, n_bins - 0.4)
    #ax.set_ylabel('Predicted canopy (%)\n(true canopy ≈ 0%)')
    ax.set_ylabel('Mean Residual\n(Predicted - Lidar % Tree Canopy)')
    ax.set_xlabel('Bins: Shrub % per Pixel', labelpad=4)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
 
    # lawn -> shrub gradient arrow below the axis label
    lo_end, hi_end = t['bin_lo'].iloc[0], t['bin_hi'].iloc[-1]
    left_txt = f'{lo_end * 100:g}% shrub, {(1 - lo_end) * 100:g}% lawn'
    right_txt = f'{hi_end * 100:g}% shrub, {(1 - hi_end) * 100:g}% lawn'
    # positioned relative to the x-axis label (x in axes fraction, y in units of the label's height)
    coords = ('axes fraction', ax.xaxis.label)
    y_arrow = -0.9
    ax.annotate(left_txt, xy=(0.0, y_arrow), xycoords=coords, ha='left', va='center',
                fontsize=9, color='0.25', annotation_clip=False)
    ax.annotate(right_txt, xy=(1.0, y_arrow), xycoords=coords, ha='right', va='center',
                fontsize=9, color='0.25', annotation_clip=False)
    ax.annotate('', xy=(0.80, y_arrow), xytext=(0.20, y_arrow), xycoords=coords, textcoords=coords,
                arrowprops=dict(arrowstyle='-|>', color='0.45', lw=1.2), annotation_clip=False)
 
    plt.tight_layout()
    plt.savefig(fname or OUT_FIG, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()