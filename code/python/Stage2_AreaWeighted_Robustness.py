"""
============================================================
STAGE 2 ROBUSTNESS: Area-weighted vs. nearest-neighbour
============================================================
This script tests whether the recovery-restriction finding is
robust to the choice of categorical-raster aggregation method
when reprojecting 30-m / 1-km land-cover rasters to the ~5.5 km
CHIRPS climate grid.

Module B (Batch2) used `rioxarray.reproject_match` with default
nearest-neighbour resampling, implementing a point-sampling
design: each ~5.5 km climate cell inherits the value of one
representative underlying 300-m / 1-km pixel. This script
recomputes Cohen's d for four key region x category combinations
using AREA-WEIGHTED aggregation: each ~5.5 km climate cell is
characterised by the FRACTION of underlying 30-m transition
pixels and the FRACTION of underlying 1-km from-state pixels
that satisfy the condition. Climate cells are then classified
by fractional thresholds and Cohen's d is recomputed.

If the area-weighted Cohen's d values are similar in sign and
order of magnitude to the nearest-neighbour values used in the
main analysis, the recovery-restriction finding is robust to
aggregation method.

KEY CELLS TESTED (the most central to the paper's thesis):
  1. Sahel recovery        (BAL_GRS, GRS_SHR, SHR_FST in SAH)
  2. Sahel degradation     (FST_SHR, SHR_GRS, FST_CRP, GRS_BAL in SAH)
  3. Continental recovery  (3 recovery pathways across all regions)
  4. Continental degradation (4 degradation pathways across all regions)

OUTPUTS:
  - statistics_for_paper/TableS3_robustness_area_weighted.csv
  - statistics_for_paper/TableS3_robustness_summary.txt

INPUTS (existing files, no re-export needed):
  - transition_{name}.tif (8 bands, 300 m native, multi-band)
  - state_{name}.tif      (8 bands, 1 km native, multi-band)
  - {timescale}_mean_{interval}.tif (per-interval mean SPEI rasters)
  - ipc_africa_5_regions.shp

Run:
  C:\\Users\\hidayat\\.conda\\envs\\zama1\\python.exe Stage2_AreaWeighted_Robustness.py

Expected runtime: 30-60 minutes (4 cells x 7 priored intervals x
fractional aggregation; faster than full Module B because we test
only 4 region x category cells, not 5 x 3).
============================================================
"""

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import geopandas as gpd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION  -- matches Batch2_StricterStablePixel_*.py
# ============================================================
TRANSITIONS_DIR = r'D:\Claude idea\PhD_Paper3_Data\PhD_Paper3_Data'
STATES_DIR      = r'D:\Claude idea\PhD_Paper3_Data\PhD_Paper3_Data'
SPEI_DIR        = r'D:\Claude idea\PhD_Paper3_Data\step5_spei_output'
SHAPEFILE       = r'D:\Claude idea\ipc_africa_5_regions.shp'

STATS_DIR = os.path.join(SPEI_DIR, 'statistics_for_paper')
os.makedirs(STATS_DIR, exist_ok=True)

TRANSITIONS = [
    {'name': 'FST_SHR', 'category': 'Degradation', 'from_state': 'FST'},
    {'name': 'SHR_GRS', 'category': 'Degradation', 'from_state': 'SHR'},
    {'name': 'FST_CRP', 'category': 'Degradation', 'from_state': 'FST'},
    {'name': 'GRS_BAL', 'category': 'Degradation', 'from_state': 'GRS'},
    {'name': 'SHR_FST', 'category': 'Recovery',    'from_state': 'SHR'},
    {'name': 'GRS_SHR', 'category': 'Recovery',    'from_state': 'GRS'},
    {'name': 'BAL_GRS', 'category': 'Recovery',    'from_state': 'BAL'},
]

REGION_ORDER = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
REGION_NAMES = {'MED': 'Mediterranean', 'SAH': 'Sahara-Sahel',
                'WAF': 'West Africa', 'EAF': 'East Africa',
                'SAF': 'Southern Africa'}

# (interval_label, start_band, end_band) -- same as Batch2
INTERVALS = [
    ('1985_1990', 1, 2),
    ('1990_1995', 2, 3),
    ('1995_2000', 3, 4),
    ('2000_2005', 4, 5),
    ('2005_2010', 5, 6),
    ('2010_2015', 6, 7),
    ('2015_2020', 7, 8),
    ('2020_2022', 8, 8),  # terminal interval
]

TIMESCALE = 'spei_12'  # primary timescale used for headline numbers
WINDOW    = 'lagged'   # primary window for RSI/DRA
MIN_TRANS_PIXELS = 30  # same threshold as Batch2

# Fractional thresholds for area-weighted classification
TRANS_FRACTION_MIN  = 0.001  # cell counts as transition if >0.1% of underlying pixels transitioned
STABLE_FRACTION_MIN = 0.50   # cell counts as stable if >50% of underlying pixels remained in from-state

# Test cells (the four most central to the paper's thesis)
TEST_CELLS = [
    {'label': 'SAH_recovery',    'region': 'SAH', 'category': 'Recovery'},
    {'label': 'SAH_degradation', 'region': 'SAH', 'category': 'Degradation'},
    {'label': 'AFR_recovery',    'region': None,  'category': 'Recovery'},     # continental
    {'label': 'AFR_degradation', 'region': None,  'category': 'Degradation'},  # continental
]


# ============================================================
# HELPERS
# ============================================================
def load_multiband(path):
    """Open a multi-band GeoTIFF; ensure CRS is set."""
    da = rioxarray.open_rasterio(path)
    if da.rio.crs is None:
        da = da.rio.write_crs('EPSG:4326')
    return da


def reproject_to_grid_areaweighted(src, target):
    """
    Area-weighted reprojection of a binary 0/1 raster to the target grid.
    Returns the fractional cover (0.0 to 1.0) of the underlying '1' pixels
    that fall inside each target cell.

    This is what reproject_match SHOULD do for categorical aggregation
    when downsampling to a coarser grid -- the fraction of underlying
    pixels meeting the binary criterion.

    Implementation: convert to float and use Resampling.average, which
    gives the mean of input pixels that fall in each output cell.
    For 0/1 binary input, the mean IS the fractional area.
    """
    src_float = src.astype('float32')
    out = src_float.rio.reproject_match(target, resampling=Resampling.average)
    # Clip to [0, 1] to suppress any floating-point overshoots
    out = out.where(out >= 0, 0.0).where(out <= 1, 1.0)
    return out


def compute_cohens_d(transition_spei, stable_spei):
    """
    Compute Cohen's d = (mean_stable - mean_transition) / sigma_pooled.
    Positive d => transition pixels were drier than stable.
    """
    if len(transition_spei) < MIN_TRANS_PIXELS or len(stable_spei) < 100:
        return np.nan, len(transition_spei), len(stable_spei)
    mu_t = np.nanmean(transition_spei)
    mu_s = np.nanmean(stable_spei)
    var_t = np.nanvar(transition_spei, ddof=1)
    var_s = np.nanvar(stable_spei, ddof=1)
    n_t, n_s = len(transition_spei), len(stable_spei)
    sigma_pooled = np.sqrt(((n_t - 1) * var_t + (n_s - 1) * var_s) / (n_t + n_s - 2))
    if sigma_pooled <= 0 or np.isnan(sigma_pooled):
        return np.nan, n_t, n_s
    d = (mu_s - mu_t) / sigma_pooled
    return float(d), n_t, n_s


def squeeze2d(a):
    while a.ndim > 2:
        a = a.squeeze(axis=0)
    return a


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("STAGE 2 ROBUSTNESS: Area-weighted aggregation comparison")
    print("=" * 70)
    print(f"\nTimescale:      {TIMESCALE}")
    print(f"Window:         {WINDOW} (immediately prior interval mean SPEI)")
    print(f"Trans fraction: > {TRANS_FRACTION_MIN*100:.1f}% of underlying pixels")
    print(f"Stable frac:    > {STABLE_FRACTION_MIN*100:.0f}% of underlying pixels")
    print()

    # ---- Load shapefile and dissolve regions ----
    print("[1/4] Loading IPCC shapefile...")
    gdf = gpd.read_file(SHAPEFILE)
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs('EPSG:4326')

    region_geoms = {}
    for code in REGION_ORDER:
        subset = gdf[gdf['LAB'] == code]
        if len(subset) > 0:
            region_geoms[code] = (subset.geometry.union_all()
                                  if hasattr(subset.geometry, 'union_all')
                                  else subset.geometry.unary_union)
    AFR_GEOM = (gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all')
                else gdf.geometry.unary_union)
    print(f"    Loaded {len(region_geoms)} regions + continental geometry")

    # ---- Load transition and state rasters (multi-band) ----
    print("\n[2/4] Loading transition and state rasters...")
    transition_rasters = {}
    for t in TRANSITIONS:
        p = os.path.join(TRANSITIONS_DIR, f'transition_{t["name"]}.tif')
        transition_rasters[t['name']] = load_multiband(p)
    state_rasters = {}
    for state in ['FST', 'SHR', 'GRS', 'BAL']:
        p = os.path.join(STATES_DIR, f'state_{state}.tif')
        state_rasters[state] = load_multiband(p)
    print(f"    Loaded {len(transition_rasters)} transition + {len(state_rasters)} state rasters")

    # ---- Run robustness loop over 4 test cells ----
    print(f"\n[3/4] Running area-weighted attribution for {len(TEST_CELLS)} test cells...")

    rows = []
    for cell in TEST_CELLS:
        cell_label = cell['label']
        region_code = cell['region']
        category = cell['category']

        # Geometry: regional or continental
        if region_code is None:
            geom = AFR_GEOM
            region_label = 'AFRICA'
        else:
            geom = region_geoms[region_code]
            region_label = region_code

        # Pathways for this category
        pathways = [t for t in TRANSITIONS if t['category'] == category]

        print(f"\n  --- Cell: {cell_label} ({region_label} x {category}) ---")
        print(f"      Pathways: {[t['name'] for t in pathways]}")

        cell_d_values = []
        cell_n_trans  = []
        cell_n_stable = []

        # Iterate over interval pairs (need a prior, so start at index 1)
        for int_idx in range(1, len(INTERVALS)):
            int_label, int_start_band, int_end_band = INTERVALS[int_idx]
            prior_label, _, _ = INTERVALS[int_idx - 1]

            # Load prior-interval mean SPEI as the lagged-window value
            spei_path = os.path.join(SPEI_DIR, f'{TIMESCALE}_mean_{prior_label}.tif')
            if not os.path.exists(spei_path):
                # Try alternate path conventions
                spei_path = os.path.join(SPEI_DIR, f'{TIMESCALE}_{prior_label}.tif')
                if not os.path.exists(spei_path):
                    print(f"      WARN: SPEI not found for prior {prior_label}, skip")
                    continue

            spei_prior = load_multiband(spei_path)
            if 'band' in spei_prior.dims:
                spei_prior = spei_prior.isel(band=0, drop=True)
            spei_prior = spei_prior.rio.write_crs('EPSG:4326')

            for t in pathways:
                trans_ds = transition_rasters[t['name']]
                from_state = t['from_state']
                state_ds = state_rasters[from_state]

                # Select the relevant interval band (start = int_start_band)
                if 'band' in trans_ds.dims:
                    trans_band = trans_ds.sel(band=int_start_band)
                else:
                    trans_band = trans_ds
                if 'band' in state_ds.dims:
                    state_start = state_ds.sel(band=int_start_band)
                    state_end   = state_ds.sel(band=int_end_band)
                else:
                    state_start = state_ds
                    state_end   = state_ds

                try:
                    # AREA-WEIGHTED reprojection (fractional cover)
                    trans_frac        = reproject_to_grid_areaweighted(trans_band,   spei_prior)
                    state_start_frac  = reproject_to_grid_areaweighted(state_start,  spei_prior)
                    state_end_frac    = reproject_to_grid_areaweighted(state_end,    spei_prior)
                except Exception as e:
                    print(f"      WARN: reproject failed for {t['name']} interval {int_label}: {e}")
                    continue

                # Clip to region geometry
                try:
                    trans_clip = trans_frac.rio.clip([geom], crs='EPSG:4326',
                                                    all_touched=True, drop=False)
                    sstart_clip = state_start_frac.rio.clip([geom], crs='EPSG:4326',
                                                            all_touched=True, drop=False)
                    send_clip = state_end_frac.rio.clip([geom], crs='EPSG:4326',
                                                       all_touched=True, drop=False)
                    spei_clip = spei_prior.rio.clip([geom], crs='EPSG:4326',
                                                   all_touched=True, drop=False)
                except Exception:
                    continue

                tv = squeeze2d(trans_clip.values)
                sv = squeeze2d(sstart_clip.values)
                ev = squeeze2d(send_clip.values)
                pv = squeeze2d(spei_clip.values.astype('float32'))

                if not (tv.shape == sv.shape == ev.shape == pv.shape):
                    continue

                # Define transition cells: cells with > TRANS_FRACTION_MIN underlying transition pixels
                # AND with the from-state present at interval start
                trans_mask = (tv > TRANS_FRACTION_MIN) & (sv > 0)

                # Define stable cells: cells with from-state present at BOTH endpoints
                # AND with no transition flag at this cell, AND fractional from-state coverage
                # above STABLE_FRACTION_MIN to avoid edge mixed cells
                stable_mask = (sv >= STABLE_FRACTION_MIN) & \
                              (ev >= STABLE_FRACTION_MIN) & \
                              (tv <= TRANS_FRACTION_MIN)

                # Drop NaN SPEI cells
                valid = ~np.isnan(pv)
                trans_mask = trans_mask & valid
                stable_mask = stable_mask & valid

                trans_spei = pv[trans_mask]
                stable_spei = pv[stable_mask]

                d, nt, ns = compute_cohens_d(trans_spei, stable_spei)
                if not np.isnan(d):
                    cell_d_values.append(d)
                    cell_n_trans.append(nt)
                    cell_n_stable.append(ns)
                    print(f"      {t['name']:14s} | {int_label} | d={d:+.3f} | n_t={nt:5d} | n_s={ns:6d}")

            # Free memory
            del spei_prior
            gc.collect()

        # Aggregate over pathways and intervals (sample-size-weighted mean d)
        if len(cell_d_values) > 0:
            arr_d = np.array(cell_d_values)
            arr_n = np.array(cell_n_trans, dtype='float64')
            mean_d_unw = float(np.mean(arr_d))
            mean_d_w   = float(np.sum(arr_d * arr_n) / np.sum(arr_n))
            t_stat, p_val = stats.ttest_1samp(arr_d, popmean=0.0, nan_policy='omit')
            n_cells = len(arr_d)
        else:
            mean_d_unw = np.nan
            mean_d_w   = np.nan
            t_stat, p_val = np.nan, np.nan
            n_cells = 0

        rows.append({
            'cell':                cell_label,
            'region':              region_label,
            'category':            category,
            'n_pathway_intervals': n_cells,
            'mean_d_unweighted':   round(mean_d_unw, 4) if not np.isnan(mean_d_unw) else np.nan,
            'mean_d_weighted':     round(mean_d_w,   4) if not np.isnan(mean_d_w)   else np.nan,
            't_stat':              round(float(t_stat), 3) if not np.isnan(t_stat)  else np.nan,
            'p_value':             round(float(p_val),  4) if not np.isnan(p_val)   else np.nan,
        })

    # ---- Compare against locked-in nearest-neighbour values ----
    print("\n[4/4] Comparing area-weighted vs nearest-neighbour Cohen's d...")
    locked_in_nn = {
        # From manuscript headline numbers (lagged window, SPEI-12)
        'SAH_recovery':    {'mean_d_weighted': -0.484, 'p_value': 0.048},
        'SAH_degradation': {'mean_d_weighted': +0.124, 'p_value': 0.272},
        'AFR_recovery':    {'mean_d_weighted': -0.178, 'p_value': None},
        'AFR_degradation': {'mean_d_weighted': +0.020, 'p_value': None},
    }

    df = pd.DataFrame(rows)
    df['mean_d_locked_in_nn'] = df['cell'].map(
        lambda c: locked_in_nn.get(c, {}).get('mean_d_weighted', np.nan))
    df['delta_aw_minus_nn'] = (df['mean_d_weighted'] - df['mean_d_locked_in_nn']).round(4)
    df['ratio_aw_to_nn'] = (df['mean_d_weighted'] / df['mean_d_locked_in_nn']).round(3)
    df['sign_match'] = (np.sign(df['mean_d_weighted']) == np.sign(df['mean_d_locked_in_nn']))

    out_csv = os.path.join(STATS_DIR, 'TableS3_robustness_area_weighted.csv')
    df.to_csv(out_csv, index=False)
    print(f"    Saved: {out_csv}")

    # ---- Write summary ----
    out_txt = os.path.join(STATS_DIR, 'TableS3_robustness_summary.txt')
    with open(out_txt, 'w') as f:
        f.write("STAGE 2 ROBUSTNESS: Area-weighted aggregation vs nearest-neighbour\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timescale: {TIMESCALE}\n")
        f.write(f"Window:    {WINDOW}\n")
        f.write(f"Fractional thresholds: trans > {TRANS_FRACTION_MIN}, stable >= {STABLE_FRACTION_MIN}\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("INTERPRETATION GUIDE:\n")
        f.write("  - 'mean_d_weighted' is the area-weighted result (this script).\n")
        f.write("  - 'mean_d_locked_in_nn' is the nearest-neighbour result\n")
        f.write("    used in the main manuscript (locked-in values).\n")
        f.write("  - 'sign_match' should be True for all 4 cells if the\n")
        f.write("    recovery-restriction pattern is robust to aggregation method.\n")
        f.write("  - Magnitude differences (|delta| < 0.20 or ratio in [0.5, 1.5])\n")
        f.write("    indicate strong robustness; larger differences should be\n")
        f.write("    interpreted with caution and reported transparently.\n")
    print(f"    Saved: {out_txt}")

    # ---- Console summary ----
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    print()
    if df['sign_match'].all():
        print("  All four test cells: sign of Cohen's d MATCHES between")
        print("  area-weighted and nearest-neighbour designs.")
        print("  -> Recovery-restriction finding is robust to aggregation method.")
    else:
        mismatched = df[~df['sign_match']]['cell'].tolist()
        print("  WARNING: sign mismatch in cells: " + ", ".join(mismatched))
        print("  -> Investigate before submission; consider revising main analysis.")

    print("\nThis output should be cited in the manuscript Section 2.4 and")
    print("Supplementary Materials as Supplementary Table S3.")
    print("=" * 70)


if __name__ == '__main__':
    main()
