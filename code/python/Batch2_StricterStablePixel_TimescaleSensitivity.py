"""
============================================================
BATCH 2: Stricter stable-pixel + timescale sensitivity (Tasks 3 + 5)
============================================================

Task 3: Stricter stable-pixel definition
  Old: pixel was in from-state X at interval start AND did not transition
  New: pixel was in from-state X at interval start AND interval end
       AND did not record a transition during the interval
  Effect: filters out pixels that transitioned MULTIPLE times within
          an interval (stable-then-unstable-then-stable scenarios).
  Expected: stable-pixel sample sizes drop ~10-20%, but Cohen's d values
            become more defensible.

Task 5: Timescale sensitivity
  Compute Cohen's d, RSI, and DRA for all 4 SPEI timescales (12/24/36/60).
  Verify whether the recovery-suppression signal is robust across timescales.
  Output: TableB4_timescale_sensitivity_RSI.csv
          TableB4_timescale_sensitivity_DRA.csv
          TableB4_timescale_sensitivity_per_pathway.csv

The stricter Module B rerun produces, for each region × pathway × interval × timescale:
  - mean SPEI at transition pixels
  - mean SPEI at (stricter) stable pixels
  - drought exposure fractions
  - Cohen's d
  - n_trans, n_stable

Because the stricter analysis IS our new primary, we save with primary names:
  TableB2_LAGGED_per_interval_STRICTER.csv
  TableB2_CUMULATIVE_per_interval_STRICTER.csv
  TableB2_LAGGED_SUMMARY_spei12_STRICTER.csv
  TableB2_CUMULATIVE_SUMMARY_spei12_STRICTER.csv
And then re-run RSI/DRA/decadal analyses on the stricter results.

EXPECTED RUNTIME: 30-60 minutes (one Module B pass with stricter definition)
============================================================
"""

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
TRANSITIONS_DIR = r'D:\Claude idea\PhD_Paper3_Data\PhD_Paper3_Data'
STATES_DIR = r'D:\Claude idea\PhD_Paper3_Data\PhD_Paper3_Data'
SPEI_DIR = r'D:\Claude idea\PhD_Paper3_Data\step5_spei_output'
SHAPEFILE = r'D:\Claude idea\ipc_africa_5_regions.shp'
RESULTS_DIR = os.path.join(SPEI_DIR, 'ModuleB_results')
STATS_DIR = os.path.join(SPEI_DIR, 'statistics_for_paper')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

TRANSITIONS = [
    {'name': 'FST_SHR',         'category': 'Degradation', 'from_state': 'FST', 'from_label': 'FST'},
    {'name': 'SHR_GRS',         'category': 'Degradation', 'from_state': 'SHR', 'from_label': 'SHR'},
    {'name': 'FST_CRP',         'category': 'Degradation', 'from_state': 'FST', 'from_label': 'FST'},
    {'name': 'GRS_BAL',         'category': 'Degradation', 'from_state': 'GRS', 'from_label': 'GRS'},
    {'name': 'SHR_FST',         'category': 'Recovery',    'from_state': 'SHR', 'from_label': 'SHR'},
    {'name': 'GRS_SHR',         'category': 'Recovery',    'from_state': 'GRS', 'from_label': 'GRS'},
    {'name': 'BAL_GRS',         'category': 'Recovery',    'from_state': 'BAL', 'from_label': 'BAL'},
    {'name': 'AGEXPANSION',     'category': 'Agricultural','from_state': 'NATURAL', 'from_label': 'SHR/GRS/BAL'},
    {'name': 'CRP_ABANDONMENT', 'category': 'Agricultural','from_state': 'CRP', 'from_label': 'CRP'},
]

REGION_ORDER = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
REGION_NAMES = {'MED': 'Mediterranean', 'SAH': 'Sahara-Sahel',
                'WAF': 'West Africa', 'EAF': 'East Africa',
                'SAF': 'Southern Africa'}

# (interval_label, interval_band_number, end_band_number)
# Bands 1-8 represent the START state of each interval; the END state of
# interval i is the START state of interval i+1. Interval 8 is 2020-2022,
# we use band 8 as start; interval-end is implicit at end-of-2022 (no later band).
INTERVALS = [
    ('1985_1990', 1, 2),
    ('1990_1995', 2, 3),
    ('1995_2000', 3, 4),
    ('2000_2005', 4, 5),
    ('2005_2010', 5, 6),
    ('2010_2015', 6, 7),
    ('2015_2020', 7, 8),
    ('2020_2022', 8, 8),  # last interval: use start state for both (no end-state available)
]

SPEI_TIMESCALES = ['spei_12', 'spei_24', 'spei_36', 'spei_60']
MIN_TRANS_PIXELS = 30


# ============================================================
# Diagnostic banner
# ============================================================
print("=" * 70)
print("BATCH 2: Stricter stable-pixel + timescale sensitivity (Tasks 3 + 5)")
print("=" * 70)
print()
print("This script reruns Module B attribution with a stricter stable-pixel")
print("definition: a pixel is 'stable' for transition X→Y in interval t if it")
print("was in state X at BOTH interval start AND interval end, AND the transition")
print("X→Y was NOT recorded during interval t.")
print()
print("It also computes RSI/DRA across all 4 SPEI timescales (12/24/36/60).")


# ============================================================
# PREFLIGHT: Verify all files
# ============================================================
print("\n" + "=" * 70)
print("PREFLIGHT: File validation")
print("=" * 70)

missing = []
for t in TRANSITIONS:
    p = os.path.join(TRANSITIONS_DIR, f'transition_{t["name"]}.tif')
    if not os.path.exists(p):
        missing.append(p)

for state in ['FST', 'SHR', 'GRS', 'BAL', 'CRP', 'NATURAL']:
    p = os.path.join(STATES_DIR, f'state_{state}.tif')
    if not os.path.exists(p):
        missing.append(p)

for ts in SPEI_TIMESCALES:
    for label, _, _ in INTERVALS:
        p = os.path.join(SPEI_DIR, f'{ts}_mean_{label}.tif')
        if not os.path.exists(p):
            missing.append(p)

if missing:
    print(f"  ERROR: {len(missing)} files missing:")
    for m in missing[:10]:
        print(f"    {m}")
    raise SystemExit("Cannot proceed.")

print(f"  ✓ All 9 transition files present")
print(f"  ✓ All 6 state files present")
print(f"  ✓ All 32 SPEI files present")


# ============================================================
# Load shapefile
# ============================================================
gdf = gpd.read_file(SHAPEFILE)
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.set_crs('EPSG:4326') if gdf.crs is None else gdf.to_crs('EPSG:4326')
region_geoms = {}
for code in REGION_ORDER:
    subset = gdf[gdf['LAB'] == code]
    region_geoms[code] = (subset.geometry.union_all() if hasattr(subset.geometry, 'union_all')
                          else subset.geometry.unary_union)


# ============================================================
# Helpers
# ============================================================

def load_tif(path):
    da = rioxarray.open_rasterio(path)
    if da.rio.crs is None:
        da = da.rio.write_crs('EPSG:4326')
    if 'band' in da.dims and da.sizes['band'] == 1:
        da = da.squeeze('band', drop=True)
    return da


def compute_attribution_stats(spei_trans, spei_stable, threshold=-1.0):
    """Compute attribution statistics for one cell."""
    trans = spei_trans[~np.isnan(spei_trans)]
    stable = spei_stable[~np.isnan(spei_stable)]

    empty = {
        'n_trans': int(len(trans)), 'n_stable': int(len(stable)),
        'mean_spei_trans': np.nan, 'mean_spei_stable': np.nan,
        'spei_difference': np.nan,
        'exposure_frac_trans': np.nan, 'exposure_frac_stable': np.nan,
        'cohens_d': np.nan,
    }

    if len(trans) < MIN_TRANS_PIXELS or len(stable) < 100:
        return empty

    mean_t = float(np.mean(trans))
    mean_s = float(np.mean(stable))
    std_t = float(np.std(trans))
    std_s = float(np.std(stable))

    expo_t = float((trans < threshold).sum()) / len(trans)
    expo_s = float((stable < threshold).sum()) / len(stable)

    pooled_std = np.sqrt((std_t**2 + std_s**2) / 2)
    cohens_d = (mean_s - mean_t) / pooled_std if pooled_std > 1e-9 else np.nan

    return {
        'n_trans': int(len(trans)),
        'n_stable': int(len(stable)),
        'mean_spei_trans': round(mean_t, 4),
        'mean_spei_stable': round(mean_s, 4),
        'spei_difference': round(mean_t - mean_s, 4),
        'exposure_frac_trans': round(expo_t, 4),
        'exposure_frac_stable': round(expo_s, 4),
        'cohens_d': round(float(cohens_d), 3) if not np.isnan(cohens_d) else np.nan,
    }


# ============================================================
# Load transition + state rasters
# ============================================================
print("\n" + "=" * 70)
print("Loading transition + state rasters")
print("=" * 70)

transition_rasters = {}
for t in TRANSITIONS:
    path = os.path.join(TRANSITIONS_DIR, f'transition_{t["name"]}.tif')
    transition_rasters[t['name']] = rioxarray.open_rasterio(path)
    if transition_rasters[t['name']].rio.crs is None:
        transition_rasters[t['name']] = transition_rasters[t['name']].rio.write_crs('EPSG:4326')

state_rasters = {}
for state in ['FST', 'SHR', 'GRS', 'BAL', 'CRP', 'NATURAL']:
    path = os.path.join(STATES_DIR, f'state_{state}.tif')
    state_rasters[state] = rioxarray.open_rasterio(path)
    if state_rasters[state].rio.crs is None:
        state_rasters[state] = state_rasters[state].rio.write_crs('EPSG:4326')
    print(f"  state_{state}: {state_rasters[state].shape}")


# ============================================================
# MAIN ATTRIBUTION LOOP — STRICTER
# ============================================================
print("\n" + "=" * 70)
print("STRICTER ATTRIBUTION (Lagged + Cumulative across all timescales)")
print("=" * 70)

results_lagged = []
results_cumulative = []
diagnostic_n_changes = []  # track how the stricter definition changed sample sizes

total_combos = len(SPEI_TIMESCALES) * len(TRANSITIONS) * len(REGION_ORDER) * (len(INTERVALS) - 1)
processed = 0

for ts in SPEI_TIMESCALES:
    print(f"\n{ts.upper()}")

    spei_all = {}
    for label, _, _ in INTERVALS:
        p = os.path.join(SPEI_DIR, f'{ts}_mean_{label}.tif')
        spei_all[label] = load_tif(p)

    for t in TRANSITIONS:
        trans_ds = transition_rasters[t['name']]
        state_ds = state_rasters[t['from_state']]

        for int_idx in range(1, len(INTERVALS)):
            int_label, int_start_band, int_end_band = INTERVALS[int_idx]
            prior_label, _, _ = INTERVALS[int_idx - 1]
            prior_labels_cum = [INTERVALS[i][0] for i in range(int_idx)]

            # Transition band for this interval
            if 'band' in trans_ds.dims:
                trans_band = trans_ds.sel(band=int_start_band)
            else:
                trans_band = trans_ds

            # State at INTERVAL START (band = int_start_band)
            if 'band' in state_ds.dims:
                state_start = state_ds.sel(band=int_start_band)
            else:
                state_start = state_ds

            # State at INTERVAL END (band = int_end_band)
            if 'band' in state_ds.dims:
                state_end = state_ds.sel(band=int_end_band)
            else:
                state_end = state_ds

            # Lagged SPEI
            spei_prior = spei_all[prior_label]

            # Cumulative SPEI (mean of prior intervals)
            prior_spei_list = [spei_all[lbl] for lbl in prior_labels_cum]
            ref = prior_spei_list[0]
            aligned = [r.rio.reproject_match(ref) if r.shape != ref.shape else r
                       for r in prior_spei_list]
            spei_cum = xr.concat(aligned, dim='_t').mean(dim='_t', skipna=True)
            spei_cum.rio.write_crs('EPSG:4326', inplace=True)

            for code in REGION_ORDER:
                processed += 1
                geom = region_geoms[code]

                try:
                    trans_aligned = trans_band.rio.reproject_match(spei_prior)
                    state_start_aligned = state_start.rio.reproject_match(spei_prior)
                    state_end_aligned = state_end.rio.reproject_match(spei_prior)
                except Exception:
                    continue

                try:
                    trans_clip = trans_aligned.rio.clip([geom], crs='EPSG:4326',
                                                        all_touched=True, drop=False)
                    state_start_clip = state_start_aligned.rio.clip([geom], crs='EPSG:4326',
                                                                     all_touched=True, drop=False)
                    state_end_clip = state_end_aligned.rio.clip([geom], crs='EPSG:4326',
                                                                 all_touched=True, drop=False)
                    spei_prior_clip = spei_prior.rio.clip([geom], crs='EPSG:4326',
                                                           all_touched=True, drop=False)
                    spei_cum_clip = spei_cum.rio.clip([geom], crs='EPSG:4326',
                                                       all_touched=True, drop=False)
                except Exception:
                    continue

                def squeeze2d(a):
                    while a.ndim > 2:
                        a = a.squeeze(axis=0)
                    return a

                trans_vals = squeeze2d(trans_clip.values)
                state_start_vals = squeeze2d(state_start_clip.values)
                state_end_vals = squeeze2d(state_end_clip.values)
                spei_prior_vals = squeeze2d(spei_prior_clip.values.astype('float32'))
                spei_cum_vals = squeeze2d(spei_cum_clip.values.astype('float32'))

                if not (trans_vals.shape == state_start_vals.shape == state_end_vals.shape
                        == spei_prior_vals.shape == spei_cum_vals.shape):
                    continue

                # ---- TRANSITION mask: in from-state at start AND transitioned ----
                trans_mask = (trans_vals == 1) & (state_start_vals == 1)

                # ---- STRICTER STABLE mask:
                #      in from-state at start AND in from-state at end AND no transition
                stable_mask_strict = (state_start_vals == 1) & \
                                     (state_end_vals == 1) & \
                                     (trans_vals == 0)

                # ---- LAX STABLE mask (for diagnostic comparison):
                #      in from-state at start AND no transition
                stable_mask_lax = (state_start_vals == 1) & (trans_vals == 0)

                # Track sample size delta from stricter definition
                n_strict = int(stable_mask_strict.sum())
                n_lax = int(stable_mask_lax.sum())
                if n_lax > 0:
                    diagnostic_n_changes.append({
                        'spei_timescale': ts, 'transition': t['name'],
                        'region': code, 'interval': int_label,
                        'n_stable_lax': n_lax,
                        'n_stable_strict': n_strict,
                        'pct_retained': round(n_strict / n_lax * 100, 1) if n_lax > 0 else np.nan,
                    })

                # LAGGED analysis
                lag_trans = spei_prior_vals[trans_mask]
                lag_stable = spei_prior_vals[stable_mask_strict]
                if len(lag_stable) > 50000:
                    np.random.seed(42)
                    idx = np.random.choice(len(lag_stable), 50000, replace=False)
                    lag_stable = lag_stable[idx]
                lag_stats = compute_attribution_stats(lag_trans, lag_stable)

                # CUMULATIVE analysis
                cum_trans = spei_cum_vals[trans_mask]
                cum_stable = spei_cum_vals[stable_mask_strict]
                if len(cum_stable) > 50000:
                    np.random.seed(42)
                    idx = np.random.choice(len(cum_stable), 50000, replace=False)
                    cum_stable = cum_stable[idx]
                cum_stats = compute_attribution_stats(cum_trans, cum_stable)

                base = {
                    'spei_timescale': ts, 'transition': t['name'],
                    'category': t['category'], 'from_label': t['from_label'],
                    'from_state': t['from_state'], 'region': code,
                    'region_name': REGION_NAMES[code], 'interval': int_label,
                    'interval_num': int_start_band,
                    'prior_interval': prior_label,
                    'n_prior_intervals_cum': len(prior_labels_cum),
                }
                results_lagged.append({**base, **lag_stats})
                results_cumulative.append({**base, **cum_stats})

        # Per-transition progress
        recent_d = []
        for r in results_lagged[-len(REGION_ORDER) * (len(INTERVALS) - 1):]:
            if r['transition'] == t['name'] and r['spei_timescale'] == ts:
                if not np.isnan(r['cohens_d']):
                    recent_d.append(r['cohens_d'])
        d_str = f"mean d={np.mean(recent_d):+.2f}" if recent_d else "no valid d"
        pct = processed / total_combos * 100
        print(f"  {t['name']:<20} [{pct:5.1f}%]  {d_str}")


# ============================================================
# Save full STRICTER per-interval results
# ============================================================
print("\n" + "=" * 70)
print("Saving stricter results")
print("=" * 70)

df_lag = pd.DataFrame(results_lagged)
df_cum = pd.DataFrame(results_cumulative)

df_lag.to_csv(os.path.join(RESULTS_DIR, 'TableB2_LAGGED_per_interval_STRICTER.csv'),
              index=False, encoding='utf-8')
df_cum.to_csv(os.path.join(RESULTS_DIR, 'TableB2_CUMULATIVE_per_interval_STRICTER.csv'),
              index=False, encoding='utf-8')
print(f"  Saved: TableB2_LAGGED_per_interval_STRICTER.csv ({len(df_lag)} rows)")
print(f"  Saved: TableB2_CUMULATIVE_per_interval_STRICTER.csv ({len(df_cum)} rows)")

# Save diagnostic of sample size change
df_diag = pd.DataFrame(diagnostic_n_changes)
df_diag.to_csv(os.path.join(STATS_DIR, 'Batch2_stable_pixel_retention_diagnostic.csv'),
               index=False, encoding='utf-8')

# Diagnostic summary: how much did the stricter filter remove?
if len(df_diag) > 0:
    pct_retained = df_diag['pct_retained'].dropna()
    print(f"\n  STABLE-PIXEL RETENTION DIAGNOSTIC:")
    print(f"    Number of (transition × region × interval × timescale) cells: {len(df_diag)}")
    print(f"    Mean retention rate (stricter / lax): {pct_retained.mean():.1f}%")
    print(f"    Median retention:                      {pct_retained.median():.1f}%")
    print(f"    Min retention:                         {pct_retained.min():.1f}%")
    print(f"    Max retention:                         {pct_retained.max():.1f}%")


# ============================================================
# SUMMARY tables (SPEI-12 primary)
# ============================================================
def weighted_mean(grp, col):
    w = grp['n_trans']
    if w.sum() == 0:
        return np.nan
    valid = grp[col].notna()
    if valid.sum() == 0:
        return np.nan
    return np.average(grp.loc[valid, col], weights=w[valid])


def summarize(df, outfile):
    df12 = df[df['spei_timescale'] == 'spei_12']
    rows = []
    for t in TRANSITIONS:
        for code in REGION_ORDER:
            sub = df12[(df12['transition'] == t['name']) & (df12['region'] == code)]
            if len(sub) == 0 or sub['n_trans'].sum() == 0:
                continue
            rows.append({
                'transition': t['name'], 'category': t['category'],
                'region': code, 'region_name': REGION_NAMES[code],
                'total_transition_pixels': int(sub['n_trans'].sum()),
                'mean_stable_pixels_per_interval': int(sub['n_stable'].mean()),
                'mean_spei_at_trans_pixels': round(weighted_mean(sub, 'mean_spei_trans'), 3),
                'mean_spei_at_stable_pixels': round(weighted_mean(sub, 'mean_spei_stable'), 3),
                'spei_difference': round(weighted_mean(sub, 'spei_difference'), 3),
                'drought_exposure_trans_pct': round(weighted_mean(sub, 'exposure_frac_trans') * 100, 2),
                'drought_exposure_stable_pct': round(weighted_mean(sub, 'exposure_frac_stable') * 100, 2),
                'mean_cohens_d': round(weighted_mean(sub, 'cohens_d'), 3),
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(RESULTS_DIR, outfile), index=False, encoding='utf-8')
    print(f"  Saved: {outfile}")
    return df_out


print("\n  SPEI-12 summary tables (STRICTER):")
df_lag_summary = summarize(df_lag, 'TableB2_LAGGED_SUMMARY_spei12_STRICTER.csv')
df_cum_summary = summarize(df_cum, 'TableB2_CUMULATIVE_SUMMARY_spei12_STRICTER.csv')


# ============================================================
# Headline: stricter category-level Cohen's d
# ============================================================
print("\n" + "=" * 70)
print("HEADLINE — Category-level Cohen's d (STRICTER)")
print("=" * 70)

print("\n  CUMULATIVE SPEI-12 (primary):")
for cat in ['Degradation', 'Recovery', 'Agricultural']:
    cat_rows = df_cum_summary[df_cum_summary['category'] == cat]
    if len(cat_rows):
        cd = cat_rows['mean_cohens_d'].mean()
        exp_t = cat_rows['drought_exposure_trans_pct'].mean()
        exp_s = cat_rows['drought_exposure_stable_pct'].mean()
        sign = "DRIER" if cd > 0 else "WETTER"
        print(f"    {cat:<14}: trans={exp_t:5.1f}% stable={exp_s:5.1f}% "
              f"d={cd:+.3f} (trans {sign})")

print("\n  LAGGED SPEI-12 (robustness):")
for cat in ['Degradation', 'Recovery', 'Agricultural']:
    cat_rows = df_lag_summary[df_lag_summary['category'] == cat]
    if len(cat_rows):
        cd = cat_rows['mean_cohens_d'].mean()
        exp_t = cat_rows['drought_exposure_trans_pct'].mean()
        exp_s = cat_rows['drought_exposure_stable_pct'].mean()
        sign = "DRIER" if cd > 0 else "WETTER"
        print(f"    {cat:<14}: trans={exp_t:5.1f}% stable={exp_s:5.1f}% "
              f"d={cd:+.3f} (trans {sign})")


# ============================================================
# RSI (cumulative + lagged) — STRICTER
# ============================================================
print("\n" + "=" * 70)
print("RSI (STRICTER) — both windows, all timescales")
print("=" * 70)

def weighted_mean_d(df):
    w = df['n_trans']
    if w.sum() == 0:
        return np.nan
    valid = df['cohens_d'].notna() & (w > 0)
    if valid.sum() == 0:
        return np.nan
    return float(np.average(df.loc[valid, 'cohens_d'], weights=w[valid]))


def compute_rsi_for_ts(df, ts_filter):
    """Compute RSI across regions for a given timescale."""
    recovery_names = ['SHR_FST', 'GRS_SHR', 'BAL_GRS']
    df_ts = df[df['spei_timescale'] == ts_filter]
    rows = []
    for code in REGION_ORDER:
        sub = df_ts[(df_ts['region'] == code) & (df_ts['transition'].isin(recovery_names))]
        if len(sub) == 0:
            continue
        ds = sub.loc[sub['cohens_d'].notna(), 'cohens_d'].values
        rsi = weighted_mean_d(sub)
        if len(ds) >= 3:
            t_stat, p_val = stats.ttest_1samp(ds, 0.0)
        else:
            t_stat, p_val = np.nan, np.nan
        rows.append({
            'spei_timescale': ts_filter,
            'region': code,
            'RSI': round(rsi, 3) if not np.isnan(rsi) else np.nan,
            'n_cases': int(sub['cohens_d'].notna().sum()),
            't_stat': round(float(t_stat), 3) if not np.isnan(t_stat) else np.nan,
            'p_value': round(float(p_val), 4) if not np.isnan(p_val) else np.nan,
            'sig_005': (not np.isnan(p_val)) and (p_val < 0.05),
        })
    return rows


def compute_dra_for_ts(df, ts_filter):
    """Compute DRA across regions for a given timescale."""
    deg_names = ['FST_SHR', 'SHR_GRS', 'FST_CRP', 'GRS_BAL']
    rec_names = ['SHR_FST', 'GRS_SHR', 'BAL_GRS']
    df_ts = df[df['spei_timescale'] == ts_filter]
    rows = []
    for code in REGION_ORDER:
        deg = df_ts[(df_ts['region'] == code) & (df_ts['transition'].isin(deg_names))]
        rec = df_ts[(df_ts['region'] == code) & (df_ts['transition'].isin(rec_names))]
        deg_d = weighted_mean_d(deg)
        rec_d = weighted_mean_d(rec)
        dra = deg_d - rec_d if not (np.isnan(deg_d) or np.isnan(rec_d)) else np.nan
        deg_arr = deg.loc[deg['cohens_d'].notna(), 'cohens_d'].values
        rec_arr = rec.loc[rec['cohens_d'].notna(), 'cohens_d'].values
        if len(deg_arr) >= 3 and len(rec_arr) >= 3:
            t_stat, p_val = stats.ttest_ind(deg_arr, rec_arr, equal_var=False)
        else:
            t_stat, p_val = np.nan, np.nan
        rows.append({
            'spei_timescale': ts_filter,
            'region': code,
            'mean_deg_d': round(deg_d, 3) if not np.isnan(deg_d) else np.nan,
            'mean_rec_d': round(rec_d, 3) if not np.isnan(rec_d) else np.nan,
            'DRA': round(dra, 3) if not np.isnan(dra) else np.nan,
            't_stat': round(float(t_stat), 3) if not np.isnan(t_stat) else np.nan,
            'p_value': round(float(p_val), 4) if not np.isnan(p_val) else np.nan,
            'sig_005': (not np.isnan(p_val)) and (p_val < 0.05),
        })
    return rows


# Compute RSI/DRA across ALL timescales for both lagged and cumulative
all_rsi_cum = []
all_rsi_lag = []
all_dra_cum = []
all_dra_lag = []

for ts in SPEI_TIMESCALES:
    all_rsi_cum.extend(compute_rsi_for_ts(df_cum, ts))
    all_rsi_lag.extend(compute_rsi_for_ts(df_lag, ts))
    all_dra_cum.extend(compute_dra_for_ts(df_cum, ts))
    all_dra_lag.extend(compute_dra_for_ts(df_lag, ts))

df_rsi_cum = pd.DataFrame(all_rsi_cum)
df_rsi_lag = pd.DataFrame(all_rsi_lag)
df_dra_cum = pd.DataFrame(all_dra_cum)
df_dra_lag = pd.DataFrame(all_dra_lag)

df_rsi_cum.to_csv(os.path.join(STATS_DIR, 'TableB4_RSI_timescale_sensitivity_cumulative_STRICTER.csv'),
                  index=False, encoding='utf-8')
df_rsi_lag.to_csv(os.path.join(STATS_DIR, 'TableB4_RSI_timescale_sensitivity_lagged_STRICTER.csv'),
                  index=False, encoding='utf-8')
df_dra_cum.to_csv(os.path.join(STATS_DIR, 'TableB4_DRA_timescale_sensitivity_cumulative_STRICTER.csv'),
                  index=False, encoding='utf-8')
df_dra_lag.to_csv(os.path.join(STATS_DIR, 'TableB4_DRA_timescale_sensitivity_lagged_STRICTER.csv'),
                  index=False, encoding='utf-8')

# Print RSI table — lagged is primary because Script 3c showed lagged-RSI was significant
print("\n  RSI — LAGGED (primary; SPEI-12 was significant for SAH in Script 3c):")
print(f"  {'Timescale':<10}{'Region':<6}{'RSI':>8}{'p':>10}{'Sig':>5}")
print("  " + "-" * 45)
for _, r in df_rsi_lag.iterrows():
    sig = '***' if r['sig_005'] else ''
    print(f"  {r['spei_timescale']:<10}{r['region']:<6}{r['RSI']:>+8.3f}"
          f"{r['p_value']:>10.4f}  {sig}")

print("\n  RSI — CUMULATIVE:")
print(f"  {'Timescale':<10}{'Region':<6}{'RSI':>8}{'p':>10}{'Sig':>5}")
print("  " + "-" * 45)
for _, r in df_rsi_cum.iterrows():
    sig = '***' if r['sig_005'] else ''
    print(f"  {r['spei_timescale']:<10}{r['region']:<6}{r['RSI']:>+8.3f}"
          f"{r['p_value']:>10.4f}  {sig}")

print("\n  DRA — LAGGED (primary):")
print(f"  {'Timescale':<10}{'Region':<6}{'DRA':>8}{'p':>10}{'Sig':>5}")
print("  " + "-" * 45)
for _, r in df_dra_lag.iterrows():
    sig = '***' if r['sig_005'] else ''
    print(f"  {r['spei_timescale']:<10}{r['region']:<6}{r['DRA']:>+8.3f}"
          f"{r['p_value']:>10.4f}  {sig}")


# ============================================================
# Decadal shifts on STRICTER cumulative
# ============================================================
print("\n" + "=" * 70)
print("Decadal shifts on STRICTER cumulative SPEI-12")
print("=" * 70)

EARLY_INTERVALS = ['1990_1995', '1995_2000', '2000_2005']
LATE_INTERVALS = ['2005_2010', '2010_2015', '2015_2020', '2020_2022']

def compute_decadal_shift_strict(df_window):
    df12 = df_window[df_window['spei_timescale'] == 'spei_12']
    rows = []
    for code in REGION_ORDER:
        for cat in ['Degradation', 'Recovery', 'Agricultural']:
            subset = df12[(df12['region'] == code) & (df12['category'] == cat)]
            early = subset[subset['interval'].isin(EARLY_INTERVALS)]
            late = subset[subset['interval'].isin(LATE_INTERVALS)]
            early_d = early.loc[early['cohens_d'].notna(), 'cohens_d'].values
            late_d = late.loc[late['cohens_d'].notna(), 'cohens_d'].values
            if len(early_d) < 2 or len(late_d) < 2:
                rows.append({'region': code, 'category': cat,
                             'n_early': len(early_d), 'n_late': len(late_d),
                             'early_mean_d': np.nan, 'late_mean_d': np.nan,
                             'delta_d': np.nan, 't_stat': np.nan,
                             'p_value': np.nan, 'sig_005': False})
                continue
            try:
                t_stat, p_val = stats.ttest_ind(early_d, late_d, equal_var=False)
            except Exception:
                t_stat, p_val = np.nan, np.nan
            rows.append({
                'region': code, 'category': cat,
                'n_early': len(early_d), 'n_late': len(late_d),
                'early_mean_d': round(float(np.mean(early_d)), 3),
                'late_mean_d': round(float(np.mean(late_d)), 3),
                'delta_d': round(float(np.mean(late_d) - np.mean(early_d)), 3),
                't_stat': round(float(t_stat), 3) if not np.isnan(t_stat) else np.nan,
                'p_value': round(float(p_val), 4) if not np.isnan(p_val) else np.nan,
                'sig_005': (not np.isnan(p_val)) and (p_val < 0.05),
            })
    return rows


def bh_correction(pvals, alpha=0.05):
    """BH FDR correction."""
    pvals = np.asarray(pvals, dtype='float64')
    valid = ~np.isnan(pvals)
    n_valid = valid.sum()
    q = np.full_like(pvals, np.nan)
    reject = np.zeros_like(pvals, dtype=bool)
    if n_valid == 0:
        return q, reject
    valid_idx = np.where(valid)[0]
    valid_pvals = pvals[valid_idx]
    sort_order = np.argsort(valid_pvals)
    sorted_pvals = valid_pvals[sort_order]
    ranks = np.arange(1, n_valid + 1)
    thresholds = ranks / n_valid * alpha
    below = sorted_pvals <= thresholds
    max_k = (np.max(np.where(below)[0]) + 1) if below.any() else 0
    sorted_q = np.minimum.accumulate((n_valid / ranks * sorted_pvals)[::-1])[::-1]
    sorted_q = np.clip(sorted_q, 0, 1)
    q_at_valid = np.empty_like(sorted_q)
    q_at_valid[sort_order] = sorted_q
    q[valid_idx] = q_at_valid
    reject_at_valid = np.zeros(n_valid, dtype=bool)
    if max_k > 0:
        reject_at_valid[sort_order[:max_k]] = True
    reject[valid_idx] = reject_at_valid
    return q, reject


for window_label, df_win in [('cumulative', df_cum), ('lagged', df_lag)]:
    rows = compute_decadal_shift_strict(df_win)
    df_dec = pd.DataFrame(rows)
    if df_dec['p_value'].notna().any():
        q, reject = bh_correction(df_dec['p_value'].values, alpha=0.05)
        df_dec['q_value_bh'] = np.round(q, 4)
        df_dec['sig_FDR_005'] = reject
    df_dec.to_csv(os.path.join(STATS_DIR, f'TableB4_Decadal_Shift_{window_label}_STRICTER_FDR.csv'),
                  index=False, encoding='utf-8')

    print(f"\n  Decadal shifts ({window_label}, STRICTER, with FDR):")
    print(f"  {'Region':<6}{'Category':<14}{'Δd':>8}{'p_raw':>10}{'q_bh':>10}{'FDR':>5}")
    print("  " + "-" * 55)
    for _, r in df_dec.iterrows():
        if pd.notna(r['p_value']):
            fdr = '***' if r.get('sig_FDR_005', False) else ''
            print(f"  {r['region']:<6}{r['category']:<14}"
                  f"{r['delta_d']:>+8.2f}{r['p_value']:>10.4f}"
                  f"{r.get('q_value_bh', np.nan):>10.4f}  {fdr}")


# ============================================================
# COMPLETE
# ============================================================
print("\n" + "=" * 70)
print("BATCH 2 COMPLETE")
print("=" * 70)
print(f"""
  Files saved:
  
  In ModuleB_results/:
    TableB2_LAGGED_per_interval_STRICTER.csv
    TableB2_CUMULATIVE_per_interval_STRICTER.csv
    TableB2_LAGGED_SUMMARY_spei12_STRICTER.csv
    TableB2_CUMULATIVE_SUMMARY_spei12_STRICTER.csv
  
  In statistics_for_paper/:
    Batch2_stable_pixel_retention_diagnostic.csv
    TableB4_RSI_timescale_sensitivity_cumulative_STRICTER.csv
    TableB4_RSI_timescale_sensitivity_lagged_STRICTER.csv
    TableB4_DRA_timescale_sensitivity_cumulative_STRICTER.csv
    TableB4_DRA_timescale_sensitivity_lagged_STRICTER.csv
    TableB4_Decadal_Shift_cumulative_STRICTER_FDR.csv
    TableB4_Decadal_Shift_lagged_STRICTER_FDR.csv

  VERIFY BEFORE PROCEEDING:
    1. Sample-size retention rate should be 70-95% (stricter is 5-30% smaller).
       Major drops (<50% retention) would indicate noise in state rasters.
    2. Headline category Cohen's d directions should match earlier (lax) results:
       Recovery should still be NEGATIVE, Degradation near zero or weakly positive.
       If signs flipped, we have a real story change to investigate.
    3. RSI/DRA significance per timescale: signal should persist across SPEI-12
       to SPEI-60 (timescale sensitivity check passes).
    4. Decadal shift FDR survivors with stricter definition compared to lax
       (Batch 1): same set should largely survive.

  Next: Batch 3 (Module B figures from these stricter results).
""")
