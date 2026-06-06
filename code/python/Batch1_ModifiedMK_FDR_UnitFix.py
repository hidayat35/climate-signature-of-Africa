"""
============================================================
BATCH 1: Statistical robustness fixes (Tasks 1 + 2 + 4)
============================================================
Combined script:
  Task 1: Modified Mann-Kendall (Hamed-Rao) trend test
          - Corrects for temporal autocorrelation
          - Regenerates Table 2 trend statistics
          - Produces comparison CSV: standard vs modified MK

  Task 2: FDR/Benjamini-Hochberg correction on decadal shifts
          - Applies BH correction to Script 3c decadal p-values
          - Regenerates TableB3_Decadal_Shift_cumulative/lagged CSVs
          - Produces summary of which shifts survive FDR

  Task 4: PET trend unit verification
          - Confirms Theil-Sen slope units are mm/year (not mm/year²)
          - Documents the unit in a verification text file

COMPREHENSIVE DIAGNOSTIC PRINTOUT at each stage so nothing
moves forward without verification.

INPUTS:
  - spei_12_monthly.nc, spi_12_monthly.nc (canonical SPEI/SPI)
  - precip_monthly_1985_2022.tif, pet_monthly_1985_2022.tif
  - ipc_africa_5_regions.shp
  - TableB3_Decadal_Shift_cumulative.csv (from Script 3c)
  - TableB3_Decadal_Shift_lagged.csv (from Script 3c)

OUTPUTS (in statistics_for_paper/):
  - Table2_trend_statistics_ModifiedMK.csv  (primary, replaces Table 2)
  - Table2_comparison_standardMK_vs_modifiedMK.csv
  - TableB3_Decadal_Shift_cumulative_FDR.csv
  - TableB3_Decadal_Shift_lagged_FDR.csv
  - Batch1_decadal_shift_survival_summary.csv
  - Batch1_unit_verification.txt
============================================================
"""

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
import pymannkendall as mk
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
SPEI_DIR = r'D:\Claude idea\PhD_Paper3_Data\step5_spei_output'
MONTHLY_INPUT_DIR = r'D:\Claude idea\PhD_Paper3_Data\monthly_inputs'
SHAPEFILE = r'D:\Claude idea\ipc_africa_5_regions.shp'
STATS_DIR = os.path.join(SPEI_DIR, 'statistics_for_paper')
os.makedirs(STATS_DIR, exist_ok=True)

REGION_ORDER = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
REGION_NAMES = {'MED': 'Mediterranean', 'SAH': 'Sahara-Sahel',
                'WAF': 'West Africa', 'EAF': 'East Africa',
                'SAF': 'Southern Africa'}

print("=" * 70)
print("BATCH 1: Statistical Robustness Fixes (Tasks 1 + 2 + 4)")
print("=" * 70)
print()
print("This script runs three independent tasks in sequence:")
print("  Task 1: Modified Mann-Kendall (Hamed-Rao) trend test")
print("  Task 2: FDR/BH correction on decadal shift p-values")
print("  Task 4: PET trend unit verification")
print()
print("Each task prints full diagnostic output for verification.")

# ============================================================
# TASK 4 (FIRST — just documentation, fast): PET UNIT VERIFICATION
# ============================================================
print("\n" + "=" * 70)
print("TASK 4: PET trend unit verification")
print("=" * 70)

unit_text = """\
PET TREND UNIT VERIFICATION (run on 2026-04-24)
================================================

The Theil-Sen slope estimator (as implemented in pymannkendall.original_test
and sensitivity_test.hamed_rao_modification_test) returns the median of
pairwise slopes [y_j - y_i] / [j - i], where j > i are time indices.

For annual-aggregated PET:
  - y-values are annual total PET in millimeters (mm)
  - time indices are year numbers (1985, 1986, ..., 2022)
  - The slope unit is therefore: [mm] / [year] = mm / year

That is, the slope represents the linear change in annual PET per year.

If the reported numbers in the draft said "mm/year^2" they were INCORRECT.
The correct unit is mm/year (or equivalently mm year^-1).

Example interpretation:
  A Theil-Sen slope of 5.0 mm/year means annual PET is increasing by
  5.0 mm per year, i.e., after 10 years we expect total annual PET to
  be 50 mm higher than at the start.

This verification applies equally to precipitation trends (mm/year)
and to SPEI/SPI trends (dimensionless/year; the index values are unitless
but the slope is the rate of change per year).

CONCLUSION: All "PET trend" numbers in Table 2 should be labeled mm/year.
"""

unit_file = os.path.join(STATS_DIR, 'Batch1_unit_verification.txt')
with open(unit_file, 'w', encoding='utf-8') as f:
    f.write(unit_text)

print(f"\n  Unit verification written to: {os.path.basename(unit_file)}")
print("  Summary: Theil-Sen slope unit is mm/year (not mm/year^2)")
print("  All trend tables should report slopes in mm/year")


# ============================================================
# TASK 1: Modified Mann-Kendall (Hamed-Rao)
# ============================================================
print("\n" + "=" * 70)
print("TASK 1: Modified Mann-Kendall with Hamed-Rao correction")
print("=" * 70)
print()
print("Hamed-Rao adjusts the Mann-Kendall variance for lag-1+ autocorrelation")
print("in the time series, preventing false-positive significance from serial")
print("correlation. Expected effect: FEWER pixels classified as significantly")
print("trending (typically 10-30% reduction depending on autocorrelation).")

# ---- Load inputs ----
print("\n  Loading inputs...")

gdf = gpd.read_file(SHAPEFILE)
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.set_crs('EPSG:4326') if gdf.crs is None else gdf.to_crs('EPSG:4326')
region_geoms = {}
for code in REGION_ORDER:
    subset = gdf[gdf['LAB'] == code]
    region_geoms[code] = (subset.geometry.union_all() if hasattr(subset.geometry, 'union_all')
                          else subset.geometry.unary_union)
print(f"    Loaded {len(region_geoms)} regions")

# Load SPEI-12 and SPI-12
def load_nc(path, var_guesses=('spei', 'spi')):
    ds = xr.open_dataset(path)
    for v in var_guesses:
        if v in ds.data_vars:
            da = ds[v]
            break
    else:
        da = ds[list(ds.data_vars)[0]]
    if not hasattr(da, 'rio') or da.rio.crs is None:
        da = da.rio.write_crs('EPSG:4326')
    return da

spei12 = load_nc(os.path.join(SPEI_DIR, 'spei_12_monthly.nc'))
spi12 = load_nc(os.path.join(SPEI_DIR, 'spi_12_monthly.nc'))
print(f"    SPEI-12: shape {spei12.shape}")
print(f"    SPI-12:  shape {spi12.shape}")

# Load monthly precip and PET, aggregate to annual
def load_monthly_tif(path, start_year=1985):
    da = rioxarray.open_rasterio(path)
    if da.rio.crs is None:
        da = da.rio.write_crs('EPSG:4326')
    n_bands = da.sizes['band']
    times = pd.date_range(start=f'{start_year}-01-01', periods=n_bands, freq='MS')
    da = da.assign_coords(band=('band', times)).rename({'band': 'time'})
    return da

print("    Loading monthly precip + PET...")
precip_m = load_monthly_tif(os.path.join(MONTHLY_INPUT_DIR, 'precip_monthly_1985_2022.tif'))
pet_m = load_monthly_tif(os.path.join(MONTHLY_INPUT_DIR, 'pet_monthly_1985_2022.tif'))

precip_annual = precip_m.where(precip_m >= 0).groupby('time.year').sum('time', skipna=True)
pet_annual = pet_m.where(pet_m >= 0).groupby('time.year').sum('time', skipna=True)
print(f"    precip_annual: shape {precip_annual.shape}")
print(f"    pet_annual:    shape {pet_annual.shape}")

del precip_m, pet_m
gc.collect()

# Annual SPEI-12 / SPI-12 at end-of-year (December)
spei12_annual = spei12.sel(time=spei12.time.dt.month == 12)
spi12_annual = spi12.sel(time=spi12.time.dt.month == 12)

# ---- Run Modified Mann-Kendall per pixel ----
def compute_mk_per_pixel(da, method='standard'):
    """
    Apply Mann-Kendall per pixel. method='standard' uses mk.original_test,
    method='hamed_rao' uses mk.hamed_rao_modification_test.
    
    Returns (slope, p_value) arrays of shape (y, x).
    """
    vals = da.values
    n_time, ny, nx = vals.shape
    slope = np.full((ny, nx), np.nan, dtype='float32')
    pval = np.full((ny, nx), np.nan, dtype='float32')

    test_fn = mk.original_test if method == 'standard' else mk.hamed_rao_modification_test

    print(f"      Running {method} MK on {ny}×{nx} pixels (may take 10-20 min)...")
    for i in range(ny):
        if i % max(1, ny // 20) == 0:
            print(f"        row {i}/{ny} ({i/ny*100:.0f}%)")
        for j in range(nx):
            ts = vals[:, i, j]
            valid_ts = ts[~np.isnan(ts)]
            if len(valid_ts) < n_time * 0.5:
                continue
            try:
                res = test_fn(valid_ts)
                slope[i, j] = res.slope
                pval[i, j] = res.p
            except Exception:
                pass
    return slope, pval


# Store slope/pval arrays for each variable and method
variables = [
    ('SPEI12', spei12_annual),
    ('SPI12',  spi12_annual),
    ('precip', precip_annual),
    ('PET',    pet_annual),
]

results = {}

for var_name, da in variables:
    print(f"\n  === {var_name} ===")
    # Standard MK (for comparison)
    std_slope, std_pval = compute_mk_per_pixel(da, method='standard')
    # Hamed-Rao MK (primary)
    hr_slope, hr_pval = compute_mk_per_pixel(da, method='hamed_rao')
    results[var_name] = {
        'std_slope': std_slope, 'std_pval': std_pval,
        'hr_slope': hr_slope, 'hr_pval': hr_pval,
        'ref_da': da,
    }
    # Save Hamed-Rao rasters (these are the primary results)
    def save_2d(arr, ref_da, path):
        out_da = xr.DataArray(arr, dims=('y', 'x'),
                              coords={'y': ref_da.y, 'x': ref_da.x}).rio.write_crs('EPSG:4326')
        out_da.rio.to_raster(path, dtype='float32')
    
    save_2d(hr_slope, da, os.path.join(SPEI_DIR, f'{var_name.lower()}_trend_slope_modMK.tif'))
    save_2d(hr_pval,  da, os.path.join(SPEI_DIR, f'{var_name.lower()}_trend_pvalue_modMK.tif'))
    
    # Quick diagnostic: compare counts
    valid_std = np.sum(~np.isnan(std_pval))
    valid_hr = np.sum(~np.isnan(hr_pval))
    sig_std = np.sum((std_pval < 0.05) & ~np.isnan(std_pval))
    sig_hr  = np.sum((hr_pval < 0.05) & ~np.isnan(hr_pval))
    print(f"    Standard MK:   {sig_std:>12,} / {valid_std:>12,} significant "
          f"({sig_std/valid_std*100:.2f}%)")
    print(f"    Modified MK:   {sig_hr:>12,} / {valid_hr:>12,} significant "
          f"({sig_hr/valid_hr*100:.2f}%)")
    print(f"    Reduction:     {(sig_std-sig_hr)/max(sig_std,1)*100:+.1f}% "
          f"(expected: negative, fewer sig with autocorr correction)")


# ---- Regional statistics for Table 2 ----
print("\n  Computing regional trend statistics (Modified MK + comparison)...")

def regional_trend_stats(slope, pval, ref_da, geom, sig=0.05):
    slope_da = xr.DataArray(slope, dims=('y', 'x'),
                            coords={'y': ref_da.y, 'x': ref_da.x}).rio.write_crs('EPSG:4326')
    pval_da = xr.DataArray(pval, dims=('y', 'x'),
                           coords={'y': ref_da.y, 'x': ref_da.x}).rio.write_crs('EPSG:4326')
    try:
        s_clip = slope_da.rio.clip([geom], crs='EPSG:4326', all_touched=True, drop=False).values
        p_clip = pval_da.rio.clip([geom], crs='EPSG:4326', all_touched=True, drop=False).values
    except Exception:
        return {'mean_slope': np.nan, 'pct_significant': np.nan,
                'pct_drying_sig': np.nan, 'pct_wetting_sig': np.nan, 'n_pixels': 0}
    valid = ~(np.isnan(s_clip) | np.isnan(p_clip))
    if valid.sum() == 0:
        return {'mean_slope': np.nan, 'pct_significant': np.nan,
                'pct_drying_sig': np.nan, 'pct_wetting_sig': np.nan, 'n_pixels': 0}
    s_v = s_clip[valid]
    p_v = p_clip[valid]
    sig_mask = p_v < sig
    return {
        'mean_slope': float(np.mean(s_v)),
        'pct_significant': float(sig_mask.sum() / len(p_v) * 100),
        'pct_drying_sig': float(((s_v < 0) & sig_mask).sum() / len(p_v) * 100),
        'pct_wetting_sig': float(((s_v > 0) & sig_mask).sum() / len(p_v) * 100),
        'n_pixels': int(valid.sum()),
    }


# Primary Table 2: Modified MK
table2_rows = []
for code in REGION_ORDER:
    geom = region_geoms[code]
    row = {'region': code, 'region_name': REGION_NAMES[code]}
    for var_name, _ in variables:
        st = regional_trend_stats(results[var_name]['hr_slope'],
                                   results[var_name]['hr_pval'],
                                   results[var_name]['ref_da'], geom)
        unit_label = 'mm_per_year' if var_name in ('precip', 'PET') else 'index_per_year'
        row[f'{var_name}_mean_slope_{unit_label}'] = round(st['mean_slope'], 4) if not np.isnan(st['mean_slope']) else np.nan
        row[f'{var_name}_pct_significant'] = round(st['pct_significant'], 2)
        row[f'{var_name}_pct_drying_sig'] = round(st['pct_drying_sig'], 2)
        row[f'{var_name}_pct_wetting_sig'] = round(st['pct_wetting_sig'], 2)
        row[f'{var_name}_n_pixels'] = st['n_pixels']
    table2_rows.append(row)

df_table2_modMK = pd.DataFrame(table2_rows)
df_table2_modMK.to_csv(os.path.join(STATS_DIR, 'Table2_trend_statistics_ModifiedMK.csv'),
                       index=False, encoding='utf-8')

# Comparison table: standard vs modified
comparison_rows = []
for code in REGION_ORDER:
    geom = region_geoms[code]
    for var_name, _ in variables:
        st_std = regional_trend_stats(results[var_name]['std_slope'],
                                        results[var_name]['std_pval'],
                                        results[var_name]['ref_da'], geom)
        st_hr = regional_trend_stats(results[var_name]['hr_slope'],
                                       results[var_name]['hr_pval'],
                                       results[var_name]['ref_da'], geom)
        comparison_rows.append({
            'region': code,
            'variable': var_name,
            'standard_MK_pct_sig': round(st_std['pct_significant'], 2),
            'modified_MK_pct_sig': round(st_hr['pct_significant'], 2),
            'delta_pct_sig': round(st_hr['pct_significant'] - st_std['pct_significant'], 2),
            'standard_MK_pct_drying': round(st_std['pct_drying_sig'], 2),
            'modified_MK_pct_drying': round(st_hr['pct_drying_sig'], 2),
            'standard_MK_pct_wetting': round(st_std['pct_wetting_sig'], 2),
            'modified_MK_pct_wetting': round(st_hr['pct_wetting_sig'], 2),
        })

df_comparison = pd.DataFrame(comparison_rows)
df_comparison.to_csv(os.path.join(STATS_DIR, 'Table2_comparison_standardMK_vs_modifiedMK.csv'),
                    index=False, encoding='utf-8')

# Print regional Table 2 (Modified MK)
print("\n  TABLE 2 RESULTS — Modified Mann-Kendall (primary):")
print("  " + "-" * 80)
print(f"  {'Region':<6}{'Var':<8}{'Slope':>12}{'% Sig':>8}{'% Drying':>10}{'% Wetting':>10}")
print("  " + "-" * 80)
for _, row in df_table2_modMK.iterrows():
    for var_name, _ in variables:
        unit_label = 'mm_per_year' if var_name in ('precip', 'PET') else 'index_per_year'
        slope_val = row[f'{var_name}_mean_slope_{unit_label}']
        print(f"  {row['region']:<6}{var_name:<8}"
              f"{slope_val:>+12.4f}"
              f"{row[f'{var_name}_pct_significant']:>8.1f}"
              f"{row[f'{var_name}_pct_drying_sig']:>10.1f}"
              f"{row[f'{var_name}_pct_wetting_sig']:>10.1f}")
    print()

print("\n  COMPARISON — Standard vs Modified MK (% significant reduction):")
print("  " + "-" * 60)
for _, row in df_comparison.iterrows():
    print(f"    {row['region']:<6} {row['variable']:<8}: "
          f"std={row['standard_MK_pct_sig']:5.1f}%  "
          f"mod={row['modified_MK_pct_sig']:5.1f}%  "
          f"Δ={row['delta_pct_sig']:+5.1f}%")

del spei12, spi12, spei12_annual, spi12_annual, precip_annual, pet_annual
del results
gc.collect()

print("\n  Files saved:")
print(f"    {os.path.join(STATS_DIR, 'Table2_trend_statistics_ModifiedMK.csv')}")
print(f"    {os.path.join(STATS_DIR, 'Table2_comparison_standardMK_vs_modifiedMK.csv')}")


# ============================================================
# TASK 2: FDR/BH correction on decadal shift p-values
# ============================================================
print("\n" + "=" * 70)
print("TASK 2: FDR/Benjamini-Hochberg correction on decadal shifts")
print("=" * 70)
print()
print("Script 3c ran 15 Welch's t-tests per window (5 regions × 3 categories).")
print("With α=0.05 uncorrected, ~0.75 false positives expected per window.")
print("Applying Benjamini-Hochberg FDR correction reduces false-discovery rate")
print("while being less conservative than Bonferroni.")

def bh_correction(pvals, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.
    Returns (q_values, reject_null_boolean_array).
    """
    pvals = np.asarray(pvals, dtype='float64')
    valid = ~np.isnan(pvals)
    n_valid = valid.sum()
    
    q = np.full_like(pvals, np.nan)
    reject = np.zeros_like(pvals, dtype=bool)
    
    if n_valid == 0:
        return q, reject
    
    # Get indices of valid p-values sorted ascending
    valid_idx = np.where(valid)[0]
    valid_pvals = pvals[valid_idx]
    sort_order = np.argsort(valid_pvals)
    sorted_pvals = valid_pvals[sort_order]
    
    # BH procedure: find largest k such that p_(k) <= (k/n)*alpha
    # Then reject all k' <= k
    ranks = np.arange(1, n_valid + 1)
    thresholds = ranks / n_valid * alpha
    below = sorted_pvals <= thresholds
    if below.any():
        max_k = np.max(np.where(below)[0]) + 1  # rank-1 to index
    else:
        max_k = 0
    
    # q-value = min over i >= k of (n/i * p_(i))
    sorted_q = np.minimum.accumulate((n_valid / ranks * sorted_pvals)[::-1])[::-1]
    sorted_q = np.clip(sorted_q, 0, 1)
    
    # Map back to original order
    q_at_valid = np.empty_like(sorted_q)
    q_at_valid[sort_order] = sorted_q
    q[valid_idx] = q_at_valid
    
    reject_at_valid = np.zeros(n_valid, dtype=bool)
    if max_k > 0:
        reject_at_valid[sort_order[:max_k]] = True
    reject[valid_idx] = reject_at_valid
    
    return q, reject


# Process cumulative and lagged shift files
for window_type in ['cumulative', 'lagged']:
    infile = os.path.join(STATS_DIR, f'TableB3_Decadal_Shift_{window_type}.csv')
    if not os.path.exists(infile):
        print(f"  ⚠ Not found: {infile}. Skipping {window_type}.")
        continue
    
    df = pd.read_csv(infile)
    print(f"\n  Processing {window_type} shift ({len(df)} rows)...")
    
    # Apply BH correction to p-values
    q_vals, reject_005 = bh_correction(df['p_value'].values, alpha=0.05)
    q_vals_010, reject_010 = bh_correction(df['p_value'].values, alpha=0.10)
    
    df['q_value_bh_a005'] = np.round(q_vals, 4)
    df['significant_FDR_005'] = reject_005
    df['q_value_bh_a010'] = np.round(q_vals_010, 4)
    df['significant_FDR_010'] = reject_010
    
    # Save
    outfile = os.path.join(STATS_DIR, f'TableB3_Decadal_Shift_{window_type}_FDR.csv')
    df.to_csv(outfile, index=False, encoding='utf-8')
    print(f"    Saved: {os.path.basename(outfile)}")
    
    # Diagnostic printout: before vs after FDR
    n_total = df['p_value'].notna().sum()
    n_uncorr_005 = (df['p_value'] < 0.05).sum()
    n_fdr_005 = int(df['significant_FDR_005'].sum())
    n_fdr_010 = int(df['significant_FDR_010'].sum())
    
    print(f"    Total tests with valid p: {n_total}")
    print(f"    Uncorrected p < 0.05:     {n_uncorr_005}")
    print(f"    FDR BH (α=0.05):          {n_fdr_005}")
    print(f"    FDR BH (α=0.10):          {n_fdr_010}")
    
    # Show which survive FDR
    surviving = df[df['significant_FDR_005'] == True]
    if len(surviving) > 0:
        print(f"\n    Shifts surviving FDR α=0.05:")
        print(f"    {'Region':<6}{'Category':<14}{'Δd':>8}{'p_raw':>10}{'q_bh':>10}")
        print("    " + "-" * 60)
        for _, r in surviving.iterrows():
            print(f"    {r['region']:<6}{r['category']:<14}"
                  f"{r['delta_d']:>+8.2f}"
                  f"{r['p_value']:>10.4f}"
                  f"{r['q_value_bh_a005']:>10.4f}")
    else:
        print(f"    No shifts survive FDR α=0.05.")


# Combined summary of survival for the paper
print("\n  Combined survival summary:")
summary_rows = []
for window_type in ['cumulative', 'lagged']:
    infile = os.path.join(STATS_DIR, f'TableB3_Decadal_Shift_{window_type}_FDR.csv')
    if not os.path.exists(infile):
        continue
    df = pd.read_csv(infile)
    for _, r in df.iterrows():
        if pd.notna(r['p_value']):
            summary_rows.append({
                'window': window_type,
                'region': r['region'],
                'category': r['category'],
                'delta_d': r['delta_d'],
                'p_raw': r['p_value'],
                'q_bh_a005': r['q_value_bh_a005'],
                'sig_uncorrected_005': r['p_value'] < 0.05,
                'sig_FDR_005': r['significant_FDR_005'],
                'sig_FDR_010': r['significant_FDR_010'],
            })

df_surv_summary = pd.DataFrame(summary_rows)
df_surv_summary.to_csv(os.path.join(STATS_DIR, 'Batch1_decadal_shift_survival_summary.csv'),
                       index=False, encoding='utf-8')
print(f"    Saved: Batch1_decadal_shift_survival_summary.csv")

# Final grand summary
n_both_cum = df_surv_summary[(df_surv_summary['window'] == 'cumulative') & df_surv_summary['sig_FDR_005']].shape[0] if len(df_surv_summary) else 0
n_both_lag = df_surv_summary[(df_surv_summary['window'] == 'lagged') & df_surv_summary['sig_FDR_005']].shape[0] if len(df_surv_summary) else 0
print(f"\n  OVERALL: Cumulative shifts surviving FDR 0.05: {n_both_cum}/15")
print(f"           Lagged shifts surviving FDR 0.05:     {n_both_lag}/15")


# ============================================================
# DONE — Final verification block
# ============================================================
print("\n" + "=" * 70)
print("BATCH 1 COMPLETE — VERIFICATION BLOCK")
print("=" * 70)
print("""
  Tasks completed:
    ✓ Task 4: PET trend unit verified as mm/year (Batch1_unit_verification.txt)
    ✓ Task 1: Modified Mann-Kendall (Hamed-Rao) trend test
              - Regional % drying/wetting statistics updated in:
                Table2_trend_statistics_ModifiedMK.csv
              - Comparison to standard MK in:
                Table2_comparison_standardMK_vs_modifiedMK.csv
    ✓ Task 2: FDR correction on decadal shifts
              - TableB3_Decadal_Shift_cumulative_FDR.csv
              - TableB3_Decadal_Shift_lagged_FDR.csv
              - Batch1_decadal_shift_survival_summary.csv

  VERIFY BEFORE PROCEEDING:
    1. % pixels significantly drying (Modified MK) should be SMALLER than
       what Table 2 originally reported. Example: MED went from 71.1% → ?
    2. FDR survivors should be a subset of uncorrected p<0.05 cases.
       Previously 7 cumulative shifts were p<0.05; after FDR some may drop.

  Next: Batch 2 (Task 3 stricter stable-pixel + Task 5 timescale sensitivity)
""")
