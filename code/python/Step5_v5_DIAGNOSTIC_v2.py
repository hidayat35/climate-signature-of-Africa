"""
============================================================
Step 5 v5 DIAGNOSTIC (v2): Validate canonical SPEI outputs
============================================================
Verifies that the canonical SPEI values from Step5_v5_FINAL
are scientifically sound by checking:

  CHECK 1: Decadal means per region (vs old annual-basis values)
  CHECK 2: Drought exposure % per region (vs old v4 figures)
  CHECK 3: Clip-bound saturation (extreme values frequency)
  CHECK 4: Reference-period mean should be ~0 (sanity of fit)
  CHECK 5: SPEI vs SPI gap (should be positive, AED signal)
  CHECK 6: Per-timescale ranges (all 4 should now reach ~±3.7)

Reads from spei_12_monthly.nc through spei_60_monthly.nc and
the corresponding SPI files. Robust to variable naming.
============================================================
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = r'D:\Claude idea\PhD_Paper3_Data\step5_spei_output'
SHAPEFILE = r'D:\Claude idea\ipc_africa_5_regions.shp'
DIAG_DIR = os.path.join(OUTPUT_DIR, 'diagnostics')
os.makedirs(DIAG_DIR, exist_ok=True)

REGION_ORDER = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
REGION_NAMES = {'MED': 'Mediterranean', 'SAH': 'Sahara-Sahel',
                'WAF': 'West Africa', 'EAF': 'East Africa',
                'SAF': 'Southern Africa'}

# Old annual-basis values from v4 Table7 (for comparison)
OLD_DECADAL_SPEI12 = {
    '1985_1994': {'MED': 0.216, 'SAH': 0.024, 'WAF': -0.072, 'EAF': -0.065, 'SAF': -0.106},
    '1995_2004': {'MED': -0.654, 'SAH': -0.339, 'WAF': -0.116, 'EAF': -0.040, 'SAF': 0.086},
    '2005_2014': {'MED': -0.821, 'SAH': -0.834, 'WAF': -0.332, 'EAF': -0.299, 'SAF': -0.004},
    '2015_2022': {'MED': -2.049, 'SAH': -1.404, 'WAF': -0.467, 'EAF': -0.239, 'SAF': -0.541},
}
OLD_DRYING_PCT_SPEI12 = {'MED': 72.1, 'SAH': 60.5, 'WAF': 28.7, 'EAF': 29.4, 'SAF': 20.3}

TIMESCALES = [12, 24, 36, 60]


# ============================================================
# Setup
# ============================================================
print("=" * 70)
print("DIAGNOSTIC v2: Canonical SPEI validation")
print("=" * 70)

gdf = gpd.read_file(SHAPEFILE)
if gdf.crs is None:
    gdf = gdf.set_crs('EPSG:4326')
elif gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs('EPSG:4326')

region_geoms = {}
for code in REGION_ORDER:
    subset = gdf[gdf['LAB'] == code]
    if len(subset) > 0:
        region_geoms[code] = (subset.geometry.union_all() if hasattr(subset.geometry, 'union_all')
                              else subset.geometry.unary_union)
print(f"  Loaded {len(region_geoms)} regions")


def get_index_var(ds, preferred_names):
    """Find the main variable in a NetCDF dataset; return DataArray."""
    # Try preferred names first
    for name in preferred_names:
        if name in ds.data_vars:
            return ds[name]
    # Fall back to first data variable
    first_var = list(ds.data_vars)[0]
    return ds[first_var]


def regional_mean(da, geom):
    try:
        clipped = da.rio.clip([geom], crs='EPSG:4326', all_touched=True, drop=False)
        return float(clipped.mean(skipna=True))
    except Exception:
        return np.nan


def regional_pct_drought(da, geom, threshold=-1.0):
    try:
        clipped = da.rio.clip([geom], crs='EPSG:4326', all_touched=True, drop=False)
        vals = clipped.values
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            return np.nan
        return float((vals < threshold)[valid].sum() / valid.sum() * 100)
    except Exception:
        return np.nan


# ============================================================
# Load SPEI and SPI NetCDFs (just the primary SPEI-12 for main checks,
# but check all 4 timescales exist)
# ============================================================
print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

spei_files = {k: os.path.join(OUTPUT_DIR, f'spei_{k}_monthly.nc') for k in TIMESCALES}
spi_files = {k: os.path.join(OUTPUT_DIR, f'spi_{k}_monthly.nc') for k in TIMESCALES}

missing = []
for k, f in spei_files.items():
    if not os.path.exists(f):
        missing.append(f)
for k, f in spi_files.items():
    if not os.path.exists(f):
        missing.append(f)

if missing:
    print(f"\n  ⚠ ERROR: {len(missing)} required files missing:")
    for m in missing:
        print(f"    {m}")
    raise SystemExit("\nRun Step5_v5_FINAL_all_timescales.py first.")

print(f"  ✓ All 8 NetCDF files present")

# Load SPEI-12 for main diagnostic checks
spei12_ds = xr.open_dataset(spei_files[12])
spei12 = get_index_var(spei12_ds, ['spei', 'SPEI', 'spei12', 'water_balance_spei'])
if not hasattr(spei12, 'rio') or spei12.rio.crs is None:
    spei12 = spei12.rio.write_crs('EPSG:4326')
print(f"  SPEI-12: shape {spei12.shape}, range {float(spei12.min()):.2f} to {float(spei12.max()):.2f}")

spi12_ds = xr.open_dataset(spi_files[12])
spi12 = get_index_var(spi12_ds, ['spi', 'SPI', 'spi12', 'precip_spi'])
if not hasattr(spi12, 'rio') or spi12.rio.crs is None:
    spi12 = spi12.rio.write_crs('EPSG:4326')
print(f"  SPI-12:  shape {spi12.shape}, range {float(spi12.min()):.2f} to {float(spi12.max()):.2f}")


# ============================================================
# CHECK 1: Decadal means per region
# ============================================================
print("\n" + "=" * 70)
print("CHECK 1: Decadal means per region (canonical vs v4 annual-basis)")
print("=" * 70)
print("\nExpected: signs/directions should match; magnitudes may differ")
print("(canonical SPEI is computed from monthly data with log-logistic fit)")

decadal_periods = [
    ('1985_1994', 1985, 1994),
    ('1995_2004', 1995, 2004),
    ('2005_2014', 2005, 2014),
    ('2015_2022', 2015, 2022),
]

rows_new = []
print(f"\n{'Period':<12}{'Region':<6}{'New (canonical)':>16}{'Old (annual)':>16}{'Diff':>10}{'Sign?':>8}")
print("-" * 70)

for period_label, yr_start, yr_end in decadal_periods:
    mask = (spei12.time.dt.year >= yr_start) & (spei12.time.dt.year <= yr_end)
    spei_period = spei12.sel(time=mask).mean(dim='time', skipna=True)

    for code in REGION_ORDER:
        new_val = regional_mean(spei_period, region_geoms[code])
        old_val = OLD_DECADAL_SPEI12[period_label][code]
        diff = new_val - old_val
        same_sign = '✓' if (np.sign(new_val) == np.sign(old_val)) or abs(new_val) < 0.1 else '⚠'
        rows_new.append({
            'period': period_label, 'region': code,
            'new_canonical_spei': round(new_val, 3),
            'old_annual_spei': old_val,
            'difference': round(diff, 3),
            'same_sign': same_sign
        })
        print(f"{period_label:<12}{code:<6}{new_val:>+16.3f}{old_val:>+16.3f}{diff:>+10.3f}{same_sign:>8}")
    print()

df_decadal = pd.DataFrame(rows_new)
df_decadal.to_csv(os.path.join(DIAG_DIR, 'diag_decadal_means.csv'), index=False)
print(f"Saved: diag_decadal_means.csv")


# ============================================================
# CHECK 2: Drought exposure per region
# ============================================================
print("\n" + "=" * 70)
print("CHECK 2: Drought exposure % per region (pixel-months SPEI<-1)")
print("=" * 70)

rows_expo = []
print(f"\n{'Region':<8}{'Canonical <-1 %':>18}{'Old drying %':>16}")
print("-" * 50)

for code in REGION_ORDER:
    pct_drought = regional_pct_drought(spei12, region_geoms[code])
    old_pct = OLD_DRYING_PCT_SPEI12[code]
    rows_expo.append({
        'region': code,
        'new_pct_drought': round(pct_drought, 1),
        'old_pct_drying_trend': old_pct,
    })
    print(f"{code:<8}{pct_drought:>+18.1f}{old_pct:>+16.1f}")

pd.DataFrame(rows_expo).to_csv(os.path.join(DIAG_DIR, 'diag_drought_exposure.csv'), index=False)
print(f"\nSaved: diag_drought_exposure.csv")
print("\nNote: Direct comparison is approximate — old 'drying %' was the")
print("percent of pixels with significant downward trends (Mann-Kendall),")
print("new is percent of pixel-months in moderate drought.")


# ============================================================
# CHECK 3: Clip-bound saturation
# ============================================================
print("\n" + "=" * 70)
print("CHECK 3: Clip-bound saturation check (SPEI-12)")
print("=" * 70)

data = spei12.values
valid = ~np.isnan(data)
total_valid = int(valid.sum())

n_at_low = int(((data <= -3.70) & valid).sum())
n_at_high = int(((data >= 3.70) & valid).sum())
n_extreme_neg = int(((data < -2.0) & valid).sum())
n_extreme_pos = int(((data > 2.0) & valid).sum())

print(f"\n  Total valid SPEI values:    {total_valid:>14,}")
print(f"  At low clip (≤-3.70):       {n_at_low:>14,} ({n_at_low/total_valid*100:.4f}%)")
print(f"  At high clip (≥+3.70):      {n_at_high:>14,} ({n_at_high/total_valid*100:.4f}%)")
print(f"  Extreme drought (<-2):      {n_extreme_neg:>14,} ({n_extreme_neg/total_valid*100:.3f}%)")
print(f"  Extreme wet (>+2):          {n_extreme_pos:>14,} ({n_extreme_pos/total_valid*100:.3f}%)")

if (n_at_low + n_at_high) / total_valid > 0.001:
    print("\n  ⚠ WARNING: Clip bounds are hit >0.1% — may truncate extremes.")
else:
    print("\n  ✓ Clip saturation OK.")


# ============================================================
# CHECK 4: Reference-period mean ~0
# ============================================================
print("\n" + "=" * 70)
print("CHECK 4: Reference-period (1985-2000) mean should be ~0")
print("=" * 70)

ref_mask = (spei12.time.dt.year >= 1985) & (spei12.time.dt.year <= 2000)
spei_ref = spei12.sel(time=ref_mask).mean(dim='time', skipna=True)

print(f"\n{'Region':<8}{'Ref mean':>14}{'Status':>10}")
print("-" * 35)
rows_ref = []
for code in REGION_ORDER:
    ref_val = regional_mean(spei_ref, region_geoms[code])
    flag = '✓' if abs(ref_val) < 0.2 else '⚠'
    rows_ref.append({'region': code, 'ref_period_mean': round(ref_val, 3), 'status': flag})
    print(f"{code:<8}{ref_val:>+14.3f}{flag:>10}")

pd.DataFrame(rows_ref).to_csv(os.path.join(DIAG_DIR, 'diag_ref_period_means.csv'), index=False)


# ============================================================
# CHECK 5: SPEI-SPI drought frequency gap (AED signal)
# ============================================================
print("\n" + "=" * 70)
print("CHECK 5: SPEI-SPI drought frequency gap (continental)")
print("=" * 70)

spei_vals = spei12.values
spi_vals = spi12.values

v_spei = ~np.isnan(spei_vals)
v_spi = ~np.isnan(spi_vals)
n_valid_spei = int(v_spei.sum())
n_valid_spi = int(v_spi.sum())

pct_spei_drought = float((spei_vals < -1)[v_spei].sum() / n_valid_spei * 100)
pct_spi_drought = float((spi_vals < -1)[v_spi].sum() / n_valid_spi * 100)
gap = pct_spei_drought - pct_spi_drought

print(f"\n  SPEI-12 drought frequency:   {pct_spei_drought:.2f}%")
print(f"  SPI-12 drought frequency:    {pct_spi_drought:.2f}%")
print(f"  Gap (SPEI - SPI):            {gap:+.2f}%")
if gap > 0:
    print("\n  ✓ Positive gap confirms warming-driven drought amplification.")
else:
    print("\n  ⚠ Gap is zero or negative — unexpected, investigate.")


# ============================================================
# CHECK 6: Ranges per timescale
# ============================================================
print("\n" + "=" * 70)
print("CHECK 6: Value ranges per timescale (all should be ±3.7)")
print("=" * 70)

print(f"\n{'Timescale':<12}{'SPEI min':>12}{'SPEI max':>12}{'SPI min':>12}{'SPI max':>12}")
print("-" * 70)

rows_ranges = []
for k in TIMESCALES:
    ds_spei = xr.open_dataset(spei_files[k])
    ds_spi = xr.open_dataset(spi_files[k])
    sp = get_index_var(ds_spei, ['spei', 'SPEI', f'spei{k}'])
    si = get_index_var(ds_spi, ['spi', 'SPI', f'spi{k}'])
    spei_min, spei_max = float(sp.min()), float(sp.max())
    spi_min, spi_max = float(si.min()), float(si.max())
    rows_ranges.append({
        'timescale_months': k,
        'spei_min': round(spei_min, 2), 'spei_max': round(spei_max, 2),
        'spi_min': round(spi_min, 2), 'spi_max': round(spi_max, 2),
    })
    print(f"{'SPEI-'+str(k):<12}{spei_min:>+12.2f}{spei_max:>+12.2f}{spi_min:>+12.2f}{spi_max:>+12.2f}")
    ds_spei.close()
    ds_spi.close()

pd.DataFrame(rows_ranges).to_csv(os.path.join(DIAG_DIR, 'diag_value_ranges.csv'), index=False)


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print(f"\nFiles saved to: {DIAG_DIR}")
print("  diag_decadal_means.csv")
print("  diag_drought_exposure.csv")
print("  diag_ref_period_means.csv")
print("  diag_value_ranges.csv")
print("\nInterpretation guide:")
print("  CHECK 1: Most/all 'same_sign' column should be ✓. Some SAF/EAF")
print("    entries near zero may show ⚠ (noise). That's OK if magnitudes")
print("    are small.")
print("  CHECK 2: New 'drought %' and old 'drying %' are different metrics")
print("    but both should show MED/SAH > WAF/EAF/SAF. Qualitative pattern")
print("    match is what matters.")
print("  CHECK 3: Clip saturation < 0.1% is fine; < 0.01% is ideal.")
print("  CHECK 4: Reference-period mean should be very close to zero.")
print("    |mean| > 0.2 indicates a problem with the fit.")
print("  CHECK 5: SPEI-SPI gap should be POSITIVE (AED amplification).")
print("  CHECK 6: All 4 timescales should have consistent ±3.7 ranges.")
print("\nIf all checks pass → canonical SPEI is ready for Module B attribution.")
print("If any check fails → investigate before continuing.")
