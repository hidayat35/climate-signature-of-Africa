"""
============================================================
STEP 5 — MODULE A CANONICAL RERUN
============================================================
Produces updated Module A continental climate numbers using
the canonical SPEI/SPI from Step5_v6_climate_indices.py.

INPUTS (must exist):
  - spei_{12,24,36,60}_monthly.nc    — canonical SPEI NetCDFs
  - spi_{12,24,36,60}_monthly.nc     — canonical SPI NetCDFs
  - precip_monthly_1985_2022.tif     — monthly precip (for trend)
  - pet_monthly_1985_2022.tif        — monthly PET (for trend)
  - ipc_africa_5_regions.shp         — IPCC region polygons

OUTPUTS:
  TABLES (in statistics_for_paper/):
    - Table1_regional_climate_summary.csv   (precip, PET, AED%)
    - Table2_trend_statistics_per_region.csv (% drying/wetting etc.)
    - Table3_annual_timeseries_per_region.csv (year×region values)
    - Table4_spei_per_interval_per_region.csv (interval means)
    - Table5_drought_class_pixel_counts.csv   (drought severity classes)
    - Table6_regional_correlations.csv        (SPEI-SPI, SPEI-PET corr)
    - Table7_decadal_comparison.csv           (4 decades × 5 regions)

  FIGURES (in figures_for_paper/):
    - Figure1_drought_trends.png       (4-panel Mann-Kendall slopes)
    - Figure2_AED_and_frequency.png    (AED contribution + drought freq maps)
    - Figure3_regional_timeseries.png  (5-panel regional SPEI series)
    - Figure4_SPEI_intervals.png       (8-panel interval-mean maps)
    - Figure5_regional_summary.png     (bar chart summary)
    - Figure6_decadal_heatmap.png      (decadal SPEI heatmap)

Expected runtime: 15-25 minutes.
============================================================
"""

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as sp_stats
import pymannkendall as mk
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
SPEI_DIR = r'D:\Claude idea\PhD_Paper3_Data\step5_spei_output'
MONTHLY_INPUT_DIR = r'D:\Claude idea\PhD_Paper3_Data\monthly_inputs'
SHAPEFILE = r'D:\Claude idea\ipc_africa_5_regions.shp'

STATS_DIR = os.path.join(SPEI_DIR, 'statistics_for_paper')
FIGS_DIR = os.path.join(SPEI_DIR, 'figures_for_paper')
os.makedirs(STATS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

REGION_ORDER = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
REGION_NAMES = {'MED': 'Mediterranean', 'SAH': 'Sahara-Sahel',
                'WAF': 'West Africa', 'EAF': 'East Africa',
                'SAF': 'Southern Africa'}
REGION_COLORS = {'MED': '#d62728', 'SAH': '#ff7f0e',
                 'WAF': '#2ca02c', 'EAF': '#1f77b4', 'SAF': '#9467bd'}

DECADES = [
    ('1985_1994', 1985, 1994),
    ('1995_2004', 1995, 2004),
    ('2005_2014', 2005, 2014),
    ('2015_2022', 2015, 2022),
]

INTERVALS = [
    ('1985_1990', 1985, 1989),
    ('1990_1995', 1990, 1994),
    ('1995_2000', 1995, 1999),
    ('2000_2005', 2000, 2004),
    ('2005_2010', 2005, 2009),
    ('2010_2015', 2010, 2014),
    ('2015_2020', 2015, 2019),
    ('2020_2022', 2020, 2022),
]

# ============================================================
# PART 1: Load shapefile and NetCDF data
# ============================================================
print("=" * 60)
print("PART 1: Loading inputs")
print("=" * 60)

gdf = gpd.read_file(SHAPEFILE)
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.set_crs('EPSG:4326') if gdf.crs is None else gdf.to_crs('EPSG:4326')

region_geoms = {}
for code in REGION_ORDER:
    subset = gdf[gdf['LAB'] == code]
    if len(subset) > 0:
        region_geoms[code] = (subset.geometry.union_all() if hasattr(subset.geometry, 'union_all')
                              else subset.geometry.unary_union)
AFRICA_GEOM = gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') else gdf.geometry.unary_union
print(f"  Loaded {len(region_geoms)} regions + continental geometry")


def load_index_nc(path, var_guesses=('spei', 'spi')):
    """Load SPEI/SPI NetCDF and return DataArray with CRS set."""
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


print("  Loading canonical SPEI-12 and SPI-12...")
spei12 = load_index_nc(os.path.join(SPEI_DIR, 'spei_12_monthly.nc'))
spi12 = load_index_nc(os.path.join(SPEI_DIR, 'spi_12_monthly.nc'))
print(f"    SPEI-12: shape {spei12.shape}")
print(f"    SPI-12:  shape {spi12.shape}")


# ============================================================
# PART 2: Regional-mean functions
# ============================================================

def regional_mean(da, geom):
    try:
        clipped = da.rio.clip([geom], crs='EPSG:4326', all_touched=True, drop=False)
        return float(np.nanmean(clipped.values))
    except Exception:
        return np.nan


def regional_pct_below(da, geom, threshold):
    try:
        clipped = da.rio.clip([geom], crs='EPSG:4326', all_touched=True, drop=False)
        vals = clipped.values
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            return np.nan
        return float((vals < threshold)[valid].sum() / valid.sum() * 100)
    except Exception:
        return np.nan


def regional_pct_above(da, geom, threshold):
    try:
        clipped = da.rio.clip([geom], crs='EPSG:4326', all_touched=True, drop=False)
        vals = clipped.values
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            return np.nan
        return float((vals > threshold)[valid].sum() / valid.sum() * 100)
    except Exception:
        return np.nan


# ============================================================
# PART 3: AED contribution calculation
# Gebrechorkos et al. (2025) formula:
#   AED_i = (S_SPEI_i - S_SPI_i) / S_SPEI_i * 100
# where S_X = sum of |negative X values| across time
# ============================================================
print("\n" + "=" * 60)
print("PART 3: Computing AED contribution (per pixel)")
print("=" * 60)


def compute_neg_severity_sum(da):
    """Per-pixel sum of |negative values| across time."""
    vals = da.values
    neg_vals = np.where(vals < 0, -vals, 0.0)
    return np.nansum(neg_vals, axis=0)  # (y, x)


print("  Computing severity sums...")
s_spei = compute_neg_severity_sum(spei12)
s_spi = compute_neg_severity_sum(spi12)

# Per-pixel AED contribution (clip to [-50, 100] per Gebrechorkos)
with np.errstate(invalid='ignore', divide='ignore'):
    aed_contrib = np.where(s_spei > 0.01, (s_spei - s_spi) / s_spei * 100, np.nan)
aed_contrib = np.clip(aed_contrib, -50, 100)

# Wrap in DataArray with same coords as spei12 (spatial only)
aed_da = xr.DataArray(
    aed_contrib, dims=('y', 'x'),
    coords={'y': spei12.y, 'x': spei12.x},
    name='aed_contribution_pct'
).rio.write_crs('EPSG:4326')

# Save AED raster for later use
aed_da.rio.to_raster(os.path.join(SPEI_DIR, 'aed_contribution_mean.tif'), dtype='float32')
print(f"  Saved: aed_contribution_mean.tif")
print(f"  Continental mean AED contribution: {float(np.nanmean(aed_contrib)):.2f}%")


# ============================================================
# PART 4: Load monthly precip and PET, compute annual totals
# ============================================================
print("\n" + "=" * 60)
print("PART 4: Annual precip and PET from monthly inputs")
print("=" * 60)


def load_monthly_tif(path, start_year=1985):
    da = rioxarray.open_rasterio(path)
    if da.rio.crs is None:
        da = da.rio.write_crs('EPSG:4326')
    n_bands = da.sizes['band']
    times = pd.date_range(start=f'{start_year}-01-01', periods=n_bands, freq='MS')
    da = da.assign_coords(band=('band', times)).rename({'band': 'time'})
    return da


print("  Loading monthly precip...")
precip_m = load_monthly_tif(os.path.join(MONTHLY_INPUT_DIR, 'precip_monthly_1985_2022.tif'))
print("  Loading monthly PET...")
pet_m = load_monthly_tif(os.path.join(MONTHLY_INPUT_DIR, 'pet_monthly_1985_2022.tif'))

# Annual aggregation: sum over calendar year
print("  Aggregating to annual totals...")
precip_annual = precip_m.where(precip_m >= 0).groupby('time.year').sum('time', skipna=True)
pet_annual = pet_m.where(pet_m >= 0).groupby('time.year').sum('time', skipna=True)
print(f"    precip_annual: shape {precip_annual.shape}")
print(f"    pet_annual:    shape {pet_annual.shape}")

del precip_m, pet_m
gc.collect()


# ============================================================
# PART 5: Per-pixel Mann-Kendall trends
# ============================================================
print("\n" + "=" * 60)
print("PART 5: Per-pixel trends (Mann-Kendall + Theil-Sen)")
print("=" * 60)


def compute_mk_per_pixel(da):
    """
    Apply Mann-Kendall per pixel over time axis.
    Returns (slope, p_value) arrays of shape (y, x).
    """
    vals = da.values  # (time, y, x)
    n_time, ny, nx = vals.shape
    slope = np.full((ny, nx), np.nan, dtype='float32')
    pval = np.full((ny, nx), np.nan, dtype='float32')

    total = ny * nx
    for i in range(ny):
        if i % max(1, ny // 20) == 0:
            print(f"    row {i}/{ny} ({i/ny*100:.0f}%)")
        for j in range(nx):
            ts = vals[:, i, j]
            if np.sum(~np.isnan(ts)) < n_time * 0.5:
                continue
            try:
                res = mk.original_test(ts[~np.isnan(ts)])
                slope[i, j] = res.slope
                pval[i, j] = res.p
            except Exception:
                pass
    return slope, pval


# Annual SPEI-12 — take end-of-year values (December of each year)
spei12_annual = spei12.sel(time=spei12.time.dt.month == 12)
spi12_annual = spi12.sel(time=spi12.time.dt.month == 12)

print("  Computing SPEI-12 trends...")
spei_slope, spei_pval = compute_mk_per_pixel(spei12_annual)
print("  Computing SPI-12 trends...")
spi_slope, spi_pval = compute_mk_per_pixel(spi12_annual)
print("  Computing precip annual trends...")
precip_slope, precip_pval = compute_mk_per_pixel(precip_annual)
print("  Computing PET annual trends...")
pet_slope, pet_pval = compute_mk_per_pixel(pet_annual)


# Save trend rasters
def save_2d(arr, ref_da, path):
    da = xr.DataArray(
        arr, dims=('y', 'x'),
        coords={'y': ref_da.y, 'x': ref_da.x},
    ).rio.write_crs('EPSG:4326')
    da.rio.to_raster(path, dtype='float32')


save_2d(spei_slope, spei12, os.path.join(SPEI_DIR, 'spei12_trend_slope.tif'))
save_2d(spei_pval, spei12, os.path.join(SPEI_DIR, 'spei12_trend_pvalue.tif'))
save_2d(spi_slope, spi12, os.path.join(SPEI_DIR, 'spi12_trend_slope.tif'))
save_2d(spi_pval, spi12, os.path.join(SPEI_DIR, 'spi12_trend_pvalue.tif'))
save_2d(precip_slope, precip_annual, os.path.join(SPEI_DIR, 'precip_trend_slope.tif'))
save_2d(precip_pval, precip_annual, os.path.join(SPEI_DIR, 'precip_trend_pvalue.tif'))
save_2d(pet_slope, pet_annual, os.path.join(SPEI_DIR, 'pet_trend_slope.tif'))
save_2d(pet_pval, pet_annual, os.path.join(SPEI_DIR, 'pet_trend_pvalue.tif'))
print("  Saved 8 trend rasters")


# ============================================================
# PART 6: Table 1 — Regional climate summary
# ============================================================
print("\n" + "=" * 60)
print("PART 6: Table 1 — Regional climate summary")
print("=" * 60)


def regional_2d_mean(arr_2d, ref_da, geom):
    """Mean of a 2D numpy array over a region."""
    da = xr.DataArray(
        arr_2d, dims=('y', 'x'),
        coords={'y': ref_da.y, 'x': ref_da.x}
    ).rio.write_crs('EPSG:4326')
    return regional_mean(da, geom)


mean_precip_full = precip_annual.mean(dim='year', skipna=True)
mean_pet_full = pet_annual.mean(dim='year', skipna=True)

table1_rows = []
for code in ['AFRICA'] + REGION_ORDER:
    geom = AFRICA_GEOM if code == 'AFRICA' else region_geoms[code]
    name = 'Africa (continental)' if code == 'AFRICA' else REGION_NAMES[code]

    row = {
        'region': code,
        'region_name': name,
        'mean_annual_precip_mm': round(regional_mean(mean_precip_full, geom), 1),
        'mean_annual_PET_mm': round(regional_mean(mean_pet_full, geom), 1),
        'mean_SPEI12': round(regional_mean(spei12.mean(dim='time', skipna=True), geom), 3),
        'mean_SPI12': round(regional_mean(spi12.mean(dim='time', skipna=True), geom), 3),
        'aed_contribution_pct': round(regional_2d_mean(aed_contrib, spei12, geom), 1),
        'pct_SPEI_drought_below_m1': round(regional_pct_below(spei12, geom, -1.0), 1),
        'pct_SPI_drought_below_m1': round(regional_pct_below(spi12, geom, -1.0), 1),
    }
    table1_rows.append(row)
    print(f"  {code:<8}: precip={row['mean_annual_precip_mm']:.0f} mm/yr, "
          f"PET={row['mean_annual_PET_mm']:.0f} mm/yr, AED={row['aed_contribution_pct']:.1f}%")

df1 = pd.DataFrame(table1_rows)
df1.to_csv(os.path.join(STATS_DIR, 'Table1_regional_climate_summary.csv'),
           index=False, encoding='utf-8')
print(f"  Saved: Table1_regional_climate_summary.csv")


# ============================================================
# PART 7: Table 2 — Trend statistics per region
# ============================================================
print("\n" + "=" * 60)
print("PART 7: Table 2 — Trend statistics")
print("=" * 60)


def regional_trend_stats(slope, pval, ref_da, geom, sig=0.05):
    """Mean slope, % significant, % drying/wetting over a region."""
    slope_da = xr.DataArray(slope, dims=('y', 'x'),
                            coords={'y': ref_da.y, 'x': ref_da.x}).rio.write_crs('EPSG:4326')
    pval_da = xr.DataArray(pval, dims=('y', 'x'),
                           coords={'y': ref_da.y, 'x': ref_da.x}).rio.write_crs('EPSG:4326')
    try:
        s_clip = slope_da.rio.clip([geom], crs='EPSG:4326', all_touched=True, drop=False).values
        p_clip = pval_da.rio.clip([geom], crs='EPSG:4326', all_touched=True, drop=False).values
    except Exception:
        return {'mean_slope': np.nan, 'pct_significant': np.nan,
                'pct_drying_sig': np.nan, 'pct_wetting_sig': np.nan}
    valid = ~(np.isnan(s_clip) | np.isnan(p_clip))
    if valid.sum() == 0:
        return {'mean_slope': np.nan, 'pct_significant': np.nan,
                'pct_drying_sig': np.nan, 'pct_wetting_sig': np.nan}
    s_v = s_clip[valid]
    p_v = p_clip[valid]
    sig_mask = p_v < sig
    return {
        'mean_slope': float(np.mean(s_v)),
        'pct_significant': float(sig_mask.sum() / len(p_v) * 100),
        'pct_drying_sig': float(((s_v < 0) & sig_mask).sum() / len(p_v) * 100),
        'pct_wetting_sig': float(((s_v > 0) & sig_mask).sum() / len(p_v) * 100),
    }


table2_rows = []
for code in REGION_ORDER:
    geom = region_geoms[code]
    row = {'region': code, 'region_name': REGION_NAMES[code]}
    for var, slope, pval, ref in [
        ('SPEI12', spei_slope, spei_pval, spei12),
        ('SPI12', spi_slope, spi_pval, spi12),
        ('precip', precip_slope, precip_pval, precip_annual),
        ('PET', pet_slope, pet_pval, pet_annual),
    ]:
        st = regional_trend_stats(slope, pval, ref, geom)
        row[f'{var}_mean_slope'] = round(st['mean_slope'], 4) if not np.isnan(st['mean_slope']) else np.nan
        row[f'{var}_pct_significant'] = round(st['pct_significant'], 1)
        row[f'{var}_pct_drying_sig'] = round(st['pct_drying_sig'], 1)
        row[f'{var}_pct_wetting_sig'] = round(st['pct_wetting_sig'], 1)
    table2_rows.append(row)
    print(f"  {code}: SPEI drying={row['SPEI12_pct_drying_sig']:.1f}%, "
          f"PET sig={row['PET_pct_significant']:.1f}%")

df2 = pd.DataFrame(table2_rows)
df2.to_csv(os.path.join(STATS_DIR, 'Table2_trend_statistics_per_region.csv'),
           index=False, encoding='utf-8')
print(f"  Saved: Table2_trend_statistics_per_region.csv")


# ============================================================
# PART 8: Table 3 — Annual time series per region (all years)
# ============================================================
print("\n" + "=" * 60)
print("PART 8: Table 3 — Annual time series")
print("=" * 60)

# For annual time series, use December SPEI-12 values
years = np.arange(1985, 2023)
table3_rows = []
for year in years:
    row = {'year': int(year)}
    # Find the December index
    try:
        t_idx = np.where((spei12.time.dt.year.values == year) & (spei12.time.dt.month.values == 12))[0]
        if len(t_idx) == 0:
            continue
        spei_yr = spei12.isel(time=t_idx[0])
        spi_yr = spi12.isel(time=t_idx[0])
        # Annual precip and PET already have 'year' coord
        precip_yr = precip_annual.sel(year=year) if year in precip_annual.year.values else None
        pet_yr = pet_annual.sel(year=year) if year in pet_annual.year.values else None
    except Exception:
        continue

    for code in REGION_ORDER:
        geom = region_geoms[code]
        row[f'{code}_SPEI12'] = round(regional_mean(spei_yr, geom), 3)
        row[f'{code}_SPI12'] = round(regional_mean(spi_yr, geom), 3)
        if precip_yr is not None:
            row[f'{code}_precip_mm'] = round(regional_mean(precip_yr, geom), 1)
        if pet_yr is not None:
            row[f'{code}_PET_mm'] = round(regional_mean(pet_yr, geom), 1)
    table3_rows.append(row)

df3 = pd.DataFrame(table3_rows)
df3.to_csv(os.path.join(STATS_DIR, 'Table3_annual_timeseries_per_region.csv'),
           index=False, encoding='utf-8')
print(f"  Saved: Table3_annual_timeseries_per_region.csv ({len(df3)} rows)")


# ============================================================
# PART 9: Table 4 — SPEI per interval per region
# ============================================================
print("\n" + "=" * 60)
print("PART 9: Table 4 — SPEI per interval per region")
print("=" * 60)

table4_rows = []
for label, yr_start, yr_end in INTERVALS:
    spei_mask = (spei12.time.dt.year >= yr_start) & (spei12.time.dt.year <= yr_end)
    spi_mask = (spi12.time.dt.year >= yr_start) & (spi12.time.dt.year <= yr_end)
    spei_int = spei12.sel(time=spei_mask).mean(dim='time', skipna=True)
    spi_int = spi12.sel(time=spi_mask).mean(dim='time', skipna=True)

    for code in REGION_ORDER:
        geom = region_geoms[code]
        table4_rows.append({
            'interval': label,
            'year_start': yr_start,
            'year_end': yr_end,
            'region': code,
            'region_name': REGION_NAMES[code],
            'mean_SPEI12': round(regional_mean(spei_int, geom), 3),
            'mean_SPI12': round(regional_mean(spi_int, geom), 3),
            'pct_SPEI_drought': round(regional_pct_below(spei_int, geom, -1), 1),
            'pct_SPI_drought': round(regional_pct_below(spi_int, geom, -1), 1),
        })

df4 = pd.DataFrame(table4_rows)
df4.to_csv(os.path.join(STATS_DIR, 'Table4_spei_per_interval_per_region.csv'),
           index=False, encoding='utf-8')
print(f"  Saved: Table4_spei_per_interval_per_region.csv ({len(df4)} rows)")


# ============================================================
# PART 10: Table 5 — Drought class pixel counts per region
# ============================================================
print("\n" + "=" * 60)
print("PART 10: Table 5 — Drought class pixel counts")
print("=" * 60)

# McKee 1993 drought classification
CLASSES = [
    ('extremely_wet', 2.0, np.inf),
    ('very_wet', 1.5, 2.0),
    ('moderately_wet', 1.0, 1.5),
    ('near_normal', -1.0, 1.0),
    ('moderate_drought', -1.5, -1.0),
    ('severe_drought', -2.0, -1.5),
    ('extreme_drought', -np.inf, -2.0),
]

table5_rows = []
mean_spei_all = spei12.mean(dim='time', skipna=True)
mean_spi_all = spi12.mean(dim='time', skipna=True)

for code in REGION_ORDER:
    geom = region_geoms[code]
    row = {'region': code, 'region_name': REGION_NAMES[code]}
    try:
        clip_spei = mean_spei_all.rio.clip([geom], crs='EPSG:4326', all_touched=True, drop=False).values
        clip_spi = mean_spi_all.rio.clip([geom], crs='EPSG:4326', all_touched=True, drop=False).values
        for cname, low, high in CLASSES:
            row[f'SPEI_{cname}_pct'] = round(float(((clip_spei >= low) & (clip_spei < high) &
                                                     ~np.isnan(clip_spei)).sum() /
                                                    (~np.isnan(clip_spei)).sum() * 100), 2)
            row[f'SPI_{cname}_pct'] = round(float(((clip_spi >= low) & (clip_spi < high) &
                                                    ~np.isnan(clip_spi)).sum() /
                                                   (~np.isnan(clip_spi)).sum() * 100), 2)
    except Exception as e:
        print(f"  Error for {code}: {e}")
    table5_rows.append(row)

df5 = pd.DataFrame(table5_rows)
df5.to_csv(os.path.join(STATS_DIR, 'Table5_drought_class_pixel_counts.csv'),
           index=False, encoding='utf-8')
print(f"  Saved: Table5_drought_class_pixel_counts.csv")


# ============================================================
# PART 11: Table 6 — Regional correlations (SPEI vs SPI, SPEI vs PET)
# ============================================================
print("\n" + "=" * 60)
print("PART 11: Table 6 — Regional correlations")
print("=" * 60)

# Annual regional means for correlation
spei_annual_regional = {}
spi_annual_regional = {}
pet_annual_regional = {}
precip_annual_regional = {}

for code in REGION_ORDER:
    geom = region_geoms[code]
    spei_annual_regional[code] = []
    spi_annual_regional[code] = []
    pet_annual_regional[code] = []
    precip_annual_regional[code] = []
    for year in years:
        t_idx = np.where((spei12.time.dt.year.values == year) & (spei12.time.dt.month.values == 12))[0]
        if len(t_idx) == 0:
            continue
        spei_annual_regional[code].append(regional_mean(spei12.isel(time=t_idx[0]), geom))
        spi_annual_regional[code].append(regional_mean(spi12.isel(time=t_idx[0]), geom))
        if year in pet_annual.year.values:
            pet_annual_regional[code].append(regional_mean(pet_annual.sel(year=year), geom))
            precip_annual_regional[code].append(regional_mean(precip_annual.sel(year=year), geom))
        else:
            pet_annual_regional[code].append(np.nan)
            precip_annual_regional[code].append(np.nan)

table6_rows = []
for code in REGION_ORDER:
    spei_arr = np.array(spei_annual_regional[code])
    spi_arr = np.array(spi_annual_regional[code])
    pet_arr = np.array(pet_annual_regional[code])
    precip_arr = np.array(precip_annual_regional[code])

    def safe_corr(a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 5:
            return np.nan, np.nan
        r, p = sp_stats.pearsonr(a[mask], b[mask])
        return r, p

    r_spei_spi, p_spei_spi = safe_corr(spei_arr, spi_arr)
    r_spei_pet, p_spei_pet = safe_corr(spei_arr, pet_arr)
    r_spei_precip, p_spei_precip = safe_corr(spei_arr, precip_arr)

    table6_rows.append({
        'region': code, 'region_name': REGION_NAMES[code],
        'r_SPEI_vs_SPI': round(r_spei_spi, 3),
        'p_SPEI_vs_SPI': round(p_spei_spi, 4),
        'r_SPEI_vs_PET': round(r_spei_pet, 3),
        'p_SPEI_vs_PET': round(p_spei_pet, 4),
        'r_SPEI_vs_precip': round(r_spei_precip, 3),
        'p_SPEI_vs_precip': round(p_spei_precip, 4),
    })

df6 = pd.DataFrame(table6_rows)
df6.to_csv(os.path.join(STATS_DIR, 'Table6_regional_correlations.csv'),
          index=False, encoding='utf-8')
print(f"  Saved: Table6_regional_correlations.csv")


# ============================================================
# PART 12: Table 7 — Decadal comparison
# ============================================================
print("\n" + "=" * 60)
print("PART 12: Table 7 — Decadal comparison")
print("=" * 60)

table7_rows = []
for decade_label, yr_start, yr_end in DECADES:
    spei_mask = (spei12.time.dt.year >= yr_start) & (spei12.time.dt.year <= yr_end)
    spi_mask = (spi12.time.dt.year >= yr_start) & (spi12.time.dt.year <= yr_end)
    spei_dec = spei12.sel(time=spei_mask).mean(dim='time', skipna=True)
    spi_dec = spi12.sel(time=spi_mask).mean(dim='time', skipna=True)

    precip_mask = (precip_annual.year >= yr_start) & (precip_annual.year <= yr_end)
    pet_mask = (pet_annual.year >= yr_start) & (pet_annual.year <= yr_end)
    precip_dec = precip_annual.sel(year=precip_mask).mean(dim='year', skipna=True)
    pet_dec = pet_annual.sel(year=pet_mask).mean(dim='year', skipna=True)

    for code in REGION_ORDER:
        geom = region_geoms[code]
        table7_rows.append({
            'decade': decade_label,
            'year_start': yr_start,
            'year_end': yr_end,
            'region': code,
            'region_name': REGION_NAMES[code],
            'mean_SPEI12': round(regional_mean(spei_dec, geom), 3),
            'mean_SPI12': round(regional_mean(spi_dec, geom), 3),
            'mean_precip_mm': round(regional_mean(precip_dec, geom), 1),
            'mean_PET_mm': round(regional_mean(pet_dec, geom), 1),
        })

df7 = pd.DataFrame(table7_rows)
df7.to_csv(os.path.join(STATS_DIR, 'Table7_decadal_comparison.csv'),
           index=False, encoding='utf-8')
print(f"  Saved: Table7_decadal_comparison.csv")

# Print Table 7 compact summary
print("\n  Decadal SPEI-12 per region:")
pivot = df7.pivot(index='region', columns='decade', values='mean_SPEI12')
print(pivot.to_string())


# ============================================================
# PART 13: FIGURES
# ============================================================
print("\n" + "=" * 60)
print("PART 13: Generating figures")
print("=" * 60)

# Set publication-style defaults
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.titlesize': 12,
    'font.family': 'DejaVu Sans',
})


def setup_map_axes(ax, title=''):
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    # Plot region outlines
    gdf.boundary.plot(ax=ax, linewidth=0.5, color='black')


# ---- Figure 1: Mann-Kendall slopes (4-panel) ----
print("  Figure 1: Mann-Kendall trend slopes")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
panels = [
    ('SPEI-12 trend', spei_slope, spei12, 'RdBu', (-0.05, 0.05)),
    ('SPI-12 trend', spi_slope, spi12, 'RdBu', (-0.05, 0.05)),
    ('Precipitation trend (mm/yr)', precip_slope, precip_annual, 'BrBG', (-5, 5)),
    ('PET trend (mm/yr)', pet_slope, pet_annual, 'RdYlBu_r', (-5, 5)),
]
for ax, (title, slope, ref, cmap, (vmin, vmax)) in zip(axes.flatten(), panels):
    slope_da = xr.DataArray(slope, dims=('y', 'x'),
                            coords={'y': ref.y, 'x': ref.x}).rio.write_crs('EPSG:4326')
    try:
        slope_clip = slope_da.rio.clip([AFRICA_GEOM], crs='EPSG:4326', all_touched=True, drop=False)
    except Exception:
        slope_clip = slope_da
    im = ax.imshow(slope_clip.values,
                   extent=[float(ref.x.min()), float(ref.x.max()),
                           float(ref.y.min()), float(ref.y.max())],
                   cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    setup_map_axes(ax, title)
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'Figure1_drought_trends.png'), dpi=150, bbox_inches='tight')
plt.close()

# ---- Figure 2: AED contribution map + drought frequency ----
print("  Figure 2: AED contribution + drought frequency")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# AED
aed_clip = aed_da.rio.clip([AFRICA_GEOM], crs='EPSG:4326', all_touched=True, drop=False)
im1 = axes[0].imshow(aed_clip.values,
                     extent=[float(aed_da.x.min()), float(aed_da.x.max()),
                             float(aed_da.y.min()), float(aed_da.y.max())],
                     cmap='YlOrRd', vmin=0, vmax=60, origin='upper')
setup_map_axes(axes[0], 'AED Contribution to Drought Severity (%)')
plt.colorbar(im1, ax=axes[0], shrink=0.7, pad=0.02, label='%')

# SPEI-SPI drought frequency difference
spei_freq = (spei12 < -1).sum(dim='time') / spei12.sizes['time'] * 100
spi_freq = (spi12 < -1).sum(dim='time') / spi12.sizes['time'] * 100
diff = spei_freq - spi_freq
diff_clip = diff.rio.write_crs('EPSG:4326').rio.clip([AFRICA_GEOM], crs='EPSG:4326', all_touched=True, drop=False)
im2 = axes[1].imshow(diff_clip.values,
                     extent=[float(spei12.x.min()), float(spei12.x.max()),
                             float(spei12.y.min()), float(spei12.y.max())],
                     cmap='YlOrRd', vmin=0, vmax=30, origin='upper')
setup_map_axes(axes[1], 'SPEI-SPI Drought Frequency Gap (%)')
plt.colorbar(im2, ax=axes[1], shrink=0.7, pad=0.02, label='Δ% drought')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'Figure2_AED_and_frequency.png'), dpi=150, bbox_inches='tight')
plt.close()

# ---- Figure 3: Regional SPEI time series ----
print("  Figure 3: Regional SPEI-12 time series")
fig, ax = plt.subplots(figsize=(12, 6))
for code in REGION_ORDER:
    ax.plot(years, spei_annual_regional[code], label=REGION_NAMES[code],
            color=REGION_COLORS[code], linewidth=1.8)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.axhline(-1, color='red', linestyle=':', linewidth=0.8, label='Drought threshold')
ax.set_xlabel('Year')
ax.set_ylabel('SPEI-12 (Dec value, regional mean)')
ax.set_title('Regional SPEI-12 Evolution (1985–2022)')
ax.legend(loc='lower left', framealpha=0.9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'Figure3_regional_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close()

# ---- Figure 4: SPEI interval means (8-panel) ----
print("  Figure 4: SPEI interval means")
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, (label, yr_start, yr_end) in zip(axes.flatten(), INTERVALS):
    spei_mask = (spei12.time.dt.year >= yr_start) & (spei12.time.dt.year <= yr_end)
    spei_int = spei12.sel(time=spei_mask).mean(dim='time', skipna=True)
    try:
        clip = spei_int.rio.clip([AFRICA_GEOM], crs='EPSG:4326', all_touched=True, drop=False)
    except Exception:
        clip = spei_int
    im = ax.imshow(clip.values,
                   extent=[float(spei12.x.min()), float(spei12.x.max()),
                           float(spei12.y.min()), float(spei12.y.max())],
                   cmap='RdBu', vmin=-2, vmax=2, origin='upper')
    setup_map_axes(ax, f'SPEI-12 {label.replace("_", "–")}')
fig.colorbar(im, ax=axes.flatten().tolist(), shrink=0.6, pad=0.02, label='SPEI-12')
plt.savefig(os.path.join(FIGS_DIR, 'Figure4_SPEI_intervals.png'), dpi=150, bbox_inches='tight')
plt.close()

# ---- Figure 5: Regional summary bar chart ----
print("  Figure 5: Regional summary bar chart")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
# Panel A: AED contribution
codes = REGION_ORDER
aed_vals = [df1[df1['region'] == c]['aed_contribution_pct'].values[0] for c in codes]
axes[0].bar(codes, aed_vals, color=[REGION_COLORS[c] for c in codes])
axes[0].set_ylabel('AED contribution (%)')
axes[0].set_title('AED Contribution to Drought')
axes[0].grid(axis='y', alpha=0.3)

# Panel B: SPEI drying %
drying_vals = [df2[df2['region'] == c]['SPEI12_pct_drying_sig'].values[0] for c in codes]
axes[1].bar(codes, drying_vals, color=[REGION_COLORS[c] for c in codes])
axes[1].set_ylabel('% pixels with significant SPEI-12 drying')
axes[1].set_title('Significant SPEI-12 Drying Trends')
axes[1].grid(axis='y', alpha=0.3)

# Panel C: PET trend mean
pet_vals = [df2[df2['region'] == c]['PET_mean_slope'].values[0] for c in codes]
axes[2].bar(codes, pet_vals, color=[REGION_COLORS[c] for c in codes])
axes[2].set_ylabel('Mean PET trend (mm/yr)')
axes[2].set_title('Mean PET Trend Slope')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'Figure5_regional_summary.png'), dpi=150, bbox_inches='tight')
plt.close()

# ---- Figure 6: Decadal SPEI heatmap ----
print("  Figure 6: Decadal SPEI heatmap")
fig, ax = plt.subplots(figsize=(8, 5))
pivot_spei = df7.pivot(index='region', columns='decade', values='mean_SPEI12')
pivot_spei = pivot_spei.reindex(REGION_ORDER)
im = ax.imshow(pivot_spei.values, cmap='RdBu', vmin=-2, vmax=2, aspect='auto')
ax.set_xticks(range(len(pivot_spei.columns)))
ax.set_xticklabels([c.replace('_', '–') for c in pivot_spei.columns])
ax.set_yticks(range(len(pivot_spei.index)))
ax.set_yticklabels([REGION_NAMES[r] for r in pivot_spei.index])
# Annotate cells
for i in range(pivot_spei.shape[0]):
    for j in range(pivot_spei.shape[1]):
        val = pivot_spei.values[i, j]
        color = 'white' if abs(val) > 1.0 else 'black'
        ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                color=color, fontsize=10, fontweight='bold')
ax.set_title('Decadal Mean SPEI-12 by Region')
plt.colorbar(im, ax=ax, shrink=0.7, label='SPEI-12')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'Figure6_decadal_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 60)
print("MODULE A CANONICAL RERUN COMPLETE")
print("=" * 60)
print(f"\nTables (in {STATS_DIR}):")
for f in sorted(os.listdir(STATS_DIR)):
    if f.startswith('Table'):
        print(f"  {f}")
print(f"\nFigures (in {FIGS_DIR}):")
for f in sorted(os.listdir(FIGS_DIR)):
    print(f"  {f}")

print("\nHEADLINE NUMBERS (for draft):")
row_a = df1[df1['region'] == 'AFRICA'].iloc[0]
print(f"  Continental AED contribution: {row_a['aed_contribution_pct']:.1f}%")
print(f"  Continental mean annual precip: {row_a['mean_annual_precip_mm']:.0f} mm/yr")
print(f"  Continental mean annual PET: {row_a['mean_annual_PET_mm']:.0f} mm/yr")
print(f"  Continental SPEI<-1 frequency: {row_a['pct_SPEI_drought_below_m1']:.1f}%")
print(f"  Continental SPI<-1 frequency: {row_a['pct_SPI_drought_below_m1']:.1f}%")
print(f"  Continental SPEI-SPI gap: "
      f"{row_a['pct_SPEI_drought_below_m1'] - row_a['pct_SPI_drought_below_m1']:+.1f}%")

print("\n  Regional AED contribution:")
for code in REGION_ORDER:
    r = df1[df1['region'] == code].iloc[0]
    print(f"    {code}: {r['aed_contribution_pct']:.1f}% "
          f"(SPEI={r['pct_SPEI_drought_below_m1']:.1f}%, SPI={r['pct_SPI_drought_below_m1']:.1f}%)")

print("\nNext: paste headline numbers → we update the draft.")
