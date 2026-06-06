"""
============================================================
GENERATE ARIDITY RASTER (1985-2022) for Study Area Figure
============================================================
Computes the UNEP/UNCCD aridity index per pixel:

    AI = mean_annual_precipitation / mean_annual_PET

over the full 1985-2022 period, from the monthly CHIRPS and
TerraClimate inputs already on disk. Output is a single-band
GeoTIFF ready to drop into ArcGIS Pro for the study area
figure (panel b).

UNEP/UNCCD aridity classes (Middleton & Thomas 1997):
    AI < 0.05         hyper-arid
    0.05 <= AI < 0.20 arid
    0.20 <= AI < 0.50 semi-arid
    0.50 <= AI < 0.65 dry sub-humid
    0.65 <= AI < 1.00 humid
    AI >= 1.00        very humid

INPUTS (must already exist; same paths as your other scripts):
  - <MONTHLY_INPUT_DIR>/precip_monthly_1985_2022.tif   (456 bands)
  - <MONTHLY_INPUT_DIR>/pet_monthly_1985_2022.tif      (456 bands)
  - <SHAPEFILE>                                        (IPCC Africa)

OUTPUT:
  - <OUTPUT_DIR>/aridity_PoverPET_1985_2022.tif        (single band)
  - <OUTPUT_DIR>/aridity_classes_UNEP_1985_2022.tif    (single band, 1-6)
  - Console report of regional mean aridity for sanity check.

Run:
  C:\\Users\\hidayat\\.conda\\envs\\zama1\\python.exe Generate_aridity_raster.py

Expected runtime: 2-5 minutes (depending on disk speed).
============================================================
"""

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION  -- matches Step5_v7_ModuleA_canonical_rerun.py
# ============================================================
MONTHLY_INPUT_DIR = r'D:\Claude idea\PhD_Paper3_Data\monthly_inputs'
SHAPEFILE         = r'D:\Claude idea\ipc_africa_5_regions.shp'

# Output goes to the same figures folder where the rest live
OUTPUT_DIR = r'D:\Claude idea\PhD_Paper3_Data\step5_spei_output\figures_for_paper'
os.makedirs(OUTPUT_DIR, exist_ok=True)

REGION_ORDER = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
REGION_NAMES = {'MED': 'Mediterranean', 'SAH': 'Sahara-Sahel',
                'WAF': 'West Africa', 'EAF': 'East Africa',
                'SAF': 'Southern Africa'}

START_YEAR = 1985
END_YEAR   = 2022


# ============================================================
# Helpers (same conventions as canonical Module A script)
# ============================================================
def load_monthly_tif(path, start_year=START_YEAR):
    """
    Read a 456-band monthly GeoTIFF and re-coordinate the band
    dimension as a pandas time axis. Mirrors the convention from
    Step5_v7_ModuleA_canonical_rerun.py.
    """
    da = rioxarray.open_rasterio(path)
    if da.rio.crs is None:
        da = da.rio.write_crs('EPSG:4326')
    n_bands = da.sizes['band']
    times = pd.date_range(start=f'{start_year}-01-01',
                          periods=n_bands, freq='MS')
    da = da.assign_coords(band=('band', times)).rename({'band': 'time'})
    return da


def regional_mean(da, geom):
    """Mean of a 2-D DataArray inside a polygon, NaN-safe."""
    try:
        clipped = da.rio.clip([geom], crs='EPSG:4326',
                              all_touched=True, drop=False)
        return float(np.nanmean(clipped.values))
    except Exception:
        return np.nan


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Generating aridity raster (P/PET) for 1985-2022")
    print("=" * 60)

    # --- Load IPCC shapefile for sanity-check regional means ---
    print("\n[1/5] Loading IPCC region shapefile...")
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
    print(f"    Loaded {len(region_geoms)} regions")

    # --- Load monthly precipitation and PET ---
    print("\n[2/5] Loading monthly precipitation...")
    precip_path = os.path.join(MONTHLY_INPUT_DIR,
                               'precip_monthly_1985_2022.tif')
    if not os.path.exists(precip_path):
        raise SystemExit(f"  ERROR: cannot find {precip_path}")
    precip_m = load_monthly_tif(precip_path)
    print(f"    shape: {precip_m.shape}")

    print("\n[3/5] Loading monthly PET...")
    pet_path = os.path.join(MONTHLY_INPUT_DIR,
                            'pet_monthly_1985_2022.tif')
    if not os.path.exists(pet_path):
        raise SystemExit(f"  ERROR: cannot find {pet_path}")
    pet_m = load_monthly_tif(pet_path)
    print(f"    shape: {pet_m.shape}")

    # --- Aggregate to mean annual totals ---
    print("\n[4/5] Computing mean annual P and mean annual PET...")
    # Mask negative fill values (consistent with canonical script).
    precip_annual = (precip_m.where(precip_m >= 0)
                              .groupby('time.year').sum('time', skipna=True))
    pet_annual    = (pet_m.where(pet_m >= 0)
                          .groupby('time.year').sum('time', skipna=True))

    # Squeeze any singleton band dim left over from rioxarray.
    if 'band' in precip_annual.dims:
        precip_annual = precip_annual.squeeze('band', drop=True)
    if 'band' in pet_annual.dims:
        pet_annual = pet_annual.squeeze('band', drop=True)

    mean_precip = precip_annual.mean(dim='year', skipna=True)
    mean_pet    = pet_annual.mean(dim='year', skipna=True)

    # Free memory
    del precip_m, pet_m, precip_annual, pet_annual
    gc.collect()

    # Make sure CRS is on the 2-D arrays (groupby can drop it)
    if not hasattr(mean_precip, 'rio') or mean_precip.rio.crs is None:
        mean_precip = mean_precip.rio.write_crs('EPSG:4326')
    if not hasattr(mean_pet, 'rio') or mean_pet.rio.crs is None:
        mean_pet = mean_pet.rio.write_crs('EPSG:4326')

    # --- Compute aridity index ---
    print("\n[5/5] Computing aridity index AI = mean_P / mean_PET...")
    # Avoid division by zero / very small PET (set to NaN)
    pet_safe = mean_pet.where(mean_pet > 1.0)  # PET < 1 mm/yr is unrealistic
    aridity = mean_precip / pet_safe
    # Clip to a reasonable display range (very humid tropics can exceed 2)
    aridity = aridity.where(aridity >= 0)

    # Ensure CRS
    if not hasattr(aridity, 'rio') or aridity.rio.crs is None:
        aridity = aridity.rio.write_crs('EPSG:4326')

    # Save continuous AI raster
    ai_out = os.path.join(OUTPUT_DIR, 'aridity_PoverPET_1985_2022.tif')
    aridity.rio.to_raster(ai_out, compress='LZW', dtype='float32')
    print(f"    Saved continuous AI: {ai_out}")

    # --- Build classified raster (UNEP/UNCCD classes, Middleton & Thomas 1997) ---
    # 1 = hyper-arid (<0.05); 2 = arid (0.05-0.20); 3 = semi-arid (0.20-0.50);
    # 4 = dry sub-humid (0.50-0.65); 5 = humid (0.65-1.00); 6 = very humid (>=1.00)
    classes = xr.full_like(aridity, np.nan)
    classes = classes.where(False, 0)  # init zeros, will overwrite
    classes = xr.where(aridity < 0.05, 1, classes)
    classes = xr.where((aridity >= 0.05) & (aridity < 0.20), 2, classes)
    classes = xr.where((aridity >= 0.20) & (aridity < 0.50), 3, classes)
    classes = xr.where((aridity >= 0.50) & (aridity < 0.65), 4, classes)
    classes = xr.where((aridity >= 0.65) & (aridity < 1.00), 5, classes)
    classes = xr.where(aridity >= 1.00, 6, classes)
    # Restore NaN where AI is NaN
    classes = classes.where(~np.isnan(aridity))

    if not hasattr(classes, 'rio') or classes.rio.crs is None:
        classes = classes.rio.write_crs('EPSG:4326')

    cls_out = os.path.join(OUTPUT_DIR, 'aridity_classes_UNEP_1985_2022.tif')
    classes.rio.to_raster(cls_out, compress='LZW', dtype='int8')
    print(f"    Saved classified AI: {cls_out}")

    # --- Sanity check: regional means against published expectations ---
    print("\n" + "-" * 60)
    print("Sanity check: regional mean aridity index")
    print("-" * 60)
    print(f"  {'Region':<22}{'Mean AI':>10}{'Class (typical)':>22}")

    EXPECTED_CLASS = {  # what we expect from your existing Table 1
        'MED': 'arid / semi-arid',
        'SAH': 'hyper-arid / arid',
        'WAF': 'humid / very humid',
        'EAF': 'semi-arid / dry sub-humid',
        'SAF': 'semi-arid / dry sub-humid',
    }

    for code in REGION_ORDER:
        ai_mean = regional_mean(aridity, region_geoms[code])
        # Determine class label
        if np.isnan(ai_mean):
            label = '(no data)'
        elif ai_mean < 0.05:
            label = 'hyper-arid'
        elif ai_mean < 0.20:
            label = 'arid'
        elif ai_mean < 0.50:
            label = 'semi-arid'
        elif ai_mean < 0.65:
            label = 'dry sub-humid'
        elif ai_mean < 1.00:
            label = 'humid'
        else:
            label = 'very humid'
        print(f"  {REGION_NAMES[code]:<22}{ai_mean:>10.3f}"
              f"  {label}  (expect {EXPECTED_CLASS[code]})")

    # --- Wrap-up advice for ArcGIS ---
    print("\n" + "=" * 60)
    print("DONE.")
    print("=" * 60)
    print("\nTo use in ArcGIS Pro:")
    print(f"  1. Add layer:  {ai_out}")
    print("     Symbology:  Stretched, Yellow-Green-Blue colormap")
    print("                 OR Classified using these breaks:")
    print("                   0.00 - 0.05  hyper-arid")
    print("                   0.05 - 0.20  arid")
    print("                   0.20 - 0.50  semi-arid")
    print("                   0.50 - 0.65  dry sub-humid")
    print("                   0.65 - 1.00  humid")
    print("                   >= 1.00      very humid")
    print(f"\n  2. Or use the pre-classified:  {cls_out}")
    print("     Symbology: Unique Values on raster value 1-6")
    print("\n  3. Overlay IPCC region boundaries from:")
    print(f"        {SHAPEFILE}")
    print("     Symbology: Hollow fill, black outline 0.6 pt")
    print("\nData citation for the figure caption:")
    print("  Aridity classes follow UNEP/UNCCD definitions")
    print("  (Middleton & Thomas, 1997, World Atlas of Desertification).")


if __name__ == '__main__':
    main()
