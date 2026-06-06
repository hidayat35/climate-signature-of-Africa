"""
============================================================
STEP 5 (v6 FIXED): Canonical SPEI/SPI via climate_indices
============================================================
FIX FROM v6: On Windows, multiprocessing uses 'spawn' mode,
meaning each worker re-runs the whole script. The previous
version loaded the 5.47 GB precip array at module level,
which caused 23 workers × ~10 GB each = OOM crash.

THIS VERSION:
  - ALL data loading moved inside if __name__ == '__main__':
  - Worker functions receive data via function arguments
  - Configuration constants kept at module level (safe, small)
  - Uses climate_indices package for canonical SPEI/SPI

Expected runtime: 1-2 hours total.
Expected peak memory: ~8-12 GB (main process holds arrays).
============================================================
"""

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings('ignore')

# ============================================================
# Safe to have at module level: imports and small config only
# ============================================================
from climate_indices import indices, compute

INPUT_DIR = r'D:\Claude idea\PhD_Paper3_Data\monthly_inputs'
OUTPUT_DIR = r'D:\Claude idea\PhD_Paper3_Data\step5_spei_output'

PRECIP_FILE = os.path.join(INPUT_DIR, 'precip_monthly_1985_2022.tif')
PET_FILE = os.path.join(INPUT_DIR, 'pet_monthly_1985_2022.tif')

DATA_START_YEAR = 1985
DATA_END_YEAR = 2022
CALIBRATION_START_YEAR = 1985
CALIBRATION_END_YEAR = 2000

SPEI_DISTRIBUTION = indices.Distribution.pearson
SPI_DISTRIBUTION = indices.Distribution.gamma

TIMESCALES_MONTHS = [12, 24, 36, 60]

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

# On Windows with large per-worker memory, keep worker count modest
N_WORKERS = max(1, min(8, cpu_count() - 1))


# ============================================================
# Worker functions (must be at module level for multiprocessing
# to pickle them, but they take all data as arguments — they do
# NOT reference global data)
# ============================================================

def compute_pixel_spei(args):
    """Worker: Compute SPEI for one pixel. Input = (idx, precip_1d, pet_1d, scale, dist_value)."""
    idx, precip_ts, pet_ts, scale, dist_value = args

    # Reconstruct Distribution enum from its value string
    distribution = indices.Distribution.pearson if dist_value == 'pearson' else indices.Distribution.gamma

    n_time = len(precip_ts)

    # Check calibration period validity
    cal_start_idx = (CALIBRATION_START_YEAR - DATA_START_YEAR) * 12
    cal_end_idx = cal_start_idx + (CALIBRATION_END_YEAR - CALIBRATION_START_YEAR + 1) * 12
    cal_precip = precip_ts[cal_start_idx:cal_end_idx]
    cal_pet = pet_ts[cal_start_idx:cal_end_idx]

    n_valid = np.sum(~np.isnan(cal_precip) & ~np.isnan(cal_pet))
    if n_valid < 0.5 * len(cal_precip):
        return idx, np.full(n_time, np.nan, dtype='float32')

    try:
        spei_values = indices.spei(
            precips_mm=precip_ts.astype('float64'),
            pet_mm=pet_ts.astype('float64'),
            scale=scale,
            distribution=distribution,
            periodicity=compute.Periodicity.monthly,
            data_start_year=DATA_START_YEAR,
            calibration_year_initial=CALIBRATION_START_YEAR,
            calibration_year_final=CALIBRATION_END_YEAR,
        )
        return idx, spei_values.astype('float32')
    except Exception:
        return idx, np.full(n_time, np.nan, dtype='float32')


def compute_pixel_spi(args):
    """Worker: Compute SPI for one pixel. Input = (idx, precip_1d, scale, dist_value)."""
    idx, precip_ts, scale, dist_value = args
    distribution = indices.Distribution.gamma if dist_value == 'gamma' else indices.Distribution.pearson

    n_time = len(precip_ts)

    cal_start_idx = (CALIBRATION_START_YEAR - DATA_START_YEAR) * 12
    cal_end_idx = cal_start_idx + (CALIBRATION_END_YEAR - CALIBRATION_START_YEAR + 1) * 12
    cal_precip = precip_ts[cal_start_idx:cal_end_idx]

    n_valid = np.sum(~np.isnan(cal_precip))
    if n_valid < 0.5 * len(cal_precip):
        return idx, np.full(n_time, np.nan, dtype='float32')

    try:
        spi_values = indices.spi(
            values=precip_ts.astype('float64'),
            scale=scale,
            distribution=distribution,
            data_start_year=DATA_START_YEAR,
            calibration_year_initial=CALIBRATION_START_YEAR,
            calibration_year_final=CALIBRATION_END_YEAR,
            periodicity=compute.Periodicity.monthly,
        )
        return idx, spi_values.astype('float32')
    except Exception:
        return idx, np.full(n_time, np.nan, dtype='float32')


def compute_spei_3d(precip_3d, pet_3d, scale, distribution, n_workers):
    """Parallel SPEI over all pixels."""
    n_time, ny, nx = precip_3d.shape
    precip_flat = precip_3d.reshape(n_time, -1).T
    pet_flat = pet_3d.reshape(n_time, -1).T
    n_pixels = precip_flat.shape[0]

    # Identify valid pixels
    cal_start_idx = (CALIBRATION_START_YEAR - DATA_START_YEAR) * 12
    cal_end_idx = cal_start_idx + (CALIBRATION_END_YEAR - CALIBRATION_START_YEAR + 1) * 12
    cal_valid_p = np.sum(~np.isnan(precip_flat[:, cal_start_idx:cal_end_idx]), axis=1)
    cal_valid_pet = np.sum(~np.isnan(pet_flat[:, cal_start_idx:cal_end_idx]), axis=1)
    min_req = 0.5 * (cal_end_idx - cal_start_idx)
    valid_pixels = (cal_valid_p >= min_req) & (cal_valid_pet >= min_req)
    valid_idx = np.where(valid_pixels)[0]

    print(f"    Valid pixels: {len(valid_idx):,} / {n_pixels:,} "
          f"({len(valid_idx)/n_pixels*100:.1f}%)")

    result_flat = np.full((n_pixels, n_time), np.nan, dtype='float32')
    dist_value = distribution.value  # 'pearson' or 'gamma'

    # Build task generator (saves memory vs full list)
    def task_gen():
        for i in valid_idx:
            yield (int(i), precip_flat[i].copy(), pet_flat[i].copy(), scale, dist_value)

    print(f"    Running SPEI-{scale} with {n_workers} workers...")
    chunksize = max(50, len(valid_idx) // (n_workers * 50))

    with Pool(processes=n_workers) as pool:
        completed = 0
        progress_step = max(1, len(valid_idx) // 20)
        for idx, values in pool.imap_unordered(compute_pixel_spei, task_gen(),
                                                chunksize=chunksize):
            result_flat[idx] = values
            completed += 1
            if completed % progress_step == 0:
                pct = completed / len(valid_idx) * 100
                print(f"      progress: {pct:5.1f}% ({completed:,}/{len(valid_idx):,})")
    print(f"      progress: 100.0% ({len(valid_idx):,}/{len(valid_idx):,})")

    return result_flat.T.reshape(n_time, ny, nx)


def compute_spi_3d(precip_3d, scale, distribution, n_workers):
    """Parallel SPI over all pixels."""
    n_time, ny, nx = precip_3d.shape
    precip_flat = precip_3d.reshape(n_time, -1).T
    n_pixels = precip_flat.shape[0]

    cal_start_idx = (CALIBRATION_START_YEAR - DATA_START_YEAR) * 12
    cal_end_idx = cal_start_idx + (CALIBRATION_END_YEAR - CALIBRATION_START_YEAR + 1) * 12
    cal_valid = np.sum(~np.isnan(precip_flat[:, cal_start_idx:cal_end_idx]), axis=1)
    min_req = 0.5 * (cal_end_idx - cal_start_idx)
    valid_pixels = cal_valid >= min_req
    valid_idx = np.where(valid_pixels)[0]

    print(f"    Valid pixels: {len(valid_idx):,} / {n_pixels:,} "
          f"({len(valid_idx)/n_pixels*100:.1f}%)")

    result_flat = np.full((n_pixels, n_time), np.nan, dtype='float32')
    dist_value = distribution.value

    def task_gen():
        for i in valid_idx:
            yield (int(i), precip_flat[i].copy(), scale, dist_value)

    print(f"    Running SPI-{scale} with {n_workers} workers...")
    chunksize = max(50, len(valid_idx) // (n_workers * 50))

    with Pool(processes=n_workers) as pool:
        completed = 0
        progress_step = max(1, len(valid_idx) // 20)
        for idx, values in pool.imap_unordered(compute_pixel_spi, task_gen(),
                                                chunksize=chunksize):
            result_flat[idx] = values
            completed += 1
            if completed % progress_step == 0:
                pct = completed / len(valid_idx) * 100
                print(f"      progress: {pct:5.1f}% ({completed:,}/{len(valid_idx):,})")
    print(f"      progress: 100.0% ({len(valid_idx):,}/{len(valid_idx):,})")

    return result_flat.T.reshape(n_time, ny, nx)


# ============================================================
# Save helpers (safe at module level, no large data)
# ============================================================

def save_netcdf(arr, time_coord, y_coord, x_coord, var_name, out_path):
    da = xr.DataArray(
        arr, dims=('time', 'y', 'x'),
        coords={'time': time_coord, 'y': y_coord, 'x': x_coord},
        name=var_name
    )
    da = da.rio.write_crs('EPSG:4326')
    encoding = {var_name: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}}
    da.to_netcdf(out_path, encoding=encoding)


def save_interval_means(arr, time_coord, y_coord, x_coord, var_prefix, intervals, output_dir):
    da = xr.DataArray(
        arr, dims=('time', 'y', 'x'),
        coords={'time': time_coord, 'y': y_coord, 'x': x_coord},
        name=var_prefix
    )
    da = da.rio.write_crs('EPSG:4326')
    for label, yr_start, yr_end in intervals:
        mask = (da.time.dt.year >= yr_start) & (da.time.dt.year <= yr_end)
        interval_mean = da.sel(time=mask).mean(dim='time', skipna=True)
        out_path = os.path.join(output_dir, f'{var_prefix}_mean_{label}.tif')
        interval_mean.rio.write_crs('EPSG:4326', inplace=True)
        interval_mean.rio.to_raster(out_path, dtype='float32')
        print(f"    {var_prefix}_mean_{label}.tif")


def load_monthly_multiband(path, var_name, start_year):
    """Load a multi-band GeoTIFF and assign time coordinate from bands."""
    da = rioxarray.open_rasterio(path)
    if da.rio.crs is None:
        da = da.rio.write_crs('EPSG:4326')

    band_names = da.attrs.get('long_name', None)
    if band_names is None or (isinstance(band_names, str) and '_m' not in band_names):
        n_bands = da.sizes['band']
        start_date = pd.Timestamp(f'{start_year}-01-01')
        times = pd.date_range(start=start_date, periods=n_bands, freq='MS')
    else:
        if isinstance(band_names, str):
            band_names = [band_names]
        times = []
        for bn in band_names:
            year = int(bn[1:5])
            month = int(bn.split('_m')[1])
            times.append(pd.Timestamp(year=year, month=month, day=1))
        times = pd.DatetimeIndex(times)

    da = da.assign_coords(band=('band', times)).rename({'band': 'time'})
    da.name = var_name
    return da


# ============================================================
# MAIN — all heavy lifting happens here, inside __main__ guard
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 5 v6 FIXED: climate_indices canonical SPEI/SPI")
    print("=" * 60)
    print(f"SPEI distribution: {SPEI_DISTRIBUTION.value}")
    print(f"SPI  distribution: {SPI_DISTRIBUTION.value}")
    print(f"Calibration period: {CALIBRATION_START_YEAR}-{CALIBRATION_END_YEAR}")
    print(f"Parallel workers: {N_WORKERS}")

    # ========================================================
    # Load inputs
    # ========================================================
    print("\n" + "=" * 60)
    print("PART 1: Loading monthly CHIRPS and TerraClimate")
    print("=" * 60)

    precip_da = load_monthly_multiband(PRECIP_FILE, 'precip', DATA_START_YEAR)
    pet_da = load_monthly_multiband(PET_FILE, 'pet', DATA_START_YEAR)

    print(f"  precip: shape {precip_da.shape}")
    print(f"  pet:    shape {pet_da.shape}")

    if precip_da.shape != pet_da.shape:
        print("  Aligning PET grid to precip grid...")
        pet_da = pet_da.rio.reproject_match(precip_da)

    TIME_COORD = precip_da.time
    X_COORD = precip_da.x
    Y_COORD = precip_da.y

    print("  Converting to numpy arrays (this takes a minute)...")
    precip_np = precip_da.values.astype('float32')
    pet_np = pet_da.values.astype('float32')

    # Clean: treat negatives as NaN (shouldn't occur in valid data)
    precip_np = np.where(precip_np >= 0, precip_np, np.nan)
    pet_np = np.where(pet_np >= 0, pet_np, np.nan)

    print(f"  Data shape: {precip_np.shape}")
    print(f"  Precip range: {np.nanmin(precip_np):.1f} to {np.nanmax(precip_np):.1f} mm")
    print(f"  PET range:    {np.nanmin(pet_np):.1f} to {np.nanmax(pet_np):.1f} mm")

    del precip_da, pet_da
    gc.collect()

    # ========================================================
    # Main compute loop
    # ========================================================
    print("\n" + "=" * 60)
    print("PART 2: Computing canonical SPEI and SPI")
    print("=" * 60)

    for k in TIMESCALES_MONTHS:
        print(f"\n{'='*60}")
        print(f"TIMESCALE: {k} months")
        print(f"{'='*60}")

        # === SPEI-k ===
        spei_nc = os.path.join(OUTPUT_DIR, f'spei_{k}_monthly.nc')
        if os.path.exists(spei_nc):
            print(f"  [skip] spei_{k}_monthly.nc already exists.")
        else:
            print(f"\n  === SPEI-{k} ===")
            spei_result = compute_spei_3d(precip_np, pet_np, k,
                                           SPEI_DISTRIBUTION, N_WORKERS)
            print(f"    SPEI-{k} range: {float(np.nanmin(spei_result)):.2f} to "
                  f"{float(np.nanmax(spei_result)):.2f}")
            print(f"    SPEI-{k} mean:  {float(np.nanmean(spei_result)):+.3f}")

            print(f"    Saving NetCDF...")
            save_netcdf(spei_result, TIME_COORD, Y_COORD, X_COORD, 'spei', spei_nc)
            print(f"    Saving interval means...")
            save_interval_means(spei_result, TIME_COORD, Y_COORD, X_COORD,
                                f'spei_{k}', INTERVALS, OUTPUT_DIR)
            del spei_result
            gc.collect()

        # === SPI-k ===
        spi_nc = os.path.join(OUTPUT_DIR, f'spi_{k}_monthly.nc')
        if os.path.exists(spi_nc):
            print(f"  [skip] spi_{k}_monthly.nc already exists.")
        else:
            print(f"\n  === SPI-{k} ===")
            spi_result = compute_spi_3d(precip_np, k, SPI_DISTRIBUTION, N_WORKERS)
            print(f"    SPI-{k} range: {float(np.nanmin(spi_result)):.2f} to "
                  f"{float(np.nanmax(spi_result)):.2f}")
            print(f"    SPI-{k} mean:  {float(np.nanmean(spi_result)):+.3f}")

            print(f"    Saving NetCDF...")
            save_netcdf(spi_result, TIME_COORD, Y_COORD, X_COORD, 'spi', spi_nc)
            print(f"    Saving interval means...")
            save_interval_means(spi_result, TIME_COORD, Y_COORD, X_COORD,
                                f'spi_{k}', INTERVALS, OUTPUT_DIR)
            del spi_result
            gc.collect()

    print("\n" + "=" * 60)
    print("STEP 5 v6 FIXED COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_DIR}")
    print("\nNext: Run Step5_v5_DIAGNOSTIC_v2.py to verify outputs")


if __name__ == '__main__':
    main()
