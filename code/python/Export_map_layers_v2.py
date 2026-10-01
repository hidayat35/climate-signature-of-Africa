"""
Export_map_layers_v2.py

Collects the layers needed to redraw the two map figures of the paper (Fig. 3, AED contribution and
SPEI-SPI drought-frequency gap; Fig. 4, SPEI-12 and PET trends with significance, regional time series and
decadal means) into one small zip file.

Two layers do not exist yet as files and are computed here, block by block so that memory use stays low:
    drought_frequency_gap.tif   percent of months with SPEI-12 < -1 minus percent with SPI-12 < -1
                                (same formula as the Figure 2 block of Step5_v7_ModuleA_canonical_rerun.py)
    precip_mean_annual.tif      mean annual precipitation 1985-2022 (mm), used to check the hyper-arid core

The trend p-values are written as small significance masks (1 = p < 0.05, 0 = not significant).

Usage (paths default to the ones used in the Step5 and Batch1 scripts):
    python Export_map_layers_v2.py
    python Export_map_layers_v2.py --spei-dir "D:\\Claude idea\\PhD_Paper3_Data\\step5_spei_output" ^
        --monthly-dir "D:\\Claude idea\\PhD_Paper3_Data\\monthly_inputs" ^
        --shapefile "D:\\Claude idea\\ipc_africa_5_regions.shp" --out figure_layers

Requires numpy, pandas, xarray and rioxarray (already used by the Step5 scripts).
"""
import argparse
import glob
import os
import shutil
import zipfile

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor)

TIF_OPTS = dict(compress='LZW', tiled=True)


def to_yx(da):
    ren = {}
    for a, b in (('lat', 'y'), ('latitude', 'y'), ('lon', 'x'), ('longitude', 'x')):
        if a in da.dims:
            ren[a] = b
    return da.rename(ren) if ren else da


def load_index_nc(path, guesses=('spei', 'spi')):
    ds = xr.open_dataset(path)
    for v in guesses:
        if v in ds.data_vars:
            return to_yx(ds[v])
    return to_yx(ds[list(ds.data_vars)[0]])


def percent_below(da, thr=-1.0, block=48):
    n = da.sizes['time']
    count = np.zeros((da.sizes['y'], da.sizes['x']), dtype=np.float64)
    valid = np.zeros_like(count)
    for t0 in range(0, n, block):
        vals = da.isel(time=slice(t0, t0 + block)).transpose('time', 'y', 'x').values
        count += np.nansum(vals < thr, axis=0)
        valid += np.sum(np.isfinite(vals), axis=0)
    pct = count / n * 100.0                     # same denominator as the Step5 figure code
    pct[valid == 0] = np.nan
    return pct


def write_2d(arr, ref, path, dtype='float32', nodata=None):
    da = xr.DataArray(arr.astype(dtype), dims=('y', 'x'), coords={'y': ref.y.values, 'x': ref.x.values})
    da = da.sortby('y', ascending=False).sortby('x')       # north-up GeoTIFF whatever the input order
    da = da.rio.write_crs('EPSG:4326')
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    kw = dict(TIF_OPTS)
    if dtype == 'float32':
        kw['predictor'] = 3
    da.rio.to_raster(path, **kw)
    print('  written', os.path.basename(path), f'{os.path.getsize(path) / 1e6:.1f} MB')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spei-dir', default=r'D:\Claude idea\PhD_Paper3_Data\step5_spei_output')
    ap.add_argument('--monthly-dir', default=r'D:\Claude idea\PhD_Paper3_Data\monthly_inputs')
    ap.add_argument('--shapefile', default=r'D:\Claude idea\ipc_africa_5_regions.shp')
    ap.add_argument('--out', default='figure_layers')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stats_dir = os.path.join(a.spei_dir, 'statistics_for_paper')
    missing = []

    # 1. SPEI-SPI drought-frequency gap
    print('Computing the drought-frequency gap from the monthly SPEI-12 and SPI-12 files...')
    f_spei = os.path.join(a.spei_dir, 'spei_12_monthly.nc')
    f_spi = os.path.join(a.spei_dir, 'spi_12_monthly.nc')
    if os.path.exists(f_spei) and os.path.exists(f_spi):
        spei = load_index_nc(f_spei)
        spi = load_index_nc(f_spi)
        gap = percent_below(spei) - percent_below(spi)
        write_2d(gap, spei, os.path.join(a.out, 'drought_frequency_gap.tif'), nodata=np.nan)
        print(f'  pixel mean of the gap: {np.nanmean(gap):.2f} percentage points')
    else:
        missing += [f for f in (f_spei, f_spi) if not os.path.exists(f)]

    # 2. mean annual precipitation
    print('Computing mean annual precipitation...')
    ptif = os.path.join(a.monthly_dir, 'precip_monthly_1985_2022.tif')
    if os.path.exists(ptif):
        pm = rioxarray.open_rasterio(ptif, masked=True)
        nb = pm.sizes['band']
        total = np.zeros((pm.sizes['y'], pm.sizes['x']))
        for b0 in range(0, nb, 12):
            vals = pm.isel(band=slice(b0, b0 + 12)).values.astype('float64')
            vals[vals < 0] = np.nan
            total += np.nansum(vals, axis=0)
        write_2d(total / (nb / 12.0), pm, os.path.join(a.out, 'precip_mean_annual.tif'), nodata=np.nan)
    else:
        missing.append(ptif)

    # 3. significance masks from the modified Mann-Kendall p-values
    print('Writing significance masks...')
    for var in ('spei12', 'pet'):
        p = os.path.join(a.spei_dir, f'{var}_trend_pvalue_modMK.tif')
        if not os.path.exists(p):
            p = os.path.join(a.spei_dir, f'{var}_trend_pvalue.tif')
        if os.path.exists(p):
            pv = rioxarray.open_rasterio(p, masked=True).squeeze()
            v = pv.values.astype('float64')
            mask = np.where(np.isfinite(v), (v < 0.05).astype('uint8'), 255).astype('uint8')
            write_2d(mask, pv, os.path.join(a.out, f'{var}_trend_significant.tif'), dtype='uint8', nodata=255)
        else:
            missing.append(p)

    # 4. files that already exist
    print('Copying existing layers and tables...')
    wanted = [os.path.join(a.spei_dir, 'aed_contribution_mean.tif'),
              os.path.join(a.spei_dir, 'spei12_trend_slope_modMK.tif'),
              os.path.join(a.spei_dir, 'pet_trend_slope_modMK.tif'),
              os.path.join(stats_dir, 'Table3_annual_timeseries_per_region.csv'),
              os.path.join(stats_dir, 'Table7_decadal_comparison.csv')]
    fallback = {'spei12_trend_slope_modMK.tif': 'spei12_trend_slope.tif',
                'pet_trend_slope_modMK.tif': 'pet_trend_slope.tif'}
    for f in wanted:
        if not os.path.exists(f) and os.path.basename(f) in fallback:
            f = os.path.join(os.path.dirname(f), fallback[os.path.basename(f)])
        if os.path.exists(f) and f.endswith('.tif'):
            r = rioxarray.open_rasterio(f, masked=True).squeeze()   # re-written compressed to keep the zip small
            write_2d(r.values.astype('float32'), r, os.path.join(a.out, os.path.basename(f)), nodata=np.nan)
        elif os.path.exists(f):
            shutil.copy2(f, a.out)
            print('  copied', os.path.basename(f))
        else:
            missing.append(f)
    stem = os.path.splitext(a.shapefile)[0]
    shp_parts = glob.glob(stem + '.*')
    if not shp_parts:
        missing.append(a.shapefile)
    for f in shp_parts:
        shutil.copy2(f, a.out)
    print(f'  copied {len(shp_parts)} shapefile parts')

    # 5. one zip to upload
    zpath = a.out.rstrip('\\/') + '.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(a.out)):
            z.write(os.path.join(a.out, f), f)
    print(f'\nDone: {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB). Please upload this zip file.')
    if missing:
        print('\nNot found (send these separately or tell me where they are):')
        for f in missing:
            print('  ', f)


if __name__ == '__main__':
    main()
