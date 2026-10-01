"""
Make_map_figures.py

Draws the two map figures of the paper at print size (6.5 in, the full text width of a journal page),
600 dpi, with lettering of 7 pt or larger:

    Fig3_AED_maps.png                  Fig. 3  (a) AED contribution to drought severity
                                               (b) SPEI-SPI drought-frequency gap
                                               dashed lines: 10 mm/year isohyet of mean annual precipitation,
                                               around the hyper-arid core where SPI is poorly defined
    Fig4_drought_intensification.png   Fig. 4  (a) SPEI-12 trend and (b) PET trend, stippled where the
                                               modified Mann-Kendall trend is significant (p < 0.05);
                                               (c) regional-mean SPEI-12, December values;
                                               (d) decadal-mean SPEI-12 by region

Input: the folder written by Export_map_layers.py:
    aed_contribution_mean.tif, drought_frequency_gap.tif, precip_mean_annual.tif,
    spei12_trend_slope_modMK.tif, spei12_trend_significant.tif,
    pet_trend_slope_modMK.tif, pet_trend_significant.tif,
    Table3_annual_timeseries_per_region.csv, Table7_decadal_comparison.csv,
    ipc_africa_5_regions.shp (with .shx, .dbf, .prj)

Usage:
    python Make_map_figures.py --layers figure_layers --out figures

Requires numpy, pandas, scipy, matplotlib (with contourpy), rioxarray and geopandas.
"""
import argparse
import os

import contourpy
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D
from PIL import Image
from scipy import ndimage

DPI = 600
REG = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
REGNAME = {'MED': 'Mediterranean', 'SAH': 'Sahara-Sahel', 'WAF': 'West Africa',
           'EAF': 'East Africa', 'SAF': 'Southern Africa'}
REGCOL = {'MED': '#d62728', 'SAH': '#ff7f0e', 'WAF': '#2ca02c', 'EAF': '#1f77b4', 'SAF': '#9467bd'}
MINUS = '−'
LON0, LON1, LAT0, LAT1 = -19.0, 52.5, -36.0, 38.5      # map window (degrees)
MAP_ASPECT = (LAT1 - LAT0) / (LON1 - LON0)
TITLE = dict(loc='left', fontsize=8.5, fontweight='bold', pad=4)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.3,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.unicode_minus': True,
})


def signed(v, nd=2):
    """+0.19 / −0.64 / 0.00, with a true minus sign."""
    if round(v, nd) == 0:
        return f'{0:.{nd}f}'
    return f'{v:+.{nd}f}'.replace('-', MINUS)


def plain(v, nd):
    return f'{v:.{nd}f}'.replace('-', MINUS)


def save(fig, path, W, H):
    """600 dpi PNG, flattened to RGB and optimized so that the Word file stays small."""
    fig.savefig(path, dpi=DPI, facecolor='white')
    plt.close(fig)
    with Image.open(path) as im:
        rgb = im.convert('RGB')
    rgb.save(path, optimize=True, dpi=(DPI, DPI))
    print('written', path, f'({W:.2f} x {H:.2f} in, {os.path.getsize(path) / 1e6:.1f} MB)')


def add_axes_in(fig, W, H, x, y, w, h):
    """Axes placed in inches from the lower-left corner of the figure."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def load(layers, name, africa):
    r = rioxarray.open_rasterio(os.path.join(layers, name), masked=True).squeeze()
    r = r.sortby('y', ascending=False).sortby('x')
    return r.rio.clip([africa], crs='EPSG:4326', all_touched=True, drop=False)


def lon_label(v):
    return '0°' if v == 0 else f'{abs(v):.0f}°{"W" if v < 0 else "E"}'


def lat_label(v):
    return '0°' if v == 0 else f'{abs(v):.0f}°{"S" if v < 0 else "N"}'


def draw_map(ax, da, cmap, norm, regions, lat_labels=True):
    dx = abs(float(da.x[1] - da.x[0]))
    dy = abs(float(da.y[1] - da.y[0]))
    ext = [float(da.x.min()) - dx / 2, float(da.x.max()) + dx / 2,
           float(da.y.min()) - dy / 2, float(da.y.max()) + dy / 2]
    im = ax.imshow(np.asarray(da.values), extent=ext, cmap=cmap, norm=norm, origin='upper',
                   interpolation='nearest', rasterized=True)
    regions.boundary.plot(ax=ax, color='#1a1a1a', linewidth=0.4)
    xt, yt = [-15, 0, 15, 30, 45], [-30, -15, 0, 15, 30]
    ax.set_xticks(xt)
    ax.set_xticklabels([lon_label(v) for v in xt])
    ax.set_yticks(yt)
    ax.set_yticklabels([lat_label(v) for v in yt])
    ax.set_xlim(LON0, LON1)             # after the ticks, which would otherwise widen the view
    ax.set_ylim(LAT0, LAT1)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(direction='out', length=2, width=0.5, labelsize=7, pad=1.5, labelleft=lat_labels)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color('#444444')
    return im


def stipple(ax, sig, step=14, size=0.28):
    """Dots on a regular lattice (every `step` grid cells) wherever the trend is significant."""
    v = np.asarray(sig.values)
    rr, cc = np.meshgrid(np.arange(step // 2, v.shape[0], step), np.arange(step // 2, v.shape[1], step),
                         indexing='ij')
    ok = v[rr, cc] == 1
    ax.scatter(np.asarray(sig.x.values)[cc[ok]], np.asarray(sig.y.values)[rr[ok]], s=size, c='#000000',
               marker='o', linewidths=0, rasterized=True, zorder=3)


ISO_MM = 10.0               # isohyet drawn on Fig. 3 (mm/year)
ISO_COLOR = '#08519c'
ISO_DASH = (0, (2.8, 1.5))


def isohyet(layers, africa, ref, level=ISO_MM, sigma=2.0, min_len=3.0):
    """Contour of mean annual precipitation at `level` mm/year, as a list of (n, 2) lon/lat arrays.

    Cells without CHIRPS data (no value in `ref`, e.g. salt pans and depressions) are left out, so that they
    do not appear as dry islands. The field is smoothed over about 10 km (sigma in grid cells) and pieces
    shorter than `min_len` degrees are dropped."""
    pr = load(layers, 'precip_mean_annual.tif', africa)
    assert pr.shape == ref.shape and np.allclose(pr.x, ref.x) and np.allclose(pr.y, ref.y)
    p = np.asarray(pr.values, dtype=float)
    p[~np.isfinite(np.asarray(ref.values))] = np.nan
    ok = np.isfinite(p)
    num = ndimage.gaussian_filter(np.where(ok, p, 0.0), sigma)
    den = ndimage.gaussian_filter(ok.astype(float), sigma)
    smooth = num / np.where(den > 0, den, np.nan)
    smooth[~ok] = np.nan
    gen = contourpy.contour_generator(np.asarray(pr.x.values), np.asarray(pr.y.values), smooth,
                                      line_type='Separate')
    return [ln for ln in gen.lines(level) if np.hypot(*np.diff(ln, axis=0).T).sum() >= min_len]


def draw_isohyet(ax, lines):
    """Blue dashes over a thin white line, legible on both the pale and the dark end of the colour scale."""
    for ln in lines:
        ax.plot(ln[:, 0], ln[:, 1], color='white', lw=1.25, solid_capstyle='round', zorder=4)
        ax.plot(ln[:, 0], ln[:, 1], color=ISO_COLOR, lw=0.7, ls=ISO_DASH, zorder=5)


def hcbar(fig, W, H, x, y, w, im, label, ticks, labels):
    cax = add_axes_in(fig, W, H, x, y, w, 0.085)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend='both', extendfrac=0.035)
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels)
    cb.ax.tick_params(labelsize=7, length=2, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label(label, fontsize=7.5, labelpad=2)
    return cb


# layout shared by the two figures (inches)
W = 6.5
LEFT = 0.36                 # room for latitude labels
GAP = 0.13
MW = (W - LEFT - GAP - 0.07) / 2
MH = MW * MAP_ASPECT
CB_W = MW * 0.8


def map_row(fig, H, y0):
    """Two map axes whose bottom edge sits at y0 (inches)."""
    return (add_axes_in(fig, W, H, LEFT, y0, MW, MH),
            add_axes_in(fig, W, H, LEFT + MW + GAP, y0, MW, MH))


# --------------------------------------------------------------------------------------------
def fig3(layers, regions, africa, out):
    aed = load(layers, 'aed_contribution_mean.tif', africa)
    gap = load(layers, 'drought_frequency_gap.tif', africa)
    y_cb, y_map = 0.31, 0.64
    H = y_map + MH + 0.22
    fig = plt.figure(figsize=(W, H))
    ax_a, ax_b = map_row(fig, H, y_map)
    cm = plt.get_cmap('YlOrRd')
    im_a = draw_map(ax_a, aed, cm, Normalize(0, 60), regions)
    im_b = draw_map(ax_b, gap, cm, Normalize(0, 30), regions, lat_labels=False)
    lines = isohyet(layers, africa, aed)
    for ax in (ax_a, ax_b):
        draw_isohyet(ax, lines)
    key = Line2D([], [], color=ISO_COLOR, lw=0.7, ls=ISO_DASH)
    ax_a.legend([key], [f'{ISO_MM:.0f} mm/year isohyet'], loc='lower left', bbox_to_anchor=(0.015, 0.02),
                frameon=False, fontsize=7, handlelength=2.6, borderaxespad=0.2)
    ax_a.set_title('(a) AED contribution to drought severity', **TITLE)
    ax_b.set_title('(b) SPEI–SPI drought-frequency gap', **TITLE)
    t_a, t_b = [0, 10, 20, 30, 40, 50, 60], [0, 5, 10, 15, 20, 25, 30]
    hcbar(fig, W, H, LEFT + (MW - CB_W) / 2, y_cb, CB_W, im_a, 'AED contribution (%)', t_a, [str(t) for t in t_a])
    hcbar(fig, W, H, LEFT + MW + GAP + (MW - CB_W) / 2, y_cb, CB_W, im_b,
          'Difference in drought frequency (percentage points)', t_b, [str(t) for t in t_b])
    save(fig, os.path.join(out, 'Fig3_AED_maps.png'), W, H)


# --------------------------------------------------------------------------------------------
def fig4(layers, regions, africa, out):
    s_slope = load(layers, 'spei12_trend_slope_modMK.tif', africa)
    s_sig = load(layers, 'spei12_trend_significant.tif', africa)
    p_slope = load(layers, 'pet_trend_slope_modMK.tif', africa)
    p_sig = load(layers, 'pet_trend_significant.tif', africa)
    t3 = pd.read_csv(os.path.join(layers, 'Table3_annual_timeseries_per_region.csv'))
    t7 = pd.read_csv(os.path.join(layers, 'Table7_decadal_comparison.csv'))

    y_low, h_low = 0.37, 1.95            # bottom row
    y_cb = y_low + h_low + 0.28 + 0.28    # colour bars of the map row
    y_map = y_cb + 0.33
    H = y_map + MH + 0.22
    fig = plt.figure(figsize=(W, H))

    # (a), (b) trend maps
    ax_a, ax_b = map_row(fig, H, y_map)
    im_a = draw_map(ax_a, s_slope, plt.get_cmap('RdBu'), TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.05), regions)
    stipple(ax_a, s_sig)
    im_b = draw_map(ax_b, p_slope, plt.get_cmap('RdYlBu_r'), TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5), regions,
                    lat_labels=False)
    stipple(ax_b, p_sig)
    ax_a.set_title('(a) SPEI-12 trend', **TITLE)
    ax_b.set_title('(b) PET trend', **TITLE)
    t_a, t_b = [-0.05, -0.025, 0, 0.025, 0.05], [-5, -2.5, 0, 2.5, 5]
    hcbar(fig, W, H, LEFT + (MW - CB_W) / 2, y_cb, CB_W, im_a, 'SPEI-12 trend (standardized units per year)', t_a,
          [plain(v, 3) if v else '0' for v in t_a])
    hcbar(fig, W, H, LEFT + MW + GAP + (MW - CB_W) / 2, y_cb, CB_W, im_b, 'PET trend (mm/year)', t_b,
          [plain(v, 1) if v else '0' for v in t_b])

    # (c) regional-mean SPEI-12, December values
    ax_c = add_axes_in(fig, W, H, 0.5, y_low, 2.85, h_low)
    for r in REG:
        ax_c.plot(t3['year'], t3[f'{r}_SPEI12'], color=REGCOL[r], lw=1.0, label=REGNAME[r])
    ax_c.axhline(0, color='#8c8c8c', lw=0.6, ls='--', zorder=0)
    ax_c.axhline(-1, color='#b2182b', lw=0.8, ls=':', label='Moderate-drought threshold', zorder=0)
    ax_c.set_xlim(1984.5, 2022.5)
    ax_c.set_xticks([1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020])
    ax_c.set_ylim(-3.1, 1.3)
    ax_c.set_yticks([-3, -2, -1, 0, 1])
    ax_c.set_ylabel('SPEI-12 (December value)', fontsize=7.8, labelpad=2)
    ax_c.set_xlabel('Year', fontsize=7.8, labelpad=2)
    ax_c.tick_params(labelsize=7, pad=1.5)
    fig.text(LEFT / W, (y_low + h_low + 0.075) / H, '(c) Regional-mean SPEI-12', fontsize=8.5,
             fontweight='bold', ha='left', va='baseline')
    ax_c.grid(alpha=0.3, ls=':', lw=0.5)
    for s in ('top', 'right'):
        ax_c.spines[s].set_visible(False)
    ax_c.legend(loc='lower left', ncol=2, frameon=False, fontsize=6.8, handlelength=1.6, columnspacing=1.0,
                labelspacing=0.25, borderaxespad=0.15)

    # (d) decadal-mean SPEI-12 by region
    x_lab = 3.62                          # left edge of the region names
    ax_d = add_axes_in(fig, W, H, x_lab + 0.8, y_low + 0.08, 1.52, h_low - 0.08)
    decades = list(dict.fromkeys(t7['decade']))
    piv = t7.pivot(index='region', columns='decade', values='mean_SPEI12').reindex(REG)[decades]
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    cm = plt.get_cmap('RdBu')
    ax_d.imshow(piv.values, cmap=cm, norm=norm, aspect='auto')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax_d.text(j, i, signed(v), ha='center', va='center', fontsize=6.8,
                      color='white' if abs(v) > 1.0 else '#111111')
    ax_d.set_xticks(range(len(decades)))
    ax_d.set_xticklabels([d.split('_')[0] + '–\n' + d.split('_')[1] for d in decades], fontsize=6.8,
                         linespacing=1.05)
    ax_d.set_yticks(range(len(REG)))
    ax_d.set_yticklabels([REGNAME[r] for r in REG], fontsize=7.2)
    ax_d.tick_params(length=0, pad=2)
    ax_d.set_xticks(np.arange(-0.5, len(decades)), minor=True)
    ax_d.set_yticks(np.arange(-0.5, len(REG)), minor=True)
    ax_d.grid(which='minor', color='white', lw=0.8)
    ax_d.tick_params(which='minor', length=0)
    for s in ax_d.spines.values():
        s.set_visible(False)
    fig.text((LEFT + MW + GAP) / W, (y_low + h_low + 0.075) / H, '(d) Decadal-mean SPEI-12', fontsize=8.5,
             fontweight='bold', ha='left', va='baseline')
    cax = add_axes_in(fig, W, H, x_lab + 0.8 + 1.52 + 0.1, y_low + 0.2, 0.07, h_low - 0.32)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax, extend='both', extendfrac=0.05)
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.set_ticklabels([plain(v, 0) for v in [-2, -1, 0, 1, 2]])
    cb.ax.tick_params(labelsize=6.8, length=2, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)

    save(fig, os.path.join(out, 'Fig4_drought_intensification.png'), W, H)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layers', default='figure_layers')
    ap.add_argument('--out', default='figures')
    ap.add_argument('--only', default=None, help='fig3, fig4 or fig3,fig4')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    regions = gpd.read_file(os.path.join(a.layers, 'ipc_africa_5_regions.shp'))
    regions = regions.set_crs('EPSG:4326') if regions.crs is None else regions.to_crs('EPSG:4326')
    geo = regions.geometry
    africa = geo.union_all() if hasattr(geo, 'union_all') else geo.unary_union
    jobs = {'fig3': fig3, 'fig4': fig4}
    for name in (a.only.split(',') if a.only else jobs):
        jobs[name](a.layers, regions, africa, a.out)


if __name__ == '__main__':
    main()
