"""
Make_publication_figures.py

Draws six figures from the per-interval effect-size tables, sized for print: each figure is drawn at its
final width (6.5-7.0 in, the full text width of a journal page), so that all lettering is 7 pt or larger
when the figure is printed at that width.

    Fig2_workflow.png    Fig. 2   analytical framework (no input data)
    Fig5_forest.png      Fig. 5   continental effect sizes under five estimators, SPEI-12
    Fig6_heatmap.png     Fig. 6   effect size by transition pathway and region, SPEI-12
    Fig7_RSI_DRA.png     Fig. 7   Recovery Suppression Index and Degradation-Recovery Asymmetry
    FigS2_mixed.png      Fig. S2  recovery-degradation contrast from the mixed-effects model (Table S7)
    FigS3_decadal.png    Fig. S3  early (1990-2004) and late (2005-2022) period means, cumulative window
    FigS4_boxplot.png    Fig. S4  distribution of interval-level effect sizes, lagged window

Inputs
    --lagged      TableB2_LAGGED_per_interval_STRICTER.csv      (Batch2 output)
    --cumulative  TableB2_CUMULATIVE_per_interval_STRICTER.csv  (Batch2 output)
    --twostage    optional JSON with the continental means and BCa intervals; its "CI" -> "spei_12"
                  entry has the form {estimator: {category: [mean, ci_low, ci_high, n_units]}}.
                  Without it, the values of Table S9 are used.
    --out         output folder (created if missing)

Example
    python Make_publication_figures.py --lagged TableB2_LAGGED_per_interval_STRICTER.csv \
        --cumulative TableB2_CUMULATIVE_per_interval_STRICTER.csv --out figures

Requires numpy, pandas, scipy and matplotlib. Each figure is written as a 600 dpi PNG and as a vector PDF
with embedded fonts.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import matplotlib.font_manager as fm

DPI = 600
REG = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
REGNAME = {'MED': 'Mediterranean', 'SAH': 'Sahara-Sahel', 'WAF': 'West Africa',
           'EAF': 'East Africa', 'SAF': 'Southern Africa'}
REGCOL = {'MED': '#d62728', 'SAH': '#ff7f0e', 'WAF': '#2ca02c', 'EAF': '#1f77b4', 'SAF': '#9467bd'}
CATS = ['Degradation', 'Recovery', 'Agricultural']
CATCOL = {'Degradation': '#B03A2E', 'Recovery': '#1E7B45', 'Agricultural': '#B9770E'}
SHORT = {'Degradation': 'Deg', 'Recovery': 'Rec', 'Agricultural': 'Agr'}
RECOVERY = ['SHR_FST', 'GRS_SHR', 'BAL_GRS']
TIMESCALES = [12, 24, 36, 60]
MINUS = '−'

# Continental means and 95% BCa intervals at SPEI-12 (Table S9): mean, low, high, number of units
TABLE_S9 = {
    'point_lagged': {'Degradation': [0.0204, -0.0745, 0.1202, 16],
                     'Recovery': [-0.1784, -0.2599, -0.1012, 12],
                     'Agricultural': [-0.138, -0.2993, -0.0152, 10]},
    'point_cumul': {'Degradation': [0.0613, -0.0093, 0.1335, 16],
                    'Recovery': [-0.1575, -0.2402, -0.038, 12],
                    'Agricultural': [-0.0552, -0.2309, 0.077, 10]},
    'area_weighted': {'Degradation': [0.0145, -0.0677, 0.105, 16],
                      'Recovery': [-0.1823, -0.2611, -0.1114, 12],
                      'Agricultural': [-0.1353, -0.2852, -0.0146, 10]},
    'area_threshold': {'Degradation': [0.0344, -0.0645, 0.1509, 16],
                       'Recovery': [-0.1706, -0.2424, -0.1065, 12],
                       'Agricultural': [-0.1209, -0.2788, -0.0067, 10]},
    'cem_matched': {'Degradation': [0.0396, -0.0378, 0.1251, 16],
                    'Recovery': [-0.1647, -0.2255, -0.0996, 12],
                    'Agricultural': [-0.0422, -0.1462, 0.037, 10]},
}

def _sans():
    """Arial where installed, otherwise its metric twin Liberation Sans."""
    for name in ('Arial', 'Liberation Sans'):
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue
    return 'DejaVu Sans'


SANS = _sans()
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'mathtext.fontset': 'custom', 'mathtext.rm': SANS, 'mathtext.it': f'{SANS}:italic',
    'mathtext.bf': f'{SANS}:bold', 'mathtext.bfit': f'{SANS}:italic:bold', 'mathtext.cal': SANS,
    'mathtext.sf': SANS,
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.unicode_minus': True, 'hatch.linewidth': 0.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def fmt(v, nd=2, plus=True):
    """Number with an explicit sign and a true minus sign."""
    s = f'{v:+.{nd}f}' if plus else f'{v:.{nd}f}'
    return s.replace('-', MINUS)


def tick_labels(values, nd=1):
    return [fmt(t, nd) if abs(t) > 1e-9 else f'{0:.{nd}f}' for t in values]


def load_effects(path):
    """Per-interval table; rows without an effect size (cells below the pixel thresholds) are dropped."""
    df = pd.read_csv(path)
    return df[df.cohens_d.notna()].copy()


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=DPI, facecolor='white')
    pdf = os.path.splitext(path)[0] + '.pdf'
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    print('written', path, 'and', pdf)


# --------------------------------------------------------------------------------------------
# Fig. 2  Analytical framework (drawn in inches: 1 data unit = 1 inch)
# --------------------------------------------------------------------------------------------
def fig2_workflow(out):
    """Monochrome flow diagram: (a) the workflow of both modules, (b) the comparison design."""
    W = 6.5
    ink, fill_out, fill_hl, note = '#1a1a1a', '#e6e6e6', '#d0d0d0', '#444444'
    lw, ts, bs = 0.6, 7.5, 7.0                      # line width, box title and body sizes (pt)
    pitch = bs * 1.24 / 72                          # body line pitch (in)
    rows_h = [0.48, 0.60, 0.60, 0.60, 0.84, 0.30]   # heights of the six workflow rows
    gap, top_a, sep, h_b = 0.15, 0.48, 0.15, 1.53
    h_a = top_a + sum(rows_h) + gap * (len(rows_h) - 1)
    H = round(h_a + sep + h_b + 0.04, 2)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    placed = []
    tau = r'$\tau$'

    def box(x0, x1, top, h, title, lines=(), fill='white', bold_lines=False):
        y = top - h
        ax.add_patch(Rectangle((x0, y), x1 - x0, h, fc=fill, ec=ink, lw=lw, joinstyle='miter'))
        tl = title.split('\n') if title else []
        tp = ts * 1.24 / 72
        block = len(tl) * tp + (0.035 if tl and lines else 0) + len(lines) * pitch
        cy = y + h / 2 + block / 2
        objs = []
        for t in tl:
            objs.append(ax.text((x0 + x1) / 2, cy - tp / 2, t, ha='center', va='center', fontsize=ts,
                                fontweight='bold'))
            cy -= tp
        if tl and lines:
            cy -= 0.035
        for t in lines:
            objs.append(ax.text((x0 + x1) / 2, cy - pitch / 2, t, ha='center', va='center', fontsize=bs,
                                fontweight='bold' if bold_lines else 'normal'))
            cy -= pitch
        placed.append(((x0, y, x1 - x0, h), objs))

    def line(pts):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=ink, lw=lw, solid_capstyle='butt', solid_joinstyle='miter')

    def arrow(*pts):
        """Orthogonal connector through the given points, with an arrowhead at the last one."""
        if len(pts) > 2:
            line(pts[:-1])
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>,head_length=0.42,head_width=0.2',
                                     mutation_scale=9, lw=lw, color=ink, shrinkA=0, shrinkB=0,
                                     joinstyle='miter', capstyle='butt'))

    # geometry (inches); the strip left of XA0 carries the connector from the climate data to the trends
    XA0, XA1 = 0.14, 1.98
    XAH = (XA0 + XA1) / 2
    XS0, XS1, XE0, XE1 = XA0, XAH - 0.05, XAH + 0.05, XA1
    XL0, XL1, XR0, XR1 = 2.27, 4.24, 4.48, 6.45
    cxS, cxE, cxL, cxR = (XS0 + XS1) / 2, (XE0 + XE1) / 2, (XL0 + XL1) / 2, (XR0 + XR1) / 2
    tops, t = [], H - top_a
    for h in rows_h:
        tops.append(t)
        t -= h + gap
    bot = [tp - h for tp, h in zip(tops, rows_h)]
    mid = [tp - h / 2 for tp, h in zip(tops, rows_h)]

    # ---- (a) workflow
    ax.text(0.02, H - 0.03, '(a) Analytical workflow', ha='left', va='top', fontsize=8.5, fontweight='bold')
    y_head = H - 0.36
    for x0, x1, label in ((XA0, XA1, 'Module A: climate'), (XL0, XR1, 'Module B: land cover')):
        ax.text(x0, y_head + 0.035, label, ha='left', va='bottom', fontsize=8, fontweight='bold')
        ax.plot([x0, x1], [y_head, y_head], color='#7f7f7f', lw=0.5)

    box(XA0, XA1, tops[0], rows_h[0], 'Climate data [2.2]', ['P: CHIRPS v2.0, monthly', 'PET: TerraClimate, monthly'])
    hw = 0.40
    wb_top = mid[1] + hw / 2
    box(XE0, XE1, wb_top, hw, 'Water balance', [f'D = P {MINUS} PET'])
    box(XS0, XS1, tops[2], rows_h[2], r'SPI-$\mathbfit{k}$ [2.3]', ['gamma fit to', r'$k$-month P'])
    box(XE0, XE1, tops[2], rows_h[2], r'SPEI-$\mathbfit{k}$ [2.3]', ['Pearson III fit to', r'$k$-month D'])
    box(XA0, XA1, tops[3], rows_h[3], 'Climate diagnostics [2.3]',
        ['AED contribution; frequency gap', 'trends in SPEI-12, SPI-12, P, PET', 'modified Mann–Kendall; Theil–Sen'])
    box(XA0, XA1, tops[4], rows_h[4], 'Drought trends and\nAED contribution [3.1]', fill=fill_out)
    arrow((cxS, bot[0]), (cxS, tops[2]))                  # precipitation alone -> SPI
    arrow((cxE, bot[0]), (cxE, wb_top))                   # P and PET -> water balance
    arrow((cxE, wb_top - hw), (cxE, tops[2]))             # water balance -> SPEI
    arrow((cxS, bot[2]), (cxS, tops[3]))
    arrow((cxE, bot[2]), (cxE, tops[3]))
    arrow((XA0, mid[0]), (0.05, mid[0]), (0.05, mid[3]), (XA0, mid[3]))   # annual P and PET -> trends
    arrow(((XA0 + XA1) / 2, bot[3]), ((XA0 + XA1) / 2, tops[4]))
    arrow((XE1, mid[2]), (XL0, mid[2]))                   # SPEI -> Module B
    ax.text((XE1 + XL0) / 2, mid[2] + 0.045, 'SPEI', ha='center', va='bottom', fontsize=7)

    box(XL0, XL1, tops[0], rows_h[0], 'Land-cover data [2.2]', ['GLC_FCS30D, 30 m', '9 maps, 1985–2022'])
    box(XL0, XL1, tops[1], rows_h[1], 'Transition and stable pixels [2.4]',
        ['9 pathways in 3 categories', 'masks at 300 m and 1 km;', 'primary: point sampling to 0.045°'])
    box(XL0, XL1, tops[2], rows_h[2], 'Antecedent SPEI [2.5]',
        [f'lagged: interval {tau} {MINUS} 1 (primary)', f'cumulative: intervals 1 to {tau} {MINUS} 1'])
    box(XL0, XL1, tops[3], rows_h[3], 'Effect sizes [2.5]',
        [r'Cohen’s $d$ for each IPCC region,', 'pathway, interval and timescale', '(point sampling, both windows)'])
    box(XL0, XL1, tops[4], rows_h[4], 'Regional indices and\ninterval-level inference [2.6, 2.7]',
        ['RSI and DRA (transition-pixel-weighted)', 'mixed-effects contrast (lagged)',
         r'early–late $\Delta d$ (both windows),', 'Welch’s tests, FDR-corrected'])
    for k in range(3):
        arrow((cxL, bot[k]), (cxL, tops[k + 1]))

    box(XR0, XR1, tops[0], rows_h[0], 'Human-pressure data [2.2]',
        ['population, travel time,', 'livestock, protected areas'])
    box(XR0, XR1, tops[1], tops[1] - bot[2], 'Alternative estimators [2.8]',
        ['area weighting and', 'area thresholding:', 'area fractions of the masks,', 'SPEI-12, lagged window', '',
         'coarsened exact matching:', 'human-pressure strata,', r'all $k$, lagged, 1990–2020'])
    box(XR0, XR1, tops[4], rows_h[4], 'Continental aggregation [2.6, 2.7]',
        ['two-stage means, equal unit weights', 'BCa CIs; equivalence tests', 'Welch’s and permutation tests'])
    arrow((cxR, bot[0]), (cxR, tops[1]))                  # covariates -> matching
    arrow((XL1, mid[1]), (XR0, mid[1]))                   # transition and from-class masks
    arrow((XL1, mid[2]), (XR0, mid[2]))                   # antecedent SPEI -> all three estimators
    arrow((cxL, bot[3]), (cxL, tops[4]))                  # primary effect sizes -> regional and interval-level
    x_alt, x_eff = cxR + 0.49, cxR - 0.49                 # two entry points on the continental box
    arrow((x_alt, bot[2]), (x_alt, tops[4]))              # alternative effect sizes -> continental aggregation
    ax.add_patch(FancyArrowPatch((XL1, mid[3]), (x_eff, tops[4]),   # primary effect sizes, rounded elbow
                                 connectionstyle=f'angle,angleA=0,angleB=90,rad={0.2 * DPI:.0f}',  # radius in pixels at 600 dpi
                                 arrowstyle='-|>,head_length=0.42,head_width=0.2', mutation_scale=9, lw=lw,
                                 color=ink, shrinkA=0, shrinkB=0, capstyle='butt'))
    box(XL0, XR1, tops[5], rows_h[5], '',
        ['Recovery–degradation asymmetry, temporal change and robustness [3.2–3.4]'], fill=fill_out,
        bold_lines=True)
    arrow((cxL, bot[4]), (cxL, tops[5]))
    arrow((cxR, bot[4]), (cxR, tops[5]))

    y_sep = H - h_a - sep / 2
    ax.plot([0.0, W], [y_sep, y_sep], color='#9a9a9a', lw=0.45)

    # ---- (b) comparison design
    yb = y_sep - sep / 2
    ax.text(0.02, yb - 0.01, r'(b) Comparison design for transition interval $\mathbfit{\tau}$', ha='left',
            va='top', fontsize=8.5, fontweight='bold')
    years = ['1985', '1990', '1995', '2000', '2005', '2010', '2015', '2020', '2022']
    tx0, cw, th, it = 0.20, 0.49, 0.25, 5                 # it: index of the example interval (tau = 6)
    ty = yb - 0.60                                        # bottom of the interval cells
    for i in range(8):
        hl = i == it
        ax.add_patch(Rectangle((tx0 + i * cw, ty), cw, th, fc=fill_hl if hl else 'white', ec=ink, lw=lw))
        ax.text(tx0 + i * cw + cw / 2, ty + th / 2, r'$\mathbfit{\tau}$ = 6' if hl else str(i + 1), ha='center',
                va='center', fontsize=7, fontweight='bold' if hl else 'normal')
    for i, yr in enumerate(years):
        ax.text(tx0 + i * cw, ty + th + 0.04, yr, ha='center', va='bottom', fontsize=7, color=note)
    ax.text(tx0 - 0.02, ty + th / 2, '', ha='right', va='center', fontsize=7)

    def bracket(xa, xb, y, drop=0.05):
        line([(xa, y + drop), (xa, y), (xb, y), (xb, y + drop)])

    lx0, lx1 = tx0 + (it - 1) * cw + 0.015, tx0 + it * cw - 0.015
    bracket(lx0, lx1, ty - 0.09)
    ax.text((lx0 + lx1) / 2, ty - 0.13, f'lagged window: interval {tau} {MINUS} 1 (primary)', ha='center',
            va='top', fontsize=7)
    bracket(tx0 + 0.015, lx1, ty - 0.37)
    ax.text((tx0 + 0.015 + lx1) / 2, ty - 0.41, f'cumulative window: intervals 1 to {tau} {MINUS} 1',
            ha='center', va='top', fontsize=7)

    px, bw_, bh_ = 5.12, 0.27, 0.23
    ex = px + 1.0
    ax.text((px + ex + bw_) / 2, yb - 0.25, f'within interval {tau}', ha='center', va='center', fontsize=7,
            style='italic')
    for yy, lab, end, fc_end in ((yb - 0.58, 'Transition pixel', 'B', fill_hl), (yb - 0.93, 'Stable pixel', 'A', 'white')):
        ax.text(px - 0.1, yy + bh_ / 2, lab, ha='right', va='center', fontsize=7)
        for xx, lt, fc in ((px, 'A', 'white'), (ex, end, fc_end)):
            ax.add_patch(Rectangle((xx, yy), bw_, bh_, fc=fc, ec=ink, lw=lw))
            ax.text(xx + bw_ / 2, yy + bh_ / 2, lt, ha='center', va='center', fontsize=7.5, fontweight='bold')
        arrow((px + bw_ + 0.05, yy + bh_ / 2), (ex - 0.05, yy + bh_ / 2))
    for xx, lt in ((px, 'start'), (ex, 'end')):
        ax.text(xx + bw_ / 2, yb - 0.98, lt, ha='center', va='top', fontsize=7, color=note)

    yf = yb - h_b + 0.20
    ax.text(W / 2, yf + 0.03, r'$d\;=\;(\mu_S\;-\;\mu_T)\;/\;\sigma_{\mathrm{pooled}}$,  where $\mu_S$ and $\mu_T$ are the '
            'mean antecedent SPEI of stable and transition pixels', ha='center', va='bottom', fontsize=7.5)
    ax.text(W / 2, yf - 0.02, r'$d$ < 0: transition pixels had wetter antecedent conditions than stable '
            r'pixels; $d$ > 0: drier', ha='center', va='top', fontsize=7)

    # every label must sit inside its box
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for (x, y, w, h), objs in placed:
        for o in objs:
            bb = o.get_window_extent(rend)
            assert (bb.x0 / fig.dpi >= x + 0.035 and bb.x1 / fig.dpi <= x + w - 0.035 and
                    bb.y0 / fig.dpi >= y + 0.02 and bb.y1 / fig.dpi <= y + h - 0.02), o.get_text()
    save(fig, out, 'Fig2_workflow.png')


# --------------------------------------------------------------------------------------------
# Fig. 5  Continental effect sizes under five estimators (SPEI-12)
# --------------------------------------------------------------------------------------------
def fig5_forest(ci, out, bound=0.10):
    order = ['point_lagged', 'point_cumul', 'area_weighted', 'area_threshold', 'cem_matched']
    label = {'point_lagged': 'Point sampling, lagged (primary)',
             'point_cumul': 'Point sampling, cumulative',
             'area_weighted': 'Area weighting',
             'area_threshold': 'Area thresholding',
             'cem_matched': 'Coarsened exact matching'}
    rows = []
    for c in CATS:
        for e in order:
            if c in ci.get(e, {}):
                m, lo, hi = ci[e][c][:3]
                rows.append((c, e, m, lo, hi))

    fig = plt.figure(figsize=(6.5, 4.8))
    ax = fig.add_axes([0.305, 0.15, 0.39, 0.79])
    ypos, y, prev = [], 0.0, None
    for c, *_ in rows:
        if prev is not None and c != prev:
            y -= 0.9
        ypos.append(y)
        y -= 1
        prev = c

    ax.axvspan(-bound, bound, color='#9E9E9E', alpha=0.16, zorder=0, lw=0)
    ax.axvline(0, color='#222', lw=0.9, zorder=1)
    for yi, (c, e, m, lo, hi) in zip(ypos, rows):
        ax.plot([lo, hi], [yi, yi], color=CATCOL[c], lw=1.5, zorder=3, solid_capstyle='butt')
        for xv in (lo, hi):
            ax.plot([xv, xv], [yi - 0.2, yi + 0.2], color=CATCOL[c], lw=1.1, zorder=3)
        ax.plot(m, yi, 'o', color=CATCOL[c], ms=4.8, zorder=4, mec='white', mew=0.6)
        ax.text(1.03, yi, f'{fmt(m, 3)}  ({fmt(lo, 3)}, {fmt(hi, 3)})', va='center', ha='left',
                fontsize=7.5, color='#222', transform=ax.get_yaxis_transform())

    ax.set_yticks(ypos)
    ax.set_yticklabels([label[e] for _, e, *_ in rows], fontsize=7.8)
    for c in CATS:
        ys = [yy for yy, r in zip(ypos, rows) if r[0] == c]
        if ys:
            ax.text(-0.765, max(ys) + 0.62, c, fontsize=8.5, fontweight='bold', color=CATCOL[c],
                    ha='left', va='bottom', transform=ax.get_yaxis_transform(), clip_on=False)
    xt = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]
    ax.set_xlim(-0.34, 0.30)
    ax.set_ylim(min(ypos) - 0.8, max(ypos) + 1.25)
    ax.set_xticks(xt)
    ax.set_xticklabels(tick_labels(xt))
    ax.set_xlabel("Continental mean Cohen's d (95% BCa confidence interval)", fontsize=8)
    ax.text(0, max(ypos) + 0.95, f'equivalence bound ±{bound:.2f}', fontsize=7, color='#444',
            ha='center', va='center', bbox=dict(fc='white', ec='none', pad=0.5))
    ax.text(1.03, max(ypos) + 0.95, 'mean  (95% CI)', fontsize=7.5, color='#222', ha='left', va='center',
            fontweight='bold', transform=ax.get_yaxis_transform())
    trans = ax.get_xaxis_transform()
    ax.annotate('', xy=(-0.33, -0.118), xytext=(-0.012, -0.118), xycoords=trans, textcoords=trans,
                annotation_clip=False, arrowprops=dict(arrowstyle='->', color='#1A3E8B', lw=0.8))
    ax.text(-0.17, -0.132, 'wetter than stable pixels', color='#1A3E8B', fontsize=7.2, ha='center',
            va='top', transform=trans, clip_on=False)
    ax.annotate('', xy=(0.29, -0.118), xytext=(0.012, -0.118), xycoords=trans, textcoords=trans,
                annotation_clip=False, arrowprops=dict(arrowstyle='->', color='#8B1A1A', lw=0.8))
    ax.text(0.15, -0.132, 'drier than stable pixels', color='#8B1A1A', fontsize=7.2, ha='center',
            va='top', transform=trans, clip_on=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.grid(axis='x', alpha=0.35, ls=':', lw=0.6)
    save(fig, out, 'Fig5_forest.png')
    for c, e, m, lo, hi in rows:
        print(f'   {c:13s} {label[e]:34s} {m:+.3f} ({lo:+.3f}, {hi:+.3f})')


# --------------------------------------------------------------------------------------------
# Fig. 6  Effect size by pathway and region (SPEI-12), lagged and cumulative windows
# --------------------------------------------------------------------------------------------
HEAT_ROWS = [('Degradation', [('FST_SHR', 'FST→SHR'), ('SHR_GRS', 'SHR→GRS'),
                              ('FST_CRP', 'FST→CRP'), ('GRS_BAL', 'GRS→BAL')]),
             ('Recovery', [('SHR_FST', 'SHR→FST'), ('GRS_SHR', 'GRS→SHR'),
                           ('BAL_GRS', 'BAL→GRS')]),
             ('Agricultural', [('AGEXPANSION', 'SHR/GRS/BAL→CRP'),
                               ('CRP_ABANDONMENT', 'CRP→FST/SHR/GRS')])]


def heat_grid(df):
    """Transition-pixel-weighted mean d over intervals (Eq. 13), and a one-sample t-test flag."""
    s = df[df.spei_timescale == 'spei_12']
    val, sig = {}, {}
    for _, items in HEAT_ROWS:
        for code, _ in items:
            for r in REG:
                g = s[(s.region == r) & (s.transition == code)]
                if len(g) == 0:
                    val[(code, r)], sig[(code, r)] = np.nan, False
                    continue
                val[(code, r)] = float(np.average(g.cohens_d, weights=g.n_trans))
                sig[(code, r)] = bool(len(g) >= 3 and stats.ttest_1samp(g.cohens_d, 0)[1] < 0.05)
    return val, sig


def fig6_heatmap(lag, cum, out):
    panels = [('(a) Lagged window', heat_grid(lag)), ('(b) Cumulative window', heat_grid(cum))]
    order = [c for _, items in HEAT_ROWS for c in items]
    nrow = len(order)
    W, H = 7.0, 4.0
    fig = plt.figure(figsize=(W, H))
    left, pw, gap, bottom, ph = 1.5, 2.33, 0.14, 0.5, 3.0
    axes = [fig.add_axes([left / W, bottom / H, pw / W, ph / H]),
            fig.add_axes([(left + pw + gap) / W, bottom / H, pw / W, ph / H])]
    cax = fig.add_axes([(left + 2 * pw + gap + 0.33) / W, (bottom + 0.35) / H, 0.09 / W, (ph - 0.7) / H])
    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0.0, vmax=0.6)
    cmap = plt.get_cmap('RdBu_r')

    for ax, (title, (val, sig)) in zip(axes, panels):
        for i, (code, _) in enumerate(order):
            y = nrow - 1 - i
            for j, r in enumerate(REG):
                v = val[(code, r)]
                if np.isnan(v):
                    ax.add_patch(Rectangle((j, y), 1, 1, facecolor='#EFEFEF', edgecolor='#BDBDBD',
                                           hatch='//////', linewidth=0.0))
                    ax.add_patch(Rectangle((j, y), 1, 1, fill=False, edgecolor='white', linewidth=0.9))
                    continue
                ax.add_patch(Rectangle((j, y), 1, 1, facecolor=cmap(norm(v)), edgecolor='white',
                                       linewidth=0.9))
                ax.text(j + 0.5, y + 0.5, fmt(v) + ('*' if sig[(code, r)] else ''),
                        ha='center', va='center', fontsize=7.4,
                        color='white' if abs(v) > 0.33 else '#111',
                        fontweight='bold' if sig[(code, r)] else 'normal')
        ax.axhline(nrow - 4, color='#222', lw=1.0)
        ax.axhline(nrow - 7, color='#222', lw=1.0)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, nrow)
        ax.set_xticks(np.arange(5) + 0.5)
        ax.set_xticklabels(REG, fontsize=7.8)
        ax.set_yticks(np.arange(nrow) + 0.5)
        ax.set_yticklabels([lab for _, lab in order][::-1], fontsize=7.8)
        ax.tick_params(length=0, pad=2.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc='left', fontsize=8.5, fontweight='bold', pad=5)
    axes[1].set_yticklabels([])

    for cat, y0, y1 in [('Degradation', nrow - 4, nrow), ('Recovery', nrow - 7, nrow - 4),
                        ('Agricultural', 0, nrow - 7)]:
        x = -1.27 / pw          # axes fraction left of the row labels
        axes[0].plot([x, x], [y0 + 0.1, y1 - 0.1], color=CATCOL[cat], lw=2.2, clip_on=False,
                     transform=axes[0].get_yaxis_transform(), solid_capstyle='butt')
        axes[0].text(x - 0.035, (y0 + y1) / 2, cat, transform=axes[0].get_yaxis_transform(),
                     rotation=90, ha='right', va='center', fontsize=7.5, fontweight='bold',
                     color=CATCOL[cat])

    ticks = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks(ticks)
    cb.set_ticklabels(tick_labels(ticks))
    cb.ax.tick_params(labelsize=7, length=2, width=0.6)
    cb.outline.set_linewidth(0.6)
    cb.set_label("Cohen's d", fontsize=7.8, labelpad=3)
    cb.ax.yaxis.set_label_position('left')
    cax.text(0.5, 1.04, 'drier', transform=cax.transAxes, ha='center', va='bottom',
             fontsize=7.5, color='#8B1A1A', fontweight='bold')
    cax.text(0.5, -0.04, 'wetter', transform=cax.transAxes, ha='center', va='top',
             fontsize=7.5, color='#1A3E8B', fontweight='bold')

    ky = 0.1
    fig.patches.append(Rectangle((left / W, (ky - 0.045) / H), 0.16 / W, 0.12 / H,
                                 transform=fig.transFigure, facecolor='#EFEFEF', edgecolor='#999',
                                 hatch='//////', lw=0.5))
    fig.text((left + 0.22) / W, (ky + 0.015) / H, 'no interval with at least 30 transition pixels',
             va='center', fontsize=7.2)
    fig.text((left + pw + gap) / W, (ky + 0.015) / H, '* p < 0.05, one-sample t-test across intervals',
             va='center', fontsize=7.2)
    save(fig, out, 'Fig6_heatmap.png')


# --------------------------------------------------------------------------------------------
# Fig. 7  Recovery Suppression Index (Eq. 15) and Degradation-Recovery Asymmetry (Eq. 16)
# --------------------------------------------------------------------------------------------
def rsi(df, region, ts):
    g = df[(df.spei_timescale == f'spei_{ts}') & (df.region == region) &
           df.transition.isin(RECOVERY)]
    if len(g) < 3:
        return np.nan, np.nan, len(g)
    return (float(np.average(g.cohens_d, weights=g.n_trans)),
            float(stats.ttest_1samp(g.cohens_d, 0.0)[1]), len(g))


def dra(df, region, ts):
    s = df[(df.spei_timescale == f'spei_{ts}') & (df.region == region)]
    dg, rc = s[s.category == 'Degradation'], s[s.category == 'Recovery']
    if len(dg) < 2 or len(rc) < 2:
        return np.nan, np.nan, 0
    md = np.average(dg.cohens_d, weights=dg.n_trans)
    mr = np.average(rc.cohens_d, weights=rc.n_trans)
    return (float(md - mr),
            float(stats.ttest_ind(dg.cohens_d, rc.cohens_d, equal_var=False)[1]), 0)


def fig7_rsi_dra(lag, cum, out):
    fig = plt.figure(figsize=(7.0, 4.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1, 1], hspace=0.5, wspace=0.42,
                          left=0.085, right=0.985, top=0.94, bottom=0.165)

    ax = fig.add_subplot(gs[:, 0])
    vals, ps, ns = zip(*[rsi(lag, r, 12) for r in REG])
    ax.bar(range(5), vals, color='#3b6fa8', edgecolor='#1c3f63', width=0.62, lw=0.6)
    for i, (v, p) in enumerate(zip(vals, ps)):
        ax.text(i, v + (-0.012 if v < 0 else 0.008), fmt(v, 3) + ('*' if p < 0.05 else ''),
                ha='center', va='top' if v < 0 else 'bottom', fontsize=7.2, fontweight='bold')
    ax.axhline(0, color='#222', lw=0.8)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'{r}\nn = {n}' for r, n in zip(REG, ns)], fontsize=7.2)
    ax.set_ylabel("Recovery Suppression Index (Cohen's d)", fontsize=7.8)
    ax.set_title('(a) RSI, lagged window, SPEI-12', loc='left', fontsize=8.3, fontweight='bold')
    yt = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1]
    ax.set_ylim(-0.60, 0.10)
    ax.set_yticks(yt)
    ax.set_yticklabels(tick_labels(yt))
    ax.text(0.97, 0.03, '* p < 0.05', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.2,
            bbox=dict(fc='white', ec='#999', lw=0.5, pad=1.5))
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    print('   RSI, lagged, SPEI-12:',
          ', '.join(f'{r} {v:+.3f} (p = {p:.3f}, n = {n})' for r, v, p, n in zip(REG, vals, ps, ns)))

    panels = [(gs[0, 1], rsi, lag, '(b1) RSI, lagged window', "RSI (Cohen's d)", False),
              (gs[0, 2], rsi, cum, '(b2) RSI, cumulative window', None, False),
              (gs[1, 1], dra, lag, '(b3) DRA, lagged window', "DRA (difference in d)", True),
              (gs[1, 2], dra, cum, '(b4) DRA, cumulative window', None, True)]
    for spec, fn, df, title, ylab, bottom_row in panels:
        a = fig.add_subplot(spec)
        for r in REG:
            ys, sig = [], []
            for ts in TIMESCALES:
                v, p, _ = fn(df, r, ts)
                ys.append(v)
                sig.append(bool(np.isfinite(p) and p < 0.05))
            a.plot(range(4), ys, '-o', color=REGCOL[r], ms=3.0, lw=1.0, label=REGNAME[r])
            for i, (yv, s_) in enumerate(zip(ys, sig)):
                if s_:
                    a.plot(i, yv, marker='*', color=REGCOL[r], ms=8.5, mec='#222', mew=0.4)
        a.axhline(0, color='#888', lw=0.7, ls=':')
        a.set_xticks(range(4))
        a.set_xticklabels([str(t) for t in TIMESCALES])
        a.set_xlim(-0.3, 3.3)
        if bottom_row:
            a.set_xlabel('SPEI timescale (months)', fontsize=7.5, labelpad=2)
        a.set_title(title, loc='left', fontsize=8.1, fontweight='bold')
        yt = a.get_yticks()
        a.set_yticks(yt)
        a.set_yticklabels([fmt(t, 2, plus=False) for t in yt])
        if ylab:
            a.set_ylabel(ylab, fontsize=7.8)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        a.grid(axis='y', alpha=0.3, ls=':', lw=0.6)

    handles = [Line2D([0], [0], color=REGCOL[r], marker='o', ms=3.0, lw=1.0, label=REGNAME[r])
               for r in REG]
    handles.append(Line2D([0], [0], color='white', marker='*', ms=8.5, mec='#222', mfc='#bbb',
                          label='p < 0.05'))
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=7.4,
               bbox_to_anchor=(0.53, 0.0), handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
    save(fig, out, 'Fig7_RSI_DRA.png')


# --------------------------------------------------------------------------------------------
# Fig. S2  Recovery-degradation contrast from the linear mixed-effects model (Table S7)
# --------------------------------------------------------------------------------------------
# beta_1 of Eq. 18 (recovery minus degradation, Cohen's d units), 95% confidence interval and p-value, lagged window
TABLE_S7 = {12: (-0.162, -0.314, -0.010, '0.037'), 24: (-0.171, -0.306, -0.037, '0.013'),
            36: (-0.188, -0.313, -0.064, '0.003'), 60: (-0.217, -0.341, -0.094, '0.0006')}


def figS2_mixed(out, est=TABLE_S7):
    fig = plt.figure(figsize=(6.5, 2.6))
    ax = fig.add_axes([0.12, 0.22, 0.50, 0.70])
    ys = np.arange(len(TIMESCALES))[::-1]
    for y, k in zip(ys, TIMESCALES):
        b, lo, hi, p = est[k]
        ax.plot([lo, hi], [y, y], color='#1E4E79', lw=1.4, solid_capstyle='butt')
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.12, y + 0.12], color='#1E4E79', lw=1.0)
        ax.plot(b, y, 'o', color='#1E4E79', ms=5.5, mec='white', mew=0.8, zorder=3)
        ax.text(1.03, y, f'{fmt(b, 3)}  ({fmt(lo, 3)}, {fmt(hi, 3)})   p = {p}', transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=7.5)
    ax.axvline(0, color='#333333', lw=0.8, ls='--')
    ax.set_yticks(ys)
    ax.set_yticklabels([f'SPEI-{k}' for k in TIMESCALES])
    ax.set_ylim(-0.6, len(TIMESCALES) - 0.4)
    ax.set_xlim(-0.36, 0.02)
    xt = [-0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0]
    ax.set_xticks(xt)
    ax.set_xticklabels(tick_labels(xt, 2))
    ax.set_xlabel(r'Recovery minus degradation, $\beta_1$ of Eq. 18 (Cohen’s $d$ units)', fontsize=8)
    ax.text(1.03, len(TIMESCALES) - 0.45, r'$\beta_1$  (95% CI)', transform=ax.get_yaxis_transform(), ha='left',
            va='bottom', fontsize=7.5, fontweight='bold')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', ls=':', lw=0.5, alpha=0.6)
    save(fig, out, 'FigS2_mixed.png')


# --------------------------------------------------------------------------------------------
# Fig. S3  Early and late period means (cumulative window, SPEI-12), Benjamini-Hochberg q
# --------------------------------------------------------------------------------------------
def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(np.argsort(p), 1))):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def figS3_decadal(cum, out):
    s = cum[cum.spei_timescale == 'spei_12'].copy()
    s['start'] = s.interval.str[:4].astype(int)
    rows = []
    for r in REG:
        for c in CATS:
            g = s[(s.region == r) & (s.category == c)]
            early, late = g[g.start < 2005].cohens_d, g[g.start >= 2005].cohens_d
            p = stats.ttest_ind(early, late, equal_var=False)[1]
            rows.append((r, c, early.mean(), late.mean(), p))
    res = pd.DataFrame(rows, columns=['region', 'category', 'early', 'late', 'p'])
    res['q'] = bh(res.p.values)

    fig = plt.figure(figsize=(7.0, 3.3))
    ax = fig.add_axes([0.085, 0.2, 0.905, 0.66])
    x0, ticks, labels, centers = 0, [], [], []
    for r in REG:
        xs = []
        for c in CATS:
            row = res[(res.region == r) & (res.category == c)].iloc[0]
            ax.bar(x0 - 0.2, row.early, 0.38, color=CATCOL[c], alpha=0.35, edgecolor='#555', lw=0.4)
            ax.bar(x0 + 0.2, row.late, 0.38, color=CATCOL[c], alpha=0.95, edgecolor='#222', lw=0.4)
            if row.q < 0.05:
                ax.text(x0, max(row.early, row.late, 0) + 0.04, '*', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')
            ticks.append(x0)
            labels.append(SHORT[c])
            xs.append(x0)
            x0 += 1.1
        centers.append((np.mean(xs), r))
        x0 += 0.7
    ax.axhline(0, color='#222', lw=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.tick_params(axis='x', length=0)
    for xc, r in centers:
        ax.text(xc, -0.13, REGNAME[r], transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=7.8, fontweight='bold')
    ax.set_ylabel("Mean Cohen's d\n(cumulative window, SPEI-12)", fontsize=7.8)
    yt = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
    ax.set_yticks(yt)
    ax.set_yticklabels(tick_labels(yt))
    ax.set_ylim(-1.0, 0.7)
    ax.set_xlim(-0.7, x0 - 0.7 - 0.4)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='y', alpha=0.3, ls=':', lw=0.6)
    handles = [Patch(facecolor='#999', alpha=0.35, edgecolor='#555', label='Early period (1990–2004)'),
               Patch(facecolor='#555', alpha=0.95, edgecolor='#222', label='Late period (2005–2022)')]
    handles += [Patch(facecolor=CATCOL[c], label=c) for c in CATS]
    ax.legend(handles=handles, loc='lower left', fontsize=7.2, frameon=False, ncol=5,
              bbox_to_anchor=(0.0, 1.02), borderaxespad=0.2, handlelength=1.4, columnspacing=1.2)
    ax.text(1.0, 1.04, '* q < 0.05', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.2)
    save(fig, out, 'FigS3_decadal.png')
    print('   shifts with q < 0.05:')
    print(res[res.q < 0.05].round(3).to_string(index=False))


# --------------------------------------------------------------------------------------------
# Fig. S4  Distribution of interval-level effect sizes (lagged window, SPEI-12)
# --------------------------------------------------------------------------------------------
def figS4_boxplot(lag, out):
    s = lag[lag.spei_timescale == 'spei_12']
    fig = plt.figure(figsize=(7.0, 3.5))
    ax = fig.add_axes([0.085, 0.235, 0.905, 0.66])
    pos, data, cols, labs, centers = [], [], [], [], []
    x0 = 0
    for r in REG:
        xs = []
        for c in CATS:
            v = s[(s.region == r) & (s.category == c)].cohens_d.values
            pos.append(x0)
            data.append(v)
            cols.append(CATCOL[c])
            labs.append(f'{SHORT[c]}\nn = {len(v)}')
            xs.append(x0)
            x0 += 1
        centers.append((np.mean(xs), r))
        x0 += 0.8
    bp = ax.boxplot(data, positions=pos, widths=0.66, patch_artist=True, showmeans=True,
                    whis=1.5, showfliers=True,
                    flierprops=dict(marker='o', ms=2.0, mfc='#666', mec='none', alpha=0.8),
                    meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='k', ms=3.2,
                                   mew=0.6),
                    medianprops=dict(color='k', lw=1.0), boxprops=dict(lw=0.6),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for patch, c in zip(bp['boxes'], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor('#222')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set_xticks(pos)
    ax.set_xticklabels(labs, fontsize=6.9)
    ax.tick_params(axis='x', length=0)
    for xc, r in centers:
        ax.text(xc, -0.17, REGNAME[r], transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=7.8, fontweight='bold')
    ax.set_ylabel("Interval-level Cohen's d\n(lagged window, SPEI-12)", fontsize=7.8)
    yt = ax.get_yticks()
    ax.set_yticks(yt)
    ax.set_yticklabels(tick_labels(yt))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='y', alpha=0.3, ls=':', lw=0.6)
    handles = [Patch(facecolor=CATCOL[c], alpha=0.75, edgecolor='#222', label=c) for c in CATS]
    ax.legend(handles=handles, loc='lower left', fontsize=7.2, ncol=3, bbox_to_anchor=(0.0, 1.02),
              frameon=False, borderaxespad=0.2, handlelength=1.4)
    save(fig, out, 'FigS4_boxplot.png')
    rec = s[s.category == 'Recovery']
    print('   recovery median / mean by region:',
          ', '.join(f'{r} {rec[rec.region == r].cohens_d.median():+.3f} / '
                    f'{rec[rec.region == r].cohens_d.mean():+.3f}' for r in REG))


# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lagged', default='TableB2_LAGGED_per_interval_STRICTER.csv')
    ap.add_argument('--cumulative', default='TableB2_CUMULATIVE_per_interval_STRICTER.csv')
    ap.add_argument('--twostage', default=None,
                    help='optional JSON with continental means and BCa intervals')
    ap.add_argument('--out', default='figures')
    ap.add_argument('--only', default=None, help='comma-separated subset, e.g. fig6,fig7')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lag, cum = load_effects(args.lagged), load_effects(args.cumulative)
    ci = TABLE_S9
    if args.twostage:
        with open(args.twostage, encoding='utf8') as f:
            ci = json.load(f)['CI']['spei_12']
    jobs = {'fig2': lambda: fig2_workflow(args.out),
            'fig5': lambda: fig5_forest(ci, args.out),
            'fig6': lambda: fig6_heatmap(lag, cum, args.out),
            'fig7': lambda: fig7_rsi_dra(lag, cum, args.out),
            'figS2': lambda: figS2_mixed(args.out),
            'figS3': lambda: figS3_decadal(cum, args.out),
            'figS4': lambda: figS4_boxplot(lag, args.out)}
    for name in (args.only.split(',') if args.only else jobs):
        jobs[name]()


if __name__ == '__main__':
    main()
