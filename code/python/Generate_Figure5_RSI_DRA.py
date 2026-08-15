"""
============================================================
FIGURE 5: Recovery Suppression Index and Degradation-Recovery
Asymmetry, by region and accumulation timescale
============================================================

Builds Figure 5 directly from the per-cell effect-size tables, so every
significance marker is computed from the data it plots and printed to
the console for checking.

Panels
------
  (a)  RSI by IPCC region, lagged window, SPEI-12, values labelled
  (b1) RSI across timescales, lagged window
  (b2) RSI across timescales, cumulative window
  (b3) DRA across timescales, lagged window
  (b4) DRA across timescales, cumulative window

Definitions follow the manuscript: RSI (Eq. 7) is the transition-pixel
weighted mean Cohen's d over the three recovery pathways within a
region; DRA (Eq. 8) is the difference between the degradation and
recovery weighted means. Significance is a one-sample t-test against
zero for RSI, and Welch's t-test between categories for DRA.

INPUTS
------
  TableB2_LAGGED_per_interval_STRICTER.csv
  TableB2_CUMULATIVE_per_interval_STRICTER.csv

OUTPUT
------
  Figure5_RSI_DRA.png   (300 dpi)
  plus a printed table of every plotted value with its p-value

RUNTIME: seconds. No rasters required.
============================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- configuration ----------------
IN_DIR = r'.'
OUT_DIR = r'.'
LAGGED = os.path.join(IN_DIR, 'TableB2_LAGGED_per_interval_STRICTER.csv')
CUMUL = os.path.join(IN_DIR, 'TableB2_CUMULATIVE_per_interval_STRICTER.csv')

RECOVERY_PATHWAYS = ['SHR_FST', 'GRS_SHR', 'BAL_GRS']
TIMESCALES = [12, 24, 36, 60]
REGIONS = [('MED', 'Mediterranean', '#d62728'),
           ('SAH', 'Sahara-Sahel', '#ff7f0e'),
           ('WAF', 'West Africa', '#2ca02c'),
           ('EAF', 'East Africa', '#1f77b4'),
           ('SAF', 'Southern Africa', '#9467bd')]


def rsi(df, region, ts):
    """Recovery Suppression Index (Eq. 7) and its one-sample t-test p-value."""
    g = df[(df.spei_timescale == f'spei_{ts}') & (df.region == region) &
           df.transition.isin(RECOVERY_PATHWAYS) & df.cohens_d.notna()]
    if len(g) < 3:
        return np.nan, np.nan
    return (float(np.average(g.cohens_d, weights=g.n_trans)),
            float(stats.ttest_1samp(g.cohens_d, 0.0)[1]))


def dra(df, region, ts):
    """Degradation-Recovery Asymmetry (Eq. 8) and its Welch t-test p-value."""
    s = df[(df.spei_timescale == f'spei_{ts}') & (df.region == region) & df.cohens_d.notna()]
    dg, rc = s[s.category == 'Degradation'], s[s.category == 'Recovery']
    if len(dg) < 2 or len(rc) < 2:
        return np.nan, np.nan
    md = np.average(dg.cohens_d, weights=dg.n_trans)
    mr = np.average(rc.cohens_d, weights=rc.n_trans)
    return (float(md - mr),
            float(stats.ttest_ind(dg.cohens_d, rc.cohens_d, equal_var=False)[1]))


lag = pd.read_csv(LAGGED)
cum = pd.read_csv(CUMUL)
print(f'lagged rows {len(lag):,} | cumulative rows {len(cum):,}')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig = plt.figure(figsize=(13.6, 7.4), dpi=300)
gs = fig.add_gridspec(2, 3, width_ratios=[1.12, 1, 1], hspace=0.42, wspace=0.30)

# ---------------- panel (a) ----------------
ax = fig.add_subplot(gs[:, 0])
vals, ps = [], []
for code, _, _ in REGIONS:
    v, p = rsi(lag, code, 12)
    vals.append(v); ps.append(p)
ax.bar(range(5), vals, color='#3b6fa8', edgecolor='#1c3f63', width=0.62)
for i, (v, p) in enumerate(zip(vals, ps)):
    ax.text(i, v + (-0.016 if v < 0 else 0.010),
            f'{v:+.3f}' + ('*' if p < 0.05 else ''),
            ha='center', va='top' if v < 0 else 'bottom',
            fontsize=8.6, fontweight='bold')
ax.axhline(0, color='#222', lw=1.2)
ax.set_xticks(range(5))
ax.set_xticklabels([n for _, n, _ in REGIONS], rotation=28, ha='right', fontsize=8.6)
ax.set_ylabel("Recovery Suppression Index (Cohen's $d$)", fontsize=9.5, fontweight='bold')
ax.set_title('(a) RSI by region \u2014 lagged window, SPEI-12', fontsize=10, fontweight='bold')
ax.set_ylim(-0.58, 0.10)
ax.text(0.97, 0.04, '* $p$ < 0.05', transform=ax.transAxes, ha='right', fontsize=8.4,
        bbox=dict(fc='white', ec='#999', lw=0.7))
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

# ---------------- panels (b1)-(b4) ----------------
panels = [(gs[0, 1], rsi, lag, '(b1) RSI \u2014 lagged', "RSI (Cohen's $d$)"),
          (gs[0, 2], rsi, cum, '(b2) RSI \u2014 cumulative', None),
          (gs[1, 1], dra, lag, '(b3) DRA \u2014 lagged', "DRA (Cohen's $d$ difference)"),
          (gs[1, 2], dra, cum, '(b4) DRA \u2014 cumulative', None)]
for spec, fn, df, title, ylab in panels:
    a = fig.add_subplot(spec)
    for code, name, col in REGIONS:
        ys, sig = [], []
        for ts in TIMESCALES:
            v, p = fn(df, code, ts)
            ys.append(v)
            sig.append(bool(np.isfinite(p) and p < 0.05))
        a.plot(range(4), ys, '-o', color=col, ms=5, lw=1.7, label=name)
        for i, (y, s_) in enumerate(zip(ys, sig)):
            if s_:
                a.plot(i, y, marker='*', color=col, ms=12, mec='#222', mew=0.5)
    a.axhline(0, color='#888', lw=0.9, ls=':')
    a.set_xticks(range(4))
    a.set_xticklabels([f'SPEI-{t}' for t in TIMESCALES], fontsize=8.4)
    a.set_title(title, fontsize=9.6, fontweight='bold')
    if ylab:
        a.set_ylabel(ylab, fontsize=9, fontweight='bold')
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    a.grid(axis='y', alpha=0.22, ls=':')

h, l = fig.axes[1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=9,
           bbox_to_anchor=(0.63, -0.005))
fig.suptitle("Stars mark one-sample $t$-test (RSI) or Welch's $t$-test (DRA) "
             "against zero at $p$ < 0.05",
             y=0.965, fontsize=9, style='italic', color='#444')
fig.subplots_adjust(bottom=0.11, top=0.91)
out = os.path.join(OUT_DIR, 'Figure5_RSI_DRA.png')
fig.savefig(out, dpi=300, facecolor='white', bbox_inches='tight')
print('written:', out)

# ---------------- printed audit of every marker ----------------
print('\nSIGNIFICANCE AUDIT (values plotted in panels b1-b4)')
print(f"{'region':16s} {'ts':>4s} {'RSI lag':>16s} {'RSI cum':>16s} "
      f"{'DRA lag':>16s} {'DRA cum':>16s}")
for code, name, _ in REGIONS:
    for ts in TIMESCALES:
        row = f'{name:16s} {ts:4d}'
        for fn, df in [(rsi, lag), (rsi, cum), (dra, lag), (dra, cum)]:
            v, p = fn(df, code, ts)
            row += f" {v:+8.3f}{'*' if np.isfinite(p) and p < 0.05 else ' '}({p:.3f})"
        print(row)
