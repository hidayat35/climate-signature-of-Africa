"""
============================================================
FIGURE 6: continental effect sizes with bootstrap confidence
intervals under five estimators
============================================================

Builds the forest plot of continental mean Cohen's d by transition
category, each with a bias-corrected and accelerated bootstrap 95%
confidence interval.

It reads the JSON written by
Batch3_TwoStage_Aggregation_and_Inference.py, so the figure and the
values reported in Supplementary Tables S9, S12 and S13 come from a
single computation.

The shaded band marks the pre-specified equivalence bound of |d| = 0.10
used in the two one-sided tests (Section 2.4 of the manuscript).

INPUT
-----
  final_twostage.json

OUTPUT
------
  Figure6_ForestPlot.png   (300 dpi)

RUNTIME: seconds.
============================================================
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IN_JSON = 'final_twostage.json'
OUT_PNG = 'Figure6_ForestPlot.png'
TIMESCALE = 'spei_12'
EQUIV_BOUND = 0.10

ORDER = ['point_lagged', 'point_cumul', 'area_weighted', 'area_threshold', 'cem_matched']
CATS = ['Degradation', 'Recovery', 'Agricultural']
COLOUR = {'Degradation': '#C0392B', 'Recovery': '#1E8449', 'Agricultural': '#D68910'}

D = json.load(open(IN_JSON))
CI, LABEL = D['CI'], D['LABEL']

rows = [(c, e) + tuple(CI[TIMESCALE][e][c])
        for c in CATS for e in ORDER if c in CI[TIMESCALE].get(e, {})]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.weight': 'bold'})
fig, ax = plt.subplots(figsize=(8.6, 6.4), dpi=300)
y = np.arange(len(rows))[::-1]

ax.axvspan(-EQUIV_BOUND, EQUIV_BOUND, color='#8E8E8E', alpha=0.13, zorder=0)
for yi, (c, e, m, lo, hi, n) in zip(y, rows):
    ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt='o',
                color=COLOUR[c], ecolor=COLOUR[c],
                capsize=4, markersize=7.5, linewidth=2.1, zorder=3)
ax.axvline(0, color='#222', lw=1.6, zorder=2)

ax.set_yticks(y)
ax.set_yticklabels([LABEL[e] for c, e, *_ in rows], fontsize=9)
for c in CATS:
    idx = [i for i, (cc, *_) in enumerate(rows) if cc == c]
    if idx:
        ax.text(-0.335, y[idx[0]] + 0.42, c, fontsize=10.5,
                fontweight='bold', color=COLOUR[c], va='bottom')

ax.set_xlim(-0.34, 0.20)
ax.set_xlabel("Continental mean Cohen's $d$ with bias-corrected bootstrap 95% CI",
              fontsize=11, fontweight='bold')
ax.set_title('Continental effect sizes under five estimators, SPEI-12\n'
             f'shaded band = pre-specified equivalence bound |$d$| = {EQUIV_BOUND:.2f}',
             fontsize=11.5, fontweight='bold', pad=10)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.grid(axis='x', alpha=0.25, linestyle=':')

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, facecolor='white')
print('written:', os.path.abspath(OUT_PNG))

print(f"\n{'category':13s} {'estimator':20s} {'mean':>8s} {'95% CI':>20s} {'excl 0':>7s}")
for c, e, m, lo, hi, n in rows:
    print(f'{c:13s} {LABEL[e]:20s} {m:+8.3f} ({lo:+7.3f},{hi:+7.3f}) '
          f"{'YES' if lo * hi > 0 else 'no':>7s}")
