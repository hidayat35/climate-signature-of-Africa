"""
============================================================
CONTINENTAL AGGREGATION AND INFERENCE
  ->  Supplementary Tables S9, S12 and S13; Figure 6 source
============================================================

Computes the continental category means reported in the paper, together
with their confidence intervals, equivalence tests and the
recovery-degradation contrast.

AGGREGATION
-----------
Continental means are computed in two stages:

  1. Within each region x pathway cell, interval-level Cohen's d values
     are collapsed by a transition-pixel-weighted mean.
  2. Those cell means are averaged with EQUAL weight across cells.

Weighting within a cell reflects the information each interval
contributes. Equal weighting across cells prevents the continental
estimate from being determined by the few region x pathway combinations
with the largest pixel counts, which would otherwise reduce it to a
Southern and East African average: Southern Africa contributes 25,972
recovery pixels against the Mediterranean's 225.

Regional indices (Eq. 7) retain transition-pixel weighting throughout,
because within a single region the disparity in cell size is far
smaller.

COMMON UNIT SET
---------------
All five estimators are evaluated on the same set of region x pathway
cells, namely those evaluable under the primary point-sampling design.
Cells falling below the minimum counts in one estimator but not another
are excluded from all, so that the estimators are directly comparable
rather than being computed on different underlying data.

INFERENCE
---------
  Confidence intervals  bias-corrected and accelerated (BCa) bootstrap,
                        10,000 resamples over the cell means.

  Equivalence           two one-sided tests against a pre-specified
                        smallest effect size of interest of |d| = 0.10,
                        half Cohen's conventional small-effect
                        threshold, with |d| = 0.20 as a secondary bound.
                        A non-significant difference from zero is not
                        evidence of equivalence; only the two one-sided
                        tests licence a positive statement that an
                        effect is negligible.

  Category contrast     Welch's t-test and a permutation test with
                        20,000 resamples on the region x pathway cell
                        means.

OUTPUTS
-------
  final_twostage.json   all results, consumed by Generate_Figure6_ForestPlot.py
  printed tables        continental means, TOST, contrast, headline counts

RUNTIME: about two minutes. Reads the per-cell effect-size tables only;
no rasters and no Earth Engine account required.
============================================================
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# CONFIGURATION - edit IN_DIR to point at your per-cell tables
# ============================================================
IN_DIR = r'.'
OUT_DIR = r'.'

CATS = ['Degradation', 'Recovery', 'Agricultural']
TIMESCALES = ['spei_12', 'spei_24', 'spei_36', 'spei_60']
N_BOOT = 10000
N_PERM = 20000
EQUIV_BOUNDS = (0.10, 0.178, 0.20)
SEED = 42

SRC = {
    'point_lagged':   ('TableB2_LAGGED_per_interval_STRICTER.csv',
                       'cohens_d', 'n_trans'),
    'point_cumul':    ('TableB2_CUMULATIVE_per_interval_STRICTER.csv',
                       'cohens_d', 'n_trans'),
    'cem_matched':    ('TableS8_CEM_per_interval.csv',
                       'cohens_d_CEM', 'n_trans'),
    'area_weighted':  ('TableS9_areaweighted_per_interval.csv',
                       'cohens_d_area_weighted', 'weight_sum_trans'),
    'area_threshold': ('TableS9b_areathresholded_per_interval.csv',
                       'cohens_d_area_weighted', 'weight_sum_trans'),
}
LABEL = {'point_lagged': 'Point, lagged', 'point_cumul': 'Point, cumulative',
         'area_weighted': 'Area-weighted', 'area_threshold': 'Area-thresholded',
         'cem_matched': 'Matched (CEM)'}
ORDER = ['point_lagged', 'point_cumul', 'area_weighted', 'area_threshold', 'cem_matched']

DATA = {}
for est, (fn, dcol, ncol) in SRC.items():
    path = os.path.join(IN_DIR, fn)
    if os.path.exists(path):
        DATA[est] = (pd.read_csv(path), dcol, ncol)
        print(f'  loaded {est:16s} {len(DATA[est][0]):,} rows')
    else:
        print(f'  MISSING {est:16s} {path}')
if 'point_lagged' not in DATA:
    raise SystemExit('The primary point-sampled table is required.')

# The primary point-sampled analysis defines the evaluable units.
base = DATA['point_lagged'][0]
b = base[(base.spei_timescale == 'spei_12') & base.cohens_d.notna()]
COMMON = set(map(tuple, b[['region', 'transition']].drop_duplicates().values))
print(f'\nCommon unit set: {len(COMMON)} region x pathway cells')


def units(df, dcol, ncol, ts, restrict=True):
    """Stage 1: collapse intervals within each region x pathway cell."""
    s = df[(df.spei_timescale == ts) & df[dcol].notna()]
    rows = []
    for (reg, pw), g in s.groupby(['region', 'transition']):
        if restrict and (reg, pw) not in COMMON:
            continue
        w = g[ncol].values
        d = float(np.average(g[dcol].values, weights=w)) if w.sum() > 0 \
            else float(g[dcol].mean())
        rows.append({'region': reg, 'pathway': pw,
                     'category': g.category.iloc[0], 'd': d, 'n_int': len(g)})
    return pd.DataFrame(rows)


def bca(x, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    """Bias-corrected and accelerated bootstrap interval for a mean."""
    x = np.asarray(x, float)
    n = len(x)
    theta = x.mean()
    if n < 3:
        return theta, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, n, (n_boot, n))].mean(1)
    prop = np.clip((boots < theta).mean(), 1 / n_boot, 1 - 1 / n_boot)
    z0 = stats.norm.ppf(prop)
    jack = np.array([np.delete(x, i).mean() for i in range(n)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den > 1e-12 else 0.0

    def adj(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo = float(np.percentile(boots, 100 * np.clip(adj(stats.norm.ppf(alpha / 2)), 0, 1)))
    hi = float(np.percentile(boots, 100 * np.clip(adj(stats.norm.ppf(1 - alpha / 2)), 0, 1)))
    return theta, lo, hi


def tost(x, bound):
    """Two one-sided tests against +/- bound; returns the larger p."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    if se <= 0:
        return np.nan
    return float(max(1 - stats.t.cdf((m + bound) / se, n - 1),
                     stats.t.cdf((m - bound) / se, n - 1)))


CI, TOSTR, CONTRAST = {}, {}, {}
rng = np.random.default_rng(SEED + 1)

for ts in TIMESCALES:
    CI[ts], TOSTR[ts] = {}, {}
    for est in ORDER:
        if est not in DATA:
            continue
        df, dcol, ncol = DATA[est]
        if ts not in set(df.spei_timescale.unique()):
            continue
        u = units(df, dcol, ncol, ts)
        CI[ts][est], TOSTR[ts][est] = {}, {}
        for c in CATS:
            x = u[u.category == c].d.values
            if len(x) < 3:
                continue
            m, lo, hi = bca(x)
            CI[ts][est][c] = [round(m, 4), round(lo, 4), round(hi, 4), len(x)]
            TOSTR[ts][est][c] = {str(bd): round(tost(x, bd), 4) for bd in EQUIV_BOUNDS}

    CONTRAST[ts] = {}
    for est in ['point_lagged', 'point_cumul']:
        if est not in DATA:
            continue
        df, dcol, ncol = DATA[est]
        u = units(df, dcol, ncol, ts)
        dg = u[u.category == 'Degradation'].d.values
        rc = u[u.category == 'Recovery'].d.values
        if len(dg) < 3 or len(rc) < 3:
            continue
        obs = dg.mean() - rc.mean()
        pool = np.concatenate([dg, rc])
        n1 = len(dg)
        cnt = sum(1 for _ in range(N_PERM)
                  if abs((pr := rng.permutation(pool))[:n1].mean() - pr[n1:].mean()) >= abs(obs))
        CONTRAST[ts][est] = {
            'diff': round(float(obs), 4),
            'welch_p': round(float(stats.ttest_ind(dg, rc, equal_var=False)[1]), 5),
            'perm_p': round((cnt + 1) / (N_PERM + 1), 5)}

with open(os.path.join(OUT_DIR, 'final_twostage.json'), 'w') as f:
    json.dump({'CI': CI, 'TOST': TOSTR, 'CONTRAST': CONTRAST, 'LABEL': LABEL}, f, indent=0)

# ============================================================
# REPORT
# ============================================================
print('\n' + '=' * 104)
print('CONTINENTAL EFFECT SIZES, TWO-STAGE AGGREGATION ON A COMMON UNIT SET')
print('=' * 104)
print(f"{'ts':9s} {'estimator':18s} {'category':13s} {'mean':>8s} {'95% CI':>21s} "
      f"{'excl0':>6s} {'TOST.10':>8s} {'TOST.20':>8s} {'n':>3s}")
for ts in TIMESCALES:
    for est in ORDER:
        if est not in CI.get(ts, {}):
            continue
        for c in CATS:
            if c not in CI[ts][est]:
                continue
            m, lo, hi, n = CI[ts][est][c]
            t = TOSTR[ts][est][c]
            print(f"{ts:9s} {LABEL[est]:18s} {c:13s} {m:+8.3f} ({lo:+7.3f},{hi:+7.3f}) "
                  f"{'YES' if lo * hi > 0 else 'no':>6s} {t['0.1']:8.4f} {t['0.2']:8.4f} {n:3d}")

print('\n' + '=' * 104)
print('HEADLINE COUNTS')
print('=' * 104)
for c in CATS:
    tot = ex = 0
    for ts in TIMESCALES:
        for est in ['point_lagged', 'point_cumul']:
            if c in CI.get(ts, {}).get(est, {}):
                tot += 1
                ex += (CI[ts][est][c][1] * CI[ts][est][c][2] > 0)
    print(f'  {c:13s}: interval excludes zero in {ex} of {tot} timescale x window combinations')

print('\n' + '=' * 104)
print('RECOVERY vs DEGRADATION CONTRAST')
print('=' * 104)
print(f"{'ts':9s} {'window':18s} {'difference':>11s} {'Welch p':>9s} {'perm p':>9s}")
for ts in TIMESCALES:
    for est, v in CONTRAST.get(ts, {}).items():
        print(f"{ts:9s} {LABEL[est]:18s} {v['diff']:+11.4f} {v['welch_p']:9.5f} {v['perm_p']:9.5f}")

print(f"\nWritten: {os.path.join(OUT_DIR, 'final_twostage.json')}")
