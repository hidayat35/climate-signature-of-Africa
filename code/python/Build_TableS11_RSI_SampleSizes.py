"""
============================================================
CELL COMPOSITION BEHIND EACH REGIONAL RECOVERY SUPPRESSION
INDEX   ->  Supplementary Table S11
============================================================

A Recovery Suppression Index is a sample-size-weighted mean of Cohen's d
over the recovery pathways within a region (Eq. 7). Interpreting a
regional value requires knowing how many pathway-interval cells produced
it, and how many distinct pathways they represent: a complete index
draws on three recovery pathways across seven intervals, giving a
maximum of 21 cells, but not every region reaches that.

This script reports, for every region, timescale and antecedent window:

  n_cells        pathway-interval cells contributing
  n_pathways     distinct recovery pathways present (max 3)
  n_intervals    distinct intervals present
  total_n_trans  summed transition pixels
  median_n_trans median transition pixels per cell
  min_n_trans    smallest contributing cell
  RSI            recomputed index, with its one-sample t-test p-value

OUTPUTS
-------
  TableS14_RSI_sample_sizes.csv   Supplementary Table S11
  R2_run_log.txt

RUNTIME: seconds. Reads the per-cell effect-size tables only.
============================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — EDIT PATHS
# ============================================================
LAGGED_CSV = r'TableB2_LAGGED_per_interval_STRICTER.csv'
CUMUL_CSV  = r'TableB2_CUMULATIVE_per_interval_STRICTER.csv'
OUT_DIR    = r'.'
os.makedirs(OUT_DIR, exist_ok=True)

RECOVERY_PATHWAYS = ['SHR_FST', 'GRS_SHR', 'BAL_GRS']
REGION_ORDER = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
REGION_NAMES = {'MED': 'Mediterranean', 'SAH': 'Sahara-Sahel', 'WAF': 'West Africa',
                'EAF': 'East Africa', 'SAF': 'Southern Africa'}
TIMESCALES = ['spei_12', 'spei_24', 'spei_36', 'spei_60']


# ============================================================
def summarise(df, window_label):
    rows = []
    for ts in TIMESCALES:
        sub = df[(df.spei_timescale == ts) &
                 df.transition.isin(RECOVERY_PATHWAYS) &
                 df.cohens_d.notna()]
        for code in REGION_ORDER:
            g = sub[sub.region == code]
            if len(g) == 0:
                rows.append({'region': REGION_NAMES[code], 'region_code': code,
                             'timescale': ts, 'window': window_label,
                             'n_cells': 0, 'n_pathways': 0, 'n_intervals': 0,
                             'total_n_trans': 0, 'median_n_trans': np.nan,
                             'min_n_trans': np.nan, 'RSI': np.nan, 'p_value': np.nan})
                continue
            rsi = float(np.average(g.cohens_d, weights=g.n_trans)) if g.n_trans.sum() > 0 \
                  else float(g.cohens_d.mean())
            p = float(stats.ttest_1samp(g.cohens_d.values, 0.0)[1]) if len(g) >= 3 else np.nan
            rows.append({
                'region': REGION_NAMES[code], 'region_code': code,
                'timescale': ts, 'window': window_label,
                'n_cells': int(len(g)),
                'n_pathways': int(g.transition.nunique()),
                'n_intervals': int(g.interval.nunique()),
                'total_n_trans': int(g.n_trans.sum()),
                'median_n_trans': float(g.n_trans.median()),
                'min_n_trans': int(g.n_trans.min()),
                'RSI': round(rsi, 4),
                'p_value': round(p, 4) if np.isfinite(p) else np.nan,
            })
    return pd.DataFrame(rows)


print("=" * 70)
print("RSI CELL COMPOSITION BY REGION")
print("=" * 70)

frames = []
lag = pd.read_csv(LAGGED_CSV)
print(f"\nlagged file  : {len(lag):,} rows")
frames.append(summarise(lag, 'lagged'))

if os.path.exists(CUMUL_CSV):
    cum = pd.read_csv(CUMUL_CSV)
    print(f"cumulative   : {len(cum):,} rows")
    frames.append(summarise(cum, 'cumulative'))
else:
    print(f"cumulative   : NOT FOUND at {CUMUL_CSV}")
    print("               lagged-window results only; edit CUMUL_CSV if you have it")

out = pd.concat(frames, ignore_index=True)
out.to_csv(os.path.join(OUT_DIR, 'TableS14_RSI_sample_sizes.csv'), index=False)

# ------------------------------------------------------------
print("\n" + "=" * 70)
print("LAGGED WINDOW, SPEI-12  (the values currently in Figure 5a)")
print("=" * 70)
show = out[(out.window == 'lagged') & (out.timescale == 'spei_12')]
print(show[['region', 'n_cells', 'n_pathways', 'n_intervals',
            'total_n_trans', 'median_n_trans', 'min_n_trans',
            'RSI', 'p_value']].to_string(index=False))

print("\nCELL COUNTS ACROSS ALL TIMESCALES (lagged)")
piv = out[out.window == 'lagged'].pivot_table(
    index='region', columns='timescale', values='n_cells', aggfunc='first')
print(piv.to_string())

# ------------------------------------------------------------
# Flag thin regions: a 3-pathway x 7-interval design has a maximum of 21
# ------------------------------------------------------------
MAX_CELLS = 21
thin = out[(out.window == 'lagged') & (out.n_cells < 0.5 * MAX_CELLS) & (out.n_cells > 0)]
print("\n" + "=" * 70)
print("REGIONS WITH FEWER THAN HALF THE MAXIMUM 21 CELLS")
print("=" * 70)
if len(thin):
    print(thin[['region', 'timescale', 'n_cells', 'RSI', 'p_value']].to_string(index=False))
    print("\nThese regional indices rest on sparse data. State this in Section 4.3")
    print("and do not present them with the same confidence as the continental result.")
else:
    print("None.")

with open(os.path.join(OUT_DIR, 'R2_run_log.txt'), 'w', encoding='utf-8') as f:
    f.write("RSI CELL COMPOSITION\n" + "=" * 60 + "\n\n")
    f.write(f"Maximum possible cells per region = 3 pathways x 7 intervals = {MAX_CELLS}\n\n")
    f.write(out.to_string(index=False) + "\n")

print(f"\nWritten to {OUT_DIR}")
print("  TableS14_RSI_sample_sizes.csv")
print("  R2_run_log.txt")
