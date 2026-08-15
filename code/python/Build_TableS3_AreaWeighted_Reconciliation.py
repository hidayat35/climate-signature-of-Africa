"""
============================================================
AREA-WEIGHTED AGGREGATION: SPECIFICATION COMPARISON AND
THRESHOLD SENSITIVITY   ->  Supplementary Tables S3 and S4
============================================================

Point sampling assigns each ~5.5 km climate cell the value of one
underlying near-centre pixel. This script tests whether the continental
degradation effect depends on that choice, by recomputing it under
area-based aggregation across a range of specifications.

Both analyses live in one script because the expensive operation is
to_fraction(), which aggregates each 300 m transition mask onto the
0.05 deg climate grid. That runs 63 times (9 pathways x 7 intervals).
Every threshold configuration and cell subset is then applied to arrays
already in memory, at negligible cost.

PART 1 - SPECIFICATION COMPARISON  (Supplementary Table S3)
-----------------------------------------------------------
A full factorial over three factors that could each move the estimate:

  3 aggregation specifications
      threshold pair A  (transition > 0.001, from-state >= 0.50)
      threshold pair B  (transition > 0.02,  from-state >= 0.50)
      direct fractional-area weighting
  x 2 domains  (continental, Sahara-Sahel)
  x 2 interval sets  (including / excluding the terminal 2020-2022)

Running the factorial isolates each factor, which a single pairwise
comparison cannot do.

PART 2 - THRESHOLD SENSITIVITY SWEEP  (Supplementary Table S4)
--------------------------------------------------------------
Continental degradation d as a function of the transition-fraction
threshold (8 values, 0.0001 to 0.05) and the from-state threshold
(7 values, 0.10 to 0.90). 56 combinations in total. A sign change
anywhere in this surface would mean no single area-weighted value is
defensible; the surface shows whether one occurs.

CONFIGURATION
-------------
The two threshold pairs are set in the CONFIGURATION block below.
After the run, read TableS12_areaweighted_reconciliation.csv and set
ADOPTED_THRESHOLD_SPEC to the specification you adopt, then re-emit the
bridge CSVs consumed by Batch3_TwoStage_Aggregation_and_Inference.py.

OUTPUTS
-------
  TableS12_areaweighted_reconciliation.csv   Supplementary Table S3
  TableS13_threshold_sweep.csv               Supplementary Table S4
  FigureS4_threshold_sensitivity.png         sweep surface
  TableS9_areaweighted_per_interval.csv      bridge, fractional weighting
  TableS9b_areathresholded_per_interval.csv  bridge, thresholded
  percell_long.csv                           per-cell values, all specs
  R1_run_log.txt

RUNTIME: 2-4 hours. Requires the exported transition and state rasters.
============================================================
"""

import os
import gc
import itertools
import numpy as np
import pandas as pd
import rioxarray
import geopandas as gpd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
TRANSITIONS_DIR = r'<DATA_ROOT>\transitions'
STATES_DIR      = r'<DATA_ROOT>\states'
SPEI_DIR        = r'<DATA_ROOT>\step5_spei_output'
SHAPEFILE       = r'<DATA_ROOT>\ipcc_africa_5_regions.shp'
OUT_DIR         = r'.'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# TWO THRESHOLD SPECIFICATIONS ARE COMPARED
#
# NAMING WARNING: "Table S3" (the supplementary table showing -0.144)
# Threshold pair A: the stricter transition-fraction setting.

# Table S3's caption states 0.1% transition fraction and 50% from-state.
# Threshold pair B: the coarser transition-fraction setting.
TABLES3_TRANS  = 0.001      # 0.1%
TABLES3_STABLE = 0.50

# Verified from S3_AreaWeighted_Primary_AllCells.py lines 74-75.
TABLES9_TRANS  = 0.02
TABLES9_STABLE = 0.50

# Sweep grids for the sensitivity surface
TRANS_SWEEP  = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05]
STABLE_SWEEP = [0.10, 0.25, 0.40, 0.50, 0.65, 0.80, 0.90]

MIN_TRANS_CELLS  = 20
MIN_STABLE_CELLS = 50
MIN_WEIGHT_SUM   = 5.0

TRANSITIONS = [
    {'name': 'FST_SHR',         'category': 'Degradation',  'from_state': 'FST'},
    {'name': 'SHR_GRS',         'category': 'Degradation',  'from_state': 'SHR'},
    {'name': 'FST_CRP',         'category': 'Degradation',  'from_state': 'FST'},
    {'name': 'GRS_BAL',         'category': 'Degradation',  'from_state': 'GRS'},
    {'name': 'SHR_FST',         'category': 'Recovery',     'from_state': 'SHR'},
    {'name': 'GRS_SHR',         'category': 'Recovery',     'from_state': 'GRS'},
    {'name': 'BAL_GRS',         'category': 'Recovery',     'from_state': 'BAL'},
    {'name': 'AGEXPANSION',     'category': 'Agricultural', 'from_state': 'NATURAL'},
    {'name': 'CRP_ABANDONMENT', 'category': 'Agricultural', 'from_state': 'CRP'},
]
REGION_ORDER = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
INTERVALS = [
    ('1985_1990', 1, 2), ('1990_1995', 2, 3), ('1995_2000', 3, 4),
    ('2000_2005', 4, 5), ('2005_2010', 5, 6), ('2010_2015', 6, 7),
    ('2015_2020', 7, 8), ('2020_2022', 8, 8),
]
PRIMARY_TS = 'spei_12'      # reconciliation and sweep run at this timescale


# ============================================================
def load_tif(p):
    da = rioxarray.open_rasterio(p)
    if da.rio.crs is None:
        da = da.rio.write_crs('EPSG:4326')
    if 'band' in da.dims and da.sizes['band'] == 1:
        da = da.squeeze('band', drop=True)
    return da


def squeeze2d(a):
    while a.ndim > 2:
        a = a.squeeze(axis=0)
    return a


def to_fraction(binary_da, target):
    return binary_da.astype('float32').rio.reproject_match(
        target, resampling=Resampling.average)


def d_from_moments(m_t, v_t, n_t, m_s, v_s, n_s):
    if not all(np.isfinite([m_t, v_t, m_s, v_s])) or n_t < 2 or n_s < 2:
        return np.nan
    pooled = np.sqrt(((n_t - 1) * v_t + (n_s - 1) * v_s) / (n_t + n_s - 2))
    return (m_s - m_t) / pooled if pooled > 1e-9 else np.nan


def d_threshold(sp, ft, fs, t_thr, s_thr):
    """Cohen's d under a fractional-threshold classification."""
    mt, ms = ft > t_thr, fs >= s_thr
    xt, xs = sp[mt], sp[ms]
    xt, xs = xt[np.isfinite(xt)], xs[np.isfinite(xs)]
    if len(xt) < MIN_TRANS_CELLS or len(xs) < MIN_STABLE_CELLS:
        return np.nan, len(xt), len(xs)
    d = d_from_moments(xt.mean(), xt.var(ddof=1), len(xt),
                       xs.mean(), xs.var(ddof=1), len(xs))
    return d, len(xt), len(xs)


def wmv(x, w):
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return np.nan, np.nan, 0.0
    x, w = x[ok], w[ok]
    sw = w.sum()
    m = float((w * x).sum() / sw)
    den = sw - ((w ** 2).sum() / sw)
    v = float((w * (x - m) ** 2).sum() / den) if den > 1e-9 else np.nan
    return m, v, float(sw)


def d_weighted(sp, ft, fs):
    mt, vt, wt = wmv(sp, ft)
    ms, vs, ws = wmv(sp, fs)
    if wt < MIN_WEIGHT_SUM or ws < MIN_WEIGHT_SUM:
        return np.nan, wt, ws
    return d_from_moments(mt, vt, wt, ms, vs, ws), wt, ws


# ============================================================
print("=" * 70)
print("AREA-WEIGHTED SPECIFICATION COMPARISON AND THRESHOLD SWEEP")
print("=" * 70)
print(f"Table S3 spec : trans > {TABLES3_TRANS}, stable >= {TABLES3_STABLE}")
print(f"Table S9 spec : trans > {TABLES9_TRANS}, stable >= {TABLES9_STABLE}")


regions = gpd.read_file(SHAPEFILE)
geoms = {r['LAB']: r['geometry'] for _, r in regions.iterrows()}

trans_r = {t['name']: load_tif(os.path.join(TRANSITIONS_DIR, f"transition_{t['name']}.tif"))
           for t in TRANSITIONS}
state_r = {s: load_tif(os.path.join(STATES_DIR, f'state_{s}.tif'))
           for s in ['FST', 'SHR', 'GRS', 'BAL', 'CRP', 'NATURAL']}
grid = load_tif(os.path.join(SPEI_DIR, f'{PRIMARY_TS}_mean_{INTERVALS[0][0]}.tif'))
SPEI = {lbl: load_tif(os.path.join(SPEI_DIR, f'{PRIMARY_TS}_mean_{lbl}.tif'))
        for lbl, _, _ in INTERVALS}

# threshold specs evaluated per cell
SPECS = {'tableS3': (TABLES3_TRANS, TABLES3_STABLE),
         'tableS9': (TABLES9_TRANS, TABLES9_STABLE)}
SWEEP = list(itertools.product(TRANS_SWEEP, STABLE_SWEEP))

rows = []
pair = 0
n_pairs = len(TRANSITIONS) * (len(INTERVALS) - 1)

for t in TRANSITIONS:
    tds, sds = trans_r[t['name']], state_r[t['from_state']]
    for idx in range(1, len(INTERVALS)):        # includes 2020_2022; filtered later
        pair += 1
        lbl, sb, eb = INTERVALS[idx]
        prior = INTERVALS[idx - 1][0]

        tb = tds.sel(band=sb) if 'band' in tds.dims else tds
        s0 = sds.sel(band=sb) if 'band' in sds.dims else sds
        s1 = sds.sel(band=eb) if 'band' in sds.dims else sds

        try:                                     # <-- the expensive part, once
            f_t = to_fraction((tb == 1), grid)
            f_0 = to_fraction((s0 == 1), grid)
            f_1 = to_fraction((s1 == 1), grid)
        except Exception as e:
            print(f"  fraction failed {t['name']} {lbl}: {e}")
            continue

        a_t = np.minimum(squeeze2d(f_t.values), squeeze2d(f_0.values))
        a_s = np.clip(np.minimum(squeeze2d(f_0.values), squeeze2d(f_1.values))
                      - squeeze2d(f_t.values), 0, 1)
        da_t = f_t.copy(data=a_t.reshape(f_t.shape))
        da_s = f_t.copy(data=a_s.reshape(f_t.shape))

        for code in REGION_ORDER:
            g = geoms.get(code)
            if g is None:
                continue
            try:
                ft = squeeze2d(da_t.rio.clip([g], crs='EPSG:4326', all_touched=False, drop=False).values)
                fs = squeeze2d(da_s.rio.clip([g], crs='EPSG:4326', all_touched=False, drop=False).values)
                sp = squeeze2d(SPEI[prior].rio.clip([g], crs='EPSG:4326',
                                                    all_touched=False, drop=False).values.astype('float32'))
            except Exception:
                continue
            if sp.shape != ft.shape:
                continue

            # flatten once; thresholds are then cheap selections
            ok = np.isfinite(sp)
            spf, ftf, fsf = sp[ok], ft[ok], fs[ok]

            base = {'pathway': t['name'], 'category': t['category'],
                    'region': code, 'interval': lbl}

            for spec, (tt, st) in SPECS.items():
                d, nt, ns = d_threshold(spf, ftf, fsf, tt, st)
                rows.append({**base, 'estimator': f'threshold_{spec}',
                             'trans_thr': tt, 'stable_thr': st,
                             'd': d, 'n_trans': nt, 'n_stable': ns})

            dw, wt, ws = d_weighted(spf, ftf, fsf)
            rows.append({**base, 'estimator': 'weighted', 'trans_thr': np.nan,
                         'stable_thr': np.nan, 'd': dw,
                         'n_trans': wt, 'n_stable': ws})

            for tt, st in SWEEP:                 # sensitivity surface
                d, nt, ns = d_threshold(spf, ftf, fsf, tt, st)
                rows.append({**base, 'estimator': 'sweep',
                             'trans_thr': tt, 'stable_thr': st,
                             'd': d, 'n_trans': nt, 'n_stable': ns})

        del f_t, f_0, f_1, da_t, da_s
        gc.collect()
        print(f"  [{pair:3d}/{n_pairs}] {t['name']:<18} {lbl}   rows: {len(rows):,}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, 'percell_long.csv'), index=False)
print(f"\nper-cell rows: {len(df):,}")

# ============================================================
# BRIDGE TO S7  (fix B) + TABLE S4 REBUILD  (fix E)
# ============================================================
# S7 expects the column layout that S3 produced. Re-emit the adopted
# specifications in that format so S7 can compute CIs and TOST on the
# reconciled numbers.
#
# Set ADOPTED_THRESHOLD_SPEC after reading TableS12 below. Until then the
# Table S9 spec is emitted so the pipeline runs end to end.
ADOPTED_THRESHOLD_SPEC = 'threshold_tableS9'   # or 'threshold_tableS3'

for spec, outname in [('weighted',             'TableS9_areaweighted_per_interval.csv'),
                      (ADOPTED_THRESHOLD_SPEC, 'TableS9b_areathresholded_per_interval.csv')]:
    b = df[df.estimator == spec].copy()
    if len(b) == 0:
        print(f"  bridge SKIPPED: no rows for {spec}")
        continue
    b['spei_timescale']         = PRIMARY_TS
    b['transition']             = b['pathway']
    b['cohens_d_area_weighted'] = b['d']
    b['weight_sum_trans']       = b['n_trans']
    b.to_csv(os.path.join(OUT_DIR, outname), index=False)
    print(f"  bridge written: {outname}  ({len(b):,} rows)")

# ---- Table S4 rebuilt under every specification ----
FROM_FOREST = ['FST_SHR', 'FST_CRP']
DRYLAND     = ['SHR_GRS', 'GRS_BAL']
s4 = []
for spec in ['threshold_tableS3', 'threshold_tableS9', 'weighted']:
    s = df[(df.estimator == spec) & (df.interval != '2020_2022') & df.d.notna()]
    grp = {}
    for name, paths in [('from_forest', FROM_FOREST), ('dryland_degradation', DRYLAND)]:
        g = s[s.pathway.isin(paths)]
        grp[name] = g.d.values
        for p_ in paths:
            gp = g[g.pathway == p_]
            if len(gp) >= 3:
                tt_, pv = stats.ttest_1samp(gp.d.values, 0.0)
                s4.append({'spec': spec, 'group': name, 'pathway': p_, 'n': len(gp),
                           'mean_d': round(float(gp.d.mean()), 4),
                           't': round(float(tt_), 3), 'p': round(float(pv), 4)})
        if len(g) >= 3:
            tt_, pv = stats.ttest_1samp(g.d.values, 0.0)
            s4.append({'spec': spec, 'group': name, 'pathway': 'GROUP MEAN', 'n': len(g),
                       'mean_d': round(float(g.d.mean()), 4),
                       't': round(float(tt_), 3), 'p': round(float(pv), 4)})
    if len(grp['from_forest']) >= 3 and len(grp['dryland_degradation']) >= 3:
        tt_, pv = stats.ttest_ind(grp['from_forest'], grp['dryland_degradation'],
                                  equal_var=False)
        s4.append({'spec': spec, 'group': 'WELCH between groups', 'pathway': '-',
                   'n': len(grp['from_forest']) + len(grp['dryland_degradation']),
                   'mean_d': np.nan, 't': round(float(tt_), 3), 'p': round(float(pv), 4)})

s4 = pd.DataFrame(s4)
s4.to_csv(os.path.join(OUT_DIR, 'TableS4_rebuilt_decomposition.csv'), index=False)
print("\nTABLE S4 REBUILT - published values were:")
print("  from-forest -0.252 (p=0.007), dryland +0.051 (p=0.41), Welch p=0.005")
print(s4.to_string(index=False))



# ============================================================
# the specification factorial
# ============================================================
def continental(sub, weight_col='n_trans'):
    s = sub[sub.d.notna()]
    if len(s) == 0:
        return np.nan, 0
    w = s[weight_col].replace(0, np.nan)
    if w.notna().sum() == 0 or w.sum() == 0:
        return float(s.d.mean()), len(s)
    return float(np.average(s.d, weights=w.fillna(0))), len(s)


fac = []
# NOTE: the second axis is SCOPE (continental vs Sahel), because Stage2
# reported continental and Sahel separately. It is not a cell subset.
# 3 estimators x 2 scopes x 2 interval sets = 12 rows.
for spec in ['threshold_tableS3', 'threshold_tableS9', 'weighted']:
    for scope in ['continental', 'sahel']:
        for intervals in ['incl_2020', 'excl_2020']:
            sub = df[(df.estimator == spec) & (df.category == 'Degradation')]
            if intervals == 'excl_2020':
                sub = sub[sub.interval != '2020_2022']
            if scope == 'sahel':
                sub = sub[sub.region == 'SAH']
            d, n = continental(sub)
            fac.append({'estimator': spec, 'scope': scope,
                        'intervals': intervals,
                        'degradation_d': round(d, 4) if np.isfinite(d) else np.nan,
                        'n_cells': n})
fac = pd.DataFrame(fac)
fac.to_csv(os.path.join(OUT_DIR, 'TableS12_areaweighted_reconciliation.csv'), index=False)

print("\n" + "=" * 70)
print("SPECIFICATION COMPARISON (continental degradation)")
print("=" * 70)
print(fac.to_string(index=False))



# ============================================================
# the sensitivity sweep
# ============================================================
sw = df[(df.estimator == 'sweep') & (df.category == 'Degradation') &
        (df.interval != '2020_2022')]
grid_rows = []
for tt, st in SWEEP:
    s = sw[(sw.trans_thr == tt) & (sw.stable_thr == st)]
    d, n = continental(s)
    grid_rows.append({'trans_thr': tt, 'stable_thr': st,
                      'continental_degradation_d': round(d, 4) if np.isfinite(d) else np.nan,
                      'n_cells': n})
swdf = pd.DataFrame(grid_rows)
swdf.to_csv(os.path.join(OUT_DIR, 'TableS13_threshold_sweep.csv'), index=False)

print("\n" + "=" * 70)
print("THRESHOLD SENSITIVITY SWEEP")
print("=" * 70)
piv = swdf.pivot(index='trans_thr', columns='stable_thr',
                 values='continental_degradation_d')
print(piv.round(3).to_string())
vals = swdf.continental_degradation_d.dropna()
if len(vals):
    print(f"\nrange {vals.min():+.3f} to {vals.max():+.3f}   "
          f"sign changes: {'YES' if (vals.min() < 0 < vals.max()) else 'no'}")

# ---- figure ----
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.weight': 'bold'})
fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=300)
for st in STABLE_SWEEP:
    s = swdf[swdf.stable_thr == st].sort_values('trans_thr')
    ax.plot(s.trans_thr, s.continental_degradation_d, '-o',
            lw=2.2, markersize=6, label=f'stable ≥ {st:.2f}')
ax.axhline(0, color='#222', lw=2)
ax.axhspan(-0.03, 0.03, color='#999', alpha=0.15, zorder=0)
ax.axhline(-0.144, color='#C00', ls='--', lw=1.8)
ax.text(TRANS_SWEEP[0], -0.144, ' Table S3 (−0.144)', color='#C00',
        va='bottom', fontsize=10, fontweight='bold')
ax.set_xscale('log')
ax.set_xlabel('transition-fraction threshold', fontsize=13, fontweight='bold')
ax.set_ylabel("continental degradation Cohen's $d$", fontsize=13, fontweight='bold')
ax.set_title('Sensitivity of the continental degradation effect\n'
             'to area-weighted classification thresholds',
             fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=9.5, frameon=False, ncol=2)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'FigureS4_threshold_sensitivity.png'),
            dpi=300, facecolor='white')

with open(os.path.join(OUT_DIR, 'R1_run_log.txt'), 'w', encoding='utf-8') as f:
    f.write("R1 RECONCILIATION AND SWEEP\n" + "=" * 60 + "\n\n")
    f.write(f"Threshold pair A (Table S3): trans > {TABLES3_TRANS} / "
            f"stable >= {TABLES3_STABLE}\n")
    f.write(f"Threshold pair B (Table S9): trans > {TABLES9_TRANS} / "
            f"stable >= {TABLES9_STABLE}\n")
    f.write(f"Primary timescale: {PRIMARY_TS}\n")
    f.write(f"Adopted bridge spec: {ADOPTED_THRESHOLD_SPEC}\n\n")
    f.write("FACTORIAL\n" + fac.to_string(index=False) + "\n\n")
    f.write("SWEEP\n" + piv.round(4).to_string() + "\n")

print(f"\nWritten to {OUT_DIR}")
print("  TableS12_areaweighted_reconciliation.csv   -> Supplementary Table S3")
print("  TableS13_threshold_sweep.csv")
print("  FigureS4_threshold_sensitivity.png")
print("  percell_long.csv")
print("\nINTERPRETATION")
print("  If the sweep changes sign across plausible thresholds, no single")
print("  area-weighted value is defensible and the degradation effect should")
print("  be reported as indistinguishable from zero across specifications.")
print("  If it is stable, adopt that specification and report it as primary.")
