"""
MixedEffects_CategoryContrast.py

Linear mixed-effects model of the category contrast in antecedent-drought effect size:

    d[r, p, tau] = b0 + b1 Rec[p] + b2 Agr[p] + gamma[r] + u[p] + e[r, p, tau]

d is the interval-level Cohen's d of region r, transition pathway p and interval tau in the lagged
window. Degradation is the reference category, so b1 and b2 are the recovery-degradation and
agricultural-degradation contrasts. IPCC region (gamma) is a fixed effect and the pathway (u) a
random intercept. The model is fitted by restricted maximum likelihood (statsmodels MixedLM)
separately at each SPEI timescale, first with all seven intervals and then without the terminal
2020-2022 interval.

Input:  TableB2_LAGGED_per_interval_STRICTER.csv (written by
        Batch2_StricterStablePixel_TimescaleSensitivity.py)
Output: mixed_effects_contrast.csv, and a printed comparison with the values reported in the paper

Usage:
    python MixedEffects_CategoryContrast.py --lagged TableB2_LAGGED_per_interval_STRICTER.csv --out .

Requires numpy, pandas and statsmodels.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

TIMESCALES = ['spei_12', 'spei_24', 'spei_36', 'spei_60']
TERMINAL = '2020_2022'
TERMS = {'Recovery': "C(category, Treatment('Degradation'))[T.Recovery]",
         'Agricultural': "C(category, Treatment('Degradation'))[T.Agricultural]"}

# Values reported in the paper: coefficient, 95% CI and p, all seven intervals, n = 260 per timescale
REPORTED = {('Recovery', 'spei_12'): (-0.162, -0.314, -0.010, 0.037),
             ('Recovery', 'spei_24'): (-0.171, -0.306, -0.037, 0.013),
             ('Recovery', 'spei_36'): (-0.188, -0.313, -0.064, 0.003),
             ('Recovery', 'spei_60'): (-0.217, -0.341, -0.093, 0.0006),
             ('Agricultural', 'spei_12'): (-0.130, -0.295, 0.035, 0.121),
             ('Agricultural', 'spei_24'): (-0.123, -0.266, 0.021, 0.094),
             ('Agricultural', 'spei_36'): (-0.126, -0.257, 0.005, 0.059),
             ('Agricultural', 'spei_60'): (-0.130, -0.260, 0.000, 0.0501)}


def fit(df):
    """REML fit with the statsmodels default optimiser; Powell and Nelder-Mead are tried only if it
    fails or does not converge."""
    model = smf.mixedlm("cohens_d ~ C(category, Treatment('Degradation')) + C(region)", df,
                        groups=df['transition'])
    res, notes = None, []
    for method in (None, 'powell', 'nm'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                res = model.fit(reml=True) if method is None else model.fit(reml=True, method=method)
            except (np.linalg.LinAlgError, ValueError) as e:
                notes.append(f'{method or "default"} optimiser failed: {e}')
                continue
        notes += sorted({str(w.message).split('\n')[0] for w in caught})
        se = np.asarray(res.bse_fe, dtype=float)
        sound = bool(res.converged) and np.all(np.isfinite(se)) and se.max() < 1.0
        if sound:
            if method is not None:
                notes.append(f'fitted with the {method} optimiser')
            break
        notes.append(f'{method or "default"} optimiser gave a degenerate fit; trying the next one')
        res = None
    if res is None:
        raise RuntimeError('the mixed model could not be fitted; see the notes above')
    return res, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lagged', default='TableB2_LAGGED_per_interval_STRICTER.csv')
    ap.add_argument('--out', default='.')
    a = ap.parse_args()

    data = pd.read_csv(a.lagged)
    data = data[data['cohens_d'].notna()].copy()
    rows = []
    for subset in ('all seven intervals', 'without 2020-2022'):
        for ts in TIMESCALES:
            df = data[data['spei_timescale'] == ts]
            if subset != 'all seven intervals':
                df = df[df['interval'].astype(str) != TERMINAL]
            res, notes = fit(df)
            ci = res.conf_int()
            for cat, term in TERMS.items():
                rows.append({'intervals': subset, 'timescale': ts, 'category': cat,
                             'beta': res.params[term], 'ci_low': ci.loc[term, 0], 'ci_high': ci.loc[term, 1],
                             'p': res.pvalues[term], 'n': int(res.nobs),
                             'n_pathways': int(df['transition'].nunique()),
                             'converged': bool(res.converged)})
            for n_ in notes:
                print(f'  note ({subset}, {ts}): {n_}')

    out = pd.DataFrame(rows)
    path = os.path.join(a.out, 'mixed_effects_contrast.csv')
    out.to_csv(path, index=False, float_format='%.6f')

    print('\nAll seven intervals, compared with the values reported in the paper (rounded to three decimals):')
    print(f"{'category':13s} {'timescale':9s} {'beta':>8s} {'95% CI':>20s} {'p':>8s} {'n':>4s}  reported")
    worst = 0.0
    for r in out[out.intervals == 'all seven intervals'].itertuples():
        pub = REPORTED[(r.category, r.timescale)]
        diff = max(abs(round(r.beta, 3) - pub[0]), abs(round(r.ci_low, 3) - pub[1]),
                   abs(round(r.ci_high, 3) - pub[2]))
        worst = max(worst, diff)
        print(f'{r.category:13s} {r.timescale:9s} {r.beta:+8.3f} ({r.ci_low:+.3f}, {r.ci_high:+.3f}) '
              f'{r.p:8.4f} {r.n:4d}  {pub[0]:+.3f} ({pub[1]:+.3f}, {pub[2]:+.3f}) p = {pub[3]}')
    print(f'\nLargest difference from the reported coefficients and limits: {worst:.3f}'
          + ('  (matches)' if worst <= 0.001 else '  (CHECK)'))
    print('Expected n = 260 per timescale.')

    w = out[(out.intervals == 'without 2020-2022') & (out.category == 'Recovery')]
    print(f'\nWithout 2020-2022, recovery contrast: beta {w.beta.max():+.3f} to {w.beta.min():+.3f}, '
          f'largest p = {w.p.max():.3f} (reported: -0.147 to -0.197, p <= 0.026)')
    print(f'\nWritten: {path}')


if __name__ == '__main__':
    main()
