"""
=============================================================================
BATCH 4 PART 1: Consolidated paper summary CSV
=============================================================================
Builds a single master CSV listing every numeric value that will appear in
the Methods or Results section of the paper, along with:
  - source CSV file
  - short description
  - section of the paper where it appears

This makes the final writeup mechanical rather than error-prone, and
provides a complete provenance trail for reviewers and supervisors.

OUTPUT: Paper3_summary_for_paper.csv
        (in statistics_for_paper/)

Categories of numbers compiled:
  A. Continental climate baseline (precip, PET, AED, drought freq)
  B. Regional climate (5 regions × precip, PET, AED, drought freq)
  C. Trend statistics (Modified MK)
  D. Decadal SPEI progression (5 regions × 4 decades)
  E. Category Cohen's d (continental, both windows, stricter)
  F. RSI per region (SPEI-12, both windows, stricter)
  G. DRA per region (SPEI-12, both windows, stricter)
  H. RSI timescale sensitivity (5 regions × 4 timescales × both windows)
  I. DRA timescale sensitivity (5 regions × 4 timescales × both windows)
  J. Decadal shift FDR-significant (cumulative, stricter)
  K. Pathway-level attribution headlines (top |d| values)
=============================================================================
"""

import os
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================
SPEI_DIR = r'D:\Claude idea\PhD_Paper3_Data\step5_spei_output'
RESULTS_DIR = os.path.join(SPEI_DIR, 'ModuleB_results')
STATS_DIR = os.path.join(SPEI_DIR, 'statistics_for_paper')

REGION_ORDER = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
REGION_NAMES = {'MED': 'Mediterranean', 'SAH': 'Sahara-Sahel',
                'WAF': 'West Africa', 'EAF': 'East Africa',
                'SAF': 'Southern Africa'}

print("=" * 75)
print("BATCH 4 PART 1: Consolidated paper summary CSV")
print("=" * 75)


# =============================================================================
# Container for the master CSV rows
# =============================================================================
rows = []

def add_row(category, paper_section, region, metric, value, source_csv,
            description, units='', notes=''):
    rows.append({
        'category': category,
        'paper_section': paper_section,
        'region': region,
        'metric': metric,
        'value': value,
        'units': units,
        'source_csv': source_csv,
        'description': description,
        'notes': notes,
    })


# =============================================================================
# CATEGORY A: Continental climate baseline
# =============================================================================
print("\n  Section A: Continental climate baseline...")
df1 = pd.read_csv(os.path.join(STATS_DIR, 'Table1_regional_climate_summary.csv'))
africa_row = df1[df1['region'] == 'AFRICA']
if len(africa_row) > 0:
    r = africa_row.iloc[0]
    add_row('A_climate', '3.1.1', 'AFRICA', 'mean_annual_precip',
            r['mean_annual_precip_mm'], 'Table1_regional_climate_summary.csv',
            'Mean annual precipitation, continental Africa, 1985-2022', 'mm/year')
    add_row('A_climate', '3.1.1', 'AFRICA', 'mean_annual_PET',
            r['mean_annual_PET_mm'], 'Table1_regional_climate_summary.csv',
            'Mean annual PET, continental Africa, 1985-2022', 'mm/year')
    add_row('A_climate', '3.1.1', 'AFRICA', 'aridity_ratio',
            round(r['mean_annual_precip_mm'] / r['mean_annual_PET_mm'], 2),
            'computed from Table1', 'P/PET aridity ratio, continental Africa', '')
    add_row('A_climate', '3.1.2', 'AFRICA', 'AED_contribution',
            r['aed_contribution_pct'], 'Table1_regional_climate_summary.csv',
            'AED contribution to drought severity (continental)', '%')
    add_row('A_climate', '3.1.2', 'AFRICA', 'pct_SPEI_drought',
            r['pct_SPEI_drought_below_m1'], 'Table1_regional_climate_summary.csv',
            'Continental % pixel-months with SPEI < -1', '%')
    add_row('A_climate', '3.1.2', 'AFRICA', 'pct_SPI_drought',
            r['pct_SPI_drought_below_m1'], 'Table1_regional_climate_summary.csv',
            'Continental % pixel-months with SPI < -1', '%')
    add_row('A_climate', '3.1.2', 'AFRICA', 'SPEI_SPI_gap',
            round(r['pct_SPEI_drought_below_m1'] - r['pct_SPI_drought_below_m1'], 1),
            'computed from Table1',
            'Continental SPEI-SPI drought-frequency gap (AED amplification)', '%')


# =============================================================================
# CATEGORY B: Regional climate (5 regions × climate metrics)
# =============================================================================
print("  Section B: Regional climate baselines...")
for code in REGION_ORDER:
    region_row = df1[df1['region'] == code]
    if len(region_row) == 0:
        continue
    r = region_row.iloc[0]
    add_row('B_climate', '3.1.1', code, 'mean_annual_precip',
            r['mean_annual_precip_mm'], 'Table1_regional_climate_summary.csv',
            f'Mean annual precipitation, {REGION_NAMES[code]}', 'mm/year')
    add_row('B_climate', '3.1.1', code, 'mean_annual_PET',
            r['mean_annual_PET_mm'], 'Table1_regional_climate_summary.csv',
            f'Mean annual PET, {REGION_NAMES[code]}', 'mm/year')
    add_row('B_climate', '3.1.2', code, 'AED_contribution',
            r['aed_contribution_pct'], 'Table1_regional_climate_summary.csv',
            f'AED contribution to drought severity, {REGION_NAMES[code]}', '%')
    add_row('B_climate', '3.1.2', code, 'pct_SPEI_drought',
            r['pct_SPEI_drought_below_m1'], 'Table1_regional_climate_summary.csv',
            f'% pixel-months with SPEI < -1, {REGION_NAMES[code]}', '%')
    add_row('B_climate', '3.1.2', code, 'pct_SPI_drought',
            r['pct_SPI_drought_below_m1'], 'Table1_regional_climate_summary.csv',
            f'% pixel-months with SPI < -1, {REGION_NAMES[code]}', '%')
    add_row('B_climate', '3.1.2', code, 'SPEI_SPI_gap',
            round(r['pct_SPEI_drought_below_m1'] - r['pct_SPI_drought_below_m1'], 1),
            'computed from Table1',
            f'SPEI-SPI drought-frequency gap, {REGION_NAMES[code]}', '%')


# =============================================================================
# CATEGORY C: Trend statistics (Modified Mann-Kendall, primary)
# =============================================================================
print("  Section C: Trend statistics (Modified MK)...")
df2_modMK = pd.read_csv(os.path.join(STATS_DIR, 'Table2_trend_statistics_ModifiedMK.csv'))

for code in REGION_ORDER:
    region_row = df2_modMK[df2_modMK['region'] == code]
    if len(region_row) == 0:
        continue
    r = region_row.iloc[0]
    add_row('C_trends', '3.1.3', code, 'pct_SPEI_drying_sig_modMK',
            r.get('SPEI12_pct_drying_sig', np.nan),
            'Table2_trend_statistics_ModifiedMK.csv',
            f'% pixels with significant SPEI-12 drying (Modified MK), {REGION_NAMES[code]}', '%',
            'Hamed-Rao autocorrelation correction')
    add_row('C_trends', '3.1.3', code, 'pct_PET_sig_trending_modMK',
            r.get('PET_pct_significant', np.nan),
            'Table2_trend_statistics_ModifiedMK.csv',
            f'% pixels with significant PET trend (Modified MK), {REGION_NAMES[code]}', '%')
    add_row('C_trends', '3.1.3', code, 'mean_PET_slope_modMK',
            r.get('PET_mean_slope_mm_per_year', np.nan),
            'Table2_trend_statistics_ModifiedMK.csv',
            f'Mean PET trend slope (Modified MK), {REGION_NAMES[code]}', 'mm/year')


# =============================================================================
# CATEGORY D: Decadal SPEI-12 progression
# =============================================================================
print("  Section D: Decadal SPEI progression...")
df7 = pd.read_csv(os.path.join(STATS_DIR, 'Table7_decadal_comparison.csv'))

for code in REGION_ORDER:
    region_data = df7[df7['region'] == code]
    for _, r in region_data.iterrows():
        add_row('D_decadal_spei', '3.1.4', code,
                f"SPEI12_{r['decade']}",
                r['mean_SPEI12'], 'Table7_decadal_comparison.csv',
                f"Decadal mean SPEI-12 for {REGION_NAMES[code]}, {r['decade'].replace('_','-')}",
                'sigma units')


# =============================================================================
# CATEGORY E: Category-level Cohen's d (stricter, both windows)
# =============================================================================
print("  Section E: Category Cohen's d (stricter)...")
df_lag_summary = pd.read_csv(os.path.join(RESULTS_DIR,
    'TableB2_LAGGED_SUMMARY_spei12_STRICTER.csv'))
df_cum_summary = pd.read_csv(os.path.join(RESULTS_DIR,
    'TableB2_CUMULATIVE_SUMMARY_spei12_STRICTER.csv'))

for cat in ['Degradation', 'Recovery', 'Agricultural']:
    cat_lag = df_lag_summary[df_lag_summary['category'] == cat]
    cat_cum = df_cum_summary[df_cum_summary['category'] == cat]
    if len(cat_lag):
        d_mean = round(cat_lag['mean_cohens_d'].mean(), 3)
        add_row('E_category_d', '3.2.2', 'AFRICA', f'd_{cat}_lagged', d_mean,
                'TableB2_LAGGED_SUMMARY_spei12_STRICTER.csv',
                f'Continental category-mean Cohen\'s d for {cat} (lagged window)',
                '', 'positive d = trans drier than stable')
    if len(cat_cum):
        d_mean = round(cat_cum['mean_cohens_d'].mean(), 3)
        add_row('E_category_d', '3.2.2', 'AFRICA', f'd_{cat}_cumulative', d_mean,
                'TableB2_CUMULATIVE_SUMMARY_spei12_STRICTER.csv',
                f'Continental category-mean Cohen\'s d for {cat} (cumulative window)',
                '', 'positive d = trans drier than stable')


# =============================================================================
# CATEGORY F: RSI per region (SPEI-12, both windows)
# =============================================================================
print("  Section F: RSI per region...")
df_rsi_lag = pd.read_csv(os.path.join(STATS_DIR,
    'TableB4_RSI_timescale_sensitivity_lagged_STRICTER.csv'))
df_rsi_cum = pd.read_csv(os.path.join(STATS_DIR,
    'TableB4_RSI_timescale_sensitivity_cumulative_STRICTER.csv'))

for code in REGION_ORDER:
    sub_lag = df_rsi_lag[(df_rsi_lag['region'] == code) &
                          (df_rsi_lag['spei_timescale'] == 'spei_12')]
    sub_cum = df_rsi_cum[(df_rsi_cum['region'] == code) &
                          (df_rsi_cum['spei_timescale'] == 'spei_12')]
    if len(sub_lag):
        r = sub_lag.iloc[0]
        add_row('F_RSI', '3.2.3', code, 'RSI_lagged_spei12', r['RSI'],
                'TableB4_RSI_timescale_sensitivity_lagged_STRICTER.csv',
                f'RSI for {REGION_NAMES[code]} (lagged, SPEI-12)', '')
        add_row('F_RSI', '3.2.3', code, 'RSI_lagged_spei12_pvalue', r['p_value'],
                'TableB4_RSI_timescale_sensitivity_lagged_STRICTER.csv',
                f'RSI p-value for {REGION_NAMES[code]} (lagged, SPEI-12)', '',
                'one-sample t-test against zero')
    if len(sub_cum):
        r = sub_cum.iloc[0]
        add_row('F_RSI', '3.2.3', code, 'RSI_cumulative_spei12', r['RSI'],
                'TableB4_RSI_timescale_sensitivity_cumulative_STRICTER.csv',
                f'RSI for {REGION_NAMES[code]} (cumulative, SPEI-12)', '')
        add_row('F_RSI', '3.2.3', code, 'RSI_cumulative_spei12_pvalue', r['p_value'],
                'TableB4_RSI_timescale_sensitivity_cumulative_STRICTER.csv',
                f'RSI p-value for {REGION_NAMES[code]} (cumulative, SPEI-12)', '')


# =============================================================================
# CATEGORY G: DRA per region (SPEI-12, both windows)
# =============================================================================
print("  Section G: DRA per region...")
df_dra_lag = pd.read_csv(os.path.join(STATS_DIR,
    'TableB4_DRA_timescale_sensitivity_lagged_STRICTER.csv'))
df_dra_cum = pd.read_csv(os.path.join(STATS_DIR,
    'TableB4_DRA_timescale_sensitivity_cumulative_STRICTER.csv'))

for code in REGION_ORDER:
    sub_lag = df_dra_lag[(df_dra_lag['region'] == code) &
                          (df_dra_lag['spei_timescale'] == 'spei_12')]
    sub_cum = df_dra_cum[(df_dra_cum['region'] == code) &
                          (df_dra_cum['spei_timescale'] == 'spei_12')]
    if len(sub_lag):
        r = sub_lag.iloc[0]
        add_row('G_DRA', '3.2.4', code, 'DRA_lagged_spei12', r['DRA'],
                'TableB4_DRA_timescale_sensitivity_lagged_STRICTER.csv',
                f'DRA for {REGION_NAMES[code]} (lagged, SPEI-12)', '',
                'positive DRA = degradation drier than recovery')
        add_row('G_DRA', '3.2.4', code, 'DRA_lagged_spei12_pvalue', r['p_value'],
                'TableB4_DRA_timescale_sensitivity_lagged_STRICTER.csv',
                f'DRA p-value for {REGION_NAMES[code]} (lagged, SPEI-12)', '',
                "Welch's t-test deg vs rec")
    if len(sub_cum):
        r = sub_cum.iloc[0]
        add_row('G_DRA', '3.2.4', code, 'DRA_cumulative_spei12', r['DRA'],
                'TableB4_DRA_timescale_sensitivity_cumulative_STRICTER.csv',
                f'DRA for {REGION_NAMES[code]} (cumulative, SPEI-12)', '')
        add_row('G_DRA', '3.2.4', code, 'DRA_cumulative_spei12_pvalue', r['p_value'],
                'TableB4_DRA_timescale_sensitivity_cumulative_STRICTER.csv',
                f'DRA p-value for {REGION_NAMES[code]} (cumulative, SPEI-12)', '')


# =============================================================================
# CATEGORY H: RSI timescale sensitivity (full grid)
# =============================================================================
print("  Section H: RSI timescale sensitivity...")
for ts in ['spei_12', 'spei_24', 'spei_36', 'spei_60']:
    for code in REGION_ORDER:
        for window_label, df_src, fname in [
            ('lagged', df_rsi_lag,
             'TableB4_RSI_timescale_sensitivity_lagged_STRICTER.csv'),
            ('cumulative', df_rsi_cum,
             'TableB4_RSI_timescale_sensitivity_cumulative_STRICTER.csv')
        ]:
            sub = df_src[(df_src['region'] == code) &
                          (df_src['spei_timescale'] == ts)]
            if len(sub):
                r = sub.iloc[0]
                add_row('H_RSI_timescale', '3.2.5', code,
                        f'RSI_{window_label}_{ts}', r['RSI'], fname,
                        f'RSI for {REGION_NAMES[code]} ({window_label}, {ts})', '')
                add_row('H_RSI_timescale', '3.2.5', code,
                        f'RSI_{window_label}_{ts}_pvalue', r['p_value'], fname,
                        f'RSI p-value for {REGION_NAMES[code]} ({window_label}, {ts})', '')


# =============================================================================
# CATEGORY I: DRA timescale sensitivity (full grid)
# =============================================================================
print("  Section I: DRA timescale sensitivity...")
for ts in ['spei_12', 'spei_24', 'spei_36', 'spei_60']:
    for code in REGION_ORDER:
        for window_label, df_src, fname in [
            ('lagged', df_dra_lag,
             'TableB4_DRA_timescale_sensitivity_lagged_STRICTER.csv'),
            ('cumulative', df_dra_cum,
             'TableB4_DRA_timescale_sensitivity_cumulative_STRICTER.csv')
        ]:
            sub = df_src[(df_src['region'] == code) &
                          (df_src['spei_timescale'] == ts)]
            if len(sub):
                r = sub.iloc[0]
                add_row('I_DRA_timescale', '3.2.5', code,
                        f'DRA_{window_label}_{ts}', r['DRA'], fname,
                        f'DRA for {REGION_NAMES[code]} ({window_label}, {ts})', '')
                add_row('I_DRA_timescale', '3.2.5', code,
                        f'DRA_{window_label}_{ts}_pvalue', r['p_value'], fname,
                        f'DRA p-value for {REGION_NAMES[code]} ({window_label}, {ts})', '')


# =============================================================================
# CATEGORY J: Decadal shift FDR-significant (cumulative, stricter)
# =============================================================================
print("  Section J: Decadal shifts (cumulative, FDR-corrected)...")
df_dec_cum = pd.read_csv(os.path.join(STATS_DIR,
    'TableB4_Decadal_Shift_cumulative_STRICTER_FDR.csv'))

for _, r in df_dec_cum.iterrows():
    if pd.notna(r.get('delta_d', np.nan)):
        add_row('J_decadal_shift', '3.2.6', r['region'],
                f"delta_d_{r['category']}", r['delta_d'],
                'TableB4_Decadal_Shift_cumulative_STRICTER_FDR.csv',
                f"Decadal shift Δd ({r['category']}, {r['region']}, "
                f"early 1990-2004 vs late 2005-2022)", '',
                f"survives FDR α=0.05: {r.get('sig_FDR_005', '')}")
        add_row('J_decadal_shift', '3.2.6', r['region'],
                f"p_raw_{r['category']}", r['p_value'],
                'TableB4_Decadal_Shift_cumulative_STRICTER_FDR.csv',
                f"Raw p-value ({r['category']}, {r['region']})", '')
        add_row('J_decadal_shift', '3.2.6', r['region'],
                f"q_BH_{r['category']}", r.get('q_value_bh', np.nan),
                'TableB4_Decadal_Shift_cumulative_STRICTER_FDR.csv',
                f"FDR-corrected q-value ({r['category']}, {r['region']})", '')


# =============================================================================
# CATEGORY K: Pathway-level attribution headlines
# (the strongest pathway × region cells from the heatmap)
# =============================================================================
print("  Section K: Pathway-level attribution headlines...")

KEY_PATHWAY_REGION = [
    ('AGEXPANSION', 'MED', '3.2.2',
     'AGEXPANSION in MED: largest negative d (lagged), Mediterranean cropland '
     'expansion strictly wet-window restricted'),
    ('BAL_GRS', 'SAH', '3.2.2',
     'BAL_GRS in SAH: bare-to-grassland recovery in Sahel, strongest '
     'recovery-suppression cell (lagged d = -0.50)'),
    ('GRS_BAL', 'SAF', '3.2.2',
     'GRS_BAL in SAF: grassland-to-bare degradation in Southern Africa, '
     'strongest desertification cell (lagged d = +0.48)'),
    ('CRP_ABANDONMENT', 'MED', '3.2.2',
     'CRP_ABANDONMENT in MED: significant negative d in lagged AND cumulative'),
    ('GRS_SHR', 'SAH', '3.2.2',
     'GRS_SHR in SAH: grass-to-shrub recovery, significant in cumulative window'),
]

for pname, code, section, desc in KEY_PATHWAY_REGION:
    sub_lag = df_lag_summary[(df_lag_summary['transition'] == pname) &
                              (df_lag_summary['region'] == code)]
    sub_cum = df_cum_summary[(df_cum_summary['transition'] == pname) &
                              (df_cum_summary['region'] == code)]
    if len(sub_lag):
        add_row('K_pathway_headlines', section, code,
                f'd_lagged_{pname}',
                round(sub_lag['mean_cohens_d'].iloc[0], 3),
                'TableB2_LAGGED_SUMMARY_spei12_STRICTER.csv',
                desc, '')
    if len(sub_cum):
        add_row('K_pathway_headlines', section, code,
                f'd_cumulative_{pname}',
                round(sub_cum['mean_cohens_d'].iloc[0], 3),
                'TableB2_CUMULATIVE_SUMMARY_spei12_STRICTER.csv',
                desc, '')


# =============================================================================
# Save consolidated CSV
# =============================================================================
df_summary = pd.DataFrame(rows)

# Sort by category, then paper section, then region
category_order = {f'{x}_': i for i, x in enumerate(
    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])}
df_summary['_sort_cat'] = df_summary['category'].apply(
    lambda x: category_order.get(x[:2], 99))

# Region sort order
region_sort = {'AFRICA': 0, 'MED': 1, 'SAH': 2, 'WAF': 3, 'EAF': 4, 'SAF': 5}
df_summary['_sort_region'] = df_summary['region'].map(region_sort).fillna(99)

df_summary = df_summary.sort_values(['_sort_cat', '_sort_region', 'metric'])
df_summary = df_summary.drop(columns=['_sort_cat', '_sort_region'])

outfile = os.path.join(STATS_DIR, 'Paper3_summary_for_paper.csv')
df_summary.to_csv(outfile, index=False, encoding='utf-8')

print(f"\n  Saved: Paper3_summary_for_paper.csv ({len(df_summary)} rows)")


# =============================================================================
# Final diagnostic: counts per category
# =============================================================================
print("\n" + "=" * 75)
print("CATEGORY COUNTS")
print("=" * 75)
counts = df_summary['category'].value_counts().sort_index()
for cat, n in counts.items():
    descriptions = {
        'A_climate': 'Continental climate baseline',
        'B_climate': 'Regional climate baselines (5 regions)',
        'C_trends': 'Trend statistics (Modified MK)',
        'D_decadal_spei': 'Decadal SPEI-12 progression',
        'E_category_d': "Continental Cohen's d per category",
        'F_RSI': 'RSI per region (SPEI-12)',
        'G_DRA': 'DRA per region (SPEI-12)',
        'H_RSI_timescale': 'RSI timescale sensitivity',
        'I_DRA_timescale': 'DRA timescale sensitivity',
        'J_decadal_shift': 'Decadal shifts (FDR-corrected)',
        'K_pathway_headlines': 'Pathway-level headline numbers',
    }
    print(f"  {cat:25} ({descriptions.get(cat, ''):<45}): {n:>3} rows")

print(f"\n  TOTAL: {len(df_summary)} numeric values cited in the paper")


# =============================================================================
# Print key headline numbers for verification
# =============================================================================
print("\n" + "=" * 75)
print("KEY HEADLINE NUMBERS (verify before writeup)")
print("=" * 75)

print("\n  Continental:")
for m in ['mean_annual_precip', 'mean_annual_PET', 'AED_contribution',
          'pct_SPEI_drought', 'pct_SPI_drought', 'SPEI_SPI_gap']:
    v = df_summary[(df_summary['region'] == 'AFRICA') & (df_summary['metric'] == m)]
    if len(v):
        print(f"    {m:30}: {v['value'].iloc[0]:>10}")

print("\n  AED contribution by region:")
for code in REGION_ORDER:
    v = df_summary[(df_summary['region'] == code) &
                    (df_summary['metric'] == 'AED_contribution')]
    if len(v):
        print(f"    {REGION_NAMES[code]:25}: {v['value'].iloc[0]:>10}%")

print("\n  Continental category Cohen's d:")
for cat in ['Degradation', 'Recovery', 'Agricultural']:
    for win in ['lagged', 'cumulative']:
        v = df_summary[(df_summary['region'] == 'AFRICA') &
                        (df_summary['metric'] == f'd_{cat}_{win}')]
        if len(v):
            print(f"    {cat:14} ({win:11}): d = {v['value'].iloc[0]:>+8.3f}")

print("\n  RSI per region (lagged SPEI-12, primary):")
for code in REGION_ORDER:
    v = df_summary[(df_summary['region'] == code) &
                    (df_summary['metric'] == 'RSI_lagged_spei12')]
    p = df_summary[(df_summary['region'] == code) &
                    (df_summary['metric'] == 'RSI_lagged_spei12_pvalue')]
    if len(v) and len(p):
        sig = ' ***' if p['value'].iloc[0] < 0.05 else ''
        print(f"    {REGION_NAMES[code]:25}: RSI = {v['value'].iloc[0]:>+7.3f}, "
              f"p = {p['value'].iloc[0]:.4f}{sig}")

print("\n  DRA per region (lagged SPEI-12, primary):")
for code in REGION_ORDER:
    v = df_summary[(df_summary['region'] == code) &
                    (df_summary['metric'] == 'DRA_lagged_spei12')]
    p = df_summary[(df_summary['region'] == code) &
                    (df_summary['metric'] == 'DRA_lagged_spei12_pvalue')]
    if len(v) and len(p):
        sig = ' ***' if p['value'].iloc[0] < 0.05 else ''
        print(f"    {REGION_NAMES[code]:25}: DRA = {v['value'].iloc[0]:>+7.3f}, "
              f"p = {p['value'].iloc[0]:.4f}{sig}")

print("\n  FDR-significant decadal shifts (cumulative, stricter):")
fdr_rows = []
for _, r in df_dec_cum.iterrows():
    if r.get('sig_FDR_005', False):
        fdr_rows.append((r['region'], r['category'], r['delta_d'],
                          r.get('q_value_bh', np.nan)))
for region, category, delta, q in fdr_rows:
    print(f"    {region:5} {category:14}: Δd = {delta:>+6.2f}, q = {q:.4f}")
print(f"    (Total: {len(fdr_rows)}/15 shifts survive FDR α=0.05)")


# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 75)
print("BATCH 4 PART 1 COMPLETE")
print("=" * 75)
print(f"""
  Single master CSV: {outfile}
  Contains {len(df_summary)} numeric values across 11 categories.

  Each row has:
    - category    : grouping (A_climate, B_climate, C_trends, etc.)
    - paper_section : where in the paper this number appears (e.g. 3.1.1, 3.2.4)
    - region      : AFRICA, MED, SAH, WAF, EAF, or SAF
    - metric      : descriptive metric name
    - value       : the actual number
    - units       : units (mm/year, %, sigma units, etc.)
    - source_csv  : which raw CSV the value comes from
    - description : human-readable description
    - notes       : caveats, methodology notes

  VERIFY before writeup:
    1. Continental AED = 26.0%
    2. SAH AED = 44.4%, MED AED = 41.9%
    3. Recovery d (lagged) = -0.18 continental
    4. SAH RSI (lagged) = -0.484, p = 0.048 ***
    5. SAH DRA (lagged) = +0.608, p = 0.031 ***
    6. 5 cumulative decadal shifts survive FDR

  Next: Batch 4 Part 2 — Final Methods + Results writeup
""")
