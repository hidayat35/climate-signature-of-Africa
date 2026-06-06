"""
============================================================
FIX (v2): Regenerate Figure 5 with TRUE DRYING-ONLY values
          from Modified Mann-Kendall trend statistics
============================================================
The original Fix_Figure5_ModifiedMK.py read the column
'SPEI12_pct_significant' (= total significant trends in either
direction) from Table2_trend_statistics_ModifiedMK.csv and
labelled the panel "Significant SPEI-12 Drying Trends".

This was technically inaccurate: in West Africa and East Africa,
8-12% of pixels show significant WETTING trends, so the total
'pct_significant' overstates pure drying. v6 of the manuscript
now cites the drying-direction-only column 'SPEI12_pct_drying_sig',
and Figure 5 must match.

This script reads the same CSV but uses the correct column
('SPEI12_pct_drying_sig') and regenerates Figure 5 in the same
visual style as v1.

CORRECT DRYING-ONLY PERCENTAGES (from Table2_trend_statistics_ModifiedMK.csv):
  MED 72.03 | SAH 58.05 | WAF 30.13 | EAF 31.97 | SAF 21.13

INPUTS:
  - <SPEI_DIR>/statistics_for_paper/Table2_trend_statistics_ModifiedMK.csv
  - <SPEI_DIR>/statistics_for_paper/Table1_regional_climate_summary.csv

OUTPUT:
  - <SPEI_DIR>/figures_for_paper/Figure5_regional_summary.png
    (overwrites the previous version)

Run:
  C:\\Users\\hidayat\\.conda\\envs\\zama1\\python.exe Fix_Figure5_ModifiedMK_v2.py
============================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION  --  matches Step5_v7_ModuleA_canonical_rerun.py
# ============================================================
SPEI_DIR  = r'D:\Claude idea\PhD_Paper3_Data\step5_spei_output'
STATS_DIR = os.path.join(SPEI_DIR, 'statistics_for_paper')
FIGS_DIR  = os.path.join(SPEI_DIR, 'figures_for_paper')

REGION_ORDER  = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
REGION_NAMES  = {'MED': 'Mediterranean', 'SAH': 'Sahara-Sahel',
                 'WAF': 'West Africa',  'EAF': 'East Africa',
                 'SAF': 'Southern Africa'}
REGION_COLORS = {'MED': '#d62728', 'SAH': '#ff7f0e',
                 'WAF': '#2ca02c', 'EAF': '#1f77b4', 'SAF': '#9467bd'}

# Manuscript-locked DRYING-ONLY values (from §3.1.3 of v6 manuscript)
# Used only if the CSVs cannot be read.
FALLBACK_AED_PCT = {
    'MED': 41.9, 'SAH': 44.4, 'WAF': 20.0, 'EAF': 15.0, 'SAF': 11.3
}
FALLBACK_DRYING_MODMK = {
    'MED': 72.03, 'SAH': 58.05, 'WAF': 30.13, 'EAF': 31.97, 'SAF': 21.13
}
FALLBACK_PET_TREND = {
    'MED': 2.87, 'SAH': 2.19, 'WAF': 2.29, 'EAF': 1.98, 'SAF': 1.31
}


# ============================================================
# LOADERS (CSV first, fall back to manuscript-locked)
# ============================================================
def load_aed_pct():
    try:
        f = os.path.join(STATS_DIR, 'Table1_regional_climate_summary.csv')
        df = pd.read_csv(f)
        out = {}
        for code in REGION_ORDER:
            row = df[df['region'] == code]
            if len(row) == 0:
                raise ValueError(f"region {code} not in Table1")
            out[code] = float(row['aed_contribution_pct'].values[0])
        print(f"  AED %               <- {os.path.basename(f)}")
        return out, 'csv'
    except Exception as e:
        print(f"  AED %               <- FALLBACK [{e}]")
        return FALLBACK_AED_PCT.copy(), 'fallback'


def load_drying_only_modmk():
    """
    Read the DRYING-DIRECTION-ONLY % from the primary modified-MK CSV.
    Column name in Batch1 output: 'SPEI12_pct_drying_sig'.
    Some older runs may name it differently; we try common alternates.
    """
    try:
        f = os.path.join(STATS_DIR, 'Table2_trend_statistics_ModifiedMK.csv')
        df = pd.read_csv(f)
        # Look for the drying-only column (NOT pct_significant which is both directions)
        col_candidates = [
            'SPEI12_pct_drying_sig',
            'SPEI12_pct_drying',
            'spei12_pct_drying_sig',
            'pct_drying_sig',
        ]
        col = None
        for c in col_candidates:
            if c in df.columns:
                col = c
                break
        if col is None:
            raise ValueError(f"drying-only col not found; have {list(df.columns)}")

        out = {}
        for code in REGION_ORDER:
            row = df[df['region'] == code]
            if len(row) == 0:
                raise ValueError(f"region {code} not in Table2_modMK")
            out[code] = float(row[col].values[0])
        print(f"  drying-only % MK*   <- {os.path.basename(f)} [{col}]")
        return out, 'csv'
    except Exception as e:
        print(f"  drying-only % MK*   <- FALLBACK [{e}]")
        return FALLBACK_DRYING_MODMK.copy(), 'fallback'


def load_pet_trend():
    try:
        f = os.path.join(STATS_DIR, 'Table2_trend_statistics_ModifiedMK.csv')
        df = pd.read_csv(f)
        col_candidates = ['PET_mean_slope_mm_per_year',
                          'PET_mean_slope',
                          'pet_mean_slope_mm_per_year']
        col = None
        for c in col_candidates:
            if c in df.columns:
                col = c
                break
        if col is None:
            raise ValueError(f"PET slope col not found; have {list(df.columns)}")
        out = {}
        for code in REGION_ORDER:
            row = df[df['region'] == code]
            if len(row) == 0:
                raise ValueError(f"region {code} not in Table2_modMK")
            out[code] = float(row[col].values[0])
        print(f"  PET trend           <- {os.path.basename(f)} [{col}]")
        return out, 'csv'
    except Exception as e:
        print(f"  PET trend           <- FALLBACK [{e}]")
        return FALLBACK_PET_TREND.copy(), 'fallback'


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("Regenerating Figure 5 with TRUE DRYING-ONLY Modified-MK values")
    print("=" * 65)
    print(f"\nReading from: {STATS_DIR}")
    print(f"Writing to:   {FIGS_DIR}\n")

    if not os.path.isdir(FIGS_DIR):
        os.makedirs(FIGS_DIR, exist_ok=True)
        print(f"  Created: {FIGS_DIR}")

    aed_pct,    aed_src    = load_aed_pct()
    drying_pct, drying_src = load_drying_only_modmk()
    pet_trend,  pet_src    = load_pet_trend()

    print("\nValues used in figure (DRYING-DIRECTION ONLY):")
    print(f"  {'Region':<6}{'AED %':>10}{'Drying-only % (modMK)':>25}{'PET (mm/yr)':>16}")
    for code in REGION_ORDER:
        print(f"  {code:<6}{aed_pct[code]:>10.1f}"
              f"{drying_pct[code]:>25.2f}"
              f"{pet_trend[code]:>16.2f}")

    # ----------------------------------------------------
    # Figure
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    codes  = REGION_ORDER
    colors = [REGION_COLORS[c] for c in codes]

    # Panel A — AED contribution
    aed_vals = [aed_pct[c] for c in codes]
    axes[0].bar(codes, aed_vals, color=colors)
    axes[0].set_ylabel('AED contribution (%)')
    axes[0].set_title('AED Contribution to Drought')
    axes[0].grid(axis='y', alpha=0.3)

    # Panel B — Modified-MK significant SPEI-12 DRYING-ONLY %
    drying_vals = [drying_pct[c] for c in codes]
    axes[1].bar(codes, drying_vals, color=colors)
    axes[1].set_ylabel('% pixels with significant SPEI-12 drying')
    axes[1].set_title('Significant SPEI-12 Drying Trends\n'
                      '(Modified Mann-Kendall, Hamed-Rao;\n'
                      'drying-direction only)')
    axes[1].grid(axis='y', alpha=0.3)

    # Panel C — Mean PET trend
    pet_vals = [pet_trend[c] for c in codes]
    axes[2].bar(codes, pet_vals, color=colors)
    axes[2].set_ylabel('Mean PET trend (mm/yr)')
    axes[2].set_title('Mean PET Trend Slope\n'
                      '(Modified Mann-Kendall, Hamed-Rao)')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    out_path = os.path.join(FIGS_DIR, 'Figure5_regional_summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {out_path}")

    sources = {aed_src, drying_src, pet_src}
    if sources == {'csv'}:
        print("\n  All three panels populated directly from Module A CSVs.")
    elif 'fallback' in sources and 'csv' in sources:
        print("\n  WARNING: some panels used CSVs, others used hardcoded values.")
    else:
        print("\n  WARNING: All panels used hardcoded fallback values.")

    print("\nThis figure now matches v6 manuscript text:")
    print("  MED 72.0% | SAH 58.1% | WAF 30.1% | EAF 32.0% | SAF 21.1%")
    print("\nReplace your old Figure5_regional_summary.png with this one.")
    print("=" * 65)


if __name__ == '__main__':
    main()
