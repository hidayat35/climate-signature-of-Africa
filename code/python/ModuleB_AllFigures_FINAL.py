"""
=============================================================================
MODULE B FIGURES — FINAL CONSOLIDATED SCRIPT
=============================================================================

Generates all 5 publication-quality Module B figures from the Tier 2 stricter
attribution analysis (Batch 2 outputs):

  FigureB1: Attribution heatmap (pathway × region × Cohen's d, both windows)
  FigureB2: Recovery Suppression Index (RSI) bar chart per region
  FigureB3: Category-level Cohen's d boxplots (with whisker clipping)
  FigureB4: Decadal shift (early 1990–2004 vs late 2005–2022)
  FigureB5: Timescale sensitivity (RSI/DRA across SPEI-12/24/36/60)

Each figure is saved as both PNG (for drafts/presentations) and PDF (for
publication) plus the underlying data CSV (for supplementary materials).

REQUIRED INPUTS (in standard project locations):
  ModuleB_results/
    TableB2_LAGGED_per_interval_STRICTER.csv
    TableB2_CUMULATIVE_per_interval_STRICTER.csv
    TableB2_LAGGED_SUMMARY_spei12_STRICTER.csv
    TableB2_CUMULATIVE_SUMMARY_spei12_STRICTER.csv

  statistics_for_paper/
    TableB4_RSI_timescale_sensitivity_lagged_STRICTER.csv
    TableB4_RSI_timescale_sensitivity_cumulative_STRICTER.csv
    TableB4_DRA_timescale_sensitivity_lagged_STRICTER.csv
    TableB4_DRA_timescale_sensitivity_cumulative_STRICTER.csv
    TableB4_Decadal_Shift_cumulative_STRICTER_FDR.csv

OUTPUTS in figures_for_paper/ModuleB/:
  FigureB1_attribution_heatmap.png/.pdf      + FigureB1_data_*.csv
  FigureB2_recovery_suppression.png/.pdf     + FigureB2_data.csv
  FigureB3_category_comparison.png/.pdf      + FigureB3_data.csv
  FigureB4_decadal_shift.png/.pdf            + FigureB4_data.csv
  FigureB5_timescale_sensitivity.png/.pdf    + FigureB5_data_*.csv

EXPECTED RUNTIME: ~30 seconds (just plotting from existing CSVs).
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
SPEI_DIR = r'D:\Claude idea\PhD_Paper3_Data\step5_spei_output'
RESULTS_DIR = os.path.join(SPEI_DIR, 'ModuleB_results')
STATS_DIR = os.path.join(SPEI_DIR, 'statistics_for_paper')
FIGS_DIR = os.path.join(SPEI_DIR, 'figures_for_paper', 'ModuleB')
os.makedirs(FIGS_DIR, exist_ok=True)

REGION_ORDER = ['MED', 'SAH', 'WAF', 'EAF', 'SAF']
REGION_NAMES = {'MED': 'Mediterranean', 'SAH': 'Sahara-Sahel',
                'WAF': 'West Africa', 'EAF': 'East Africa',
                'SAF': 'Southern Africa'}

CATEGORY_COLORS = {
    'Degradation': '#d62728',     # Red
    'Recovery':    '#2ca02c',     # Green
    'Agricultural':'#ff7f0e',     # Orange
}

REGION_COLORS = {
    'MED': '#d62728', 'SAH': '#ff7f0e', 'WAF': '#2ca02c',
    'EAF': '#1f77b4', 'SAF': '#9467bd'
}

PATHWAY_ORDER = [
    ('FST_SHR',        'Degradation'),
    ('SHR_GRS',        'Degradation'),
    ('FST_CRP',        'Degradation'),
    ('GRS_BAL',        'Degradation'),
    ('SHR_FST',        'Recovery'),
    ('GRS_SHR',        'Recovery'),
    ('BAL_GRS',        'Recovery'),
    ('AGEXPANSION',    'Agricultural'),
    ('CRP_ABANDONMENT','Agricultural'),
]

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.titlesize': 13,
    'font.family': 'DejaVu Sans',
})


def save_figure(fig, name):
    """Save both PNG (200 DPI) and PDF versions."""
    fig.savefig(os.path.join(FIGS_DIR, f'{name}.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS_DIR, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Load all input CSVs
# =============================================================================
print("=" * 75)
print("MODULE B FIGURES — FINAL CONSOLIDATED GENERATION")
print("=" * 75)

print("\nLoading input CSVs...")
df_lag = pd.read_csv(os.path.join(RESULTS_DIR,
    'TableB2_LAGGED_per_interval_STRICTER.csv'))
df_cum = pd.read_csv(os.path.join(RESULTS_DIR,
    'TableB2_CUMULATIVE_per_interval_STRICTER.csv'))
df_lag_summary = pd.read_csv(os.path.join(RESULTS_DIR,
    'TableB2_LAGGED_SUMMARY_spei12_STRICTER.csv'))
df_cum_summary = pd.read_csv(os.path.join(RESULTS_DIR,
    'TableB2_CUMULATIVE_SUMMARY_spei12_STRICTER.csv'))
df_rsi_lag = pd.read_csv(os.path.join(STATS_DIR,
    'TableB4_RSI_timescale_sensitivity_lagged_STRICTER.csv'))
df_rsi_cum = pd.read_csv(os.path.join(STATS_DIR,
    'TableB4_RSI_timescale_sensitivity_cumulative_STRICTER.csv'))
df_dra_lag = pd.read_csv(os.path.join(STATS_DIR,
    'TableB4_DRA_timescale_sensitivity_lagged_STRICTER.csv'))
df_dra_cum = pd.read_csv(os.path.join(STATS_DIR,
    'TableB4_DRA_timescale_sensitivity_cumulative_STRICTER.csv'))
df_dec_cum = pd.read_csv(os.path.join(STATS_DIR,
    'TableB4_Decadal_Shift_cumulative_STRICTER_FDR.csv'))

print(f"  Lagged per-interval:     {len(df_lag)} rows")
print(f"  Cumulative per-interval: {len(df_cum)} rows")
print(f"  RSI/DRA tables:          {len(df_rsi_lag)} rows each")
print(f"  Decadal shift:           {len(df_dec_cum)} rows")


# =============================================================================
# Helper: pathway × region p-values (for heatmap significance markers)
# =============================================================================
def pathway_region_pvalues(df_per_interval, ts='spei_12'):
    """One-sample t-test of cohens_d distribution against zero for each
    pathway × region cell."""
    df_ts = df_per_interval[df_per_interval['spei_timescale'] == ts]
    pvals = {}
    for p, _ in PATHWAY_ORDER:
        for code in REGION_ORDER:
            sub = df_ts[(df_ts['transition'] == p) & (df_ts['region'] == code)]
            ds = sub.loc[sub['cohens_d'].notna(), 'cohens_d'].values
            if len(ds) >= 3:
                _, pval = stats.ttest_1samp(ds, 0.0)
                pvals[(p, code)] = pval
            else:
                pvals[(p, code)] = np.nan
    return pvals


# =============================================================================
# FIGURE B1: Attribution heatmap (pathway × region × Cohen's d)
# =============================================================================
print("\n" + "=" * 75)
print("FIGURE B1: Attribution heatmap")
print("=" * 75)

pathway_names = [p[0] for p in PATHWAY_ORDER]
mat_lag = np.full((len(pathway_names), len(REGION_ORDER)), np.nan)
mat_cum = np.full((len(pathway_names), len(REGION_ORDER)), np.nan)

for i, (pname, _) in enumerate(PATHWAY_ORDER):
    for j, code in enumerate(REGION_ORDER):
        sub_l = df_lag_summary[(df_lag_summary['transition'] == pname) &
                                (df_lag_summary['region'] == code)]
        sub_c = df_cum_summary[(df_cum_summary['transition'] == pname) &
                                (df_cum_summary['region'] == code)]
        if len(sub_l) > 0 and pd.notna(sub_l['mean_cohens_d'].iloc[0]):
            mat_lag[i, j] = sub_l['mean_cohens_d'].iloc[0]
        if len(sub_c) > 0 and pd.notna(sub_c['mean_cohens_d'].iloc[0]):
            mat_cum[i, j] = sub_c['mean_cohens_d'].iloc[0]

pvals_lag = pathway_region_pvalues(df_lag)
pvals_cum = pathway_region_pvalues(df_cum)

fig, (ax_lag, ax_cum) = plt.subplots(1, 2, figsize=(15, 8), sharey=True,
                                       gridspec_kw={'wspace': 0.05})
norm = TwoSlopeNorm(vmin=-0.7, vcenter=0, vmax=0.7)

# Lagged panel
im = ax_lag.imshow(mat_lag, cmap='RdBu_r', norm=norm, aspect='auto')
ax_lag.set_xticks(range(len(REGION_ORDER)))
ax_lag.set_xticklabels(REGION_ORDER)
ax_lag.set_yticks(range(len(pathway_names)))
ax_lag.set_yticklabels(pathway_names, fontsize=9)
ax_lag.set_title('Lagged window (primary)')

for i in range(len(pathway_names)):
    for j in range(len(REGION_ORDER)):
        v = mat_lag[i, j]
        if np.isnan(v):
            continue
        color = 'white' if abs(v) > 0.4 else 'black'
        p = pvals_lag.get((PATHWAY_ORDER[i][0], REGION_ORDER[j]), np.nan)
        sig = '*' if (not np.isnan(p) and p < 0.05) else ''
        ax_lag.text(j, i, f'{v:+.2f}{sig}', ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')

# Category color strip on the left
for i, (_, cat) in enumerate(PATHWAY_ORDER):
    ax_lag.add_patch(plt.Rectangle((-0.7, i - 0.45), 0.15, 0.9,
                                     color=CATEGORY_COLORS[cat], alpha=0.7,
                                     transform=ax_lag.transData, clip_on=False))
ax_lag.set_xlim(-0.8, len(REGION_ORDER) - 0.5)

# Cumulative panel (y-ticks shared)
ax_cum.imshow(mat_cum, cmap='RdBu_r', norm=norm, aspect='auto')
ax_cum.set_xticks(range(len(REGION_ORDER)))
ax_cum.set_xticklabels(REGION_ORDER)
ax_cum.set_title('Cumulative window (robustness)')
ax_cum.tick_params(axis='y', which='both', left=False, labelleft=False)

for i in range(len(pathway_names)):
    for j in range(len(REGION_ORDER)):
        v = mat_cum[i, j]
        if np.isnan(v):
            continue
        color = 'white' if abs(v) > 0.4 else 'black'
        p = pvals_cum.get((PATHWAY_ORDER[i][0], REGION_ORDER[j]), np.nan)
        sig = '*' if (not np.isnan(p) and p < 0.05) else ''
        ax_cum.text(j, i, f'{v:+.2f}{sig}', ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Cohen's d (transition vs stable)", rotation=270,
               labelpad=18, fontsize=10)
cbar_ax.text(2.8, 0.95, 'drier\ntransitions', transform=cbar_ax.transAxes,
              ha='center', va='top', fontsize=8, fontweight='bold',
              color='#5b1010')
cbar_ax.text(2.8, 0.05, 'wetter\ntransitions', transform=cbar_ax.transAxes,
              ha='center', va='bottom', fontsize=8, fontweight='bold',
              color='#0a3057')

# Category legend at top
cat_patches = [Patch(color=CATEGORY_COLORS[c], alpha=0.7, label=c)
               for c in ['Degradation', 'Recovery', 'Agricultural']]
fig.legend(handles=cat_patches, loc='upper center', ncol=3,
           bbox_to_anchor=(0.46, 0.98), fontsize=10, framealpha=0.9)

fig.suptitle("Attribution heatmap — pathway × region (SPEI-12, stricter stable reference)\n"
             "* indicates one-sample t-test against zero p < 0.05",
             y=1.06, fontsize=12)
plt.subplots_adjust(left=0.08, right=0.90, top=0.88, bottom=0.08)
save_figure(fig, 'FigureB1_attribution_heatmap')

pd.DataFrame(mat_lag, index=pathway_names, columns=REGION_ORDER).to_csv(
    os.path.join(FIGS_DIR, 'FigureB1_data_lagged.csv'), encoding='utf-8')
pd.DataFrame(mat_cum, index=pathway_names, columns=REGION_ORDER).to_csv(
    os.path.join(FIGS_DIR, 'FigureB1_data_cumulative.csv'), encoding='utf-8')
print("  ✓ Saved FigureB1_attribution_heatmap.png/.pdf + data CSVs")


# =============================================================================
# FIGURE B2: Recovery Suppression Index per region
# =============================================================================
print("\n" + "=" * 75)
print("FIGURE B2: Recovery Suppression Index")
print("=" * 75)

fig, ax = plt.subplots(figsize=(11, 6))

rsi_lag_12 = df_rsi_lag[df_rsi_lag['spei_timescale'] == 'spei_12'].set_index('region')
rsi_cum_12 = df_rsi_cum[df_rsi_cum['spei_timescale'] == 'spei_12'].set_index('region')

x = np.arange(len(REGION_ORDER))
width = 0.35

lag_values = [rsi_lag_12.loc[r, 'RSI'] if r in rsi_lag_12.index else np.nan for r in REGION_ORDER]
cum_values = [rsi_cum_12.loc[r, 'RSI'] if r in rsi_cum_12.index else np.nan for r in REGION_ORDER]
lag_pvals = [rsi_lag_12.loc[r, 'p_value'] if r in rsi_lag_12.index else np.nan for r in REGION_ORDER]
cum_pvals = [rsi_cum_12.loc[r, 'p_value'] if r in rsi_cum_12.index else np.nan for r in REGION_ORDER]

ax.bar(x - width/2, lag_values, width, label='Lagged window (primary)',
       color='#1f77b4', edgecolor='black', linewidth=0.8)
ax.bar(x + width/2, cum_values, width, label='Cumulative window',
       color='#9467bd', edgecolor='black', linewidth=0.8)

# Significance asterisks
for i, (p, v) in enumerate(zip(lag_pvals, lag_values)):
    if not np.isnan(p) and p < 0.05:
        ax.text(i - width/2, v - 0.02, '*', ha='center', va='top',
                fontsize=20, fontweight='bold', color='black')
for i, (p, v) in enumerate(zip(cum_pvals, cum_values)):
    if not np.isnan(p) and p < 0.05:
        ax.text(i + width/2, v - 0.02, '*', ha='center', va='top',
                fontsize=20, fontweight='bold', color='black')

# Value annotations
for i, v in enumerate(lag_values):
    if not np.isnan(v):
        offset = 0.012 if v > 0 else -0.012
        ax.text(i - width/2, v + offset, f'{v:+.2f}', ha='center',
                va='bottom' if v > 0 else 'top', fontsize=9)
for i, v in enumerate(cum_values):
    if not np.isnan(v):
        offset = 0.012 if v > 0 else -0.012
        ax.text(i + width/2, v + offset, f'{v:+.2f}', ha='center',
                va='bottom' if v > 0 else 'top', fontsize=9)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([REGION_NAMES[r] for r in REGION_ORDER])
ax.set_ylabel("Recovery Suppression Index (Cohen's d)")
ax.set_title("Recovery Suppression Index by region (SPEI-12, stricter stable reference)\n"
             "Negative values indicate recovery occurred in wetter-than-typical conditions")
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(-0.6, 0.15)
ax.text(0.02, 0.97, "* p < 0.05 (one-sample t-test against zero)",
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

save_figure(fig, 'FigureB2_recovery_suppression')

pd.DataFrame({
    'region': REGION_ORDER,
    'RSI_lagged': lag_values, 'p_lagged': lag_pvals,
    'RSI_cumulative': cum_values, 'p_cumulative': cum_pvals,
}).to_csv(os.path.join(FIGS_DIR, 'FigureB2_data.csv'), index=False)
print("  ✓ Saved FigureB2_recovery_suppression.png/.pdf + data CSV")


# =============================================================================
# FIGURE B3: Category boxplots (with whisker clipping)
# =============================================================================
print("\n" + "=" * 75)
print("FIGURE B3: Category comparison boxplots")
print("=" * 75)

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
Y_MIN, Y_MAX = -1.0, 1.0

for ax, df, win in [(axes[0], df_lag, 'Lagged'), (axes[1], df_cum, 'Cumulative')]:
    df12 = df[(df['spei_timescale'] == 'spei_12') & df['cohens_d'].notna()]

    box_data = []
    box_labels = []
    box_colors = []

    for code in REGION_ORDER:
        for cat in ['Degradation', 'Recovery', 'Agricultural']:
            sub = df12[(df12['region'] == code) & (df12['category'] == cat)]
            ds = sub['cohens_d'].dropna().values
            box_data.append(ds if len(ds) > 0 else np.array([0]))
            box_labels.append(f'{code}\n{cat[:3]}')
            box_colors.append(CATEGORY_COLORS[cat])

    ax.set_ylim(Y_MIN, Y_MAX)

    # Detect which boxes have whiskers exceeding the cap, then clip data values
    box_data_for_render = []
    whisker_above = []
    whisker_below = []
    for ds in box_data:
        if len(ds) > 0:
            q1, q3 = np.percentile(ds, [25, 75])
            iqr = q3 - q1
            whisker_lo = ds[ds >= q1 - 1.5 * iqr].min() if (ds >= q1 - 1.5 * iqr).any() else q1
            whisker_hi = ds[ds <= q3 + 1.5 * iqr].max() if (ds <= q3 + 1.5 * iqr).any() else q3
            whisker_above.append(int(whisker_hi > Y_MAX))
            whisker_below.append(int(whisker_lo < Y_MIN))
            box_data_for_render.append(np.clip(ds, Y_MIN + 1e-3, Y_MAX - 1e-3))
        else:
            box_data_for_render.append(ds)
            whisker_above.append(0)
            whisker_below.append(0)

    bp = ax.boxplot(box_data_for_render, positions=range(len(box_data_for_render)),
                     widths=0.6, patch_artist=True, showmeans=True, showfliers=False,
                     meanprops={'marker': 'D', 'markerfacecolor': 'white',
                                'markeredgecolor': 'black', 'markersize': 6},
                     medianprops={'color': 'black', 'linewidth': 1.5})
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Truncation arrows
    for i, (above, below) in enumerate(zip(whisker_above, whisker_below)):
        if above:
            ax.text(i, Y_MAX - 0.04, '↑', ha='center', va='top',
                    fontsize=14, fontweight='bold', color='black')
        if below:
            ax.text(i, Y_MIN + 0.04, '↓', ha='center', va='bottom',
                    fontsize=14, fontweight='bold', color='black')

    # Outlier counts (true Tukey outliers)
    for i, ds in enumerate(box_data):
        if len(ds) > 4:
            q1, q3 = np.percentile(ds, [25, 75])
            iqr = q3 - q1
            n_out = ((ds < q1 - 1.5 * iqr) | (ds > q3 + 1.5 * iqr)).sum()
            if n_out > 0:
                ax.text(i, 0.96, f'(+{n_out})', ha='center',
                        transform=ax.get_xaxis_transform(),
                        fontsize=7, color='gray', alpha=0.8)

    ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax.set_xticks(range(len(box_labels)))
    ax.set_xticklabels(box_labels, fontsize=8)
    ax.set_ylabel("Cohen's d")
    ax.set_title(f'{win} window')
    ax.grid(axis='y', alpha=0.3)
    for sep in [3, 6, 9, 12]:
        ax.axvline(sep - 0.5, color='gray', linestyle=':', alpha=0.5)

patches = [Patch(color=c, alpha=0.7, label=cat)
           for cat, c in CATEGORY_COLORS.items()]
fig.legend(handles=patches, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 1.02), fontsize=10, framealpha=0.9)
fig.suptitle("Distribution of Cohen's d per category × region (SPEI-12, stricter stable reference)\n"
             "↑/↓ marks boxes with whiskers extending beyond ±1; outlier counts in parentheses",
             y=1.06, fontsize=12)
save_figure(fig, 'FigureB3_category_comparison')

# Underlying data
df_b3a = df_lag[(df_lag['spei_timescale'] == 'spei_12')][
    ['region', 'transition', 'category', 'interval', 'n_trans', 'cohens_d']
].rename(columns={'cohens_d': 'd_lagged'})
df_b3b = df_cum[(df_cum['spei_timescale'] == 'spei_12')][
    ['region', 'transition', 'category', 'interval', 'cohens_d']
].rename(columns={'cohens_d': 'd_cumulative'})
df_b3a.merge(df_b3b, on=['region', 'transition', 'category', 'interval'], how='outer'
              ).to_csv(os.path.join(FIGS_DIR, 'FigureB3_data.csv'), index=False)
print("  ✓ Saved FigureB3_category_comparison.png/.pdf + data CSV")


# =============================================================================
# FIGURE B4: Decadal shift (early vs late)
# =============================================================================
print("\n" + "=" * 75)
print("FIGURE B4: Decadal shift in attribution")
print("=" * 75)

n_regions = len(REGION_ORDER)
n_cats = 3
group_width = 1.0
gap_within = 0.1
cell_width = (group_width - gap_within * (n_cats - 1)) / n_cats
bar_width = cell_width * 0.4

x_positions = []
group_centers = []
labels_text = []
colors_for_cells = []
early_vals, late_vals, fdr_sigs = [], [], []

x_offset = 0
for ri, code in enumerate(REGION_ORDER):
    region_start = x_offset
    for ci, cat in enumerate(['Degradation', 'Recovery', 'Agricultural']):
        sub = df_dec_cum[(df_dec_cum['region'] == code) & (df_dec_cum['category'] == cat)]
        cell_center = x_offset + cell_width / 2
        early_x = cell_center - bar_width * 0.6
        late_x = cell_center + bar_width * 0.6
        x_positions.append((early_x, late_x, cell_center))
        labels_text.append(cat[:3])
        colors_for_cells.append(CATEGORY_COLORS[cat])
        if len(sub) > 0 and pd.notna(sub['delta_d'].iloc[0]):
            early_vals.append(sub['early_mean_d'].iloc[0])
            late_vals.append(sub['late_mean_d'].iloc[0])
            fdr_sigs.append(sub.get('sig_FDR_005', pd.Series([False])).iloc[0]
                            if 'sig_FDR_005' in sub.columns else False)
        else:
            early_vals.append(np.nan)
            late_vals.append(np.nan)
            fdr_sigs.append(False)
        x_offset += cell_width + gap_within
    group_centers.append((region_start + x_offset - gap_within) / 2)
    x_offset += 0.5

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)

all_vals = [v for v in early_vals + late_vals if not np.isnan(v)]
y_min = min(all_vals) - 0.18
y_max = max(all_vals) + 0.25
ax.set_ylim(y_min, y_max)

for (ex, lx, _), e, l, c, sig in zip(x_positions, early_vals, late_vals,
                                       colors_for_cells, fdr_sigs):
    if not np.isnan(e):
        ax.bar(ex, e, width=bar_width, color=c, alpha=0.4,
               edgecolor='black', linewidth=0.6)
    if not np.isnan(l):
        ax.bar(lx, l, width=bar_width, color=c, alpha=1.0,
               edgecolor='black', linewidth=0.6)
    if sig and not (np.isnan(e) or np.isnan(l)):
        max_bar = max(e, l)
        cell_center = (ex + lx) / 2
        ax.text(cell_center, max_bar + 0.05, '*', ha='center', va='bottom',
                fontsize=22, fontweight='bold', color='black')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(group_centers)
ax.set_xticklabels([REGION_NAMES[r] for r in REGION_ORDER], fontsize=11)
ax.tick_params(axis='x', pad=22)

# Sub-labels (Deg/Rec/Agr) just below x-axis line
for (_, _, cx), lbl in zip(x_positions, labels_text):
    ax.annotate(lbl, xy=(cx, 0), xytext=(0, -8),
                xycoords=('data', 'axes fraction'),
                textcoords='offset points',
                ha='center', va='top', fontsize=8, color='gray', clip_on=False)

ax.set_ylabel("Cohen's d")
ax.set_title("Decadal shift in attribution: early period (1990–2004) vs late period (2005–2022)\n"
             "Cumulative SPEI-12, stricter stable reference, Welch's t-test with Benjamini-Hochberg FDR")
ax.grid(axis='y', alpha=0.3)

# Legend in upper-right corner
legend_patches = []
for cat in ['Degradation', 'Recovery', 'Agricultural']:
    c = CATEGORY_COLORS[cat]
    legend_patches.append(Patch(facecolor=c, alpha=0.4, edgecolor='black',
                                 label=f'{cat} early (1990-2004)'))
    legend_patches.append(Patch(facecolor=c, alpha=1.0, edgecolor='black',
                                 label=f'{cat} late (2005-2022)'))
ax.legend(handles=legend_patches, loc='upper right', fontsize=8, framealpha=0.95,
          ncol=2, columnspacing=1.0)

ax.text(0.02, 0.98, "* survives FDR (Benjamini-Hochberg, α = 0.05)",
        transform=ax.transAxes, fontsize=9, va='top', style='italic',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.13)
save_figure(fig, 'FigureB4_decadal_shift')
df_dec_cum.to_csv(os.path.join(FIGS_DIR, 'FigureB4_data.csv'), index=False)
print("  ✓ Saved FigureB4_decadal_shift.png/.pdf + data CSV")


# =============================================================================
# FIGURE B5: Timescale sensitivity (RSI/DRA across SPEI-12/24/36/60)
# =============================================================================
print("\n" + "=" * 75)
print("FIGURE B5: Timescale sensitivity")
print("=" * 75)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
TS_VALUES = [12, 24, 36, 60]
TS_LABELS = ['spei_12', 'spei_24', 'spei_36', 'spei_60']


def plot_metric_panel(ax, df, metric, marker_shape, title, ylabel, sig_above):
    """Plot one panel of timescale sensitivity (RSI or DRA)."""
    for code in REGION_ORDER:
        df_r = df[df['region'] == code].sort_values('spei_timescale')
        if len(df_r) == 0:
            continue
        vals, pvals = [], []
        for ts in TS_LABELS:
            row = df_r[df_r['spei_timescale'] == ts]
            if len(row) > 0:
                vals.append(row[metric].iloc[0])
                pvals.append(row['p_value'].iloc[0])
            else:
                vals.append(np.nan)
                pvals.append(np.nan)

        ax.plot(TS_VALUES, vals, marker=marker_shape, markersize=9,
                color=REGION_COLORS[code], label=REGION_NAMES[code],
                linewidth=1.8, markeredgecolor='black', markeredgewidth=0.6)

        for x, y, p in zip(TS_VALUES, vals, pvals):
            if not np.isnan(y) and not np.isnan(p) and p < 0.05:
                if sig_above:
                    ax.text(x, y + 0.025, '*', ha='center', va='bottom',
                            fontsize=14, fontweight='bold', color=REGION_COLORS[code])
                else:
                    ax.text(x, y - 0.025, '*', ha='center', va='top',
                            fontsize=14, fontweight='bold', color=REGION_COLORS[code])

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(TS_VALUES)
    ax.set_xlabel('SPEI accumulation timescale (months)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)


# Set y-limits before plotting so significance markers fit
all_rsi_vals = pd.concat([df_rsi_lag['RSI'], df_rsi_cum['RSI']]).dropna()
rsi_ylim = (all_rsi_vals.min() - 0.07, all_rsi_vals.max() + 0.05)
for a in [axes[0, 0], axes[0, 1]]:
    a.set_ylim(rsi_ylim)

all_dra_vals = pd.concat([df_dra_lag['DRA'], df_dra_cum['DRA']]).dropna()
dra_ylim = (all_dra_vals.min() - 0.05, all_dra_vals.max() + 0.10)
for a in [axes[1, 0], axes[1, 1]]:
    a.set_ylim(dra_ylim)

plot_metric_panel(axes[0, 0], df_rsi_lag, 'RSI', 'o',
                   'RSI — Lagged window (primary)',
                   "RSI (Cohen's d)", sig_above=False)
plot_metric_panel(axes[0, 1], df_rsi_cum, 'RSI', 'o',
                   'RSI — Cumulative window',
                   "RSI (Cohen's d)", sig_above=False)
plot_metric_panel(axes[1, 0], df_dra_lag, 'DRA', 's',
                   'DRA — Lagged window (primary)',
                   "DRA (Cohen's d difference)", sig_above=True)
plot_metric_panel(axes[1, 1], df_dra_cum, 'DRA', 's',
                   'DRA — Cumulative window',
                   "DRA (Cohen's d difference)", sig_above=True)

# Single shared legend at bottom
handles = [plt.Line2D([0], [0], marker='o', color=REGION_COLORS[c],
                       linewidth=1.8, markersize=9,
                       markeredgecolor='black', markeredgewidth=0.6)
           for c in REGION_ORDER]
labels = [REGION_NAMES[c] for c in REGION_ORDER]
fig.legend(handles, labels, loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.01), fontsize=10, framealpha=0.9,
           columnspacing=2.0)

fig.suptitle("Timescale sensitivity of Recovery Suppression Index (RSI) "
             "and Degradation-Recovery Asymmetry (DRA)",
             y=0.995, fontsize=13)
fig.text(0.5, 0.04, "* indicates p < 0.05 (one-sample t-test against zero for RSI; "
                    "Welch's t-test deg vs rec for DRA)",
         ha='center', fontsize=9, style='italic')

plt.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.12,
                    hspace=0.32, wspace=0.22)
save_figure(fig, 'FigureB5_timescale_sensitivity')

pd.concat([df_rsi_lag.assign(window='lagged'),
           df_rsi_cum.assign(window='cumulative')]
          ).to_csv(os.path.join(FIGS_DIR, 'FigureB5_data_RSI.csv'), index=False)
pd.concat([df_dra_lag.assign(window='lagged'),
           df_dra_cum.assign(window='cumulative')]
          ).to_csv(os.path.join(FIGS_DIR, 'FigureB5_data_DRA.csv'), index=False)
print("  ✓ Saved FigureB5_timescale_sensitivity.png/.pdf + data CSVs")


# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 75)
print("ALL 5 MODULE B FIGURES GENERATED")
print("=" * 75)
print(f"""
  Output folder: {FIGS_DIR}

  Figures (PNG + PDF for each):
    1. FigureB1_attribution_heatmap         — pathway × region Cohen's d heatmap
    2. FigureB2_recovery_suppression        — RSI bar chart per region
    3. FigureB3_category_comparison         — category boxplots with whisker clipping
    4. FigureB4_decadal_shift               — early vs late period bars with FDR
    5. FigureB5_timescale_sensitivity       — RSI/DRA across SPEI-12/24/36/60

  Underlying data CSVs (for supplementary materials):
    FigureB1_data_lagged.csv, FigureB1_data_cumulative.csv
    FigureB2_data.csv
    FigureB3_data.csv
    FigureB4_data.csv
    FigureB5_data_RSI.csv, FigureB5_data_DRA.csv

  All figures use the Tier 2 stricter stable-pixel definition (Batch 2)
  and FDR-corrected p-values (Batch 1) for the decadal shift analysis.
""")
