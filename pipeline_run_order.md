# Pipeline run order

Each stage lists its scripts in order, with their main inputs and outputs. Stage A runs in the browser; Stages B to D run locally.

```bash
pip install -r requirements.txt
cp config.py.template config.py     # then edit the paths in config.py and at the top of each script
```

## Stage A: Earth Engine exports

Run these at <https://code.earthengine.google.com> and download the exports from Google Drive. Expect about 2 to 4 hours of export time.

| Order | Script | Output |
|---|---|---|
| A1 | `code/gee/Step2_v2_GEE_Monthly_Export.js` | `precip_monthly_1985_2022.tif`, `pet_monthly_1985_2022.tif` (456 monthly bands each) |
| A2 | `code/gee/ModuleB_GEE_Step1c_StateRasters.js` | `state_*.tif` from-class rasters (1 km) |
| A3 | `code/gee/ModuleB_Step1_GEE_v2_9transitions.js` | `transition_*.tif` rasters for the nine pathways (300 m) |

## Stage B: Climate indices and effect sizes

| Order | Script | Main outputs |
|---|---|---|
| B1 | `Step5_v6_FIXED_climate_indices.py` | SPEI (Pearson III) and SPI (gamma) at 12, 24, 36 and 60 months; interval-mean SPEI rasters |
| B2 | `Step5_v7_ModuleA_canonical_rerun.py` | `Table1_regional_climate_summary.csv`, `Table3_annual_timeseries_per_region.csv`, `Table7_decadal_comparison.csv`; `aed_contribution_mean.tif`; trend rasters with the standard test |
| B3 | `Batch1_ModifiedMK_FDR_UnitFix.py` | `*_trend_slope_modMK.tif`, `*_trend_pvalue_modMK.tif`; `Table2_trend_statistics_ModifiedMK.csv`; `Table2_comparison_standardMK_vs_modifiedMK.csv` |
| B4 | `Batch2_StricterStablePixel_TimescaleSensitivity.py` | `TableB2_LAGGED_per_interval_STRICTER.csv`, `TableB2_CUMULATIVE_per_interval_STRICTER.csv`; summaries at SPEI-12; RSI, DRA and early–late tables (`TableB4_*`); stable-pixel retention |
| B5 | `Generate_aridity_raster.py` | aridity index and classes |

The two `TableB2_*_per_interval_STRICTER.csv` files are read by every script in Stages C and D.

## Stage C: Robustness, aggregation and inference

| Order | Script | Main outputs |
|---|---|---|
| C1 | `Build_TableS11_RSI_SampleSizes.py` | `TableS14_RSI_sample_sizes.csv` |
| C2 | `Build_TableS3_AreaWeighted_Reconciliation.py` | `TableS9_areaweighted_per_interval.csv`, `TableS9b_areathresholded_per_interval.csv`, `TableS12_areaweighted_reconciliation.csv`, `TableS13_threshold_sweep.csv`, per-cell values |
| C3 | [matching script] | `TableS8_CEM_per_interval.csv`; covariate balance before matching |
| C4 | `Batch3_TwoStage_Aggregation_and_Inference.py` | `final_twostage.json`: continental means, BCa intervals, equivalence tests, Welch's and permutation tests, and the primary estimator over the six matched intervals |
| C5 | `MixedEffects_CategoryContrast.py` | `mixed_effects_contrast.csv` |

Only C2 and C3 need the exported rasters. C1, C4 and C5 read CSV files and finish within minutes.

C2 has its threshold settings at the top of the script. Area thresholding uses a transition fraction above 0.02 and a from-class fraction of at least 0.50. The header comment of the script explains how the per-interval files read by C4 are written.

## Stage D: Figures

| Order | Script | Output |
|---|---|---|
| D1 | `Export_map_layers_v2.py` | map layers, including `drought_frequency_gap.tif` and `precip_mean_annual.tif` |
| D2 | `Make_map_figures.py` | climate maps and regional series |
| D3 | `Make_publication_figures.py` | workflow, forest plot, heat map, regional indices and interval-level distributions |

The figure files keep the names of an earlier draft (for example `Fig5_forest.png`). They do not follow the numbering of the published article.
