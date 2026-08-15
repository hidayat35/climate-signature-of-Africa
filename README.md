# Antecedent drought and African land-cover transitions, 1985–2022

Analysis code for the paper:

> **Antecedent drought is more strongly associated with restricted vegetation recovery than with accelerated degradation across Africa, 1985–2022**
> Hidayat Ullah, Wilson Kalisa, Shawkat Ali, Jiahua Zhang

The study links 30-m land-cover transitions (GLC_FCS30D, 1985–2022) to antecedent drought across the five IPCC AR5 Africa reference regions, using canonical SPEI and SPI series computed from CHIRPS precipitation and TerraClimate potential evapotranspiration. It tests whether the climate signature acts through degradation acceleration, recovery restriction, or both.

The **derived data** (analysis rasters and published-table / figure-source CSVs) are archived on Zenodo at **https://doi.org/10.5281/zenodo.20572259**, because the rasters exceed GitHub's file-size limits.

## Headline findings

| Finding | Value | Reference |
|---|---|---|
| Continental recovery effect (Cohen's *d*), five estimators | −0.158 to −0.182, every interval excluding zero | §3.2 |
| Continental degradation effect | +0.015 to +0.040, no interval excluding zero | §3.2 |
| Recovery interval excludes zero | 8 of 8 timescale × window combinations | §3.3 |
| Recovery–degradation contrast | significant in all 8 combinations (Welch, permutation, mixed-effects) | §3.3, §3.4 |
| Degradation equivalent to zero within \|*d*\| = 0.20 | 13 of 14 estimator × timescale tests | §3.2 |
| Sahel Recovery Suppression Index | −0.484 (*p* = 0.048) | §3.2 |
| Continental AED contribution to drought severity | 26.0%, scaling from 11.3% (SAF) to 44.4% (SAH) | §3.1 |
| Wetting vs drying precipitation pixels, Sahara-Sahel | 5.4 : 1, while SPEI-12 dries over 58.1% of pixels | §3.1 |

## Aggregation rule

Continental category means are computed in **two stages**, and this matters for anyone reproducing the numbers:

1. Within each region × pathway cell, interval-level *d* values are collapsed by a **transition-pixel-weighted** mean.
2. Those cell means are then averaged with **equal weight** across cells.

Weighting within a cell reflects the information each interval contributes. Equal weighting across cells prevents the continental estimate from being determined by the few region × pathway combinations with the largest pixel counts, which would otherwise reduce it to a Southern and East African average: Southern Africa contributes 25,972 recovery pixels against the Mediterranean's 225.

Regional indices (Eq. 7) retain transition-pixel weighting throughout, because within a single region the disparity in cell size is far smaller.

All five estimators are evaluated on the **same set of region × pathway cells**, namely those evaluable under the primary point-sampling design, so that the estimators are directly comparable.

`Batch3_TwoStage_Aggregation_and_Inference.py` implements this and is the source of every continental figure in the paper.

## Repository layout

```
.
├── README.md                                   ← this file
├── LICENSE                                     ← MIT
├── requirements.txt                            ← pinned Python dependencies
├── pipeline_run_order.md                       ← full input/output map, stage by stage
├── config.py.template                          ← path-config template (copy → config.py)
└── code/
    ├── gee/                                    ← Earth Engine scripts (run in browser)
    │   ├── Step2_v2_GEE_Monthly_Export.js              ← CHIRPS + TerraClimate monthly
    │   ├── ModuleB_Step1_GEE_v2_9transitions.js        ← 9 transition pathways, 300 m
    │   └── ModuleB_GEE_Step1c_StateRasters.js          ← from-state rasters, 1 km
    └── python/
        ├── Step5_v6_FIXED_climate_indices.py           ← canonical SPEI (Pearson III) + SPI (gamma)
        ├── Step5_v7_ModuleA_canonical_rerun.py         ← Module A regional stats + map rasters
        ├── Step5_v5_DIAGNOSTIC_v2.py                   ← SPEI/SPI six-check validation
        ├── Batch1_ModifiedMK_FDR_UnitFix.py            ← Modified Mann-Kendall + Hamed-Rao + BH-FDR
        ├── Batch2_StricterStablePixel_TimescaleSensitivity.py
        │                                                 ← Module B effect sizes + RSI + DRA
        ├── Batch3_TwoStage_Aggregation_and_Inference.py ← continental means, CIs, TOST, contrasts
        ├── Build_TableS1_SPEI_Validation.py            ← Supplementary Table S1
        ├── Build_TableS2_MK_with_lag1.py               ← Supplementary Table S2
        ├── Build_TableS3_AreaWeighted_Reconciliation.py ← Supplementary Tables S3, S4
        ├── Build_TableS11_RSI_SampleSizes.py           ← Supplementary Table S11
        ├── Generate_Figure5_RSI_DRA.py                 ← Figure 5
        ├── Generate_Figure6_ForestPlot.py              ← Figure 6
        ├── Generate_All_Publication_Figures_Tables_v2.py ← consolidated figure/table generation
        ├── ModuleB_AllFigures_FINAL.py                 ← Module B figure set
        ├── Generate_aridity_raster.py                  ← aridity-index rasters for Figure 1
        └── Batch4_Part1_PaperSummaryCSV.py             ← master numerical-summary CSV
```

## Which script produces what

| Script | Produces | Rasters needed |
|---|---|---|
| `Step5_v6_FIXED_climate_indices.py` | SPEI and SPI series over the 1985–2000 calibration | yes |
| `Step5_v7_ModuleA_canonical_rerun.py` | regional climate statistics; AED and trend rasters | yes |
| `Step5_v5_DIAGNOSTIC_v2.py` | six-check SPEI/SPI validation | yes |
| `Batch1_ModifiedMK_FDR_UnitFix.py` | Modified Mann-Kendall trends; decadal-shift BH-FDR | yes |
| `Batch2_StricterStablePixel_TimescaleSensitivity.py` | per-cell effect sizes; RSI and DRA timescale tables | yes |
| `Batch3_TwoStage_Aggregation_and_Inference.py` | continental means, BCa intervals, TOST, Welch and permutation contrasts | no |
| `Build_TableS3_AreaWeighted_Reconciliation.py` | area-weighted specification comparison; 56-combination threshold sweep | yes |
| `Build_TableS11_RSI_SampleSizes.py` | cells, pathways and pixel counts behind every regional RSI | no |
| `Generate_Figure5_RSI_DRA.py` | Figure 5, with a printed audit of every plotted marker | no |
| `Generate_Figure6_ForestPlot.py` | Figure 6, from `final_twostage.json` | no |

| Manuscript item | Script |
|---|---|
| Table 1, Figures 2–3 | `Step5_v7_ModuleA_canonical_rerun.py` |
| Table 2 | `Batch1_ModifiedMK_FDR_UnitFix.py` |
| Figure 1 | `Generate_aridity_raster.py` |
| Figure 4 | `ModuleB_AllFigures_FINAL.py` |
| Figure 5 | `Generate_Figure5_RSI_DRA.py` |
| Figure 6 | `Generate_Figure6_ForestPlot.py` |
| Tables S1, S2 | `Build_TableS1_SPEI_Validation.py`, `Build_TableS2_MK_with_lag1.py` |
| Tables S3, S4 | `Build_TableS3_AreaWeighted_Reconciliation.py` |
| Tables S5, S14 | `Batch1_ModifiedMK_FDR_UnitFix.py` |
| Table S6 | `Batch2_StricterStablePixel_TimescaleSensitivity.py` |
| Table S11 | `Build_TableS11_RSI_SampleSizes.py` |
| Tables S9, S12, S13 | `Batch3_TwoStage_Aggregation_and_Inference.py` |

## How to reproduce

Stage-by-stage inputs and outputs are in **`pipeline_run_order.md`**. In brief:

```bash
pip install -r requirements.txt
cp config.py.template config.py     # edit with your local paths
```

**Stage A** ;  run the three `code/gee/` scripts in the Earth Engine Code Editor, download the exports.
**Stage B** ;  run the core pipeline (`Step5_*`, `Batch1`, `Batch2`, `Generate_aridity_raster`).
**Stage C** ;  run `Build_TableS11_RSI_SampleSizes.py`, `Build_TableS3_AreaWeighted_Reconciliation.py`, then `Batch3_TwoStage_Aggregation_and_Inference.py`.
**Stage D** ;  run the figure and table builders.

### Minimum path to the headline numbers

With the per-interval CSVs in hand, no rasters and no Earth Engine account are required:

```bash
python code/python/Batch3_TwoStage_Aggregation_and_Inference.py
python code/python/Generate_Figure6_ForestPlot.py
```

Expected output at SPEI-12, lagged window:

| Category | Mean *d* | 95% CI |
|---|---|---|
| Degradation | +0.020 | (−0.074, +0.120) |
| Recovery | **−0.178** | **(−0.260, −0.101)** |
| Agricultural | −0.138 | (−0.299, −0.015) |

## Derived data on Zenodo: https://doi.org/10.5281/zenodo.20572259

Organised as `rasters/` and `tables/`.

### `rasters`: single-band, ~5 km CHIRPS grid, continental Africa

| File | Produced by | Used in |
|---|---|---|
| `aridity_PoverPET_1985_2022.tif` | `Generate_aridity_raster.py` | Fig. 1 (P/PET gradient) |
| `aridity_classes_UNEP_1985_2022.tif` | `Generate_aridity_raster.py` | Fig. 1 (UNEP aridity classes) |
| `aed_contribution_mean.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | Fig. 2a |
| `spei12_trend_slope.tif`, `spei12_trend_pvalue.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | Fig. 3a |
| `pet_trend_slope.tif`, `pet_trend_pvalue.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | Fig. 3b |
| `spi12_trend_slope.tif`, `spi12_trend_pvalue.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | SPEI–SPI comparison |
| `precip_trend_slope.tif`, `precip_trend_pvalue.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | precipitation-trend reference |

### `tables`: published-table and figure-source CSVs

| File | Content |
|---|---|
| `Paper3_summary_for_paper.csv` | **Master numerical summary; every paper-cited number traceable to source** |
| `Table1_regional_climate_summary.csv` | Table 1 |
| `TableB2_LAGGED_per_interval_STRICTER.csv` | Per-cell effect sizes, lagged window |
| `TableB2_CUMULATIVE_per_interval_STRICTER.csv` | Per-cell effect sizes, cumulative window |
| `TableS8_CEM_per_interval.csv` | Coarsened-exact-matched per-cell effect sizes |
| `TableS9_areaweighted_per_interval.csv`, `TableS9b_areathresholded_per_interval.csv` | Area-based per-cell effect sizes |
| `final_twostage.json` | Continental means, intervals, TOST and contrast results |
| `Table2_trend_statistics_ModifiedMK.csv` | Supplementary Table S14 |
| `Table2_comparison_standardMK_vs_modifiedMK.csv` | Supplementary Table S2 |
| `Table3_FDR_significant_decadal_shifts.csv` | Table 2 (main text) |
| `TableB4_*_timescale_sensitivity_*.csv` | Supplementary Table S6; Figure 5b source |
| `TableB4_Decadal_Shift_cumulative_STRICTER_FDR.csv` | Supplementary Table S5 |
| `TableS1_SPEI_validation_diagnostics.csv` | Supplementary Table S1 |

The five per-cell CSVs plus `final_twostage.json` are the minimum set needed to regenerate every continental figure in the paper without rasters.

> **Not archived** (regenerable from `code/gee/`): the large intermediate GeoTIFFs, namely the 30-m transition rasters, 1-km from-state rasters, and the monthly precipitation and PET stacks.

## Data sources (third-party; not redistributed)

| Source | Variable | Resolution | DOI / URL |
|---|---|---|---|
| CHIRPS v2.0 | Precipitation | 0.05° monthly | [doi:10.1038/sdata.2015.66](https://doi.org/10.1038/sdata.2015.66) |
| TerraClimate | PET | ~4 km monthly | [doi:10.1038/sdata.2017.191](https://doi.org/10.1038/sdata.2017.191) |
| GLC_FCS30D | Land cover | 30 m, 1985–2022 | [doi:10.5194/essd-16-1353-2024](https://doi.org/10.5194/essd-16-1353-2024) |
| IPCC AR5 reference regions | Region polygons | vector | [doi:10.5194/essd-12-2959-2020](https://doi.org/10.5194/essd-12-2959-2020) |
| GHS-POP (JRC/GHSL/P2023A) | Population density | 5-year epochs | https://human-settlement.emergency.copernicus.eu |
| Accessibility to cities | Travel time to nearest city | ~1 km | https://malariaatlas.org |
| Gridded Livestock of the World v3, v4 | Livestock density | ~10 km | https://dataverse.harvard.edu/dataverse/glw |
| World Database on Protected Areas | Protected status | vector | https://www.protectedplanet.net |



## License

Code is released under the **MIT License** (see `LICENSE`). Derived data on Zenodo are released under **CC BY 4.0**. Third-party input datasets remain under their providers' licences.

## Contact

Hidayat Ullah (ullahhidayat@qdu.edu.cn)
Corresponding author: Jiahua Zhang (zhangjh@radi.ac.cn)

## Acknowledgements

This work depends on the open-data policies of CHIRPS, TerraClimate and GLC_FCS30D, and on the Google Earth Engine platform. Supported by the Natural Science Foundation of Shandong Province (ZR2023QD073; ZR2024LQX005) and the Qingdao Science and Technology Benefiting People Demonstration Project (25-1-5-xdny-11-nsh).
