# Across Africa, climate constrains vegetation recovery rather than accelerating degradation (1985–2022)

This repository contains the **analysis code** for the paper:

> **Across Africa, climate constrains vegetation recovery rather than accelerating degradation (1985–2022)**

The **derived data** (analysis rasters and published-table / figure-source CSVs) are archived separately on Zenodo at **https://doi.org/10.5281/zenodo.20572259** because the rasters exceed GitHub's file-size limits.

The paper attributes 30-m land-cover transitions (GLC_FCS30D, 1985–2022) to antecedent drought conditions across the five IPCC AR5 Africa reference regions, using canonical SPEI/SPI series computed from CHIRPS precipitation and TerraClimate potential evapotranspiration. The central finding is that climate-driven African land-system change operates primarily through **recovery restriction** (climate gates which areas can recover) rather than **degradation acceleration** (climate forcing additional areas to degrade).

## Headline findings

| Finding | Value | Reference |
|---|---|---|
| Continental Recovery Suppression Index (Cohen's *d*) | −0.178 | §3.2 |
| Continental Degradation Cohen's *d* | +0.020 (n.s.) | §3.2 |
| Sahel Recovery Suppression Index | −0.484 (*p* = 0.048) | §3.2 |
| Continental AED contribution to drought severity | 26.0% | §3.1 |
| FDR-significant decadal shifts (post-2005, BH α = 0.05) | 5 of 15 (all negative) | §3.3, Table 3 |

## Where to find each component

| Component | Location |
|---|---|
| Analysis code (this repo) | GitHub: **https://github.com/hidayat35/dominant-climate-signature-of-African-land-cover-change** |
| Derived rasters + CSVs | Zenodo: **https://doi.org/10.5281/zenodo.20572259** |
| Third-party input datasets | Original providers (see "Data sources") |

## Repository layout (code only)

```
.
├── README.md                              ← this file
├── LICENSE                                ← MIT
├── requirements.txt                       ← Python dependencies (pinned)
├── config.py.template                     ← path-config template (copy, edit, save as config.py)
└── code/
    ├── gee/                               ← Google Earth Engine scripts (run in browser at code.earthengine.google.com)
    │   ├── Step2_v2_GEE_Monthly_Export.js          ← export CHIRPS + TerraClimate monthly to GeoTIFF
    │   ├── ModuleB_Step1_GEE_v2_9transitions.js    ← export 9 transition pathways at 300 m
    │   └── ModuleB_GEE_Step1c_StateRasters.js      ← export from-state rasters at 1 km
    └── python/                            ← Python pipeline (run order: see pipeline_run_order.md)
        ├── Step5_v6_FIXED_climate_indices.py       ← canonical SPEI (Pearson III) + SPI (gamma)
        ├── Step5_v7_ModuleA_canonical_rerun.py     ← Module A regional climate stats (Table 1) + map rasters
        ├── Step5_v5_DIAGNOSTIC_v2.py               ← SPEI/SPI six-check validation diagnostics
        ├── Batch1_ModifiedMK_FDR_UnitFix.py        ← Modified Mann-Kendall + Hamed-Rao + BH-FDR
        ├── Batch2_StricterStablePixel_TimescaleSensitivity.py
        │                                             ← Module B attribution + RSI + DRA + timescale sensitivity
        ├── ModuleB_AllFigures_FINAL.py             ← generate Figures B1–B5
        ├── Generate_aridity_raster.py              ← generate aridity-index raster for Figure 1b
        ├── Stage2_AreaWeighted_Robustness.py       ← Sup. Table S3 (area-weighted robustness)
        ├── Stage2b_DegradationDecomposition.py     ← Sup. Table S3b (forest-frontier vs dryland)
        ├── Build_TableS1_SPEI_Validation.py        ← assemble Sup. Table S1
        ├── Build_TableS2_MK_with_lag1.py           ← assemble Sup. Table S2 (with lag-1 autocorrelation)
        └── Batch4_Part1_PaperSummaryCSV.py         ← master numerical-summary CSV (333 entries; provenance trail)
```

## Derived data on Zenodo — https://doi.org/10.5281/zenodo.20572259

The Zenodo archive is organised as `rasters/` and `tables/`.

### `rasters` — derived analysis rasters (single-band, ~5 km CHIRPS grid, continental Africa)

| File | Produced by | Used in |
|---|---|---|
| `aridity_PoverPET_1985_2022.tif` | `Generate_aridity_raster.py` | Figure 1b (P/PET gradient) |
| `aridity_classes_UNEP_1985_2022.tif` | `Generate_aridity_raster.py` | Figure 1b (UNEP aridity classes) |
| `aed_contribution_mean.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | Figure 2a (AED contribution %) |
| `spei12_trend_slope.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | Figure 3a (SPEI-12 trend) |
| `spei12_trend_pvalue.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | Figure 3a (trend significance) |
| `pet_trend_slope.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | Figure 3b (PET trend) |
| `pet_trend_pvalue.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | Figure 3b (trend significance) |
| `spi12_trend_slope.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | SPEI–SPI comparison |
| `spi12_trend_pvalue.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | SPEI–SPI comparison |
| `precip_trend_slope.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | precipitation-trend reference |
| `precip_trend_pvalue.tif` | `Step5_v7_ModuleA_canonical_rerun.py` | precipitation-trend reference |

### `tables` — published-table and figure-source CSVs

| File | Content |
|---|---|
| `Paper3_summary_for_paper.csv` | **Master numerical summary (333 entries); every paper-cited number traceable to source** |
| `Table1_regional_climate_summary.csv` | Table 1 (regional climate summary) |
| `Table2_trend_statistics_ModifiedMK.csv` | Table 2 (Modified Mann-Kendall trend statistics) |
| `Table2_comparison_standardMK_vs_modifiedMK.csv` | Sup. Table S2 (standard vs modified MK) |
| `Table3_FDR_significant_decadal_shifts.csv` | Table 3 (FDR-significant decadal shifts) |
| `Table7_decadal_comparison.csv` | Decadal SPEI/precip/PET by region |
| `TableB2_LAGGED_SUMMARY_spei12_STRICTER.csv` | Continental category-mean Cohen's *d* (lagged) |
| `TableB2_CUMULATIVE_SUMMARY_spei12_STRICTER.csv` | Continental category-mean Cohen's *d* (cumulative) |
| `TableB4_RSI_timescale_sensitivity_lagged_STRICTER.csv` | RSI vs timescale (lagged) — Figure 5b / Sup. Table S5 |
| `TableB4_RSI_timescale_sensitivity_cumulative_STRICTER.csv` | RSI vs timescale (cumulative) — Figure 5b / Sup. Table S5 |
| `TableB4_DRA_timescale_sensitivity_lagged_STRICTER.csv` | DRA vs timescale (lagged) — Figure 5b / Sup. Table S5 |
| `TableB4_DRA_timescale_sensitivity_cumulative_STRICTER.csv` | DRA vs timescale (cumulative) — Figure 5b / Sup. Table S5 |
| `TableB4_Decadal_Shift_cumulative_STRICTER_FDR.csv` | Decadal-shift Welch tests + BH-FDR (15 tests) — Table 3 / Sup. Table S4 |
| `TableS1_SPEI_validation_diagnostics.csv` | Sup. Table S1 (SPEI/SPI validation) |
| `TableS3_robustness_area_weighted.csv` | Sup. Table S3 (area-weighted robustness) |
| `TableS3b_continental_degradation_by_pathway.csv` | Sup. Table S3b (degradation decomposition) |

> **Not archived** (regenerable from the `code/gee/` scripts): the large intermediate GeoTIFFs — multi-band 30-m transition rasters (`transition_*.tif`), 1-km from-state rasters (`state_*.tif`), and the monthly precipitation/PET stacks (`precip_monthly_1985_2022.tif`, `pet_monthly_1985_2022.tif`). The three third-party input datasets are available at the DOIs in "Data sources" below.

## How to reproduce the analysis

The pipeline runs in three stages. Full input/output filenames are tabulated in `pipeline_run_order.md`.

### Stage A — Earth Engine exports (browser, 2–4 hours)

Open the GEE Code Editor at <https://code.earthengine.google.com> and run, in order:

1. `code/gee/Step2_v2_GEE_Monthly_Export.js` — monthly CHIRPS precipitation and TerraClimate PET at ~5 km, 1985–2022.
2. `code/gee/ModuleB_GEE_Step1c_StateRasters.js` — multi-band from-state rasters (FST, SHR, GRS, BAL) at 1 km.
3. `code/gee/ModuleB_Step1_GEE_v2_9transitions.js` — multi-band per-pathway transition rasters at 300 m for the nine pathways.

Outputs are written to your Google Drive; download to local disk before Stage B. (These large GeoTIFFs are the intermediates not archived on Zenodo.)

### Stage B — Python analytical pipeline (local machine, ~6–8 hours)

```bash
pip install -r requirements.txt
cp config.py.template config.py     # then edit config.py with your local paths
```

Run the Python scripts in this order:

1. `Step5_v6_FIXED_climate_indices.py` — fits canonical SPEI (Pearson III) and SPI (gamma) over the 1985–2000 calibration; writes per-interval mean SPEI rasters.
2. `Step5_v7_ModuleA_canonical_rerun.py` — regional climate stats and the analysis map rasters (Zenodo `rasters/`); produces **Table 1**.
3. `Step5_v5_DIAGNOSTIC_v2.py` — six-check SPEI/SPI validation; inputs for **Sup. Table S1**.
4. `Batch1_ModifiedMK_FDR_UnitFix.py` — Modified Mann-Kendall trends + decadal-shift BH-FDR; produces **Table 2**, **Table 3**, and **Sup. Table S4**.
5. `Batch2_StricterStablePixel_TimescaleSensitivity.py` — Module B attribution; produces inputs for **Figures 4, 5, 7** plus the RSI/DRA timescale tables.
6. `ModuleB_AllFigures_FINAL.py` — generates **Figures B1–B5**.

### Stage C — Supplementary tables + provenance (local machine, ~30 minutes)

7. `Build_TableS1_SPEI_Validation.py` — assembles **Sup. Table S1**.
8. `Build_TableS2_MK_with_lag1.py` — per-region lag-1 autocorrelation; assembles **Sup. Table S2**.
9. `Stage2_AreaWeighted_Robustness.py` — **Sup. Table S3** (area-weighted aggregation robustness).
10. `Stage2b_DegradationDecomposition.py` — **Sup. Table S3b** (from-forest vs dryland-degradation decomposition).
11. `Generate_aridity_raster.py` — generates the aridity-index rasters (Zenodo `rasters/`; used to build Figure 1b in ArcGIS).
12. `Batch4_Part1_PaperSummaryCSV.py` — compiles the master numerical-summary CSV (`Paper3_summary_for_paper.csv`; every paper-cited number traceable to a source CSV).

## Data sources (third-party; not redistributed here or on Zenodo)

| Source | Variable | Resolution | DOI / URL |
|---|---|---|---|
| CHIRPS v2.0 | Precipitation | 0.05° (~5.5 km) monthly | [doi:10.1038/sdata.2015.66](https://doi.org/10.1038/sdata.2015.66) |
| TerraClimate | PET | ~4 km monthly | [doi:10.1038/sdata.2017.191](https://doi.org/10.1038/sdata.2017.191) |
| GLC_FCS30D | Land cover | 30 m, 1985–2022 | [doi:10.5194/essd-16-1353-2024](https://doi.org/10.5194/essd-16-1353-2024) |
| IPCC AR5 reference regions | Region polygons | vector | [doi:10.5194/essd-12-2959-2020](https://doi.org/10.5194/essd-12-2959-2020) |

## Citation

If you use this code, data, or methodology in your own work, please cite both the paper and the archived data:

> Ullah, H. et al. (2026). Across Africa, climate constrains vegetation recovery rather than accelerating degradation (1985–2022). *(Journal TBD).* DOI: TBD.
>
> Ullah, H. et al. (2026). Derived data for "Across Africa, climate constrains vegetation recovery rather than accelerating degradation (1985–2022)" Data set. Zenodo. **https://doi.org/10.5281/zenodo.20572259**

## License

The code is released under the **MIT License** (see `LICENSE`). The derived data archived on Zenodo are released under **CC BY 4.0**. The third-party input datasets remain under their respective providers' licenses (see "Data sources").

## Contact

Hidayat Ullah — ullahhidayat@qdu.edu.cn

## Acknowledgements

This work depends on the generous open-data policies of CHIRPS, TerraClimate, and GLC_FCS30D, and on the Google Earth Engine platform.
