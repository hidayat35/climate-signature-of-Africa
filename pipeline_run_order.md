# Pipeline run order

Full input/output map for reproducing the analysis. Stage A runs in the browser; Stages B to D run locally.

Set your paths first:

```bash
pip install -r requirements.txt
cp config.py.template config.py     # edit config.py with your local paths
```

---

## Stage A: Earth Engine exports

Run at <https://code.earthengine.google.com>. Outputs land in your Google Drive; download them before Stage B. Roughly 2 to 4 hours of export time.

| Order | Script | Output |
|---|---|---|
| A1 | `code/gee/Step2_v2_GEE_Monthly_Export.js` | `precip_monthly_1985_2022.tif`, `pet_monthly_1985_2022.tif` (456 bands each, ~5 km) |
| A2 | `code/gee/ModuleB_GEE_Step1c_StateRasters.js` | `state_*.tif` from-state rasters (1 km, multi-band) |
| A3 | `code/gee/ModuleB_Step1_GEE_v2_9transitions.js` | `transition_*.tif` per-pathway rasters (300 m, multi-band) |

---

## Stage B: Core analysis

| Order | Script | Key outputs | Manuscript |
|---|---|---|---|
| B1 | `Step5_v6_FIXED_climate_indices.py` | SPEI (Pearson III) and SPI (gamma) NetCDFs; per-interval mean SPEI rasters | ;  |
| B2 | `Step5_v7_ModuleA_canonical_rerun.py` | regional climate statistics; AED and trend rasters | Table 1, Figs. 2–3 |
| B3 | `Step5_v5_DIAGNOSTIC_v2.py` | six-check SPEI/SPI validation | inputs for Table S1 |
| B4 | `Batch1_ModifiedMK_FDR_UnitFix.py` | Modified Mann-Kendall trends; decadal-shift BH-FDR | Table 2, Tables S5, S14 |
| B5 | `Batch2_StricterStablePixel_TimescaleSensitivity.py` | `TableB2_LAGGED_per_interval_STRICTER.csv`, `TableB2_CUMULATIVE_per_interval_STRICTER.csv`, RSI/DRA timescale tables | Table S6; inputs for Figs. 4–6 |
| B6 | `Generate_aridity_raster.py` | aridity-index rasters | Fig. 1 |

The two `TableB2_*_per_interval_STRICTER.csv` files from B5 are the pivot of everything downstream. Every script in Stages C and D reads them.

---

## Stage C: Aggregation, inference and robustness

| Order | Script | Key outputs | Manuscript |
|---|---|---|---|
| C1 | `Build_TableS11_RSI_SampleSizes.py` | `TableS14_RSI_sample_sizes.csv` | Table S11 |
| C2 | `Build_TableS3_AreaWeighted_Reconciliation.py` | specification factorial; 56-combination threshold sweep; bridge CSVs for C3 | Tables S3, S4 |
| C3 | `Batch3_TwoStage_Aggregation_and_Inference.py` | `final_twostage.json`; continental means, BCa intervals, TOST, contrast tests | Tables S9, S12, S13 |

Only **C2 needs the exported rasters** and takes 2 to 4 hours. C1 and C3 read CSVs and finish in seconds to minutes.

**Before running C2**, confirm the two threshold pairs in the configuration block:

```python
TABLES3_TRANS  = 0.001   # threshold pair A
TABLES3_STABLE = 0.50
TABLES9_TRANS  = 0.02    # threshold pair B
TABLES9_STABLE = 0.50
```

**After running C2**, read `TableS12_areaweighted_reconciliation.csv`, then set `ADOPTED_THRESHOLD_SPEC` to the specification you adopt and re-emit the bridge CSVs consumed by C3. The header comment in the script explains this.

---

## Stage D: Figures and provenance

| Order | Script | Output |
|---|---|---|
| D1 | `Generate_Figure5_RSI_DRA.py` | `Figure5_RSI_DRA.png` plus a printed audit of every plotted marker |
| D2 | `Generate_Figure6_ForestPlot.py` | `Figure6_ForestPlot.png` (reads `final_twostage.json` from C3) |
| D3 | `ModuleB_AllFigures_FINAL.py` | Module B figure set, including Figure 4 |
| D4 | `Generate_All_Publication_Figures_Tables_v2.py` | consolidated publication figure and table set |
| D5 | `Build_TableS1_SPEI_Validation.py` | Table S1 |
| D6 | `Build_TableS2_MK_with_lag1.py` | Table S2 |
| D7 | `Batch4_Part1_PaperSummaryCSV.py` | `Paper3_summary_for_paper.csv`, the master provenance file |

---

## Minimum path to the headline numbers

If you only want to reproduce the continental effect sizes, confidence intervals and equivalence tests, you need the two `TableB2_*_per_interval_STRICTER.csv` files plus the three matched and area-weighted per-interval CSVs, then:

```bash
python code/python/Batch3_TwoStage_Aggregation_and_Inference.py
python code/python/Generate_Figure6_ForestPlot.py
```

No rasters and no Earth Engine account required.
