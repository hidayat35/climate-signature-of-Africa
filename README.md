# Antecedent drought and land-cover transitions in Africa, 1985–2022

This repository holds the analysis code for a continental comparison of antecedent drought at pixels that changed land cover with pixels of the same class that stayed stable. The analysis covers the five IPCC AR5 reference regions of Africa:

- Mediterranean (MED)
- Sahara-Sahel (SAH)
- West Africa (WAF)
- East Africa (EAF)
- Southern Africa (SAF)

**Derived data:** Zenodo, <https://zenodo.org/records/21853532>. This DOI covers all versions and opens the latest one.

## What the code does

1. **Climate.**
   - SPEI and SPI at 12, 24, 36 and 60 months, from CHIRPS precipitation and TerraClimate potential evapotranspiration (PET), calibrated on 1985–2000.
   - The contribution of atmospheric evaporative demand to cumulative drought severity, and the SPEI–SPI drought-frequency gap.
   - Theil–Sen trends, tested with the Mann–Kendall test and the Hamed–Rao autocorrelation correction.
2. **Land cover.**
   - Nine GLC_FCS30D transition pathways: four degradation, three recovery and two agricultural.
   - Antecedent SPEI in a lagged window (the preceding interval) and a cumulative window (all preceding intervals).
   - Cohen's d between transition pixels and stable pixels of the same starting class. A stable pixel is in that class at both ends of the interval.
3. **Aggregation and inference.**
   - Two-stage continental means, with bias-corrected and accelerated (BCa) bootstrap intervals.
   - Two one-sided tests of equivalence, Welch's and permutation tests, and a linear mixed-effects model.
   - Regional indices: the Recovery Suppression Index (RSI) and the Degradation–Recovery Asymmetry (DRA).
   - Early–late comparisons with Benjamini–Hochberg correction.
4. **Robustness.** Area weighting and area thresholding of the transition masks, and coarsened exact matching on population density, travel time, livestock density and protected status.

**Sign convention:** d = (mean SPEI of stable pixels − mean SPEI of transition pixels) / pooled SD. Negative d means the transition pixels had wetter antecedent conditions than the stable pixels.

## Layout

```
code/gee/       Earth Engine scripts (run in the Code Editor): monthly climate, transitions, from-class rasters
code/python/    analysis, aggregation, robustness and figure scripts
pipeline_run_order.md   every script in order, with its inputs and outputs
requirements.txt        Python packages
config.py.template      template for local paths
```

## Main scripts

| Script | What it produces |
|---|---|
| `Step5_v6_FIXED_climate_indices.py` | SPEI and SPI series; interval-mean SPEI rasters |
| `Step5_v7_ModuleA_canonical_rerun.py` | regional climate summaries; evaporative-demand contribution and trend rasters |
| `Batch1_ModifiedMK_FDR_UnitFix.py` | trends with the modified Mann–Kendall test; comparison with the standard test |
| `Batch2_StricterStablePixel_TimescaleSensitivity.py` | interval-level effect sizes for both windows (`TableB2_*_per_interval_STRICTER.csv`); RSI, DRA and early–late tests |
| `Build_TableS3_AreaWeighted_Reconciliation.py` | area-weighted and area-thresholded effect sizes; threshold sweep |
| [matching script] | effect sizes after coarsened exact matching (`TableS8_CEM_per_interval.csv`); covariate balance |
| `Batch3_TwoStage_Aggregation_and_Inference.py` | continental means, BCa intervals, equivalence and contrast tests (`final_twostage.json`) |
| `MixedEffects_CategoryContrast.py` | mixed-effects estimates of the recovery–degradation and agricultural–degradation contrasts |
| `Build_TableS11_RSI_SampleSizes.py` | cells, pathways and pixel counts behind each regional index |
| `Generate_aridity_raster.py` | aridity index and classes |
| `Export_map_layers_v2.py` | map layers for the climate figures |
| `Make_map_figures.py`, `Make_publication_figures.py` | the figures of the paper |

The other scripts in `code/python/` are earlier figure and table builders. They are kept for reference.

**File names.** Some scripts and output files carry table or figure numbers from an earlier draft of the paper, for example `TableS8_CEM_per_interval.csv` or `Fig5_forest.png`. The names are kept so that the scripts and the archived files match. They do not follow the numbering of the published article.

## Reproducing the continental results without rasters

Download these interval-level tables from Zenodo into the repository folder:

- `TableB2_LAGGED_per_interval_STRICTER.csv`
- `TableB2_CUMULATIVE_per_interval_STRICTER.csv`
- `TableS8_CEM_per_interval.csv`
- `TableS9_areaweighted_per_interval.csv`
- `TableS9b_areathresholded_per_interval.csv`

Then run, from that folder:

```bash
pip install -r requirements.txt
python code/python/Batch3_TwoStage_Aggregation_and_Inference.py   # writes final_twostage.json
python code/python/MixedEffects_CategoryContrast.py                # writes mixed_effects_contrast.csv
```

No rasters or Earth Engine account are needed. The figures can then be drawn, in an environment with matplotlib 3.10, with:

```bash
python code/python/Make_publication_figures.py --twostage final_twostage.json
```

Continental means are computed in two stages:

1. Interval-level d values are averaged within each region × pathway unit, weighted by the number of transition pixels.
2. The unit means are averaged with equal weight.

This keeps the few units with very large pixel counts from dominating the continental estimate. All estimators use the same set of units.

## Data sources (third party, not redistributed)

| Dataset | Variable | Reference |
|---|---|---|
| CHIRPS v2.0 | Precipitation | [doi:10.1038/sdata.2015.66](https://doi.org/10.1038/sdata.2015.66) |
| TerraClimate | Potential evapotranspiration | [doi:10.1038/sdata.2017.191](https://doi.org/10.1038/sdata.2017.191) |
| GLC_FCS30D | Land cover, 30 m | [doi:10.5194/essd-16-1353-2024](https://doi.org/10.5194/essd-16-1353-2024) |
| GHS-POP R2023A | Population density | [doi:10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE](https://doi.org/10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE) |
| Travel time to cities | Accessibility | [doi:10.1038/nature25181](https://doi.org/10.1038/nature25181) |
| Gridded Livestock of the World (GLW 3, GLW 4) | Livestock density | [doi:10.1038/sdata.2018.227](https://doi.org/10.1038/sdata.2018.227); FAO |
| World Database on Protected Areas | Protected status | <https://www.protectedplanet.net> |
| IPCC AR5 reference regions | Region boundaries | [doi:10.5194/essd-12-2959-2020](https://doi.org/10.5194/essd-12-2959-2020) |

## Requirements

- **Analysis scripts:** Python 3.9 with the packages in `requirements.txt`.
- **Figure scripts:** Python 3.10 or later and matplotlib 3.10.
- **Earth Engine scripts:** a Google Earth Engine account.

## Licence and citation

- Code: MIT (see `LICENSE`).
- Derived data on Zenodo: CC BY 4.0.
- Third-party data remain under their providers' licences.

If you use this code or the derived data, please cite the Zenodo record above. The article reference will be added on publication.
