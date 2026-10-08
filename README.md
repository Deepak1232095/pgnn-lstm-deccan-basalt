# PGNN-LSTM: hydrostratigraphy, connectivity and physics constraints in groundwater-level forecasting (Deccan Trap basalt)

Code for the manuscript *"Hydrostratigraphy, connectivity and physics constraints in deep-learning groundwater forecasting: a controlled ablation study in Deccan Trap basalt"* (Indore and Ujjain districts, Madhya Pradesh, India).

The study tests whether hydrostratigraphic information improves groundwater-level forecasts. Seven LSTM variants add rainfall inputs, a hydraulic-conductance well graph, aquifer-zone branches and a physics-guided loss one at a time, together with a capacity-matched control, and are compared with five non-recurrent benchmarks on identical test samples.

Data: processed datasets are archived on Zenodo, https://doi.org/10.5281/zenodo.20422776. Raw groundwater-level and rainfall data are available from India-WRIS (https://indiawris.gov.in).

## Run order

| Step | File | What it does | Manuscript output |
|---|---|---|---|
| 1 | `01_indore_main_pipeline.py` | Data loading, depth-to-water to head conversion, zone classification, well filtering and rainfall processing. Steps 2 and 3 execute only these data blocks from this file. | Sections 2–3.1 |
| 2 | `04_baseline_benchmarks.py` | Five benchmarks (persistence, seasonal persistence, Ridge ARX, zone GBM, pooled GBM) on the same wells, windows and splits as the LSTMs. Writes `baseline_perwell_metrics.csv` and `baseline_test_predictions.csv`. | Table 2 |
| 3 | `12_clean_ablation_5models.ipynb` | Seven LSTM variants (M1, M2, M2big, M3, M3.5, M3.5G, M4), 3 seeds each; train ≤ 2017, validation 2018–2019, test 2020–2025; paired Wilcoxon tests and bootstrap CIs; permutation importance; MC dropout; free-run hindcast 2020–2025; 2040 scenario with reversion diagnostic. | Tables 1, 3, 4; Figs 3, 7, 8 |
| 4 | `06_forecast_diagnostics.py` | Per-well Theil–Sen trends and Mann–Kendall tests (pre-monsoon, post-monsoon, annual). Writes `diag_perwell.csv`. | Section 4.6 |
| 5 | `13_figures_clean.py` | Figures 4, 5, 6 and 10 from the CSVs written by steps 2 and 3 (no training). | Figs 4, 5, 6, 10 |
| 6 | `11_fig9_head_trends.py` | Head-trend maps: IDW surface of per-well slopes, district and block boundaries. Needs `diag_perwell.csv`. | Fig 9 |
| 7 | `02_crossdistrict_ujjain.py` | Separate cross-district pipeline (Indore + 122 Ujjain wells): T_eff zone classification, finite-difference baseline, residual LSTM. Not part of the ablation. | Section 4.8, Fig 11 |

`legacy/03_statistical_validation.py` belongs to an earlier version of the analysis (36-well M3.5 bootstrap) and is not used for the current results.

## Key settings (notebook 12)

- 47 wells trained, 37 evaluated (wells with ≥ 36 monthly values before 2020 and ≥ 5 test predictions in both the LSTM and benchmark runs).
- Inputs: 24-month head history; 7 causal rainfall features from the block station; conductance graph (wells within 15 km, C = T_harm / L; T_eff = 30 / 5 / 100 m² d⁻¹ for Weathered / Massive / Fractured).
- LSTM: 2 layers, 64 hidden units (112 for M2big), dropout 0.2. GCN: 2 layers, 16-dimensional embedding.
- Training: Adam (lr 1e-3, weight decay 1e-5), gradient clipping 1.0, ReduceLROnPlateau, max 300 epochs, early stopping (patience 30) on 2018–2019 validation RMSE.
- Physics loss (M4 only): L = MSE + λ(L_sp + L_st), λ selected on validation from {0.001, 0.01, 0.1}.
- All scaling and node features use pre-test data only.

## Before running

Paths at the top of each script and in the first notebook cell point to the original local folders (e.g. `J:\Indore_gw\...`). Change them to your own paths. If `01_indore_main_pipeline.py` is not found locally, notebook 12 and script 04 download it from this repository.

Requirements: Python ≥ 3.10, PyTorch, NumPy, pandas, SciPy, scikit-learn, matplotlib, geopandas, shapely, openpyxl. A GPU is not required.

## Contact

Deepak Mishra (phd2301104004@iiti.ac.in), Department of Civil Engineering, IIT Indore.
