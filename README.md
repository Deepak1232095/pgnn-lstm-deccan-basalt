# PGNN-LSTM: Zone-Stratified Physics-Guided Groundwater Forecasting

Code accompanying "Geology Over Architecture: A Zone-Stratified Physics-Guided
Neural Network for Groundwater Forecasting in Deccan Trap Basalt" (Indore and
Ujjain districts, Madhya Pradesh, India).

## Files

- **01_indore_main_pipeline.py** — Main 48-well Indore analysis: data loading,
  BGL→MSL conversion, hydraulics-based aquifer zone classification,
  geology-informed spatial graph, PGNN-LSTM architecture (M1–M4, M3.5),
  training, ablation, SHAP, MC Dropout, and zone-wise evaluation. This
  produced Tables 2 and 3 in the manuscript.

- **02_crossdistrict_ujjain.py** — Separate cross-district pipeline (48
  Indore + 122 Ujjain wells): T_eff-based zone classification, hydraulic-
  conductance graph, finite-difference physical baseline, and a
  zone-stratified residual LSTM (h_true = h_phys + h_residual). This
  produced Section 4.8 and Figure 11.

- **03_statistical_validation.py** — Well-level bootstrap significance
  testing (M3.5 vs M1, M3.5 vs M4) and the Indore-vs-Ujjain Mann–Whitney
  comparison. Produced Table 2a and the Section 4.8 significance results.

## Before running

All file paths in these scripts are still your original local paths
(e.g. `J:\Indore_gw\...`) — update `CONFIG`/path constants at the top of
each section to match your environment.

`01_indore_main_pipeline.py`'s ablation+SHAP+MC-Dropout section was captured
mid-debugging in the original notebook (see docstring at the top of that
file) — verify it runs cleanly end-to-end on your data before relying on it.
