# PGNN-LSTM: Zone-Stratified Physics-Guided Groundwater Forecasting

Code accompanying "Geology Over Architecture: A Zone-Stratified Physics-Guided
Neural Network for Groundwater Forecasting in Deccan Trap Basalt" (Indore and
Ujjain districts, Madhya Pradesh, India).

## Files

- **01_indore_main_pipeline.py** — Main 36-well Indore analysis: data loading,
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

## What was removed from the original working notebooks

The original development notebooks (~85 cells combined) contained, in
addition to the pipeline above:

- Exact duplicate cells (same code pasted twice/three times while iterating)
- Superseded bug-fix drafts (e.g. four progressive attempts at the
  ablation+MC-Dropout+SHAP cell while chasing a tensor-dimension bug —
  only the final, most-fixed version is kept here)
- ~10 iterations of publication-figure styling scripts for a manuscript
  version originally targeted at a different journal (Water Resources
  Research), later superseded by the current EMS submission and its figures
- Exploratory diagnostic scripts written while trying to recover a lost
  M3.5 model checkpoint for MC Dropout uncertainty quantification. This
  effort did not reach a validated result (Massive-zone R² would not
  reproduce Table 2/3), so it is **not included** here. The manuscript
  notes this as a limitation (MC Dropout was completed for M1/M4 only).

## Before running

All file paths in these scripts are still your original local paths
(e.g. `J:\Indore_gw\...`) — update `CONFIG`/path constants at the top of
each section to match your environment.

`01_indore_main_pipeline.py`'s ablation+SHAP+MC-Dropout section was captured
mid-debugging in the original notebook (see docstring at the top of that
file) — verify it runs cleanly end-to-end on your data before relying on it.
