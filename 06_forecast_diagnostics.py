"""
06_forecast_diagnostics.py
==========================
Checks needed before the 2040 forecasts can be defended in the manuscript.

Q1  Which way are heads actually moving?
    Theil-Sen slope + Mann-Kendall p-value per well, for three series:
      pre-monsoon  (Apr-Jun mean of each year)
      post-monsoon (Oct-Dec mean of each year)
      annual mean  (years with >= 6 months)
    -> tells you whether "depletion" is true for pre-monsoon only, or overall.

Q2  Are recent heads outside the training range?
    Training = months up to Dec 2019 (same split as the models).
    Current  = mean of the last 12 observed months.
    If current > training max, any model trained on <=2019 that reverts to
    its training range will show a "decline" that is not physics.

Q3  Does each model's 2040 decline track that exceedance?
    Spearman correlation between (current - training max) and 2040 change.
    Strong negative correlation = the decline is mostly reversion to the
    training range. Run for the baselines (from 05) and, if you give the
    files, for M1 / M3.5 / M4.

Q4  Where do records end?
    Wells whose record ends before 2023 start their forecast early, so their
    "2040" value is really earlier and their "current" head is old.

HOW TO RUN: same kernel as 04/05 (or fresh; data loads the same way).

OUTPUTS (OUT_DIR):
  diag_perwell.csv     every well: record end, trends, p-values, training range,
                       exceedance, and 2040 change for every model supplied
  diag_summary.csv     zone-level counts
  diag_reversion.csv   Q3 correlations per model
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, kendalltau, spearmanr

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG
# ==============================================================================
OUT_DIR       = r"J:\Indore_gw\baselines"
PIPELINE_FILE = r"J:\Indore_gw\01_indore_main_pipeline.py"
XLSX_OVERRIDE     = None
RAIN_DIR_OVERRIDE = None

SPLIT_YR   = 2019
MIN_YEARS  = 5          # minimum years for a trend
ALPHA      = 0.05
OLD_RECORD = 2023       # records ending before this year are flagged
ZONES      = ["Weathered", "Massive", "Fractured"]

# 2040 forecasts from your LSTM models, if you have them.
# Each CSV needs a 'Well' column and a change column (name given in the tuple).
# Example for the M1 file written by 01 pipeline section 10:
#   "M1": (r"J:\Indore_gw\PGNN_forecast_2040_MSL.csv", "Change_m"),
# Files written by the Colab GPU run (PGNN_LSTM_Colab_Training.py, Cell 9):
MODEL_FORECASTS = {
    "M1":   (r"J:\Indore_gw\forecast2040_m1.csv",  "Change_m"),
    "M3.5": (r"J:\Indore_gw\forecast2040_m35.csv", "Change_m"),
    "M4":   (r"J:\Indore_gw\forecast2040_m4.csv",  "Change_m"),
}
BASELINE_FC_CSV = os.path.join(OUT_DIR, "forecast2040_perwell.csv")   # from 05

# ==============================================================================
# 0. LOAD DATA (identical to 04/05; skipped if already in session)
# ==============================================================================
_needed = ["monthly_v3", "well_list_v3", "aq_info", "wells"]


def _block(lines, start_marker, end_marker, include_end=False):
    s = next(i for i, l in enumerate(lines) if start_marker in l)
    e = next(i for i in range(s + 1, len(lines)) if end_marker in lines[i])
    return "\n".join(lines[s:e + 1 if include_end else e])


if any(n not in globals() for n in _needed):
    print("Loading data blocks ...")
    if os.path.exists(PIPELINE_FILE):
        with open(PIPELINE_FILE, encoding="utf-8") as f:
            _src = f.read().splitlines()
    else:
        import urllib.request
        _url = ("https://raw.githubusercontent.com/Deepak1232095/"
                "pgnn-lstm-deccan-basalt/main/01_indore_main_pipeline.py")
        _src = urllib.request.urlopen(_url).read().decode("utf-8").splitlines()
    _A = _block(_src, "DATA LOADING + BGL->MSL CONVERSION", "4. GRAPH CONSTRUCTION")
    _B = _block(_src, "EXCLUDE_WELLS = [", "well_list_v3 = sorted", include_end=True)
    _A = _A.replace("import shap", "")
    if XLSX_OVERRIDE:
        _A = re.sub(r"file_path\s*=\s*r?['\"].*?['\"]",
                    f"file_path = r'{XLSX_OVERRIDE}'", _A)
    for _code in (_A, _B):
        exec(compile(_code, "<01 pipeline>", "exec"), globals())


# ==============================================================================
# 1. HELPERS
# ==============================================================================
def seasonal_series(ms, months, min_count=1):
    s = ms[ms.index.month.isin(months)]
    g = s.groupby(s.index.year).agg(["mean", "count"])
    return g[g["count"] >= min_count]["mean"]


def trend(series):
    """Theil-Sen slope (m/yr) and Mann-Kendall p (Kendall tau vs time)."""
    if len(series) < MIN_YEARS:
        return np.nan, np.nan
    x, y = series.index.values.astype(float), series.values
    return theilslopes(y, x)[0], kendalltau(x, y).pvalue


def classify(slope, p):
    if not np.isfinite(slope):
        return "n/a"
    if p < ALPHA:
        return "declining" if slope < 0 else "rising"
    return "no sig. trend"


# ==============================================================================
# 2. PER-WELL DIAGNOSTICS
# ==============================================================================
rows = []
for well in well_list_v3:
    ms = monthly_v3[well]
    trn = ms[ms.index.year <= SPLIT_YR]
    if len(trn) < 36:                    # same rule as the models (24 + 12)
        continue
    zone = aq_info.get(well, {}).get("dominant", "Other")

    pre = seasonal_series(ms, [4, 5, 6])
    post = seasonal_series(ms, [10, 11, 12])
    ann = seasonal_series(ms, range(1, 13), min_count=6)
    s_pre, p_pre = trend(pre)
    s_post, p_post = trend(post)
    s_ann, p_ann = trend(ann)

    current = float(ms.iloc[-12:].mean())
    tmax, tmin, tmean = float(trn.max()), float(trn.min()), float(trn.mean())
    rng_ = tmax - tmin if tmax > tmin else np.nan

    rows.append({
        "Well": well, "Zone": zone,
        "Record_start": ms.index[0].date(), "Record_end": ms.index[-1].date(),
        "Old_record": ms.index[-1].year < OLD_RECORD,
        "Pre_slope_m_yr": s_pre, "Pre_p": p_pre, "Pre_class": classify(s_pre, p_pre),
        "Post_slope_m_yr": s_post, "Post_p": p_post, "Post_class": classify(s_post, p_post),
        "Ann_slope_m_yr": s_ann, "Ann_p": p_ann, "Ann_class": classify(s_ann, p_ann),
        "Train_min": tmin, "Train_max": tmax, "Train_mean": tmean,
        "Current_12m": current,
        "Exceed_train_max_m": current - tmax,
        "Above_train_max": current > tmax,
        "Current_minus_train_mean_m": current - tmean,
        "Current_pos_in_train_range": (current - tmin) / rng_ if rng_ else np.nan,
    })

pw = pd.DataFrame(rows)

# ---- attach 2040 changes -----------------------------------------------------
change_cols = []
if os.path.exists(BASELINE_FC_CSV):
    b = pd.read_csv(BASELINE_FC_CSV)
    for c in [c for c in b.columns if c.startswith("Change_")]:
        pw = pw.merge(b[["Well", c]], on="Well", how="left")
        change_cols.append(c)
    if "Trend_change_m" in b.columns:
        pw = pw.merge(b[["Well", "Trend_change_m"]], on="Well", how="left")
else:
    print(f"(No {BASELINE_FC_CSV} — run 05 first to include baselines in Q3)")

for name, (path, col) in MODEL_FORECASTS.items():
    if not os.path.exists(path):
        print(f"  {name}: file not found -> {path}")
        continue
    m = pd.read_csv(path)
    if "Well" not in m.columns and "WellID" in m.columns:   # Colab GPU files
        m = m.rename(columns={"WellID": "Well"})
    if "Well" not in m.columns or col not in m.columns:
        print(f"  {name}: needs columns 'Well' and '{col}'. Has: {list(m.columns)}")
        continue
    cname = f"Change_{name}"
    pw = pw.merge(m[["Well", col]].rename(columns={col: cname}), on="Well", how="left")
    change_cols.append(cname)

pw.round(3).to_csv(os.path.join(OUT_DIR, "diag_perwell.csv"), index=False)

# ==============================================================================
# 3. SUMMARIES
# ==============================================================================
summ = []
for zone in ZONES + ["ALL"]:
    z = pw if zone == "ALL" else pw[pw.Zone == zone]
    r = {"Zone": zone, "N": len(z),
         "Records_end_before_" + str(OLD_RECORD): int(z.Old_record.sum()),
         "Current_above_train_max": int(z.Above_train_max.sum()),
         "Median_exceed_train_max_m": z.Exceed_train_max_m.median()}
    for tag in ["Pre", "Post", "Ann"]:
        vc = z[f"{tag}_class"].value_counts()
        r[f"{tag}_declining"] = int(vc.get("declining", 0))
        r[f"{tag}_rising"] = int(vc.get("rising", 0))
        r[f"{tag}_nosig"] = int(vc.get("no sig. trend", 0))
        r[f"{tag}_median_slope"] = z[f"{tag}_slope_m_yr"].median()
    summ.append(r)
summ = pd.DataFrame(summ)
summ.round(3).to_csv(os.path.join(OUT_DIR, "diag_summary.csv"), index=False)

rev = []
for c in change_cols:
    ok = pw[c].notna() & pw.Exceed_train_max_m.notna()
    if ok.sum() < 5:
        continue
    rho, p = spearmanr(pw.loc[ok, "Exceed_train_max_m"], pw.loc[ok, c])
    rho2, p2 = spearmanr(pw.loc[ok, "Current_minus_train_mean_m"], pw.loc[ok, c])
    dec_above = pw.loc[ok & pw.Above_train_max, c]
    dec_below = pw.loc[ok & ~pw.Above_train_max, c]
    rev.append({
        "Model": c.replace("Change_", ""), "N": int(ok.sum()),
        "Spearman_vs_exceed_max": rho, "p_max": p,
        "Spearman_vs_dev_from_train_mean": rho2, "p_mean": p2,
        "Mean_change_wells_above_max": dec_above.mean(),
        "Mean_change_wells_within_range": dec_below.mean(),
    })
rev = pd.DataFrame(rev)
rev.round(3).to_csv(os.path.join(OUT_DIR, "diag_reversion.csv"), index=False)

# ==============================================================================
# 4. PRINT
# ==============================================================================
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

print("\n" + "=" * 100)
print("Q1  TREND DIRECTION (Theil-Sen + Mann-Kendall, p < 0.05)")
print("=" * 100)
cols = ["Zone", "N"] + [f"{t}_{k}" for t in ["Pre", "Post", "Ann"]
                        for k in ["declining", "rising", "nosig", "median_slope"]]
print(summ[cols].round(3).to_string(index=False))

print("\n" + "=" * 100)
print("Q2 + Q4  TRAINING-RANGE EXCEEDANCE AND RECORD END")
print("=" * 100)
print(summ[["Zone", "N", "Current_above_train_max", "Median_exceed_train_max_m",
            "Records_end_before_" + str(OLD_RECORD)]].round(2).to_string(index=False))

print("\n" + "=" * 100)
print("Q3  IS THE 2040 CHANGE EXPLAINED BY REVERSION TO THE TRAINING RANGE?")
print("    Strong negative Spearman = wells that are furthest above their")
print("    training range get the biggest 'decline'  -> artefact, not physics")
print("=" * 100)
print(rev.round(3).to_string(index=False) if len(rev) else "  (no forecast files)")

old = pw[pw.Old_record][["Well", "Zone", "Record_end"]]
if len(old):
    print(f"\nWells whose record ends before {OLD_RECORD}:")
    print(old.to_string(index=False))

print(f"\nSaved to: {OUT_DIR}")
