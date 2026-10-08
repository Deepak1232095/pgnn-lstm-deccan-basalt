"""
04_baseline_benchmarks.py
=========================
External baselines for EMS comment 6:
"I still don't understand whether these new models represent an improvement
over other types of modeling."

HOW TO RUN
----------
Run this as the LAST cell of 01_indore_main_pipeline.py, in the same Python
session, after the rainfall-feature cell (get_rain_features / block_rain) has
run. It reuses these objects from that session, so the data, wells, zones,
scaling and train/test split are identical to the LSTM variants:

    monthly_v3, well_list_v3, aq_info, wells, block_rain, get_rain_features

Setup mirrored exactly from the M2/M3.5 evaluation code:
  * SEQ_LEN = 24, HORIZON = 12, SPLIT_YR = 2019
  * Same window loop: range(len(ms) - SEQ_LEN - HORIZON + 1)
  * A sample is "test" if its forecast date (ms.index[i+SEQ_LEN]) is after 2019
  * One-step-ahead prediction of h(t), inputs available up to t-1 only
  * Wells skipped if fewer than SEQ_LEN + HORIZON training months (same rule)
  * Zone = aq_info[well]['dominant']

BASELINES
---------
B1 Persistence          h(t) = h(t-1)
B2 Seasonal naive       h(t) = h(t-12)
B3 Seasonal persistence h(t) = h(t-12) + [h(t-1) - h(t-13)]
B4 Ridge ARX (per well) linear regression on head lags 1,2,3,12 +
                        7 rainfall features at t-1, t-2, t-3 + month sin/cos
B5 Zone-stratified GBM  one HistGradientBoosting model per aquifer zone,
                        same features as B4. Tests whether zone stratification
                        works without the LSTM/graph architecture.
B6 Pooled GBM           same as B5 but one model for all wells (no zones).
                        B5 vs B6 = effect of zone stratification alone.

B1-B3 have no fitted parameters. B4-B5 are fitted on training windows only.

OUTPUTS (in OUT_DIR)
--------------------
baseline_perwell_metrics.csv   per well x model: R2, RMSE, NSE, MAE, Bias, N
baseline_zone_summary.csv      per zone x model: mean and median per-well R2,
                               mean RMSE, pooled R2/RMSE
baseline_test_predictions.csv  every test point, every baseline (for plots)
baseline_vs_M35_paired.csv     only if M35_PERWELL_CSV is set (see CONFIG)
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# CONFIG
# ==============================================================================
OUT_DIR  = r"J:\Indore_gw\baselines"   # change to your path
SEQ_LEN  = 24
HORIZON  = 12
SPLIT_YR = 2019
SEED     = 42
N_BOOT   = 10000

# Optional: per-well M3.5 test metrics, one row per well, columns at least
# 'Well', 'R2', 'RMSE'. Must come from the SAME M3.5 run reported in the paper.
# Leave as None to skip the paired comparison.
M35_PERWELL_CSV = r"J:\Ground_water\Indore_gw\saved_results\saved_results\eval_m35.csv"

ZONES = ["Weathered", "Massive", "Fractured"]
MODELS = ["B1_Persistence", "B2_SeasonalNaive", "B3_SeasonalPersist",
          "B4_RidgeARX", "B5_ZoneGBM", "B6_PooledGBM"]

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ==============================================================================
# 0. LOAD DATA FROM 01_indore_main_pipeline.py  (no model training)
#    If the objects already exist in this session they are reused. Otherwise
#    only three blocks of the pipeline file are executed:
#      A  data loading, BGL->MSL, zone classification, monthly series
#      B  EXCLUDE_WELLS -> monthly_v3, well_list_v3
#      C  rainfall reading, block_rain, get_rain_features
#    Graph building, model definition and all training are skipped.
# ==============================================================================
PIPELINE_FILE = r"J:\Indore_gw\01_indore_main_pipeline.py"   # <- UPDATE
# Leave as None to use the paths already written inside the pipeline file
XLSX_OVERRIDE     = None   # e.g. r"J:\Indore_gw\Book2.xlsx"
RAIN_DIR_OVERRIDE = None   # e.g. r"J:\NDCQ-2026-03-471\datafiles"

_needed = ["monthly_v3", "well_list_v3", "aq_info", "wells",
           "block_rain", "get_rain_features"]


def _block(lines, start_marker, end_marker, include_end=False):
    s = next(i for i, l in enumerate(lines) if start_marker in l)
    e = next(i for i in range(s + 1, len(lines)) if end_marker in lines[i])
    return "\n".join(lines[s:e + 1 if include_end else e])


if any(n not in globals() for n in _needed):
    import re
    print(f"Loading data blocks from {PIPELINE_FILE} ...")
    if os.path.exists(PIPELINE_FILE):
        with open(PIPELINE_FILE, encoding="utf-8") as f:
            _src = f.read().splitlines()
    else:
        # Fall back to the public GitHub copy of the same file
        import urllib.request
        _url = ("https://raw.githubusercontent.com/Deepak1232095/"
                "pgnn-lstm-deccan-basalt/main/01_indore_main_pipeline.py")
        print(f"  {PIPELINE_FILE} not found -> downloading from GitHub")
        _src = urllib.request.urlopen(_url).read().decode("utf-8").splitlines()
    _A = _block(_src, "DATA LOADING + BGL->MSL CONVERSION", "4. GRAPH CONSTRUCTION")
    _B = _block(_src, "EXCLUDE_WELLS = [", "well_list_v3 = sorted", include_end=True)
    _C = _block(_src, "COMPLETE RAINFALL INTEGRATION", "# Test on one well")
    _A = _A.replace("import shap", "")          # shap not needed here
    if XLSX_OVERRIDE:
        _A = re.sub(r"file_path\s*=\s*r?['\"].*?['\"]",
                    f"file_path = r'{XLSX_OVERRIDE}'", _A)
    if RAIN_DIR_OVERRIDE:
        _C = re.sub(r"DATA_DIR\s*=\s*r?['\"].*?['\"]",
                    f"DATA_DIR = r'{RAIN_DIR_OVERRIDE}'", _C)
    for _name, _code in [("A data+zones", _A), ("B well filter", _B),
                         ("C rainfall", _C)]:
        print(f"  running block {_name} ...")
        exec(compile(_code, f"<01 pipeline: {_name}>", "exec"), globals())

_missing = [n for n in _needed if n not in globals()]
if _missing:
    raise RuntimeError(f"Still missing after loading: {_missing}")

# Sanity check: these counts should match the manuscript (36 wells; 20/5/11)
_zc = pd.Series({w: aq_info.get(w, {}).get("dominant", "Other")
                 for w in well_list_v3}).value_counts()
print(f"\nWells in monthly_v3: {len(well_list_v3)}")
print("Zone counts (before short-record filter):")
print(_zc.to_string())


# ==============================================================================
# 1. HELPERS
# ==============================================================================
def metrics(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    if len(obs) < 5:
        return None
    sse = np.sum((obs - pred) ** 2)
    sst = np.sum((obs - obs.mean()) ** 2)
    return {
        "R2":   r2_score(obs, pred) if np.std(obs) > 0 else 0.0,
        "RMSE": float(np.sqrt(mean_squared_error(obs, pred))),
        "NSE":  float(1 - sse / sst) if sst > 0 else np.nan,
        "MAE":  float(np.mean(np.abs(obs - pred))),
        "Bias": float(np.mean(pred - obs)),
        "N":    int(len(obs)),
    }


def calendar_lag(ms, dt, months, pos_fallback):
    """Value `months` calendar months before dt. If that month is missing
    (gaps after dropna), fall back to the positional value."""
    key = dt - pd.DateOffset(months=months)
    if key in ms.index:
        return float(ms.loc[key]), True
    return float(pos_fallback), False


def build_features(ms_sc, rain_f, i, dt):
    """Features for the window starting at position i, target at i+SEQ_LEN.
    Uses only information available up to t-1 (same as the LSTM inputs)."""
    t = i + SEQ_LEN
    head_lags = [ms_sc[t - 1], ms_sc[t - 2], ms_sc[t - 3], ms_sc[t - 12]]
    rain_lags = np.concatenate([rain_f[t - 1], rain_f[t - 2], rain_f[t - 3]])
    m = dt.month
    season = [np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]
    return np.concatenate([head_lags, rain_lags, season])


# ==============================================================================
# 2. BUILD WINDOWS (identical loop to the LSTM pipeline)
# ==============================================================================
rows = []          # one row per (well, window)
skipped, cal_fallbacks, cal_total = [], 0, 0

for well in well_list_v3:
    ms = monthly_v3[well]
    trn = ms[ms.index.year <= SPLIT_YR].values.reshape(-1, 1)
    if len(trn) < SEQ_LEN + HORIZON:
        skipped.append(well)
        continue

    sc = MinMaxScaler().fit(trn)             # fitted on training years only
    ms_sc = sc.transform(ms.values.reshape(-1, 1)).flatten()

    blk = wells[wells["Well No"] == well]["Block / Mandal"]
    block = str(blk.values[0]) if len(blk) else "Depalpur"
    rain_f = get_rain_features(well, block, block_rain, ms.index)
    zone = aq_info.get(well, {}).get("dominant", "Other")

    for i in range(len(ms_sc) - SEQ_LEN - HORIZON + 1):
        t = i + SEQ_LEN
        dt = ms.index[t]
        obs = float(ms.values[t])
        h1 = float(ms.values[t - 1])

        h12, ok12 = calendar_lag(ms, dt, 12, ms.values[t - 12])
        h13, ok13 = calendar_lag(ms, dt, 13, ms.values[t - 13])
        cal_total += 2
        cal_fallbacks += (not ok12) + (not ok13)

        rows.append({
            "Well": well, "Zone": zone, "Date": dt,
            "Split": "train" if dt.year <= SPLIT_YR else "test",
            "Obs": obs,
            "B1_Persistence": h1,
            "B2_SeasonalNaive": h12,
            "B3_SeasonalPersist": h12 + (h1 - h13),
            "_X": build_features(ms_sc, rain_f, i, dt),
            "_y_sc": ms_sc[t],
            "_scaler": sc,
        })

df = pd.DataFrame(rows)
print(f"Wells used: {df.Well.nunique()}  |  skipped (short record): {len(skipped)}")
print(df.groupby(["Zone", "Split"]).Well.nunique().unstack())
print(f"Calendar-lag fallbacks (data gaps): {cal_fallbacks}/{cal_total} "
      f"({100 * cal_fallbacks / max(cal_total, 1):.1f}%)")


def inv(sc, v):
    return float(sc.inverse_transform(np.array([[v]]))[0, 0])


# ==============================================================================
# 3. B4 RIDGE ARX — one model per well, trained on that well's train windows
# ==============================================================================
df["B4_RidgeARX"] = np.nan
alphas = np.logspace(-3, 3, 13)
for well, g in df.groupby("Well"):
    tr, te = g[g.Split == "train"], g[g.Split == "test"]
    if len(tr) < 24 or len(te) == 0:
        continue
    Xtr = np.vstack(tr._X.values); ytr = tr._y_sc.values
    Xte = np.vstack(te._X.values)
    model = RidgeCV(alphas=alphas).fit(np.nan_to_num(Xtr), ytr)
    p_sc = model.predict(np.nan_to_num(Xte))
    sc = te._scaler.iloc[0]
    df.loc[te.index, "B4_RidgeARX"] = [inv(sc, v) for v in p_sc]

# ==============================================================================
# 4. B5 ZONE-STRATIFIED GBM — one model per aquifer zone (pooled wells)
#    Features are per-well min-max scaled, so wells at different elevations
#    share one model, as in the zone-stratified LSTM.
# ==============================================================================
df["B5_ZoneGBM"] = np.nan
for zone, g in df.groupby("Zone"):
    tr, te = g[g.Split == "train"], g[g.Split == "test"]
    if len(tr) < 50 or len(te) == 0:
        continue
    gbm = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=SEED)
    gbm.fit(np.vstack(tr._X.values), tr._y_sc.values)
    p_sc = gbm.predict(np.vstack(te._X.values))
    df.loc[te.index, "B5_ZoneGBM"] = [
        inv(s, v) for s, v in zip(te._scaler.values, p_sc)]

# ==============================================================================
# 4b. B6 POOLED GBM — identical to B5 but ONE model for all wells (no zones).
#     B5 vs B6 isolates the effect of zone stratification itself.
# ==============================================================================
df["B6_PooledGBM"] = np.nan
_tr, _te = df[df.Split == "train"], df[df.Split == "test"]
gbm_all = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
    min_samples_leaf=20, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.15, random_state=SEED)
gbm_all.fit(np.vstack(_tr._X.values), _tr._y_sc.values)
_p = gbm_all.predict(np.vstack(_te._X.values))
df.loc[_te.index, "B6_PooledGBM"] = [inv(s, v) for s, v in zip(_te._scaler.values, _p)]

# ==============================================================================
# 5. PER-WELL TEST METRICS
# ==============================================================================
test = df[df.Split == "test"].copy()
pw = []
for (well, zone), g in test.groupby(["Well", "Zone"]):
    for m in MODELS:
        r = metrics(g.Obs, g[m])
        if r:
            pw.append({"Well": well, "Zone": zone, "Model": m, **r})
pw = pd.DataFrame(pw)

# Skill vs persistence: 1 - RMSE_model / RMSE_persistence (>0 = beats B1)
p_rmse = pw[pw.Model == "B1_Persistence"].set_index("Well").RMSE
pw["Skill_vs_Persistence"] = 1 - pw.RMSE / pw.Well.map(p_rmse)
pw.round(4).to_csv(os.path.join(OUT_DIR, "baseline_perwell_metrics.csv"),
                   index=False)

# ==============================================================================
# 6. ZONE SUMMARY (mean per-well AND pooled, so it matches whichever
#    definition Table 2 uses — check which one before comparing)
# ==============================================================================
summ = []
for zone in ZONES + ["ALL"]:
    tz = test if zone == "ALL" else test[test.Zone == zone]
    pz = pw if zone == "ALL" else pw[pw.Zone == zone]
    for m in MODELS:
        pm = pz[pz.Model == m]
        pooled = metrics(tz.Obs, tz[m])
        if pooled is None or pm.empty:
            continue
        summ.append({
            "Zone": zone, "Model": m, "N_wells": len(pm),
            "MeanWell_R2": pm.R2.mean(), "MedianWell_R2": pm.R2.median(),
            "MeanWell_RMSE": pm.RMSE.mean(),
            "MeanWell_Skill_vs_Persistence": pm.Skill_vs_Persistence.mean(),
            "Pooled_R2": pooled["R2"], "Pooled_RMSE": pooled["RMSE"],
        })
summ = pd.DataFrame(summ)
summ.round(3).to_csv(os.path.join(OUT_DIR, "baseline_zone_summary.csv"),
                     index=False)

keep = ["Well", "Zone", "Date", "Obs"] + MODELS
test[keep].to_csv(os.path.join(OUT_DIR, "baseline_test_predictions.csv"),
                  index=False)

pd.set_option("display.width", 160)
print("\n" + "=" * 90)
print("BASELINE ZONE SUMMARY — test period (forecast year > 2019), one-step-ahead")
print("=" * 90)
print(summ.round(3).to_string(index=False))

# ==============================================================================
# 7. OPTIONAL: PAIRED WELL-LEVEL COMPARISON  M3.5 vs EACH BASELINE
# ==============================================================================
if M35_PERWELL_CSV:
    m35 = pd.read_csv(M35_PERWELL_CSV)
    m35 = m35.rename(columns={"WellID": "Well", "r2": "R2", "rmse": "RMSE"})
    print("M3.5 file columns:", list(m35.columns))
    m35 = m35[["Well", "R2", "RMSE"]]
    m35 = m35.rename(columns={"R2": "R2_M35", "RMSE": "RMSE_M35"})
    out = []
    for m in MODELS:
        b = pw[pw.Model == m][["Well", "Zone", "R2", "RMSE"]]
        j = b.merge(m35, on="Well", how="inner")
        if len(j) < 5:
            print(f"{m}: only {len(j)} wells matched M3.5 — check well IDs")
            continue
        for zone in ZONES + ["ALL"]:
            jz = j if zone == "ALL" else j[j.Zone == zone]
            if len(jz) < 3:
                continue
            d = (jz.RMSE - jz.RMSE_M35).values   # >0 means M3.5 has lower error
            boots = np.array([rng.choice(d, len(d), replace=True).mean()
                              for _ in range(N_BOOT)])
            try:
                p_w = wilcoxon(jz.RMSE_M35, jz.RMSE).pvalue if len(jz) >= 6 else np.nan
            except ValueError:
                p_w = np.nan
            out.append({
                "Baseline": m, "Zone": zone, "N_wells": len(jz),
                "Mean_RMSE_baseline": jz.RMSE.mean(),
                "Mean_RMSE_M35": jz.RMSE_M35.mean(),
                "Mean_dRMSE(base-M35)": d.mean(),
                "CI95_low": np.percentile(boots, 2.5),
                "CI95_high": np.percentile(boots, 97.5),
                "Boot_p(M35 not better)": float(np.mean(boots <= 0)),
                "Wilcoxon_p": p_w,
                "Wells_M35_better": int((d > 0).sum()),
                "Mean_R2_baseline": jz.R2.mean(),
                "Mean_R2_M35": jz.R2_M35.mean(),
            })
    out = pd.DataFrame(out)
    out.round(4).to_csv(os.path.join(OUT_DIR, "baseline_vs_M35_paired.csv"),
                        index=False)
    print("\n" + "=" * 90)
    print("PAIRED COMPARISON — positive dRMSE = M3.5 more accurate than baseline")
    print("=" * 90)
    print(out.round(3).to_string(index=False))

print(f"\nSaved to: {OUT_DIR}")
