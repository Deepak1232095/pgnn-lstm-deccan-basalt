"""
13_figures_clean.py — new Figures 4, 5, 6, 10 from the clean run (notebook 12)
No training. Reads the CSVs that notebook 12 and script 04 already saved.

  Fig04_zone_accuracy      zone-wise RMSE, all models + baselines, same wells, 95% CI
  Fig05_median_wells       observed vs predicted for the MEDIAN well of each zone
                           (not the best wells), M1 / M3 / M3.5 + persistence
  Fig06_component_effects  (a) test RMSE per model ± SD over seeds
                           (b) effect of each component: ΔRMSE, 95% CI, Wilcoxon p
  Fig10_forecast_skill     (a) free-run hindcast 2020-25 vs trend / climatology
                           (b) level drift in hindcast
                           (c) 2040 change vs deviation from 1998-2019 mean (reversion)
Edit only the PATHS block.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# ============================ PATHS ===========================================
CLEAN = r"J:\Indore_gw\clean_run"
BASE  = r"J:\Indore_gw\baselines"
OUT   = r"J:\Indore_gw\clean_run\figures"

# ============================ STYLE ===========================================
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.titlelocation": "left", "savefig.dpi": 300})
ZONES = ["Weathered", "Massive", "Fractured"]
ZCOL = {"Weathered": "#3B6D11", "Massive": "#185FA5", "Fractured": "#BA7517"}
MODELS = ["M1", "M2", "M2big", "M3", "M3.5", "M3.5G", "M4"]
BASEL = {"B1_Persistence": "Persistence", "B4_RidgeARX": "Ridge ARX", "B5_ZoneGBM": "Zone GBM"}
MCOL = {"M1": "#4D4D4D", "M2": "#5E81AC", "M2big": "#9AB0CE", "M3": "#2E7D6B",
        "M3.5": "#C0504D", "M3.5G": "#E39A97", "M4": "#7B5EA7",
        "Persistence": "#BDBDBD", "Ridge ARX": "#A6A6A6", "Zone GBM": "#8C8C8C"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.show()
    print("saved", os.path.join(OUT, name + ".png"))


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


# ============================ DATA ============================================
ev = pd.read_csv(os.path.join(CLEAN, "eval_all_clean.csv"))
abl = pd.read_csv(os.path.join(CLEAN, "Table_ablation_clean.csv")).set_index("Model")
pt = pd.read_csv(os.path.join(CLEAN, "Table_paired_tests_clean.csv"))
pr = pd.read_csv(os.path.join(CLEAN, "test_predictions_all.csv"), parse_dates=["Date"])
hc = pd.read_csv(os.path.join(CLEAN, "hindcast_freerun_2020_2025.csv"))
fc = pd.read_csv(os.path.join(CLEAN, "fc_all_clean.csv"))
MODELS = [m for m in MODELS if m in set(ev.Model)]
zone_of = ev.drop_duplicates("Well").set_index("Well").Zone

bl_path = os.path.join(BASE, "baseline_perwell_metrics.csv")
blp_path = os.path.join(BASE, "baseline_test_predictions.csv")
if os.path.exists(bl_path):
    bl = pd.read_csv(bl_path)
    bl = bl[bl.Model.isin(BASEL)].assign(Model=lambda d: d.Model.map(BASEL))
    bl["Zone"] = bl.Well.map(zone_of)
    allm = pd.concat([ev[["Model", "Well", "Zone", "RMSE"]],
                      bl[["Model", "Well", "Zone", "RMSE"]]], ignore_index=True)
else:
    print("baseline_perwell_metrics.csv not found -> figures without baselines")
    allm = ev[["Model", "Well", "Zone", "RMSE"]].copy()
order = [b for b in BASEL.values() if b in set(allm.Model)] + MODELS
common = allm.groupby("Well").Model.nunique()
common = common[common == len(order)].index          # same wells for every method
allm = allm[allm.Well.isin(common)].dropna(subset=["Zone"])
print(f"Wells common to all {len(order)} methods: {len(common)}")

# =========================== FIG 4 — ZONE ACCURACY ============================
fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
for ax, zone in zip(axes, ZONES + ["All wells"]):
    d = allm if zone == "All wells" else allm[allm.Zone == zone]
    n = d.Well.nunique()
    for i, m in enumerate(order):
        v = d[d.Model == m].RMSE.values
        if len(v) == 0:
            continue
        lo, hi = boot_ci(v)
        ax.plot([lo, hi], [i, i], color=MCOL[m], lw=2.2, solid_capstyle="round")
        ax.plot(v.mean(), i, "o", ms=7, color=MCOL[m], mec="white", mew=0.8, zorder=3)
    ax.set_title(f"{zone} (n = {n})", color=ZCOL.get(zone, "#222"))
    ax.set_xlabel("Test RMSE (m), mean ± 95% CI")
    ax.grid(axis="x", alpha=0.3)
    nb = len([b for b in BASEL.values() if b in order])
    if nb:
        ax.axhspan(-0.5, nb - 0.5, color="#F2F2F2", zorder=0)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(order)
axes[0].invert_yaxis()
fig.suptitle("Test-period accuracy by aquifer zone, 2020–2025 (grey band = non-deep-learning baselines)",
             x=0.01, ha="left", fontsize=11.5, fontweight="bold")
fig.tight_layout()
save(fig, "Fig04_zone_accuracy")

# ======================= FIG 5 — MEDIAN WELL PER ZONE =========================
ens = pr.groupby(["Model", "Well", "Date"]).agg(Obs=("Obs", "first"), Pred=("Pred", "mean")).reset_index()
REF = "M3" if "M3" in MODELS else MODELS[0]
SHOW = [m for m in ["M1", "M3", "M3.5"] if m in MODELS]
pers = None
if os.path.exists(blp_path):
    bp = pd.read_csv(blp_path, parse_dates=["Date"])
    if "B1_Persistence" in bp.columns:
        pers = bp[["Well", "Date", "B1_Persistence"]]

fig, axes = plt.subplots(3, 2, figsize=(14, 10), gridspec_kw={"width_ratios": [2.4, 1]})
for r, zone in enumerate(ZONES):
    e = ev[(ev.Model == REF) & (ev.Zone == zone) & ev.Well.isin(common)].sort_values("RMSE")
    if e.empty:
        axes[r, 0].axis("off"); axes[r, 1].axis("off"); continue
    w = e.iloc[len(e) // 2].Well                      # median-RMSE well
    a1, a2 = axes[r]
    obs = ens[(ens.Model == REF) & (ens.Well == w)].sort_values("Date")
    a1.plot(obs.Date, obs.Obs, color="black", lw=2, label="Observed")
    if pers is not None:
        p = pers[pers.Well == w].sort_values("Date")
        a1.plot(p.Date, p.B1_Persistence, color=MCOL["Persistence"], lw=1.2, ls=":",
                label="Persistence")
    lab = []
    for m in SHOW:
        g = ens[(ens.Model == m) & (ens.Well == w)].sort_values("Date")
        rm = ev[(ev.Model == m) & (ev.Well == w)].RMSE.values[0]
        a1.plot(g.Date, g.Pred, color=MCOL[m], lw=1.4, label=f"{m} (RMSE {rm:.2f} m)")
        a2.scatter(g.Obs, g.Pred, s=16, color=MCOL[m], alpha=0.75, label=m)
    lim = [min(obs.Obs.min(), ens[ens.Well == w].Pred.min()) - 0.5,
           max(obs.Obs.max(), ens[ens.Well == w].Pred.max()) + 0.5]
    a2.plot(lim, lim, "k--", lw=1); a2.set_xlim(lim); a2.set_ylim(lim)
    a1.set_title(f"({'abcdef'[2 * r]}) {zone}: {w} — median well of {len(e)} by {REF} RMSE",
                 color=ZCOL[zone])
    a2.set_title(f"({'abcdef'[2 * r + 1]}) Observed vs predicted", color=ZCOL[zone])
    a1.set_ylabel("Hydraulic head (m MSL)")
    a2.set_xlabel("Observed (m MSL)"); a2.set_ylabel("Predicted (m MSL)")
    a1.legend(fontsize=8.5, ncol=3, loc="best", frameon=False)
    a1.grid(alpha=0.25); a2.grid(alpha=0.25)
fig.tight_layout()
save(fig, "Fig05_median_wells")

# ===================== FIG 6 — COMPONENT EFFECTS (ABLATION) ===================
EFFECTS = [("M2", "M1", "Rainfall inputs"),
           ("M3", "M2", "Graph (GCN), pooled LSTM"),
           ("M3.5G", "M3.5", "Graph (GCN), zone LSTM"),
           ("M3.5", "M2", "Zone branches"),
           ("M3.5", "M2big", "Zone branches, equal parameters"),
           ("M2big", "M2", "Capacity only (hidden 64 to 112)"),
           ("M4", "M3.5G", "Physics loss")]
EFFECTS = [e for e in EFFECTS if ((pt.A == e[0]) & (pt.B == e[1])).any()]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.35]})
am = abl.reindex(MODELS)
y = np.arange(len(MODELS))
a1.errorbar(am.RMSE_mean, y, xerr=am.RMSE_sd, fmt="none", ecolor="#555", capsize=3, lw=1.2)
a1.scatter(am.RMSE_mean, y, s=60, c=[MCOL[m] for m in MODELS], zorder=3, edgecolor="white")
for yi, m in enumerate(MODELS):
    a1.text(1.01, yi, f"{int(am.params[m]) / 1000:.0f}k", transform=a1.get_yaxis_transform(),
            va="center", fontsize=8.5, color="#555")
a1.set_yticks(y); a1.set_yticklabels(MODELS); a1.invert_yaxis()
a1.set_xlabel("Test RMSE (m), mean ± SD over seeds")
a1.set_title("(a) Model variants (right: parameters)")
a1.grid(axis="x", alpha=0.3)

for i, (A, B, lab) in enumerate(EFFECTS):
    r = pt[(pt.A == A) & (pt.B == B)].iloc[0]
    sig = r.wilcoxon_p < 0.05
    col = ("#2E7D6B" if r.mean_dRMSE_A_minus_B < 0 else "#C0504D") if sig else "#9E9E9E"
    a2.plot([r.CI95_low, r.CI95_high], [i, i], color=col, lw=2.4, solid_capstyle="round")
    a2.plot(r.mean_dRMSE_A_minus_B, i, "o", ms=8, color=col, mec="white", zorder=3)
    a2.text(1.02, i, f"p = {r.wilcoxon_p:.3f}   better in {int(r.A_better_in_wells)}/{int(r.n_wells)}",
            transform=a2.get_yaxis_transform(), va="center", fontsize=8.5)
a2.axvline(0, color="black", lw=1)
a2.set_yticks(range(len(EFFECTS)))
a2.set_yticklabels([f"{lab}\n({A} vs {B})" for A, B, lab in EFFECTS], fontsize=9)
a2.invert_yaxis()
a2.set_xlabel("Change in test RMSE (m), 95% bootstrap CI  (negative = component helps)")
a2.set_title("(b) Effect of each component (paired over wells)")
a2.grid(axis="x", alpha=0.3)
a2.legend(handles=[Line2D([], [], marker="o", color=c, lw=2, label=l) for c, l in
                   [("#2E7D6B", "helps, p < 0.05"), ("#C0504D", "hurts, p < 0.05"),
                    ("#9E9E9E", "not significant")]],
          loc="lower right", fontsize=8.5, frameon=False)
fig.tight_layout()
save(fig, "Fig06_component_effects")

# ======================= FIG 10 — FORECAST SKILL + 2040 =======================
FM = [m for m in ["M1", "M2", "M3", "M3.5", "M4"] if f"{m}_RMSE" in hc.columns]
meth = ["Climatology", "TheilSen"] + FM
lbl = {"Climatology": "Clim.", "TheilSen": "Trend"}
fig = plt.figure(figsize=(16, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1.15], wspace=0.35)

a = fig.add_subplot(gs[0])
rng = np.random.default_rng(0)
for i, m in enumerate(meth):
    v = hc[f"{m}_RMSE"].dropna().values
    c = MCOL.get(m, "#BDBDBD")
    a.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=10, color=c, alpha=0.45)
    a.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color="black", lw=2)
    a.text(i, v.mean(), f"{v.mean():.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
a.set_xticks(range(len(meth))); a.set_xticklabels([lbl.get(m, m) for m in meth], fontsize=9)
a.set_ylabel("RMSE 2020–2025 (m)")
a.set_title(f"(a) Free-run hindcast 2020–25\n(no head data after 2019; n = {len(hc)} wells)")
a.grid(axis="y", alpha=0.3)

b = fig.add_subplot(gs[1])
data = [hc[f"{m}_meanerr"].dropna().values for m in FM]
bp_ = b.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                medianprops=dict(color="black"))
for patch, m in zip(bp_["boxes"], FM):
    patch.set_facecolor(MCOL[m]); patch.set_alpha(0.6)
b.axhline(0, color="black", lw=1, ls="--")
b.set_xticks(range(1, len(FM) + 1)); b.set_xticklabels(FM)
b.set_ylabel("Mean error 2020–2025 (m)\n(negative = drifts below observed)")
b.set_title("(b) Level drift in free run\n ")
b.grid(axis="y", alpha=0.3)

c = fig.add_subplot(gs[2])
SC = [m for m in ["M1", "M3", "M4"] if m in set(fc.Model)]
for m in SC:
    g = fc[fc.Model == m]
    rho = stats.spearmanr(g.Dev_from_train_mean_m, g.Change_m).correlation
    c.scatter(g.Dev_from_train_mean_m, g.Change_m, s=22, color=MCOL[m], alpha=0.8,
              label=f"{m} (ρ = {rho:.2f})")
lim = np.nanmax(np.abs(fc.Dev_from_train_mean_m)) * 1.05
c.plot([-lim, lim], [lim, -lim], color="#888", ls=":", lw=1.2, label="full reversion (y = −x)")
c.axhline(0, color="black", lw=0.8); c.axvline(0, color="black", lw=0.8)
c.set_xlabel("Head 2025 minus 1998–2019 mean (m)")
c.set_ylabel("Projected change 2025–2040 (m)")
c.set_title("(c) 2040 scenario: change vs distance\nfrom 1998–2019 mean")
c.legend(fontsize=8.5, frameon=False, loc="upper right")
c.grid(alpha=0.3)
fig.text(0.01, -0.04, "Clim. = 2015–19 monthly climatology; Trend = Theil–Sen trend (≤2019) + monthly climatology. "
         "Bars = mean over wells.", fontsize=8.5, color="#555")
save(fig, "Fig10_forecast_skill")

print("\nDone. Fig 9 replacement = Fig07_HeadTrend_IDW from script 11.")
