"""
PGNN-LSTM -- Statistical Validation
Well-level bootstrap significance testing (M3.5 vs M1, M3.5 vs M4) and the
Indore-vs-Ujjain per-zone Mann-Whitney comparison. This is the exact script
that produced Table 2a and the Section 4.8 significance figures in the
manuscript.

Reads: eval_m1.csv, eval_m35.csv, eval_m4.csv (per-well R^2/RMSE, from
01_indore_main_pipeline.py) and eval_step5c_per_well_depth_scale.csv (from
02_crossdistrict_ujjain.py). Edit the CONFIG paths below to match your
output directory.
"""

"""
Statistical validation for PGNN-LSTM paper.
Three analyses in one script:
  1. Bootstrap comparison of per-well R^2 between model variants
  2. MC Dropout uncertainty quantification for M3.5
  3. Indore vs Ujjain per-zone two-sample comparison (Mann-Whitney + bootstrap CI)

THIS VERSION IS MATCHED TO YOUR ACTUAL PIPELINE (confirmed from your own
evaluate_per_well() code and training script):
  - eval_m1.csv, eval_m35.csv, eval_m4.csv already exist in OUTPUT_DIR,
    one row per well, columns: WellID, Zone, R2, RMSE, Bias.
    These are produced by your `evaluate_per_well()` function and are
    EXACTLY what Part 1 needs -- no raw predictions required, since the
    bootstrap resamples across wells, not across individual time steps.
  - M2/M3 are intentionally EXCLUDED from the formal significance test,
    since your Methods states they come from an earlier preprocessing
    pipeline -- comparing them statistically against M3.5 would not be a
    fair like-for-like test. Only M1 vs M3.5 and M3.5 vs M4 are tested,
    since all three share your current pipeline (m1_best.pt / m35_best.pt
    / m4_best.pt, trained in the same script).
  - M3.5's checkpoint is m35_best.pt (torch.save(m35.state_dict(), ...)),
    same directory as the other checkpoints.
  - Ujjain per-well results are assumed to follow the same naming pattern
    (eval_ujjain_m35.csv) -- if your cross-district script saved it under
    a different name, just change UJJAIN_EVAL_FILE below.

If any path or column name below doesn't match what's actually on disk,
only the CONFIG block needs to change -- the statistics do not.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ----------------------------------------------------------------------
# CONFIG -- EDIT THESE PATHS IF THEY DON'T MATCH YOUR OUTPUT_DIR
# ----------------------------------------------------------------------
OUTPUT_DIR = Path(r"J:\Ground_water\Indore_gw\saved_results\saved_results")
# Confirmed from your folder listing -- eval_m1.csv, eval_m35.csv, eval_m4.csv
# live inside this nested saved_results\saved_results folder (from a zip
# extraction), NOT directly in J:\Indore_gw as originally assumed.

EVAL_FILES = {
    "M1":   OUTPUT_DIR / "eval_m1.csv",
    "M3.5": OUTPUT_DIR / "eval_m35.csv",
    "M4":   OUTPUT_DIR / "eval_m4.csv",
}
# Columns expected (matches your evaluate_per_well() output exactly):
#   WellID, Zone, R2, RMSE, Bias

UJJAIN_EVAL_FILE = Path(r"J:\Ground_water\eval_step5c_per_well_depth_scale.csv")
# NOT CONFIRMED to have the same WellID/Zone/R2 column names -- this is
# the file your make_ujjain_figure.py script already reads from, so it's
# the most likely candidate, but check its actual column headers first
# (see check_ujjain_columns() below) and adjust load_eval() if they differ.

# NOT FOUND: no checkpoint file is named m35_best.pt or similar in your
# folder listing. Candidates worth checking:
#   J:\Ground_water\residual_lstm_model.pt   <- likely candidate: your
#       Results Section 4.8 text says "the residual LSTM framework was
#       applied to 122 additional wells", which sounds like M3.5's
#       zone-stratified architecture (no GCN, no physics -- just residual
#       LSTM per zone). This is the current best guess.
#   J:\Indore_gw\pgnn_v3_best.pt or pgnn_v4_best.pt  <- possible, but
#       naming doesn't confirm which ablation variant these are.
# CONFIRM which one is actually M3.5 before running Part 2 -- loading the
# wrong checkpoint will silently give meaningless uncertainty numbers.
M35_CHECKPOINT = Path(r"J:\Ground_water\residual_lstm_model.pt")

N_BOOTSTRAP = 2000
N_MC_PASSES = 100
RANDOM_SEED = 42


def check_ujjain_columns():
    """Run this alone first to confirm the Ujjain file's actual column names."""
    df = pd.read_csv(UJJAIN_EVAL_FILE, nrows=5)
    print("Ujjain file columns:", list(df.columns))
    print(df)

# ----------------------------------------------------------------------
# PART 1: Bootstrap comparison of per-well R^2 between variants
# ----------------------------------------------------------------------

def load_eval(path: Path) -> pd.DataFrame:
    """Load one of your eval_m*.csv files and standardise column names."""
    df = pd.read_csv(path)
    df = df.rename(columns={"WellID": "well_id", "Zone": "zone", "R2": "r2"})
    return df[["well_id", "zone", "r2"]].dropna()


def load_step5c(path: Path, district: str, min_n_test: int = 15) -> pd.DataFrame:
    """
    Load the combined Indore+Ujjain cross-district file
    (eval_step5c_per_well_depth_scale.csv), filter to one district, and
    apply the same test-period stability filter used in Section 4.8
    (n_test >= 15 months) so this matches the numbers already reported
    there (Indore R^2=0.49 / Ujjain R^2=0.47 for Massive, etc.).
    Columns confirmed from your file: Well No, zone, district, n_test, R2, RMSE_m, bias_m
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Well No": "well_id", "R2": "r2"})
    df = df[df["district"].str.lower() == district.lower()]
    df = df[df["n_test"] >= min_n_test]
    return df[["well_id", "zone", "r2"]].dropna()


def bootstrap_mean_diff(r2_a: np.ndarray, r2_b: np.ndarray, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Well-level bootstrap on the difference in mean per-well R^2 between two
    model variants evaluated on the SAME wells. Returns observed diff, 95% CI,
    and a two-sided bootstrap p-value (proportion of resamples where the sign
    of the difference flips).
    """
    assert len(r2_a) == len(r2_b), "Models must be evaluated on the same well set for a paired bootstrap."
    rng = np.random.default_rng(seed)
    n = len(r2_a)
    diffs = np.empty(n_boot)
    obs_diff = np.mean(r2_a) - np.mean(r2_b)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.mean(r2_a[idx]) - np.mean(r2_b[idx])
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        "obs_diff": obs_diff,
        "ci_95": (ci_low, ci_high),
        "p_value": min(p_value, 1.0),
        "significant": not (ci_low <= 0 <= ci_high),
    }


def run_part1():
    print("=" * 70)
    print("PART 1: Bootstrap comparison of per-well R^2 (paired, well-matched)")
    print("=" * 70)

    r2_tables = {}
    for name, path in EVAL_FILES.items():
        r2_tables[name] = load_eval(path)

    # Align on common wells so the paired bootstrap is valid
    common_wells = set(r2_tables["M1"]["well_id"])
    for t in r2_tables.values():
        common_wells &= set(t["well_id"])
    common_wells = sorted(common_wells)
    print(f"Wells common to M1 / M3.5 / M4: {len(common_wells)}")

    aligned = {}
    for name, t in r2_tables.items():
        t = t.set_index("well_id").loc[common_wells]
        aligned[name] = t

    comparisons = [("M3.5", "M1"), ("M3.5", "M4")]
    results = []
    for a, b in comparisons:
        res = bootstrap_mean_diff(aligned[a]["r2"].values, aligned[b]["r2"].values)
        results.append({"comparison": f"{a} vs {b}", **res})
        print(f"\n{a} vs {b}:")
        print(f"  Mean R^2 difference: {res['obs_diff']:.4f}")
        print(f"  95% bootstrap CI:    ({res['ci_95'][0]:.4f}, {res['ci_95'][1]:.4f})")
        print(f"  Bootstrap p-value:   {res['p_value']:.4f}")
        print(f"  Significant at 0.05: {res['significant']}")

    # Per-zone breakdown for the same comparisons
    print("\n--- Per-zone breakdown ---")
    zone_results = []
    zones = aligned["M3.5"]["zone"].unique()
    for a, b in comparisons:
        for zone in zones:
            wells_z = aligned[a][aligned[a]["zone"] == zone].index
            ra = aligned[a].loc[wells_z, "r2"].values
            rb = aligned[b].loc[wells_z, "r2"].values
            if len(ra) < 3:
                continue
            res = bootstrap_mean_diff(ra, rb)
            zone_results.append({"comparison": f"{a} vs {b}", "zone": zone, "n_wells": len(ra), **res})
            print(f"{a} vs {b} | {zone} (N={len(ra)}): diff={res['obs_diff']:.4f}, "
                  f"CI=({res['ci_95'][0]:.4f},{res['ci_95'][1]:.4f}), p={res['p_value']:.4f}")

    pd.DataFrame(results).to_csv("bootstrap_comparison_overall.csv", index=False)
    pd.DataFrame(zone_results).to_csv("bootstrap_comparison_by_zone.csv", index=False)
    print("\nSaved: bootstrap_comparison_overall.csv, bootstrap_comparison_by_zone.csv")


# ----------------------------------------------------------------------
# PART 2: MC Dropout uncertainty quantification for M3.5
# ----------------------------------------------------------------------

def run_part2():
    print("\n" + "=" * 70)
    print("PART 2: MC Dropout for M3.5 (same protocol as M1/M4, N=100 passes)")
    print("=" * 70)
    try:
        import torch
    except ImportError:
        print("PyTorch not available in this environment -- run Part 2 in Colab.")
        return

    # This mirrors your own evaluate_per_well() loading pattern (same model
    # forward-call signature: model(h_seq, r_seq, nf, zone, well_idx=...)),
    # just with dropout left ON and repeated N_MC_PASSES times instead of
    # a single model.eval() pass. Drop this into the same notebook cell
    # where `data`, `m35`, and `GWDataset`/`DataLoader` are already defined.
    #
    # m35.load_state_dict(torch.load(M35_CHECKPOINT, map_location=DEVICE))
    # m35.train()  # keep dropout ACTIVE -- do not call .eval()
    #
    # all_preds, tgts_m = [], None
    # for mc_pass in range(N_MC_PASSES):
    #     preds_pass = []
    #     with torch.no_grad():
    #         for wid in data['well_ids']:
    #             ds = GWDataset(data['test_df'], wid, data['well_meta'],
    #                            data['scalers'], data['node_features'],
    #                            data['RAIN_COLS'], SEQ_LEN)
    #             if len(ds) == 0:
    #                 continue
    #             loader = DataLoader(ds, batch_size=256, shuffle=False)
    #             for h_seq, r_seq, nf, tgt, zone, widx in loader:
    #                 h_seq, r_seq, nf, zone, widx = [t.to(DEVICE) for t in
    #                                                  (h_seq, r_seq, nf, zone, widx)]
    #                 pred = m35(h_seq, r_seq, nf, zone, well_idx=None)  # no physics term for M3.5
    #                 sc = data['scalers'][wid]
    #                 preds_pass.extend((pred.cpu().numpy() * sc.scale_[0] + sc.mean_[0]))
    #                 if mc_pass == 0:
    #                     tgts_m = tgts_m or []
    #                     tgts_m.extend((tgt.numpy() * sc.scale_[0] + sc.mean_[0]))
    #     all_preds.append(preds_pass)
    #
    # all_preds = np.array(all_preds)          # shape: (N_MC_PASSES, n_test_points)
    # tgts_m = np.array(tgts_m)
    # mu, sigma = all_preds.mean(axis=0), all_preds.std(axis=0)
    # ci_low, ci_high = mu - 1.96 * sigma, mu + 1.96 * sigma
    # coverage = np.mean((tgts_m >= ci_low) & (tgts_m <= ci_high))
    # ci_width = np.mean(ci_high - ci_low)
    #
    # print(f"M3.5 95% CI coverage: {coverage*100:.1f}%  (M1: 36.9%, M4: 29.5%)")
    # print(f"M3.5 mean CI width:   {ci_width:.3f} m  (M1: 0.412, M4: 0.366)")

    print("Skeleton matched to your evaluate_per_well() pattern -- uncomment "
          "and run inside the same notebook cell where `data` and `m35` "
          "already exist (don't call m35.eval(), keep dropout active).")


# ----------------------------------------------------------------------
# PART 3: Indore vs Ujjain per-zone comparison
# ----------------------------------------------------------------------

def run_part3():
    print("\n" + "=" * 70)
    print("PART 3: Indore vs Ujjain per-zone comparison (two-sample, unmatched)")
    print("=" * 70)

    indore_r2 = load_step5c(UJJAIN_EVAL_FILE, district="Indore")
    ujjain_r2 = load_step5c(UJJAIN_EVAL_FILE, district="Ujjain")
    print(f"Indore wells (n_test>=15 filter): {len(indore_r2)}")
    print(f"Ujjain wells (n_test>=15 filter): {len(ujjain_r2)}")

    results = []
    for zone in ["Massive", "Fractured", "Weathered"]:
        ind = indore_r2.loc[indore_r2["zone"] == zone, "r2"].values
        ujj = ujjain_r2.loc[ujjain_r2["zone"] == zone, "r2"].values
        if len(ind) < 3 or len(ujj) < 3:
            print(f"{zone}: insufficient wells for a formal test (Indore N={len(ind)}, Ujjain N={len(ujj)}) -- skipped.")
            continue

        # Mann-Whitney U: no assumption of paired wells or equal variance
        u_stat, p_mw = stats.mannwhitneyu(ind, ujj, alternative="two-sided")

        # Bootstrap CI on the mean difference (unpaired: resample each group independently)
        rng = np.random.default_rng(RANDOM_SEED)
        diffs = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            bi = rng.choice(ind, len(ind), replace=True)
            bj = rng.choice(ujj, len(ujj), replace=True)
            diffs[i] = np.mean(bi) - np.mean(bj)
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

        results.append({
            "zone": zone, "n_indore": len(ind), "n_ujjain": len(ujj),
            "mean_r2_indore": np.mean(ind), "mean_r2_ujjain": np.mean(ujj),
            "mean_diff": np.mean(ind) - np.mean(ujj),
            "ci_95_low": ci_low, "ci_95_high": ci_high,
            "mannwhitney_p": p_mw,
        })
        print(f"{zone}: Indore R^2={np.mean(ind):.3f} (N={len(ind)}), "
              f"Ujjain R^2={np.mean(ujj):.3f} (N={len(ujj)})")
        print(f"  Diff={np.mean(ind)-np.mean(ujj):.3f}, 95% CI=({ci_low:.3f},{ci_high:.3f}), "
              f"Mann-Whitney p={p_mw:.4f}")

    pd.DataFrame(results).to_csv("indore_ujjain_comparison.csv", index=False)
    print("\nSaved: indore_ujjain_comparison.csv")


if __name__ == "__main__":
    run_part1()
    run_part2()
    run_part3()