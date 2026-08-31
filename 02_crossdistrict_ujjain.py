"""
PGNN-LSTM -- Cross-District Extension (Indore + Ujjain)
Physics-residual pipeline used for Section 4.8 / Figure 11 in the
manuscript: T_eff-based zone classification, hydraulic-conductance
graph, a finite-difference physical baseline (Todd & Mays Ch.9), and
a zone-stratified residual LSTM (h_true = h_phys + h_residual) trained
jointly on 48 Indore + 122 Ujjain wells.

IMPORTANT: this is a SEPARATE pipeline from 01_indore_main_pipeline.py.
It does not share model weights or well subsets with the 36-well main
Indore analysis (Tables 2/3). It is the source of the Indore-vs-Ujjain
cross-district comparison numbers only (Section 4.8, Figure 11).

Several cells below say "UPDATE THE PATHS BELOW before running" -- these
are your original notes; edit the path constants for your environment.
"""


# ==============================================================================
# STEP 0 -- install xlrd (for reading legacy .xls files)
# ==============================================================================

# Run once in your shell/notebook: pip install xlrd

# ==============================================================================
# STEP 2a -- Ujjain lithology-based T_eff calculation (thickness-weighted, Todd & Mays Eq 3.4.5)
# ==============================================================================

"""
Ujjain — Step 2: Lithology-based T_eff calculation (thickness-weighted)
Same method as Indore (Todd & Mays Eq 3.4.5), applied to Ujjain OW + PZ wells.

T values (Singhal & Gupta 2010, Deccan Trap basalt):
  Weathered  = 30 m2/day
  Massive    = 5  m2/day
  Fractured  = 100 m2/day

Logic:
- Each row in the lithology sheet is a layer, with 'Depth To' = cumulative
  depth to bottom of that layer. Layer thickness = Depth_To[i] - Depth_To[i-1]
  (first layer starts at depth 0).
- Each lithology description is classified into Weathered / Massive / Fractured
  / Overburden (soil, clay, topsoil — non-aquifer material) using keyword rules.
- T_eff per well = thickness-weighted average T, computed ONLY over aquifer
  layers (Weathered/Massive/Fractured). Overburden thickness is tracked
  separately for QC but excluded from the T_eff weighting (consistent with
  Todd & Mays treating this as layered aquifer, not vadose zone).
- Dominant zone = the classification with the largest total thickness for
  that well -> used for zone_classification label.

Output: one row per well -> Well No, source (OW/PZ), total_aquifer_thickness,
T_eff, zone_classification, overburden_thickness, n_layers, n_unclassified
"""

import pandas as pd
import os

FOLDER = r"J:\Ground_water\Ujjain_gwdata"

T_VALUES = {
    "Weathered": 30.0,
    "Massive": 5.0,
    "Fractured": 100.0,
}

def classify_lithology(text):
    """Return one of: 'Weathered', 'Massive', 'Fractured', 'Overburden', 'Unclassified'"""
    if pd.isna(text):
        return "Unclassified"
    t = str(text).strip().lower()

    # Explicit fixes for reviewed unclassified strings
    explicit_map = {
        "basalt": "Massive",          # generic/unqualified -> assume fresh/massive rock
        "basalt fine": "Massive",     # fine-grained texture -> massive
        "red bole": "Overburden",     # inter-trappean weathered clay horizon -> aquitard, excluded from T_eff
        "yello siol": "Overburden",   # typo of "yellow soil"
    }
    if t in explicit_map:
        return explicit_map[t]

    # Overburden / non-aquifer material first
    overburden_kw = ["soil", "clay", "topsoil", "gravel", "silt", "sand"]
    if any(k in t for k in overburden_kw) and "basalt" not in t:
        return "Overburden"

    # Fractured takes priority if both weathered+fractured mentioned
    fractured_kw = ["fractured", "jointed"]
    if any(k in t for k in fractured_kw):
        return "Fractured"

    weathered_kw = ["weathered", "vesicular", "highly weathered"]
    if any(k in t for k in weathered_kw):
        return "Weathered"

    massive_kw = ["massive", "hard"]
    if any(k in t for k in massive_kw):
        return "Massive"

    return "Unclassified"


def compute_teff_for_file(filepath, source_tag):
    df = pd.read_excel(filepath)
    df = df.sort_values(["Well No", "Depth To"]).reset_index(drop=True)

    df["Category"] = df["Lithology"].apply(classify_lithology)

    results = []
    unclassified_log = []

    for well, grp in df.groupby("Well No"):
        grp = grp.sort_values("Depth To").reset_index(drop=True)
        prev_depth = 0.0
        aquifer_thickness_total = 0.0
        weighted_T_sum = 0.0
        overburden_thickness = 0.0
        thickness_by_cat = {"Weathered": 0.0, "Massive": 0.0, "Fractured": 0.0}
        n_unclassified = 0

        for _, row in grp.iterrows():
            depth_to = row["Depth To"]
            if pd.isna(depth_to):
                continue
            thickness = depth_to - prev_depth
            prev_depth = depth_to
            if thickness <= 0:
                continue

            cat = row["Category"]
            if cat == "Overburden":
                overburden_thickness += thickness
            elif cat in T_VALUES:
                weighted_T_sum += T_VALUES[cat] * thickness
                aquifer_thickness_total += thickness
                thickness_by_cat[cat] += thickness
            else:
                n_unclassified += 1
                unclassified_log.append((well, row["Lithology"]))

        if aquifer_thickness_total > 0:
            t_eff = weighted_T_sum / aquifer_thickness_total
            dominant_zone = max(thickness_by_cat, key=thickness_by_cat.get)
        else:
            t_eff = None
            dominant_zone = "No aquifer layers classified"

        results.append({
            "Well No": well,
            "source": source_tag,
            "n_layers": len(grp),
            "aquifer_thickness_m": round(aquifer_thickness_total, 2),
            "overburden_thickness_m": round(overburden_thickness, 2),
            "T_eff_m2_per_day": round(t_eff, 2) if t_eff is not None else None,
            "zone_classification": dominant_zone,
            "n_unclassified_layers": n_unclassified,
        })

    return pd.DataFrame(results), unclassified_log


if __name__ == "__main__":
    ow_path = os.path.join(FOLDER, "Well Lithology.xls")
    pz_path = os.path.join(FOLDER, "Well Lithology_pizzo.xls")

    ow_result, ow_unclassified = compute_teff_for_file(ow_path, "OW")
    pz_result, pz_unclassified = compute_teff_for_file(pz_path, "PZ")

    combined = pd.concat([ow_result, pz_result], ignore_index=True)

    print(f"Total wells processed: {len(combined)}")
    print(f"  OW wells: {len(ow_result)}")
    print(f"  PZ wells: {len(pz_result)}\n")

    print("Zone classification distribution:")
    print(combined["zone_classification"].value_counts().to_string())
    print()

    n_no_teff = combined["T_eff_m2_per_day"].isna().sum()
    print(f"Wells with NO T_eff computed (no aquifer layers classified): {n_no_teff}")
    if n_no_teff > 0:
        print(combined[combined["T_eff_m2_per_day"].isna()][["Well No", "source"]].to_string(index=False))
    print()

    all_unclassified = ow_unclassified + pz_unclassified
    if all_unclassified:
        unique_unclassified = sorted(set(text for _, text in all_unclassified))
        print(f"Unclassified lithology strings found ({len(unique_unclassified)} unique) — review these:")
        for s in unique_unclassified:
            print(f"  - {s}")
    else:
        print("No unclassified lithology strings.")
    print()

    print("T_eff summary stats:")
    print(combined["T_eff_m2_per_day"].describe().to_string())
    print()

    out_path = os.path.join(FOLDER, "Ujjain_Teff_by_well.xlsx")
    combined.to_excel(out_path, index=False)
    print(f"Saved combined output to: {out_path}")

# ==============================================================================
# STEP 1b -- filter Indore Book2.xlsx (53 wells) down to the confirmed 48-well set
# ==============================================================================

"""
Filter Indore Book2.xlsx (53 wells) down to the 48 wells used in the
final PGNN-LSTM model, by excluding the known problematic wells.

Excluded (per past project notes — data gaps / well filled/dead):
  SIND-006-PZ, SIND-PTW-01 NEW, SIND-06-PTW, SIND-34-PTW-NEW

NOTE: past summaries disagreed on count (4 named vs "5 excluded").
This script checks the actual math against Sheet1 and flags if the
4 named wells don't get us to exactly 48 — so we know if a 5th
well is missing from the list.
"""

import pandas as pd
import os

BOOK2_PATH = r"J:\Ground_water\Indore_gw\Book2.xlsx"

EXCLUDED_WELLS = [
    "SIND-006-PZ",
    "SIND-PTW-01 NEW",
    "SIND-06-PTW",
    "SIND-34-PTW-NEW",
    "SIND-035-PZ",   # replaced by SIND-035-NEW after being filled up Jan 2011
]

if not os.path.exists(BOOK2_PATH):
    print(f">> File NOT FOUND at: {BOOK2_PATH}")
    print(">> Update BOOK2_PATH and re-run.")
else:
    wells = pd.read_excel(BOOK2_PATH, sheet_name="Sheet1")
    total = len(wells)
    print(f"Total wells in Sheet1: {total}")

    # check which of the named excluded wells actually exist in the sheet
    all_well_ids = wells["Well No"].astype(str).str.strip().tolist()
    found = [w for w in EXCLUDED_WELLS if w in all_well_ids]
    not_found = [w for w in EXCLUDED_WELLS if w not in all_well_ids]

    print(f"\nExcluded-well-name matches found in Sheet1: {found}")
    if not_found:
        print(f">> NOT FOUND in Sheet1 (name mismatch — check spelling/format): {not_found}")

    remaining = total - len(found)
    print(f"\nWells remaining after excluding {len(found)} named wells: {remaining}")
    if remaining == 48:
        print(">> Matches expected 48 — exclusion list is CONFIRMED correct.")
    elif remaining == 49 and len(not_found) == 1:
        print(">> One named well not found (likely naming mismatch) — 49 remain, need 1 more exclusion.")
        print(">> Search Additional Information / stratigraphy columns for other 'filled/collapsed/not working' flags:")
        flag_kw = ["fill", "collaps", "not working", "defunc", "closed", "dead"]
        for col in ["Additonal Information"]:
            if col in wells.columns:
                flagged = wells[wells[col].astype(str).str.lower().str.contains("|".join(flag_kw), na=False)]
                if len(flagged) > 0:
                    print(f"\nWells flagged via '{col}':")
                    print(flagged[["Well No", col]].to_string(index=False))
    else:
        print(f">> Mismatch — expected 48, got {remaining}. Manual review needed.")
        print(">> Wells NOT in the exclude list still present:")
        remaining_ids = [w for w in all_well_ids if w not in EXCLUDED_WELLS]
        print(remaining_ids)

    # save the filtered 48-well set (or however many, for review)
    filtered = wells[~wells["Well No"].astype(str).str.strip().isin(EXCLUDED_WELLS)].copy()
    out_path = os.path.join(os.path.dirname(BOOK2_PATH), "Indore_48wells_filtered.xlsx")
    filtered.to_excel(out_path, index=False)
    print(f"\nSaved filtered well list ({len(filtered)} wells) to: {out_path}")

# ==============================================================================
# STEP 2b -- Indore lithology-based T_eff calculation (same method as Ujjain, on the 48-well set)
# ==============================================================================

"""
Indore — Step 2: Lithology-based T_eff calculation (thickness-weighted)
Same method as Ujjain, applied to the confirmed 48-well Indore set.

T values (Singhal & Gupta 2010, Deccan Trap basalt):
  Weathered  = 30 m2/day
  Massive    = 5  m2/day
  Fractured  = 100 m2/day
"""

import pandas as pd
import os

BOOK2_PATH = r"J:\Ground_water\Indore_gw\Book2.xlsx"
FILTERED_48_PATH = r"J:\Ground_water\Indore_gw\Indore_48wells_filtered.xlsx"
OUTPUT_PATH = r"J:\Ground_water\Indore_gw\Indore_Teff_by_well.xlsx"

EXCLUDED_WELLS = [
    "SIND-006-PZ",
    "SIND-PTW-01 NEW",
    "SIND-06-PTW",
    "SIND-34-PTW-NEW",
    "SIND-035-PZ",
]

T_VALUES = {
    "Weathered": 30.0,
    "Massive": 5.0,
    "Fractured": 100.0,
}


def classify_lithology(text):
    """Return one of: 'Weathered', 'Massive', 'Fractured', 'Overburden', 'Unclassified'"""
    if pd.isna(text):
        return "Unclassified"
    t = str(text).strip().lower()

    explicit_map = {
        "basalt": "Massive",
        "basalt fine": "Massive",
        "red bole": "Overburden",
        "yello siol": "Overburden",
        "b.c.soil": "Overburden",
        "b.c. soil": "Overburden",
        "black cotton": "Overburden",
        "joint basalt": "Fractured",
        "ash bed": "Overburden",
        "bolder": "Overburden",
        "red bole +vecicular basalt": "Weathered",
    }
    if t in explicit_map:
        return explicit_map[t]

    overburden_kw = ["soil", "clay", "topsoil", "gravel", "silt", "sand"]
    if any(k in t for k in overburden_kw) and "basalt" not in t:
        return "Overburden"

    fractured_kw = ["fractured", "jointed"]
    if any(k in t for k in fractured_kw):
        return "Fractured"

    weathered_kw = ["weathered", "vesicular", "highly weathered"]
    if any(k in t for k in weathered_kw):
        return "Weathered"

    massive_kw = ["massive", "hard"]
    if any(k in t for k in massive_kw):
        return "Massive"

    return "Unclassified"


def compute_teff(lith_df):
    lith_df = lith_df.rename(columns={lith_df.columns[0]: "Well No"})
    lith_df = lith_df.sort_values(["Well No", "Depth To"]).reset_index(drop=True)
    lith_df["Category"] = lith_df["Lithology"].apply(classify_lithology)

    results = []
    unclassified_log = []

    for well, grp in lith_df.groupby("Well No"):
        grp = grp.sort_values("Depth To").reset_index(drop=True)
        prev_depth = 0.0
        aquifer_thickness_total = 0.0
        weighted_T_sum = 0.0
        overburden_thickness = 0.0
        thickness_by_cat = {"Weathered": 0.0, "Massive": 0.0, "Fractured": 0.0}
        n_unclassified = 0

        for _, row in grp.iterrows():
            depth_to = row["Depth To"]
            if pd.isna(depth_to):
                continue
            thickness = depth_to - prev_depth
            prev_depth = depth_to
            if thickness <= 0:
                continue

            cat = row["Category"]
            if cat == "Overburden":
                overburden_thickness += thickness
            elif cat in T_VALUES:
                weighted_T_sum += T_VALUES[cat] * thickness
                aquifer_thickness_total += thickness
                thickness_by_cat[cat] += thickness
            else:
                n_unclassified += 1
                unclassified_log.append((well, row["Lithology"]))

        if aquifer_thickness_total > 0:
            t_eff = weighted_T_sum / aquifer_thickness_total
            dominant_zone = max(thickness_by_cat, key=thickness_by_cat.get)
        else:
            t_eff = None
            dominant_zone = "No aquifer layers classified"

        results.append({
            "Well No": well,
            "n_layers": len(grp),
            "aquifer_thickness_m": round(aquifer_thickness_total, 2),
            "overburden_thickness_m": round(overburden_thickness, 2),
            "T_eff_m2_per_day": round(t_eff, 2) if t_eff is not None else None,
            "zone_classification": dominant_zone,
            "n_unclassified_layers": n_unclassified,
        })

    return pd.DataFrame(results), unclassified_log


if __name__ == "__main__":
    lith_df = pd.read_excel(BOOK2_PATH, sheet_name="Sheet2")
    result, unclassified = compute_teff(lith_df)

    # keep only the confirmed 48 wells
    result_48 = result[~result["Well No"].isin(EXCLUDED_WELLS)].reset_index(drop=True)

    print(f"Total wells in lithology sheet: {len(result)}")
    print(f"Wells after excluding the 5 known-bad wells: {len(result_48)}")
    print()

    print("Zone classification distribution:")
    print(result_48["zone_classification"].value_counts().to_string())
    print()

    n_no_teff = result_48["T_eff_m2_per_day"].isna().sum()
    print(f"Wells with NO T_eff computed: {n_no_teff}")
    if n_no_teff > 0:
        print(result_48[result_48["T_eff_m2_per_day"].isna()][["Well No"]].to_string(index=False))
    print()

    if unclassified:
        unique_unclassified = sorted(set(text for _, text in unclassified))
        print(f"Unclassified lithology strings ({len(unique_unclassified)} unique) — review:")
        for s in unique_unclassified:
            print(f"  - {s}")

        # impact check: how much total thickness do these unclassified rows represent?
        lith_df_check = lith_df.copy()
        lith_df_check["is_unclassified"] = lith_df_check["Category"] == "Unclassified"
        total_thickness_all = 0.0
        total_thickness_unclassified = 0.0
        for well, grp in lith_df_check.groupby("Well No"):
            grp = grp.sort_values("Depth To").reset_index(drop=True)
            prev = 0.0
            for _, row in grp.iterrows():
                d = row["Depth To"]
                if pd.isna(d):
                    continue
                th = d - prev
                prev = d
                if th <= 0:
                    continue
                total_thickness_all += th
                if row["is_unclassified"]:
                    total_thickness_unclassified += th
        pct = 100 * total_thickness_unclassified / total_thickness_all if total_thickness_all > 0 else 0
        print(f"\nImpact check: unclassified rows = {total_thickness_unclassified:.1f}m out of "
              f"{total_thickness_all:.1f}m total logged thickness ({pct:.2f}%)")
    else:
        print("No unclassified lithology strings.")
    print()

    print("T_eff summary stats (48 wells):")
    print(result_48["T_eff_m2_per_day"].describe().to_string())

    result_48.to_excel(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")

# ==============================================================================
# STEP 2c -- combine Indore + Ujjain T_eff datasets into one file, tagged by district
# ==============================================================================

"""
Combine Indore (Book2.xlsx-based) + Ujjain (Ujjain_Teff_by_well.xlsx) T_eff
datasets into ONE single file, tagged by district, with lat/long attached.

UPDATE THE PATHS BELOW before running.
"""

import pandas as pd
import os

# ---- UPDATE THESE PATHS IF DIFFERENT ----
INDORE_TEFF_PATH = r"J:\Ground_water\Indore_gw\Indore_Teff_by_well.xlsx"
INDORE_METADATA_PATH = r"J:\Ground_water\Indore_gw\Book2.xlsx"
INDORE_METADATA_SHEET = "Sheet1"

UJJAIN_TEFF_PATH = r"J:\Ground_water\Ujjain_gwdata\Ujjain_Teff_by_well.xlsx"
UJJAIN_OW_PATH = r"J:\Ground_water\Ujjain_gwdata\Ujjain_gw.xls"            # for lat/long lookup (OW wells)
UJJAIN_PZ_PATH = r"J:\Ground_water\Ujjain_gwdata\GroundWater-General_pizzo.xls"  # for lat/long lookup (PZ wells)

OUTPUT_PATH = r"J:\Ground_water\Combined_Teff_Indore_Ujjain.xlsx"


def load_ujjain():
    teff = pd.read_excel(UJJAIN_TEFF_PATH)

    # NOTE: Latitude/Longitude in these files are int64 (DMS-encoded, unusable) —
    # same issue as Indore's Book2.xlsx. Use Easting/Northing (float64, decimal) instead.
    ow = pd.read_excel(UJJAIN_OW_PATH)[["Well No", "Easting", "Northing", "Geology", "District"]]
    pz = pd.read_excel(UJJAIN_PZ_PATH)[["Well No", "Easting", "Northing", "Geology", "District"]]
    latlong = pd.concat([ow, pz], ignore_index=True).drop_duplicates(subset="Well No")
    latlong = latlong.rename(columns={"Easting": "Longitude", "Northing": "Latitude"})

    merged = teff.merge(latlong, on="Well No", how="left")
    merged["district"] = "Ujjain"
    return merged


def load_indore():
    if not os.path.exists(INDORE_TEFF_PATH):
        print(f">> Indore T_eff file not found at: {INDORE_TEFF_PATH}")
        print(">> Update INDORE_TEFF_PATH at the top of this script, then re-run.")
        return None

    teff = pd.read_excel(INDORE_TEFF_PATH)

    if os.path.exists(INDORE_METADATA_PATH):
        try:
            meta = pd.read_excel(INDORE_METADATA_PATH, sheet_name=INDORE_METADATA_SHEET)
            # Easting/Northing hold decimal lon/lat (Latitude/Longitude cols are DMS strings, unusable)
            meta_sub = meta[["Well No", "Easting", "Northing", "Geology"]].rename(columns={
                "Easting": "Longitude", "Northing": "Latitude"
            })
            teff = teff.merge(meta_sub, on="Well No", how="left")
        except Exception as e:
            print(f">> Could not read Indore metadata sheet: {e}")

    teff["district"] = "Indore"
    return teff


if __name__ == "__main__":
    ujjain_df = load_ujjain()
    indore_df = load_indore()

    if indore_df is None:
        print("\nStopping — fix INDORE_TEFF_PATH and re-run before combining.")
    else:
        # align columns (union of both, missing filled with NaN)
        combined = pd.concat([indore_df, ujjain_df], ignore_index=True, sort=False)

        # drop duplicate District column (Ujjain source files had their own 'District' col)
        if "District" in combined.columns:
            combined = combined.drop(columns=["District"])

        # --- MANUAL COORDINATE CORRECTION ---
        # SUJN022-OW: raw longitude = 77.325836 (DMS: 77°19'33"), well's own
        # metadata lists Tahsil/Block = "Barnagar", which sits ~75.3-75.5E in
        # Ujjain district -- NOT 77.3E (that's ~150km away, near Bhopal/Vidisha).
        # Likely a single-digit transposition (77 -> 75) during data entry.
        # Correction: subtract 2.0 deg from longitude, keep minutes/seconds intact.
        # NOTE: this is an inferred correction, not independently verified against
        # a GPS/survey source -- flag as an assumption in the manuscript's data
        # limitations / methods section.
        mask = combined["Well No"] == "SUJN022-OW"
        if mask.any():
            old_lon = combined.loc[mask, "Longitude"].values[0]
            combined.loc[mask, "Longitude"] = old_lon - 2.0
            new_lon = combined.loc[mask, "Longitude"].values[0]
            print(f"\n>> Applied coordinate correction: SUJN022-OW longitude {old_lon:.4f} -> {new_lon:.4f}")
            print(">> (inferred digit-transposition fix, based on Barnagar tahsil location — NOT independently verified)")

        print(f"Combined total wells: {len(combined)}")
        print(combined["district"].value_counts().to_string())
        print()
        print("Columns in combined file:", list(combined.columns))
        print()
        print("Lat/Long sanity check (should be ~21-24 lat, ~74-78 long for MP region):")
        print(f"  Latitude range: {combined['Latitude'].min()} to {combined['Latitude'].max()}")
        print(f"  Longitude range: {combined['Longitude'].min()} to {combined['Longitude'].max()}")

        combined.to_excel(OUTPUT_PATH, index=False)
        print(f"\nSaved combined file to: {OUTPUT_PATH}")

# ==============================================================================
# STEP 3 -- hydraulic-conductance graph edges over all 171 wells (Todd & Mays Ch.9 style)
# ==============================================================================

"""
Step 3 — Hydraulic conductance graph edges (Todd & Mays Ch.9 style)

Method (confirmed via literature check — Bai et al. 2023, Taccari et al. 2024,
PINN-GDA 2026):
  - Single graph over ALL 171 wells (Indore + Ujjain combined), not split by
    district — cross-district edges allowed where hydraulically plausible.
  - k-Nearest-Neighbor restriction (k=8, tunable) keeps the graph sparse and
    physically sensible (no edges between wells 50+ km apart).
  - Edge weight = harmonic-mean transmissivity conductance proxy:
        C_ij = [2 * T_i * T_j / (T_i + T_j)] / L_ij
    where T_i, T_j = T_eff (m2/day) of well i, j (already thickness-integrated,
    so no separate cross-sectional area term needed), and L_ij = distance (km)
    between wells i and j (haversine, since we have lat/long not projected coords).

Output: edge list (well_i, well_j, distance_km, T_i, T_j, conductance) +
        adjacency-style summary, saved to CSV for direct use in the GNN pipeline.
"""

import pandas as pd
import numpy as np
import os

COMBINED_PATH = r"J:\Ground_water\Combined_Teff_Indore_Ujjain.xlsx"
OUTPUT_EDGES_PATH = r"J:\Ground_water\graph_edges_knn8_conductance.csv"

K_NEIGHBORS = 8  # tunable — start with 8, can sweep 5/8/10 later for sensitivity


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


if __name__ == "__main__":
    if not os.path.exists(COMBINED_PATH):
        print(f">> File NOT FOUND at: {COMBINED_PATH}")
        print(">> Update COMBINED_PATH and re-run.")
        exit()

    df = pd.read_excel(COMBINED_PATH)

    required_cols = ["Well No", "Latitude", "Longitude", "T_eff_m2_per_day"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f">> Missing required columns: {missing}")
        print(f">> Available columns: {list(df.columns)}")
        exit()

    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f">> Dropped {before - after} wells with missing lat/long/T_eff (can't build edges for them)")

    n = len(df)
    print(f"Building graph for {n} wells (k={K_NEIGHBORS})...")

    lats = df["Latitude"].values
    lons = df["Longitude"].values
    teffs = df["T_eff_m2_per_day"].values
    well_ids = df["Well No"].values
    districts = df["district"].values if "district" in df.columns else ["unknown"] * n

    # pairwise distance matrix (n x n) — fine for ~171 wells, avoid for very large n
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i, :] = haversine_km(lats[i], lons[i], lats, lons)

    edges = []
    for i in range(n):
        # exclude self (distance 0), get k nearest
        order = np.argsort(dist_matrix[i, :])
        order = order[order != i][:K_NEIGHBORS]

        for j in order:
            L_ij = dist_matrix[i, j]
            if L_ij <= 0:
                continue  # skip coincident/duplicate coordinates
            T_i, T_j = teffs[i], teffs[j]
            conductance = (2 * T_i * T_j / (T_i + T_j)) / L_ij

            edges.append({
                "well_i": well_ids[i],
                "well_j": well_ids[j],
                "district_i": districts[i],
                "district_j": districts[j],
                "distance_km": round(L_ij, 3),
                "T_i": T_i,
                "T_j": T_j,
                "conductance": round(conductance, 4),
                "cross_district": districts[i] != districts[j],
            })

    edges_df = pd.DataFrame(edges)

    print(f"\nTotal directed edges: {len(edges_df)}")
    print(f"Cross-district edges: {edges_df['cross_district'].sum()} "
          f"({100 * edges_df['cross_district'].mean():.1f}%)")
    print(f"\nDistance stats (km):")
    print(edges_df["distance_km"].describe().to_string())
    print(f"\nConductance stats:")
    print(edges_df["conductance"].describe().to_string())

    # flag suspiciously long edges (sanity check — e.g. > 100km would be odd for kNN)
    long_edges = edges_df[edges_df["distance_km"] > 100]
    if len(long_edges) > 0:
        print(f"\n>> WARNING: {len(long_edges)} edges >100km — check for coordinate errors:")
        print(long_edges[["well_i", "well_j", "distance_km"]].to_string(index=False))

    edges_df.to_csv(OUTPUT_EDGES_PATH, index=False)
    print(f"\nSaved edge list to: {OUTPUT_EDGES_PATH}")

# ==============================================================================
# STEP 4 -- finite-difference physical baseline (2D transient groundwater flow, implicit backward-Euler)
# ==============================================================================

"""
Step 4 — FD Physical Baseline
2D transient groundwater flow, implicit (backward Euler) finite difference,
following Todd & Mays Ch.9 governing equation:

    S * dh/dt = d/dx(T * dh/dx) + d/dy(T * dh/dy) + R

Design decisions (documented for manuscript methods/limitations section):
  - Grid: ~1km cells over the combined Indore+Ujjain bounding box
  - T field: IDW interpolation from well T_eff (Step 2 output)
  - S/Sy: LITERATURE PROXY (not measured) — Singhal & Gupta 2010 style values:
        Weathered -> Sy = 0.02   (unconfined)
        Massive   -> S  = 0.0005 (confined/semi-confined)
        Fractured -> S  = 0.001  (confined-like, fracture flow)
    FLAG: these S values are inferred defaults, not independently sourced —
    verify against Singhal & Gupta (2010) table before using in manuscript.
  - Recharge: monthly rainfall x infiltration factor (10% default, GEC-style)
  - Boundary condition: SPECIFIED-HEAD (Dirichlet, fixed at reference datum)
    at domain edges -- allows lateral outflow, representing unmodeled
    regional baseflow/discharge. A pure no-flow boundary with no pumping/ET
    sink causes heads to rise without bound under any positive recharge,
    which is unphysical (confirmed empirically: first FD run hit +100m in
    one wet season with no-flow boundaries).
  - ADDITIONAL distributed drainage term (-h/tau, tau=730 days default): the
    domain is too large relative to the diffusion length scale over the
    simulation period, so boundary outflow alone doesn't reach the interior
    in time -- heads still accumulated unboundedly (2000+ m by month 168)
    even WITH the Dirichlet boundary fix. This lumped local-drainage term
    (proxy for streams/tanks/ET not explicitly in the dataset) is a second
    layer of assumption -- FLAG STRONGLY for manuscript limitations. Ideally
    tau should be calibrated against observed water-level recession curves
    rather than left as a literature-style default.
  - No pumping/abstraction data available -> baseline is natural-recharge-driven
    only. Anthropogenic abstraction effects are NOT in h_phys; they get
    captured by the LSTM residual in Step 5 (by design).
  - Time stepping: monthly
"""

import numpy as np
import pandas as pd
import os
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# ==================== PATHS ====================
COMBINED_WELLS_PATH = r"J:\Ground_water\Combined_Teff_Indore_Ujjain.xlsx"
UJJAIN_RAINFALL_PATH = r"J:\Ground_water\Ujjain_gwdata\Data-Rainfall1.xls"  # deduped version
INDORE_RAINFALL_DIR = r"J:\NDCQ-2026-03-471\datafiles"
INDORE_WATERLEVEL_PATH = r"J:\Ground_water\Indore_gw\Book2.xlsx"  # Sheet3 for initial condition
OUTPUT_HPHYS_PATH = r"J:\Ground_water\FD_baseline_h_phys.csv"

# ==================== PARAMETERS ====================
CELL_SIZE_DEG = 0.01   # ~1.1km at this latitude
INFILTRATION_FACTOR = 0.10  # 10% of rainfall becomes recharge (GEC-style default)
IDW_POWER = 2
START_DATE = "2010-01-01"   # keep runtime manageable; extend later if needed
END_DATE = "2025-12-31"
TIME_STEP = "MS"  # monthly

S_LOOKUP = {
    "Weathered": 0.02,
    "Massive": 0.0005,
    "Fractured": 0.001,
    "No aquifer layers classified": 0.001,  # fallback
}

INDORE_RAIN_FILES = {
    'Depalpur':   'DEPALPUR (REV)_RAINFALL_DAILY_NDCQ-2026-03-471.csv',
    'Gautampura': 'GAUTAMPURA (REV)_RAINFALL_DAILY_NDCQ-2026-03-471.csv',
    'Indore_obs': 'INDORE (OBSY)_RAINFALL_DAILY_NDCQ-2026-03-471.csv',
    'Indore_aws': 'INDORE (AWS)_RAINFALL_DAILY_NDCQ-2026-03-471.csv',
    'Mhow':       'MHOW (REV)_RAINFALL_DAILY_NDCQ-2026-03-471.csv',
    'Sanwer':     'SANWER (REV)_RAINFALL_DAILY_NDCQ-2026-03-471.csv',
}


# ==================== STEP 4a: Load wells + build grid ====================
def load_wells():
    df = pd.read_excel(COMBINED_WELLS_PATH)
    df = df.dropna(subset=["Latitude", "Longitude", "T_eff_m2_per_day"]).reset_index(drop=True)
    df["S"] = df["zone_classification"].map(S_LOOKUP).fillna(0.001)
    return df


def build_grid(wells_df, cell_size=CELL_SIZE_DEG, buffer=0.05):
    lat_min = wells_df["Latitude"].min() - buffer
    lat_max = wells_df["Latitude"].max() + buffer
    lon_min = wells_df["Longitude"].min() - buffer
    lon_max = wells_df["Longitude"].max() + buffer

    lats = np.arange(lat_min, lat_max, cell_size)
    lons = np.arange(lon_min, lon_max, cell_size)
    ny, nx = len(lats), len(lons)
    print(f"Grid: {ny} rows x {nx} cols = {ny*nx} cells (cell size {cell_size} deg)")
    return lats, lons, ny, nx


def idw_interpolate(grid_lats, grid_lons, well_lats, well_lons, well_values, power=IDW_POWER):
    """Simple IDW interpolation of well_values onto the grid."""
    ny, nx = len(grid_lats), len(grid_lons)
    field = np.zeros((ny, nx))
    for i, glat in enumerate(grid_lats):
        for j, glon in enumerate(grid_lons):
            d = np.sqrt((well_lats - glat) ** 2 + (well_lons - glon) ** 2)
            d = np.where(d < 1e-6, 1e-6, d)
            w = 1.0 / (d ** power)
            field[i, j] = np.sum(w * well_values) / np.sum(w)
    return field


# ==================== STEP 4b: Recharge time series ====================
def read_wris_rainfall(fpath):
    df = pd.read_csv(fpath, skiprows=2, encoding='latin-1', header=0)
    df.columns = df.columns.str.strip()
    day_cols = [c for c in df.columns if c.startswith('DRF')]
    df = df[['YEAR', 'MONTH'] + day_cols].copy()
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce')
    df = df.dropna(subset=['YEAR', 'MONTH'])
    df[day_cols] = df[day_cols].apply(pd.to_numeric, errors='coerce')
    df['Monthly_mm'] = df[day_cols].sum(axis=1, skipna=True)
    df['date'] = pd.to_datetime(
        df['YEAR'].astype(int).astype(str) + '-' +
        df['MONTH'].astype(int).astype(str).str.zfill(2) + '-01'
    )
    return df.set_index('date')['Monthly_mm'].sort_index()


def load_indore_rainfall():
    series = {}
    for stn, fname in INDORE_RAIN_FILES.items():
        fpath = os.path.join(INDORE_RAINFALL_DIR, fname)
        if os.path.exists(fpath):
            try:
                series[stn] = read_wris_rainfall(fpath)
            except Exception as e:
                print(f">> Could not read {fname}: {e}")
    if not series:
        print(">> WARNING: no Indore rainfall files found — check INDORE_RAINFALL_DIR")
        return pd.Series(dtype=float)
    combined = pd.DataFrame(series).mean(axis=1)  # simple station average
    return combined


def load_ujjain_rainfall():
    if not os.path.exists(UJJAIN_RAINFALL_PATH):
        print(f">> Ujjain rainfall file not found: {UJJAIN_RAINFALL_PATH}")
        return pd.Series(dtype=float)
    df = pd.read_excel(UJJAIN_RAINFALL_PATH)
    df['date'] = pd.to_datetime(df['Date'])
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    monthly = df.groupby('month')['Rainfall'].sum()
    return monthly


def build_regional_recharge(date_range):
    """Average Indore + Ujjain monthly rainfall -> single regional R(t) time series (mm/month -> m/day)."""
    indore_rain = load_indore_rainfall()
    ujjain_rain = load_ujjain_rainfall()

    combined = pd.DataFrame({"indore": indore_rain, "ujjain": ujjain_rain})
    combined = combined.reindex(date_range)
    combined["regional_mm"] = combined.mean(axis=1)
    combined["regional_mm"] = combined["regional_mm"].fillna(combined["regional_mm"].mean())

    # convert mm/month -> m/day recharge rate, apply infiltration factor
    days_in_month = date_range.days_in_month
    recharge_m_per_day = (combined["regional_mm"].values / 1000.0) * INFILTRATION_FACTOR / days_in_month
    return recharge_m_per_day


# ==================== STEP 4c: FD solver ====================
DRAINAGE_TAU_DAYS = 730  # ~2 years — literature-proxy drainage/recession timescale, FLAG: assumption, not calibrated

def solve_fd(T_field, S_field, recharge_series, dt_days, ny, nx, dx_m, h_init, tau_days=DRAINAGE_TAU_DAYS):
    """
    Implicit (backward Euler) 2D FD solve.
    T_field, S_field: (ny, nx) arrays
    recharge_series: array of length n_timesteps, uniform recharge rate (m/day) per step
    Returns: h array of shape (n_timesteps+1, ny, nx)

    Boundary condition: specified-head (Dirichlet) at domain edges.

    Additional DISTRIBUTED DRAINAGE TERM (-h/tau at every cell): needed because
    the domain (~150-200km) is far larger than the diffusion length scale over
    the simulation period (~11km over 16 years, given T~11-100 m2/day and
    S~0.0005-0.02) -- so boundary outflow alone never reaches the interior and
    heads accumulate without bound locally. Real aquifers drain locally via
    streams/tanks/ET (e.g., Yashwant Sagar, Mendakwas Tank appear in the well
    metadata) that aren't explicitly in this dataset. This term is a lumped
    proxy for that local drainage. tau_days is an assumption (not calibrated
    against observed recession curves) -- FLAG for manuscript limitations,
    and consider calibrating tau against observed water-level recession rates
    before treating h_phys as a rigorous physical baseline.
    """
    n_cells = ny * nx
    n_steps = len(recharge_series)
    h = np.zeros((n_steps + 1, ny, nx))
    h[0] = h_init

    def idx(i, j):
        return i * nx + j

    boundary_mask = np.zeros((ny, nx), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True

    for t in range(n_steps):
        A = lil_matrix((n_cells, n_cells))
        b = np.zeros(n_cells)
        h_prev = h[t]

        for i in range(ny):
            for j in range(nx):
                k = idx(i, j)

                if boundary_mask[i, j]:
                    A[k, k] = 1.0
                    b[k] = h_init[i, j]
                    continue

                Sc = S_field[i, j]
                A[k, k] += Sc / dt_days
                A[k, k] += 1.0 / tau_days  # distributed drainage sink
                b[k] += Sc / dt_days * h_prev[i, j]
                b[k] += recharge_series[t]

                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < ny and 0 <= nj < nx:
                        T_harm = 2 * T_field[i, j] * T_field[ni, nj] / (T_field[i, j] + T_field[ni, nj] + 1e-9)
                        coef = T_harm / (dx_m ** 2)
                        A[k, k] += coef
                        A[k, idx(ni, nj)] -= coef

        A_csr = csr_matrix(A)
        h_new = spsolve(A_csr, b)
        h[t + 1] = h_new.reshape(ny, nx)

        if t % 24 == 0:
            print(f"  solved step {t}/{n_steps}  (max head so far: {h[t+1].max():.2f})")

    return h


# ==================== STEP 4d: Extract h_phys at well locations ====================
def extract_at_wells(h_grid, grid_lats, grid_lons, wells_df, date_range):
    records = []
    for _, well in wells_df.iterrows():
        i = np.argmin(np.abs(grid_lats - well["Latitude"]))
        j = np.argmin(np.abs(grid_lons - well["Longitude"]))
        h_series = h_grid[1:, i, j]  # skip t=0 initial condition
        for t, date in enumerate(date_range):
            records.append({
                "Well No": well["Well No"],
                "date": date,
                "h_phys": h_series[t],
            })
    return pd.DataFrame(records)


# ==================== MAIN ====================
if __name__ == "__main__":
    print("Loading wells...")
    wells = load_wells()
    print(f"Wells with valid T_eff/lat/long: {len(wells)}")

    lats, lons, ny, nx = build_grid(wells)

    print("\nInterpolating T field (IDW)...")
    T_field = idw_interpolate(lats, lons, wells["Latitude"].values, wells["Longitude"].values,
                               wells["T_eff_m2_per_day"].values)
    print(f"  T range: {T_field.min():.1f} - {T_field.max():.1f} m2/day")

    print("\nInterpolating S field (IDW, from literature-proxy S per well)...")
    S_field = idw_interpolate(lats, lons, wells["Latitude"].values, wells["Longitude"].values,
                               wells["S"].values)
    print(f"  S range: {S_field.min():.5f} - {S_field.max():.5f}")

    date_range = pd.date_range(START_DATE, END_DATE, freq=TIME_STEP)
    print(f"\nBuilding regional recharge series ({len(date_range)} months)...")
    recharge = build_regional_recharge(date_range)
    print(f"  Recharge rate range: {recharge.min():.6f} - {recharge.max():.6f} m/day")

    dx_m = CELL_SIZE_DEG * 111000  # rough deg->m conversion at this latitude
    dt_days = 30.0  # approx month

    # initial condition: flat field at a nominal starting head (0 = relative datum)
    # NOTE: using relative heads since absolute elevation/GL data wasn't merged in.
    h_init = np.zeros((ny, nx))

    print("\nSolving FD system (this may take a few minutes)...")
    h_grid = solve_fd(T_field, S_field, recharge, dt_days, ny, nx, dx_m, h_init)

    print("\nExtracting h_phys at well locations...")
    h_phys_df = extract_at_wells(h_grid, lats, lons, wells, date_range)

    h_phys_df.to_csv(OUTPUT_HPHYS_PATH, index=False)
    print(f"\nSaved physical baseline to: {OUTPUT_HPHYS_PATH}")
    print(f"Shape: {h_phys_df.shape}")
    print(h_phys_df.head(10).to_string())

    print(f"\nSanity check — h_phys range across all wells/dates:")
    print(h_phys_df["h_phys"].describe().to_string())
    print("\n(Compare to real observed water-level fluctuation range, roughly "
          "0-15m in the raw Sheet3/Water Levels data — if h_phys is wildly "
          "outside that range, tau or S/T assumptions need revisiting.)")

# ==============================================================================
# STEP 5a -- build residual training dataset (h_residual = h_true - h_phys)
# ==============================================================================

"""
Step 5a — Build residual training dataset
h_true = h_phys + h_residual  ->  h_residual = h_true - h_phys

IMPORTANT sign/datum handling:
  - Raw "Water Level" data = depth-to-water BELOW GROUND (m bgl). Larger value
    = water table deeper = LOWER head. Confirmed from Book2 Sheet3 pattern
    (values drop after monsoon, e.g. May->Aug decrease = water rises).
  - h_phys = relative head from FD model (flat 0 reference datum, NOT absolute
    elevation). Larger value = more water = HIGHER head.
  - These are on different, incompatible absolute scales (one is depth-below-
    ground per well's own local ground elevation, other is a synthetic
    regional relative-head field). We can't validly subtract them directly.
  - FIX: convert both to per-well ANOMALIES (deviation from that well's own
    mean over the overlap period), with a sign flip on the observed depth
    series (multiply by -1, since deeper = lower head). This makes both
    series comparable as "departure from normal," regardless of absolute
    datum -- which is what the residual LSTM actually needs to learn
    (the physically-unexplained fluctuation on top of the FD baseline).
  - FLAG for manuscript: this anomaly-based approach means h_residual models
    RELATIVE dynamics, not absolute head. If absolute-head prediction is
    needed later, a per-well bias/datum-correction term must be added
    separately (e.g. learned per-well offset, or tied to Elevation of Ground
    Level from Book2 Sheet1).
"""

import pandas as pd
import numpy as np
import os

BOOK2_PATH = r"J:\Ground_water\Indore_gw\Book2.xlsx"
UJJAIN_OW_WL_PATH = r"J:\Ground_water\Ujjain_gwdata\Water Levels.xls"
UJJAIN_PZ_WL_PATH = r"J:\Ground_water\Ujjain_gwdata\Water Levels_pizzo.xls"
HPHYS_PATH = r"J:\Ground_water\FD_baseline_h_phys.csv"
COMBINED_WELLS_PATH = r"J:\Ground_water\Combined_Teff_Indore_Ujjain.xlsx"
EDGES_PATH = r"J:\Ground_water\graph_edges_knn8_conductance.csv"

EXCLUDED_INDORE_WELLS = [
    "SIND-006-PZ", "SIND-PTW-01 NEW", "SIND-06-PTW", "SIND-34-PTW-NEW", "SIND-035-PZ",
]

OUTPUT_PATH = r"J:\Ground_water\Step5_training_data.csv"


def load_observed_water_levels():
    # Indore
    indore_wl = pd.read_excel(BOOK2_PATH, sheet_name="Sheet3")
    indore_wl = indore_wl[~indore_wl["Well No"].isin(EXCLUDED_INDORE_WELLS)]
    indore_wl = indore_wl[["Well No", "date", "Water Level"]].dropna(subset=["date", "Water Level"])
    indore_wl["district"] = "Indore"

    # Ujjain OW + PZ
    ujjain_ow = pd.read_excel(UJJAIN_OW_WL_PATH)[["Well No", "date", "Water Level"]]
    ujjain_pz = pd.read_excel(UJJAIN_PZ_WL_PATH)[["Well No", "date", "Water Level"]]
    ujjain_wl = pd.concat([ujjain_ow, ujjain_pz], ignore_index=True).dropna(subset=["date", "Water Level"])
    ujjain_wl["district"] = "Ujjain"

    all_wl = pd.concat([indore_wl, ujjain_wl], ignore_index=True)

    # robust date parsing: coerce bad dates to NaT instead of crashing
    all_wl["date"] = pd.to_datetime(all_wl["date"], errors="coerce")

    n_before = len(all_wl)
    bad_dates = all_wl[all_wl["date"].isna()]
    if len(bad_dates) > 0:
        print(f">> WARNING: {len(bad_dates)} rows had unparseable dates — dropped")

    # also filter to a sane range (catches things like year 207, or year 2099 typos)
    all_wl = all_wl.dropna(subset=["date"])
    out_of_range = all_wl[(all_wl["date"] < "1990-01-01") | (all_wl["date"] > "2026-12-31")]
    if len(out_of_range) > 0:
        print(f">> WARNING: {len(out_of_range)} rows outside sane date range (1990-2026) — flagged:")
        print(out_of_range[["Well No", "date"]].to_string(index=False))
    all_wl = all_wl[(all_wl["date"] >= "1990-01-01") & (all_wl["date"] <= "2026-12-31")]

    n_after = len(all_wl)
    print(f"Date cleaning: {n_before} -> {n_after} rows ({n_before - n_after} removed)")

    all_wl["month"] = all_wl["date"].dt.to_period("M").dt.to_timestamp()

    # --- OUTLIER CLEANING on depth_to_water_m ---
    # 1) Hard physical sanity bound: these are shallow basalt-terrain piezometers,
    #    observed range in this dataset is ~0-40m bgl. Anything >100m is almost
    #    certainly a decimal-point/digit data-entry error (confirmed cases found:
    #    SIND-004-C-PZ 850.00, SIND-019-A-PZ 180.80 -- both ~10x their well's
    #    normal range, consistent with a misplaced decimal).
    n_before_hard = len(all_wl)
    hard_outliers = all_wl[(all_wl["Water Level"] < 0) | (all_wl["Water Level"] > 100)]
    if len(hard_outliers) > 0:
        print(f"\n>> Hard physical-bound outliers (depth <0 or >100m) — DROPPED:")
        print(hard_outliers[["Well No", "date", "Water Level"]].to_string(index=False))
    all_wl = all_wl[(all_wl["Water Level"] >= 0) & (all_wl["Water Level"] <= 100)]

    # 2) Per-well statistical outlier filter (median absolute deviation), catches
    #    smaller but still implausible errors (e.g. SIND-039-A-PZ 64.65 vs well
    #    mean ~18m) that pass the hard bound but are still well-specific anomalies.
    def flag_mad_outliers(group, thresh=6.0):
        med = group["Water Level"].median()
        mad = (group["Water Level"] - med).abs().median()
        if mad == 0:
            return pd.Series([False] * len(group), index=group.index)
        modified_z = 0.6745 * (group["Water Level"] - med) / mad
        return modified_z.abs() > thresh

    outlier_mask = all_wl.groupby("Well No", group_keys=False).apply(flag_mad_outliers)
    mad_outliers = all_wl[outlier_mask]
    if len(mad_outliers) > 0:
        print(f"\n>> Per-well statistical outliers (MAD-based) — DROPPED:")
        print(mad_outliers[["Well No", "date", "Water Level"]].to_string(index=False))
    all_wl = all_wl[~outlier_mask]

    n_after_outlier = len(all_wl)
    print(f"\nOutlier cleaning: {n_before_hard} -> {n_after_outlier} rows "
          f"({n_before_hard - n_after_outlier} removed)")

    # if multiple readings in same month, average
    monthly = all_wl.groupby(["Well No", "month"])["Water Level"].mean().reset_index()
    monthly = monthly.rename(columns={"Water Level": "depth_to_water_m", "month": "date"})
    return monthly


def build_anomalies(obs_df, hphys_df):
    merged = obs_df.merge(hphys_df, on=["Well No", "date"], how="inner")
    print(f"Matched observed+h_phys records: {len(merged)}")

    # per-well means over the overlap period
    well_means = merged.groupby("Well No").agg(
        mean_depth=("depth_to_water_m", "mean"),
        mean_hphys=("h_phys", "mean"),
    ).reset_index()

    merged = merged.merge(well_means, on="Well No", how="left")

    # anomalies (sign-flipped for depth, since deeper = lower head)
    merged["obs_head_anomaly"] = -1 * (merged["depth_to_water_m"] - merged["mean_depth"])
    merged["hphys_anomaly"] = merged["h_phys"] - merged["mean_hphys"]

    # residual target: what the physics baseline does NOT explain
    merged["residual_target"] = merged["obs_head_anomaly"] - merged["hphys_anomaly"]

    return merged


if __name__ == "__main__":
    print("Loading observed water levels...")
    obs = load_observed_water_levels()
    print(f"  Total observed monthly records: {len(obs)}, wells: {obs['Well No'].nunique()}")

    print("\nLoading h_phys...")
    hphys = pd.read_csv(HPHYS_PATH, parse_dates=["date"])
    print(f"  h_phys records: {len(hphys)}, wells: {hphys['Well No'].nunique()}")

    print("\nBuilding anomalies + residual target...")
    training_df = build_anomalies(obs, hphys)

    print("\nMerging well features (T_eff, zone, district)...")
    wells = pd.read_excel(COMBINED_WELLS_PATH)
    feature_cols = ["Well No", "T_eff_m2_per_day", "zone_classification", "district",
                     "aquifer_thickness_m", "Latitude", "Longitude"]
    training_df = training_df.merge(wells[feature_cols], on="Well No", how="left")

    print(f"\nFinal training dataset shape: {training_df.shape}")
    print(f"Wells covered: {training_df['Well No'].nunique()}")
    print(f"Date range: {training_df['date'].min()} to {training_df['date'].max()}")

    print("\nResidual target stats:")
    print(training_df["residual_target"].describe().to_string())

    print("\nWells with fewest observed months (potential data-sparsity issue):")
    counts = training_df.groupby("Well No").size().sort_values()
    print(counts.head(10).to_string())

    all_well_ids = set(wells["Well No"])
    covered_well_ids = set(training_df["Well No"].unique())
    missing_wells = all_well_ids - covered_well_ids
    if missing_wells:
        print(f"\n>> Wells with ZERO matched records (missing from training data): {missing_wells}")
        for w in missing_wells:
            n_obs = len(obs[obs["Well No"] == w])
            n_hphys = len(hphys[hphys["Well No"] == w])
            print(f"   {w}: {n_obs} observed records (post-cleaning), {n_hphys} h_phys records")

    training_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")

# ==============================================================================
# STEP 5b -- zone-stratified, graph-augmented residual LSTM (predicts h_residual)
# ==============================================================================

"""
Step 5b — Residual LSTM (Zone-stratified, graph-augmented)

h_true = h_phys + h_residual
This model predicts h_residual (residual_target from Step 5a).

Architecture:
  - Per well-month features: hphys_anomaly, T_eff, zone one-hot, cyclical
    month encoding, graph-conductance-weighted neighbor residual (lagged)
  - Zone-stratified LSTM: separate LSTM branch per zone (Weathered/Massive/
    Fractured), matching the M3.5 Zone-LSTM design that was the best
    performer in the original PGNN-LSTM work (R2~0.599)
  - Sequence length: 6 months lookback -> predict next-month residual
  - Split: TEMPORAL per well (first 80% time = train, last 20% = test) --
    NOT random split, to avoid leakage in a forecasting task
  - Evaluation: per-well and per-zone R2 ONLY (never pooled -- confirmed
    project learning that pooled R2 is an artifact, R2~0.99 is meaningless)

NOTE: designed to run on Colab (T4 GPU) for full training; can sanity-check
on CPU locally first with a small number of epochs.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

# ==================== PATHS ====================
TRAINING_DATA_PATH = r"J:\Ground_water\Step5_training_data.csv"
EDGES_PATH = r"J:\Ground_water\graph_edges_knn8_conductance.csv"
MODEL_OUT_PATH = r"J:\Ground_water\residual_lstm_model.pt"

# ==================== PARAMETERS ====================
LOOKBACK = 6          # months of history used to predict next residual
TEST_FRACTION = 0.2   # last 20% of each well's timeline = test
HIDDEN_DIM = 32
NUM_EPOCHS = 150
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
ZONES = ["Weathered", "Massive", "Fractured"]


# ==================== 1. Graph-conductance spatial feature ====================
def add_neighbor_residual_feature(df, edges):
    """
    For each well-month, compute the conductance-weighted average
    residual_target of its k-NN neighbors AT THE SAME TIMESTEP.
    This is the lightweight 'graph' signal (spatial coupling) without a
    full GCN layer -- neighboring wells' anomalies inform this well's
    residual, weighted by hydraulic conductance (higher C = more connected).
    """
    neighbor_map = {}
    for _, row in edges.iterrows():
        neighbor_map.setdefault(row["well_i"], []).append((row["well_j"], row["conductance"]))

    df = df.sort_values(["Well No", "date"]).reset_index(drop=True)
    residual_lookup = df.set_index(["Well No", "date"])["residual_target"].to_dict()

    neighbor_feature = []
    for _, row in df.iterrows():
        well, date = row["Well No"], row["date"]
        neighbors = neighbor_map.get(well, [])
        if not neighbors:
            neighbor_feature.append(0.0)
            continue
        weighted_sum, weight_total = 0.0, 0.0
        for nb_well, weight in neighbors:
            val = residual_lookup.get((nb_well, date))
            if val is not None:
                weighted_sum += weight * val
                weight_total += weight
        neighbor_feature.append(weighted_sum / weight_total if weight_total > 0 else 0.0)

    df["neighbor_residual"] = neighbor_feature
    return df


# ==================== 2. Sequence dataset ====================
class WellSequenceDataset(Dataset):
    def __init__(self, sequences, targets, zone_idx, well_idx):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.zone_idx = torch.tensor(zone_idx, dtype=torch.long)
        self.well_idx = torch.tensor(well_idx, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.zone_idx[idx], self.well_idx[idx], self.targets[idx]


def build_sequences(df, feature_cols, lookback=LOOKBACK):
    """Build (lookback x n_features) sequences per well, sliding window."""
    sequences, targets, zone_idx, well_num_idx, well_ids, dates = [], [], [], [], [], []
    zone_to_idx = {z: i for i, z in enumerate(ZONES)}
    well_to_idx = {w: i for i, w in enumerate(sorted(df["Well No"].unique()))}

    for well, grp in df.groupby("Well No"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) <= lookback:
            continue
        feats = grp[feature_cols].astype(float).fillna(0.0).values
        target = grp["residual_target"].astype(float).fillna(0.0).values
        zone = grp["zone_classification"].iloc[0]
        z_idx = zone_to_idx.get(zone, 1)  # default Massive if unknown
        w_idx = well_to_idx[well]

        for i in range(lookback, len(grp)):
            sequences.append(feats[i - lookback:i])
            targets.append(target[i])
            zone_idx.append(z_idx)
            well_num_idx.append(w_idx)
            well_ids.append(well)
            dates.append(grp["date"].iloc[i])

    return (np.array(sequences), np.array(targets), np.array(zone_idx), np.array(well_num_idx),
            np.array(well_ids), np.array(dates), len(well_to_idx))


# ==================== 3. Zone-stratified LSTM model ====================
class ZoneLSTM(nn.Module):
    def __init__(self, n_features, n_wells, hidden_dim=HIDDEN_DIM, n_zones=len(ZONES), well_emb_dim=8):
        super().__init__()
        self.well_embedding = nn.Embedding(n_wells, well_emb_dim)
        self.zone_lstms = nn.ModuleList([
            nn.LSTM(n_features, hidden_dim, batch_first=True) for _ in range(n_zones)
        ])
        self.zone_heads = nn.ModuleList([
            nn.Linear(hidden_dim + well_emb_dim, 1) for _ in range(n_zones)
        ])

    def forward(self, x, zone_idx, well_idx):
        batch_size = x.size(0)
        out = torch.zeros(batch_size, device=x.device)
        well_emb = self.well_embedding(well_idx)
        for z in range(len(self.zone_lstms)):
            mask = zone_idx == z
            if mask.sum() == 0:
                continue
            x_z = x[mask]
            lstm_out, _ = self.zone_lstms[z](x_z)
            last_hidden = lstm_out[:, -1, :]
            combined = torch.cat([last_hidden, well_emb[mask]], dim=-1)
            pred = self.zone_heads[z](combined).squeeze(-1)
            out[mask] = pred
        return out


# ==================== 4. Temporal train/test split (per well) ====================
def temporal_split(sequences, targets, zone_idx, well_num_idx, well_ids, dates, test_frac=TEST_FRACTION):
    train_idx, test_idx = [], []
    for well in np.unique(well_ids):
        idx = np.where(well_ids == well)[0]
        order = idx[np.argsort(dates[idx])]
        n_test = max(1, int(len(order) * test_frac))
        train_idx.extend(order[:-n_test])
        test_idx.extend(order[-n_test:])
    return np.array(train_idx), np.array(test_idx)


# ==================== 5. Evaluation (per-well and per-zone R2, NEVER pooled) ====================
def evaluate(model, sequences, targets, zone_idx, well_num_idx, well_ids, device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(sequences, dtype=torch.float32).to(device)
        z = torch.tensor(zone_idx, dtype=torch.long).to(device)
        w = torch.tensor(well_num_idx, dtype=torch.long).to(device)
        preds = model(x, z, w).cpu().numpy()

    results_df = pd.DataFrame({
        "Well No": well_ids, "zone_idx": zone_idx, "actual": targets, "pred": preds
    })

    print("\nPer-well R2 (only wells with >=5 test points shown):")
    per_well = []
    for well, grp in results_df.groupby("Well No"):
        if len(grp) >= 5:
            r2 = r2_score(grp["actual"], grp["pred"])
            per_well.append({"Well No": well, "n": len(grp), "R2": r2})
    per_well_df = pd.DataFrame(per_well).sort_values("R2", ascending=False)
    print(per_well_df.to_string(index=False))
    print(f"\nMean per-well R2: {per_well_df['R2'].mean():.4f}")
    print(f"Median per-well R2: {per_well_df['R2'].median():.4f}")

    print("\nPer-zone R2:")
    for z_idx, zone_name in enumerate(ZONES):
        grp = results_df[results_df["zone_idx"] == z_idx]
        if len(grp) >= 5:
            r2 = r2_score(grp["actual"], grp["pred"])
            print(f"  {zone_name}: R2={r2:.4f} (n={len(grp)})")

    return results_df, per_well_df


# ==================== MAIN ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading data...")
    df = pd.read_csv(TRAINING_DATA_PATH, parse_dates=["date"])
    edges = pd.read_csv(EDGES_PATH)

    print("Adding graph-conductance neighbor feature...")
    df = add_neighbor_residual_feature(df, edges)

    df["month_num"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    zone_dummies = pd.get_dummies(df["zone_classification"], prefix="zone").astype(int)
    df = pd.concat([df, zone_dummies], axis=1)

    feature_cols = ["residual_target", "hphys_anomaly", "T_eff_m2_per_day", "neighbor_residual",
                     "month_sin", "month_cos"] + list(zone_dummies.columns)
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Features used: {feature_cols}")
    print(">> NOTE: 'residual_target' is included as an INPUT feature here because the")
    print(">> sequence uses only the PAST 6 months (lookback window) to predict the NEXT")
    print(">> month -- this is the autoregressive lag, not a leak of the current target.")
    print(">> build_sequences() only uses feats[i-lookback:i] (strictly before i) as input")
    print(">> and target[i] as the label, so no leakage occurs.")

    print("\nBuilding sequences (lookback=6 months)...")
    sequences, targets, zone_idx, well_num_idx, well_ids, dates, n_wells = build_sequences(df, feature_cols)
    print(f"Total sequences: {len(sequences)}, feature dim: {sequences.shape[-1]}, n_wells: {n_wells}")

    print("\nTemporal train/test split (per well)...")
    train_idx, test_idx = temporal_split(sequences, targets, zone_idx, well_num_idx, well_ids, dates)
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    train_ds = WellSequenceDataset(sequences[train_idx], targets[train_idx], zone_idx[train_idx], well_num_idx[train_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = ZoneLSTM(n_features=sequences.shape[-1], n_wells=n_wells).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print("\nTraining...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for x, z, w, y in train_loader:
            x, z, w, y = x.to(device), z.to(device), w.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, z, w)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y)
        epoch_loss /= len(train_ds)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train MSE = {epoch_loss:.4f}")

    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    results_df, per_well_df = evaluate(
        model, sequences[test_idx], targets[test_idx], zone_idx[test_idx], well_num_idx[test_idx],
        well_ids[test_idx], device
    )

    torch.save(model.state_dict(), MODEL_OUT_PATH)
    print(f"\nModel saved to: {MODEL_OUT_PATH}")

# ==============================================================================
# STEP 5c -- reconstruct h_true = h_phys + h_residual and evaluate on the real depth-to-water scale (FINAL step; produces eval_step5c_per_well_depth_scale.csv used in Section 4.8)
# ==============================================================================

"""
Step 5c — Reconstruct h_true and evaluate on the real depth-to-water scale
============================================================================
FINAL STEP of the Ujjain+Indore residual pipeline (follows your Step 5a/5b
cells exactly — same paths, same feature engineering, same ZoneLSTM
architecture with well embeddings from cell 24).

WHY THIS STEP IS NEEDED
------------------------
Step 5b's evaluate() reports R² on `residual_target`, which is an ANOMALY
(obs_head_anomaly - hphys_anomaly), not the actual observed water level.
Reporting R² on residual_target only tells you how well the model predicts
the leftover fluctuation — it does NOT tell you how well h_true = h_phys +
h_residual reconstructs the real observed depth-to-water, which is what the
manuscript comparison against M1 (R²=0.683) and M4 (R²=0.564) needs.

RECONSTRUCTION CHAIN (inverting Step 5a's transforms exactly):
  residual_target        = obs_head_anomaly - hphys_anomaly
  => obs_head_anomaly     = residual_target + hphys_anomaly
  obs_head_anomaly        = -1 * (depth_to_water_m - mean_depth)     [Step 5a]
  => depth_to_water_m     = mean_depth - obs_head_anomaly

So: predicted_depth_to_water_m = mean_depth - (pred_residual + hphys_anomaly)

This script rebuilds the EXACT same sequences/split as Step 5b (cell 24),
loads the saved model, predicts residual_target on the test set only, then
runs the above chain to get a real depth-to-water prediction per well-month,
and evaluates per-well / per-zone / per-district R², RMSE (in metres),
never pooled (per your project rule).

NOTE ON COMPARABILITY TO M1/M4
--------------------------------
M1/M4 R² (0.683 / 0.564) were computed on ABSOLUTE hydraulic head (m MSL)
for the 48-well Indore-only set. This script evaluates on DEPTH-TO-WATER
(m bgl) for the combined 171-well Indore+Ujjain set. Same units of error
(metres) but not a strict like-for-like R² comparison — flag this
explicitly in the manuscript if you report both numbers together.

WHICH MODEL VERSION THIS ASSUMES
-----------------------------------
Your notebook has TWO versions of Step 5b: cell 23 (no well embedding) and
cell 24 (WITH well embedding + residual_target as an autoregressive input
feature). Both save to the same MODEL_OUT_PATH, so whichever you ran LAST
is what's actually on disk. This script assumes cell 24's architecture
(the one with well embeddings, matching your "per-well embeddings" note
from the last session). If your saved model.pt is actually from cell 23,
load_state_dict below will fail with a key-mismatch error — in that case
tell me and I'll give you the cell-23-compatible version instead.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error

# ==================== PATHS — identical to your Step 5a/5b cells ====================
TRAINING_DATA_PATH = r"J:\Ground_water\Step5_training_data.csv"
EDGES_PATH = r"J:\Ground_water\graph_edges_knn8_conductance.csv"
MODEL_PATH = r"J:\Ground_water\residual_lstm_model.pt"

OUT_PER_WELL_CSV = r"J:\Ground_water\eval_step5c_per_well_depth_scale.csv"
OUT_SUMMARY_CSV  = r"J:\Ground_water\eval_step5c_summary.csv"

# ==================== PARAMETERS — must match Step 5b (cell 24) exactly ====================
LOOKBACK = 6
TEST_FRACTION = 0.2
HIDDEN_DIM = 32
WELL_EMB_DIM = 8
ZONES = ["Weathered", "Massive", "Fractured"]

# Original Indore-only manuscript baselines, absolute head (m MSL), 48 wells
# — shown for context only, see comparability note above.
BASELINE_R2 = {
    "M1 Plain LSTM (Indore-only, absolute head, 48 wells)": 0.683,
    "M4 Full PGNN-LSTM (Indore-only, absolute head, 48 wells)": 0.564,
}


# ==================== 1. Graph-conductance neighbor feature — same as Step 5b ====================
def add_neighbor_residual_feature(df, edges):
    neighbor_map = {}
    for _, row in edges.iterrows():
        neighbor_map.setdefault(row["well_i"], []).append((row["well_j"], row["conductance"]))

    df = df.sort_values(["Well No", "date"]).reset_index(drop=True)
    residual_lookup = df.set_index(["Well No", "date"])["residual_target"].to_dict()

    neighbor_feature = []
    for _, row in df.iterrows():
        well, date = row["Well No"], row["date"]
        neighbors = neighbor_map.get(well, [])
        if not neighbors:
            neighbor_feature.append(0.0)
            continue
        weighted_sum, weight_total = 0.0, 0.0
        for nb_well, weight in neighbors:
            val = residual_lookup.get((nb_well, date))
            if val is not None:
                weighted_sum += weight * val
                weight_total += weight
        neighbor_feature.append(weighted_sum / weight_total if weight_total > 0 else 0.0)

    df["neighbor_residual"] = neighbor_feature
    return df


# ==================== 2. Sequence building — same as Step 5b (cell 24) ====================
def build_sequences(df, feature_cols, lookback=LOOKBACK):
    sequences, targets, zone_idx, well_num_idx, well_ids, dates = [], [], [], [], [], []
    zone_to_idx = {z: i for i, z in enumerate(ZONES)}
    well_to_idx = {w: i for i, w in enumerate(sorted(df["Well No"].unique()))}

    for well, grp in df.groupby("Well No"):
        grp = grp.sort_values("date").reset_index(drop=True)
        if len(grp) <= lookback:
            continue
        feats = grp[feature_cols].astype(float).fillna(0.0).values
        target = grp["residual_target"].astype(float).fillna(0.0).values
        zone = grp["zone_classification"].iloc[0]
        z_idx = zone_to_idx.get(zone, 1)
        w_idx = well_to_idx[well]

        for i in range(lookback, len(grp)):
            sequences.append(feats[i - lookback:i])
            targets.append(target[i])
            zone_idx.append(z_idx)
            well_num_idx.append(w_idx)
            well_ids.append(well)
            dates.append(grp["date"].iloc[i])

    return (np.array(sequences), np.array(targets), np.array(zone_idx), np.array(well_num_idx),
            np.array(well_ids), np.array(dates), len(well_to_idx))


def temporal_split(sequences, targets, zone_idx, well_num_idx, well_ids, dates, test_frac=TEST_FRACTION):
    train_idx, test_idx = [], []
    for well in np.unique(well_ids):
        idx = np.where(well_ids == well)[0]
        order = idx[np.argsort(dates[idx])]
        n_test = max(1, int(len(order) * test_frac))
        train_idx.extend(order[:-n_test])
        test_idx.extend(order[-n_test:])
    return np.array(train_idx), np.array(test_idx)


# ==================== 3. Model — identical to Step 5b (cell 24) ====================
class ZoneLSTM(nn.Module):
    def __init__(self, n_features, n_wells, hidden_dim=HIDDEN_DIM, n_zones=len(ZONES), well_emb_dim=WELL_EMB_DIM):
        super().__init__()
        self.well_embedding = nn.Embedding(n_wells, well_emb_dim)
        self.zone_lstms = nn.ModuleList([
            nn.LSTM(n_features, hidden_dim, batch_first=True) for _ in range(n_zones)
        ])
        self.zone_heads = nn.ModuleList([
            nn.Linear(hidden_dim + well_emb_dim, 1) for _ in range(n_zones)
        ])

    def forward(self, x, zone_idx, well_idx):
        batch_size = x.size(0)
        out = torch.zeros(batch_size, device=x.device)
        well_emb = self.well_embedding(well_idx)
        for z in range(len(self.zone_lstms)):
            mask = zone_idx == z
            if mask.sum() == 0:
                continue
            x_z = x[mask]
            lstm_out, _ = self.zone_lstms[z](x_z)
            last_hidden = lstm_out[:, -1, :]
            combined = torch.cat([last_hidden, well_emb[mask]], dim=-1)
            pred = self.zone_heads[z](combined).squeeze(-1)
            out[mask] = pred
        return out


# ==================== MAIN ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n[1/7] Loading Step5_training_data.csv + graph edges...")
    df = pd.read_csv(TRAINING_DATA_PATH, parse_dates=["date"])
    edges = pd.read_csv(EDGES_PATH)
    print(f"  {len(df)} rows, {df['Well No'].nunique()} wells")

    print("[2/7] Rebuilding features exactly as Step 5b (neighbor_residual, month sin/cos, zone dummies)...")
    df = add_neighbor_residual_feature(df, edges)
    df["month_num"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    zone_dummies = pd.get_dummies(df["zone_classification"], prefix="zone").astype(int)
    df = pd.concat([df, zone_dummies], axis=1)

    feature_cols = ["residual_target", "hphys_anomaly", "T_eff_m2_per_day", "neighbor_residual",
                     "month_sin", "month_cos"] + list(zone_dummies.columns)
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"  Features: {feature_cols}")

    print("[3/7] Rebuilding sequences + temporal split (must match training order)...")
    sequences, targets, zone_idx, well_num_idx, well_ids, dates, n_wells = build_sequences(df, feature_cols)
    train_idx, test_idx = temporal_split(sequences, targets, zone_idx, well_num_idx, well_ids, dates)
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}, n_wells: {n_wells}")

    print("[4/7] Loading trained ZoneLSTM weights...")
    model = ZoneLSTM(n_features=sequences.shape[-1], n_wells=n_wells).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("[5/7] Running inference on test set (predicting residual_target)...")
    with torch.no_grad():
        x = torch.tensor(sequences[test_idx], dtype=torch.float32).to(device)
        z = torch.tensor(zone_idx[test_idx], dtype=torch.long).to(device)
        w = torch.tensor(well_num_idx[test_idx], dtype=torch.long).to(device)
        pred_residual = model(x, z, w).cpu().numpy()

    test_wells = well_ids[test_idx]
    test_dates = dates[test_idx]

    print("[6/7] Reconstructing depth-to-water_m from predicted residuals...")
    lookup = df.set_index(["Well No", "date"])
    rows = []
    n_missing = 0
    for well, date, pred_r in zip(test_wells, test_dates, pred_residual):
        try:
            row = lookup.loc[(well, date)]
        except KeyError:
            n_missing += 1
            continue
        if isinstance(row, pd.DataFrame):   # duplicate (well, date) safety
            row = row.iloc[0]
        rows.append({
            "Well No": well,
            "date": date,
            "zone": row["zone_classification"],
            "district": row["district"] if "district" in row else np.nan,
            "hphys_anomaly": row["hphys_anomaly"],
            "mean_depth": row["mean_depth"],
            "pred_residual": pred_r,
            "actual_depth_to_water_m": row["depth_to_water_m"],
        })
    if n_missing:
        print(f"  WARNING: {n_missing} test predictions could not be matched back to source rows")

    pred_df = pd.DataFrame(rows)
    print(f"  Matched {len(pred_df)} test predictions")

    # ---- the actual reconstruction chain (inverts Step 5a) ----
    pred_df["pred_obs_head_anomaly"] = pred_df["pred_residual"] + pred_df["hphys_anomaly"]
    pred_df["pred_depth_to_water_m"] = pred_df["mean_depth"] - pred_df["pred_obs_head_anomaly"]

    print("[7/7] Computing per-well / per-zone / per-district R², RMSE, bias...")
    eval_rows = []
    for well, g in pred_df.groupby("Well No"):
        if len(g) < 5 or g["actual_depth_to_water_m"].nunique() < 2:
            continue
        r2 = r2_score(g["actual_depth_to_water_m"], g["pred_depth_to_water_m"])
        rmse = np.sqrt(mean_squared_error(g["actual_depth_to_water_m"], g["pred_depth_to_water_m"]))
        bias = (g["pred_depth_to_water_m"] - g["actual_depth_to_water_m"]).mean()
        eval_rows.append({
            "Well No": well, "zone": g["zone"].iloc[0], "district": g["district"].iloc[0],
            "n_test": len(g), "R2": r2, "RMSE_m": rmse, "bias_m": bias,
        })

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(OUT_PER_WELL_CSV, index=False)

    print("\n" + "=" * 78)
    print("PER-WELL RESULTS — reconstructed depth-to-water scale (h_true)")
    print("=" * 78)
    print(eval_df.sort_values("R2", ascending=False).to_string(index=False))

    summary_rows = [{
        "group": "ALL WELLS (mean of per-well)", "N": len(eval_df),
        "mean_R2": eval_df["R2"].mean(), "median_R2": eval_df["R2"].median(),
        "mean_RMSE_m": eval_df["RMSE_m"].mean(), "mean_bias_m": eval_df["bias_m"].mean(),
    }]
    for zone, g in eval_df.groupby("zone"):
        summary_rows.append({
            "group": f"Zone: {zone}", "N": len(g),
            "mean_R2": g["R2"].mean(), "median_R2": g["R2"].median(),
            "mean_RMSE_m": g["RMSE_m"].mean(), "mean_bias_m": g["bias_m"].mean(),
        })
    for dist, g in eval_df.groupby("district"):
        summary_rows.append({
            "group": f"District: {dist}", "N": len(g),
            "mean_R2": g["R2"].mean(), "median_R2": g["R2"].median(),
            "mean_RMSE_m": g["RMSE_m"].mean(), "mean_bias_m": g["bias_m"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

    print("\n" + "=" * 78)
    print("SUMMARY — h_true reconstruction (never pooled, per-well means only)")
    print("=" * 78)
    print(summary_df.to_string(index=False))

    print("\nContext only (different scale/well-set — NOT strict like-for-like):")
    for name, r2 in BASELINE_R2.items():
        print(f"  {name:58s} R² = {r2:.3f}")
    overall_r2 = summary_df.loc[summary_df["group"] == "ALL WELLS (mean of per-well)", "mean_R2"].values[0]
    print(f"  {'Residual model (this run, depth scale, 171 wells)':58s} R² = {overall_r2:.3f}")

    print(f"\nSaved: {OUT_PER_WELL_CSV}")
    print(f"Saved: {OUT_SUMMARY_CSV}")

# ==============================================================================
# DIAGNOSTIC -- data-sparsity check explaining why Ujjain R^2 is lower than Indore (referenced in Discussion 5.1/5.5)
# ==============================================================================

"""
Diagnostic — Data sparsity check: Indore vs Ujjain
=====================================================
Goal: figure out WHY Ujjain wells show much lower R² (mean 0.187) than
Indore wells (mean 0.420), and why Fractured zone (0.048) is far worse
than Weathered/Massive (~0.48-0.51).

This does NOT retrain anything — it just inspects Step5_training_data.csv
(the merged observed + h_phys + features file) to answer:

  1. How many total months of data does each well have (before train/test
     split)? Are Ujjain wells systematically shorter records than Indore?
  2. What is the actual DATE RANGE per well/district? Is there a period
     mismatch (e.g. Ujjain data starts later, so less history + overlap
     with h_phys)?
  3. How many wells have test-set n < 15 (statistically unstable R²) vs
     n >= 15, split by district?
  4. Does residual_target itself have much higher variance/spread in
     Ujjain-Fractured wells (would mean the FD physics baseline h_phys is
     a worse fit there BEFORE the LSTM even gets involved — i.e. it's a
     Step 4 physics-baseline issue, not a Step 5 LSTM issue)?
  5. Simple sanity: what fraction of each well's months are covered by
     h_phys at all (join completeness from Step 5a's merge)?

Run this and read the printed tables — it will point to one of two very
different fixes:
  (a) if it's a DATA-LENGTH problem -> Ujjain wells just don't have enough
      history for a 6-month-lookback LSTM to learn well; more months of
      data or a shorter lookback would help.
  (b) if it's a PHYSICS-BASELINE problem -> h_phys itself is a poor fit for
      Ujjain Fractured wells (likely, since T_eff/S values used for h_phys
      were literature values partly calibrated on Indore geology) -> the
      LSTM residual model is being asked to correct a badly-wrong physics
      baseline, which is much harder than correcting a roughly-right one.
"""

import pandas as pd
import numpy as np

TRAINING_DATA_PATH = r"J:\Ground_water\Step5_training_data.csv"
HPHYS_PATH = r"J:\Ground_water\FD_baseline_h_phys.csv"

pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 60)

print("Loading Step5_training_data.csv ...")
df = pd.read_csv(TRAINING_DATA_PATH, parse_dates=["date"])
print(f"  {len(df)} rows, {df['Well No'].nunique()} wells")

# ══════════════════════════════════════════════════════════════
# 1. Per-well record length (months) and date range
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("1. RECORD LENGTH PER WELL (months of matched observed+h_phys data)")
print("=" * 78)

well_len = df.groupby(["Well No", "zone_classification", "district"]).agg(
    n_months=("date", "count"),
    first_date=("date", "min"),
    last_date=("date", "max"),
).reset_index()
well_len["span_months"] = ((well_len["last_date"] - well_len["first_date"]).dt.days / 30.44).round(1)

print("\nBy district:")
print(well_len.groupby("district")["n_months"].describe().to_string())

print("\nBy district x zone:")
print(well_len.groupby(["district", "zone_classification"])["n_months"].describe().to_string())

# ══════════════════════════════════════════════════════════════
# 2. Date range comparison — does Ujjain start later / end earlier?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("2. DATE RANGE COVERAGE — Indore vs Ujjain")
print("=" * 78)
date_range = df.groupby("district").agg(
    earliest=("date", "min"),
    latest=("date", "max"),
    n_wells=("Well No", "nunique"),
    n_rows=("date", "count"),
).reset_index()
print(date_range.to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 3. Test-set stability — how many wells have too few test points
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("3. TEST-SET SIZE DISTRIBUTION (last 20% of each well's timeline)")
print("=" * 78)
# reproduce the same 80/20 temporal split logic used in Step 5b, per well
TEST_FRACTION = 0.2
split_info = []
for well, grp in df.groupby("Well No"):
    grp = grp.sort_values("date")
    n = len(grp)
    n_test = max(1, int(n * TEST_FRACTION))
    zone = grp["zone_classification"].iloc[0]
    dist = grp["district"].iloc[0]
    split_info.append({"Well No": well, "zone": zone, "district": dist,
                        "n_total": n, "n_test": n_test})
split_df = pd.DataFrame(split_info)

print("\nTest-set size (n_test) by district:")
print(split_df.groupby("district")["n_test"].describe().to_string())

print("\nWells with n_test < 15 (statistically unstable R² — treat with caution):")
unstable = split_df[split_df["n_test"] < 15]
print(f"  Total: {len(unstable)} / {len(split_df)} wells")
print(unstable.groupby("district").size().to_string())
print(unstable.groupby(["district", "zone"]).size().to_string())

# ══════════════════════════════════════════════════════════════
# 4. residual_target variance — is h_phys itself a bad fit?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("4. RESIDUAL_TARGET SPREAD — is the FD physics baseline (h_phys) a")
print("   worse starting point for Ujjain / Fractured wells BEFORE the LSTM?")
print("=" * 78)
resid_stats = df.groupby(["district", "zone_classification"])["residual_target"].agg(
    ["mean", "std", "min", "max", "count"]
).reset_index()
print(resid_stats.to_string(index=False))

print("\nInterpretation: if std(residual_target) is much larger for Ujjain")
print("Fractured than Indore Massive/Weathered, it means the physics baseline")
print("h_phys is leaving a bigger, harder-to-learn gap for the LSTM in that")
print("group — independent of how good the LSTM itself is.")

# ══════════════════════════════════════════════════════════════
# 5. h_phys join completeness (from Step 5a's inner merge)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("5. H_PHYS COVERAGE — how much of each well's raw observed record")
print("   actually survived the Step 5a inner merge with h_phys?")
print("=" * 78)
try:
    hphys = pd.read_csv(HPHYS_PATH, parse_dates=["date"])
    hphys_counts = hphys.groupby("Well No").size().rename("n_hphys_available")
    matched_counts = df.groupby("Well No").size().rename("n_matched_in_training")
    coverage = pd.concat([hphys_counts, matched_counts], axis=1).reset_index()
    coverage = coverage.rename(columns={"index": "Well No"})
    coverage["match_rate_pct"] = (100 * coverage["n_matched_in_training"] /
                                   coverage["n_hphys_available"]).round(1)
    # attach district for grouping
    dist_map = df.drop_duplicates("Well No").set_index("Well No")["district"]
    coverage["district"] = coverage["Well No"].map(dist_map)
    print("\nMatch rate (%) by district:")
    print(coverage.groupby("district")["match_rate_pct"].describe().to_string())

    print("\nWells with worst match rate (<50%) — h_phys and observed data barely overlap:")
    worst = coverage[coverage["match_rate_pct"] < 50].sort_values("match_rate_pct")
    print(worst.to_string(index=False))
except FileNotFoundError:
    print(f"  (Skipped — {HPHYS_PATH} not found in this environment; "
          f"run this section locally where the file exists)")

print("\n" + "=" * 78)
print("DONE — read sections 1-5 above to diagnose the Ujjain/Fractured gap.")
print("=" * 78)

# ==============================================================================
# FIGURE 11 -- cross-district transferability plot (Indore vs Ujjain, per-zone R^2)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CSV_PATH = r"J:\Ground_water\eval_step5c_per_well_depth_scale.csv"
df = pd.read_csv(CSV_PATH)

ZONE_COLORS = {"Weathered": "#3B6D11", "Massive": "#185FA5", "Fractured": "#BA7517"}
ZONES = ["Weathered", "Massive", "Fractured"]
DISTRICTS = ["Indore", "Ujjain"]

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)

# ============ PANEL A: raincloud-style strip + box of per-well R2, by zone, split by district ============
ax = axes[0]
np.random.seed(42)
positions = {}
pos = 0
xticks, xticklabels = [], []
box_data = []
box_positions = []
box_colors = []

for zone in ZONES:
    for dist in DISTRICTS:
        sub = df[(df.zone == zone) & (df.district == dist)]["R2"].values
        if len(sub) == 0:
            pos += 1
            continue
        jitter = np.random.normal(0, 0.06, size=len(sub))
        marker = "o" if dist == "Indore" else "^"
        alpha = 0.85 if dist == "Indore" else 0.55
        ax.scatter(np.full(len(sub), pos) + jitter, sub, s=28, color=ZONE_COLORS[zone],
                   marker=marker, alpha=alpha, edgecolors="white", linewidths=0.4, zorder=3)
        bp = ax.boxplot([sub], positions=[pos], widths=0.32, patch_artist=True,
                         showfliers=False, zorder=2,
                         boxprops=dict(facecolor=ZONE_COLORS[zone], alpha=0.18, edgecolor=ZONE_COLORS[zone], linewidth=1.4),
                         medianprops=dict(color=ZONE_COLORS[zone], linewidth=2.2),
                         whiskerprops=dict(color=ZONE_COLORS[zone], linewidth=1.2),
                         capprops=dict(color=ZONE_COLORS[zone], linewidth=1.2))
        xticks.append(pos)
        xticklabels.append(f"{zone[:4]}.\n{dist}\n(n={len(sub)})")
        pos += 1
    pos += 0.6  # gap between zone groups

ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=1)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=8.5)
ax.set_ylabel("Per-well R\u00b2 (depth-to-water scale)", fontsize=11)
ax.set_title("(a) Per-well R\u00b2 distribution by zone \u00d7 district", fontsize=11.5)
ax.set_ylim(-1.2, 1.05)
legend_elems = [
    mpatches.Patch(facecolor="grey", alpha=0.85, label="Indore (\u25cf)"),
    mpatches.Patch(facecolor="grey", alpha=0.55, label="Ujjain (\u25b2)"),
]
ax.legend(handles=legend_elems, loc="lower left", fontsize=8.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

# ============ PANEL B: bar chart of mean R2 by district x zone, stable wells only (n_test>=15) ============
ax2 = axes[1]
stable = df[df["n_test"] >= 15]

bar_width = 0.35
x = np.arange(len(ZONES))
for i, dist in enumerate(DISTRICTS):
    means, ns = [], []
    for zone in ZONES:
        sub = stable[(stable.zone == zone) & (stable.district == dist)]["R2"]
        means.append(sub.mean() if len(sub) > 0 else np.nan)
        ns.append(len(sub))
    offset = (i - 0.5) * bar_width
    bars = ax2.bar(x + offset, means, bar_width,
                    color=[ZONE_COLORS[z] for z in ZONES],
                    alpha=0.9 if dist == "Indore" else 0.55,
                    edgecolor="black", linewidth=1.0,
                    hatch="" if dist == "Indore" else "///",
                    label=dist)
    for xi, m, n in zip(x + offset, means, ns):
        if not np.isnan(m):
            ax2.text(xi, m + 0.015, f"{m:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(ZONES, fontsize=10.5)
ax2.set_ylabel("Mean R\u00b2 (stable wells, n\u2265\u200915 test months)", fontsize=11)
ax2.set_title("(b) Cross-district comparison \u2014 statistically stable wells only", fontsize=11.5)
ax2.set_ylim(0, 0.65)
ax2.legend(fontsize=9.5, frameon=False, loc="upper right")
ax2.spines[["top", "right"]].set_visible(False)
ax2.axhline(0, color="black", linewidth=0.8)

plt.suptitle("Cross-district transferability: Indore vs Ujjain (Deccan Trap basalt)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("Ujjain_transferability_figure.png", dpi=300, bbox_inches="tight", facecolor="white")
print("Saved: Ujjain_transferability_figure.png")