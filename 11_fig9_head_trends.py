"""
11_fig7_idw_final.py — Figure 7: seasonal head-trend maps (Malda-style)
Interpolated Theil-Sen slope surface clipped to Indore district, block lines,
Ganga/Narmada basin divide, wells on top, legend box on the right.

Run after 06 (needs diag_perwell.csv). Only edit the PATHS block if a path differs.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import cKDTree

# ============================ PATHS ===========================================
DIAG_CSV  = r"J:\Indore_gw\baselines\diag_perwell.csv"
BOOK2     = r"J:\Ground_water\Indore_gw\Book2.xlsx"
INDORE_SHP = r"J:\Indore_gw\Indore\Export_Output.shp"          # blocks of Indore
BASIN_SHP  = r"J:\Indore_gw\Narmada_ganga\Narmada_ganga\NAramada_ganga.shp"
OUT_DIR   = r"J:\Indore_gw\baselines\manuscript"

# ============================ OPTIONS =========================================
MASK_KM   = None    # e.g. 8 -> blank areas >8 km from any well; None = fill district
IDW_POWER = 2
IDW_K     = 8       # nearest wells used per grid cell
SHOW_WELLS = True

BINS   = [-np.inf, -0.3, -0.1, 0.1, 0.3, np.inf]                # m/yr, head
LABELS = ["< −0.3", "−0.3 to −0.1", "−0.1 to +0.1", "+0.1 to +0.3", "> +0.3"]
COLORS = ["#C0392B", "#F0A35E", "#F7F1C9", "#7FC97F", "#2C7FB8"]  # falling→rising
ZONES  = ["Weathered", "Massive", "Fractured"]
MARKER = {"Weathered": "o", "Massive": "s", "Fractured": "^"}
SERIES = [("Pre", "Pre-monsoon (Apr–Jun)"), ("Post", "Post-monsoon (Oct–Dec)"),
          ("Ann", "Annual mean")]
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
cmap = ListedColormap(COLORS)
norm = BoundaryNorm(BINS[1:-1], cmap.N, extend="both")

# ============================ DATA ============================================
dg = pd.read_csv(DIAG_CSV)
meta = pd.read_excel(BOOK2, sheet_name="Sheet1")[["Well No", "Easting", "Northing"]]
dg = dg.merge(meta.rename(columns={"Well No": "Well"}), on="Well", how="left")
dg = dg.dropna(subset=["Easting", "Northing"])


def to_ll(g):
    if g.crs is None:
        print("  WARNING: shapefile has no CRS; assuming WGS84 lat/lon")
        return g.set_crs(4326)
    return g.to_crs(4326)


blocks = to_ll(gpd.read_file(INDORE_SHP))
district = blocks.dissolve()
dpoly = district.geometry.iloc[0]
print("Indore shapefile columns:", list(blocks.columns))
name_col = next((c for c in blocks.columns if c.lower() in
                 ("block", "block_name", "blk_name", "name", "sub_dist", "tehsil", "subdist")),
                None)

basins = None
if BASIN_SHP and os.path.exists(BASIN_SHP):
    basins = to_ll(gpd.read_file(BASIN_SHP))
    print("Basin shapefile columns:", list(basins.columns), "| features:", len(basins))
    # keep only the divide line inside the district
    divide = basins.boundary.union_all() if hasattr(basins.boundary, "union_all") \
        else basins.boundary.unary_union
    divide_in = divide.intersection(dpoly)
    bname = next((c for c in basins.columns if basins[c].dtype == object
                  and c != "geometry"), None)

x0, y0, x1, y1 = district.total_bounds
pad = 0.02
XL, YL = (x0 - pad, x1 + pad), (y0 - pad, y1 + pad)
KMX, KMY = 111.32 * np.cos(np.radians(np.mean(YL))), 111.32

gx, gy = np.meshgrid(np.linspace(*XL, 400), np.linspace(*YL, 400))
inside = shapely.contains_xy(dpoly, gx.ravel(), gy.ravel()).reshape(gx.shape)
if MASK_KM:
    dist, _ = cKDTree(np.c_[dg.Easting * KMX, dg.Northing * KMY]).query(
        np.c_[gx.ravel() * KMX, gy.ravel() * KMY])
    inside &= (dist <= MASK_KM).reshape(gx.shape)


def idw(x, y, v):
    tree = cKDTree(np.c_[x * KMX, y * KMY])
    d, i = tree.query(np.c_[gx.ravel() * KMX, gy.ravel() * KMY], k=min(IDW_K, len(v)))
    w = 1 / np.maximum(d, 1e-6) ** IDW_POWER
    return (np.sum(w * v[i], axis=1) / w.sum(axis=1)).reshape(gx.shape)


# ============================ FIGURE ==========================================
fig = plt.figure(figsize=(17, 6.8))
gsp = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.62], wspace=0.05)
for k, (tag, title) in enumerate(SERIES):
    ax = fig.add_subplot(gsp[0, k])
    d = dg.dropna(subset=[f"{tag}_slope_m_yr"])
    z = np.ma.masked_where(~inside, idw(d.Easting.values, d.Northing.values,
                                        d[f"{tag}_slope_m_yr"].values))
    ax.pcolormesh(gx, gy, z, cmap=cmap, norm=norm, shading="auto", zorder=1)
    blocks.boundary.plot(ax=ax, color="#555", lw=0.7, zorder=3)
    district.boundary.plot(ax=ax, color="#111", lw=1.5, zorder=4)
    if basins is not None and not divide_in.is_empty:
        gpd.GeoSeries([divide_in], crs=4326).plot(ax=ax, color="#6A3D9A", lw=2.0,
                                                  ls="--", zorder=4)
    if name_col:
        for _, r in blocks.iterrows():
            p = r.geometry.representative_point()
            ax.text(p.x, p.y, str(r[name_col]), fontsize=8.5, fontweight="bold",
                    color="#333", ha="center", va="center", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.55, pad=0.8))
    if SHOW_WELLS:
        for zone in ZONES:
            zz = d[d.Zone == zone]
            sig = zz[f"{tag}_class"].isin(["rising", "declining"]).values
            ax.scatter(zz.Easting, zz.Northing, marker=MARKER[zone], s=26,
                       facecolor=np.where(sig, "#111", "white"), edgecolor="#111",
                       lw=0.8, zorder=7)
    c = d[f"{tag}_class"].value_counts()
    ax.text(0.03, 0.03, f"MK p<0.05: {c.get('rising', 0)} rising, "
            f"{c.get('declining', 0)} falling (n={len(d)})", transform=ax.transAxes,
            fontsize=8.5, zorder=8, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999"))
    ax.set_title(f"({'abc'[k]}) {title}", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal")
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}°E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}°N"))
    ax.tick_params(labelsize=8.5)
    ax.set_xlabel(""); ax.set_ylabel("")
    if k > 0:
        ax.set_yticklabels([])
    ax.annotate("N", xy=(0.92, 0.97), xytext=(0.92, 0.86), xycoords="axes fraction",
                ha="center", fontsize=10, fontweight="bold", zorder=8,
                arrowprops=dict(arrowstyle="-|>", color="#111", lw=1.3))
    km = 10 / KMX
    sx, sy = XL[1] - km - 0.02, YL[0] + 0.04
    ax.plot([sx, sx + km], [sy, sy], color="#111", lw=3, solid_capstyle="butt", zorder=8)
    ax.text(sx + km / 2, sy + 0.012, "10 km", ha="center", fontsize=8, zorder=8)

# legend box
lg = fig.add_subplot(gsp[0, 3]); lg.axis("off")
lg.add_patch(plt.Rectangle((0, 0), 1, 1, transform=lg.transAxes, fill=False, ec="#333"))
lg.text(0.5, 0.96, "Groundwater head trend\n(Theil–Sen, 1998–2025)", ha="center",
        va="top", fontsize=11.5, fontweight="bold", transform=lg.transAxes)
l1 = lg.legend(handles=[Patch(fc=c, ec="#666", label=l) for c, l in
                        zip(COLORS[::-1], LABELS[::-1])],
               title="Trend (m yr⁻¹)", loc="upper center", bbox_to_anchor=(0.5, 0.82),
               frameon=False, fontsize=9.5, title_fontsize=10)
lg.add_artist(l1)
h2 = [Line2D([], [], color="#111", lw=1.5, label="District boundary"),
      Line2D([], [], color="#555", lw=0.7, label="Block boundary")]
if basins is not None:
    h2.append(Line2D([], [], color="#6A3D9A", lw=2, ls="--", label="Ganga–Narmada divide"))
if SHOW_WELLS:
    h2 += [Line2D([], [], marker=MARKER[z], ls="", mfc="white", mec="#111", ms=7, label=z)
           for z in ZONES]
    h2.append(Line2D([], [], marker="o", ls="", mfc="#111", mec="#111", ms=7,
                     label="filled = MK p < 0.05"))
lg.legend(handles=h2, loc="upper center", bbox_to_anchor=(0.5, 0.45), frameon=False,
          fontsize=9)
lg.text(0.5, 0.03, "Positive = head rising\nInverse-distance interpolation of\nper-well slopes",
        ha="center", fontsize=8, style="italic", transform=lg.transAxes)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT_DIR, f"Fig07_HeadTrend_IDW.{ext}"), dpi=300,
                bbox_inches="tight")
plt.show()
print("Saved:", os.path.join(OUT_DIR, "Fig07_HeadTrend_IDW.png"))
