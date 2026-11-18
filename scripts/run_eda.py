"""PathSnap dataset exploration (brief: "Dataset Exploration.pdf").

Run from the repo root:
    .venv/Scripts/python scripts/run_eda.py          (Windows)
    python scripts/run_eda.py                         (any OS, inside the venv)

Reads every trip in every LOCAL partition (train, student_test). The evaluator
test partition is not on disk and is never touched. Nothing here is fitted /
learned, so exploring student_test does not leak into model training.

Outputs (all regenerated on each run, deterministic):
    outputs/eda/tables/*.csv    inventory, per-trip stats, partition summaries
    outputs/eda/figures/*.png   distribution plots, trajectory small-multiples
    outputs/eda/maps/*.html     interactive Folium maps of selected trips
    outputs/eda/EDA_REPORT.md   auto-generated summary with the key numbers
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import folium  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from pathsnap import eda_metrics as M  # noqa: E402
from pathsnap.data_io import data_root, list_partitions, list_trips, load_trip  # noqa: E402
from pathsnap.geo import Projector  # noqa: E402

OUT = REPO / "outputs" / "eda"
TAB, FIG, MAP = OUT / "tables", OUT / "figures", OUT / "maps"
for d in (TAB, FIG, MAP):
    d.mkdir(parents=True, exist_ok=True)

PART_COLORS = {"train": "#2a6fdb", "student_test": "#e8833a"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


# ----------------------------------------------------------------------------
# 1. per-trip statistics
# ----------------------------------------------------------------------------
def collect():
    root = data_root()
    rows, pooled = [], {}
    for part in list_partitions(root):
        for tid in list_trips(part, root):
            t0 = time.time()
            s, arr = M.trip_stats(load_trip(part, tid, root))
            rows.append(s)
            for k, v in arr.items():
                pooled.setdefault(part, {}).setdefault(k, []).append(v)
            print(f"  {part:13s} {tid}  pts={s['n_points']:6d}  nodes={s['n_nodes']:7d}  "
                  f"({time.time() - t0:4.1f}s)  {s['flags']}")
    df = pd.DataFrame(rows)
    pooled = {p: {k: np.concatenate(v) for k, v in d.items()} for p, d in pooled.items()}
    return df, pooled


# ----------------------------------------------------------------------------
# 2. tables
# ----------------------------------------------------------------------------
def describe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby("partition")[cols].describe(percentiles=[0.5]).T
    allp = df[cols].describe(percentiles=[0.5]).T.stack().rename("all")
    return g.join(allp).round(3)


def write_tables(df: pd.DataFrame):
    df.to_csv(TAB / "trip_summary.csv", index=False)

    inv_cols = ["partition", "trip_id"] + [f"{k}_{x}" for k in ("track", "nodes", "arcs", "route")
                                           for x in ("exists", "bytes", "rows", "colcounts", "bad_rows")]
    df[inv_cols].to_csv(TAB / "inventory.csv", index=False)

    part = df.groupby("partition").agg(
        trips=("trip_id", "count"),
        files_present=("track_exists", lambda s: int(s.sum()) * 4),
        gps_points_total=("n_points", "sum"),
        gps_points_median=("n_points", "median"),
        nodes_total=("n_nodes", "sum"),
        arcs_total=("n_arcs", "sum"),
        route_arcs_total=("route_len_arcs", "sum"),
        disk_mb=("track_bytes", lambda s: 0),
    )
    part["disk_mb"] = df.groupby("partition")[[c for c in df if c.endswith("_bytes")]].sum().sum(axis=1) / 1e6
    part.round(2).to_csv(TAB / "partition_summary.csv")

    describe(df, ["n_points", "duration_s", "path_len_km", "extent_x_km", "extent_y_km", "frac_dt_eq_1",
                  "n_gaps", "n_large_gaps", "max_gap_s", "missing_seconds", "speed_median", "speed_max",
                  "n_speed_jumps", "frac_stationary", "frac_parked"]).to_csv(TAB / "trajectory_stats.csv")
    describe(df, ["n_nodes", "n_arcs", "net_total_km", "arc_len_median_m", "arc_len_max_m",
                  "frac_arcs_with_reverse", "n_self_loops", "n_duplicate_arcs", "n_isolated_nodes",
                  "n_weak_components", "largest_wcc_frac", "n_strong_components", "largest_scc_frac",
                  "n_dead_end_nodes"]).to_csv(TAB / "network_stats.csv")
    describe(df, ["route_len_arcs", "route_len_km", "route_to_gps_len_ratio", "route_n_breaks",
                  "route_n_uturns", "route_n_revisited_arcs", "route_frac_in_largest_scc", "d_net_median_m",
                  "d_net_p95_m", "d_route_median_m", "d_route_p95_m", "d_route_max_m", "frac_far_from_route",
                  "start_gap_m", "end_gap_m"]).to_csv(TAB / "route_stats.csv")

    an = df[df.n_flags > 0][["partition", "trip_id", "n_points", "max_gap_s", "n_speed_jumps",
                             "frac_far_from_route", "route_n_breaks", "flags"]]
    an.sort_values(["partition", "trip_id"]).to_csv(TAB / "anomalies.csv", index=False)

    df[["partition", "trip_id", "utm_epsg", "lon_min", "lon_max", "lat_min", "lat_max", "lon_c", "lat_c",
        "extent_x_km", "extent_y_km", "frac_track_in_net_bbox"]].to_csv(TAB / "geographic_extent.csv", index=False)


# ----------------------------------------------------------------------------
# 3. figures
# ----------------------------------------------------------------------------
def hist_by_part(ax, df, col, bins, log=False, xlabel=None):
    for p, g in df.groupby("partition"):
        ax.hist(g[col], bins=bins, alpha=0.65, label=f"{p} (n={len(g)})", color=PART_COLORS.get(p))
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel or col)
    ax.set_ylabel("trips")
    ax.legend(fontsize=8)


def fig_trip_overview(df):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    lb = lambda a, b: np.logspace(np.log10(max(a, 1)), np.log10(b), 25)  # noqa: E731
    hist_by_part(axs[0, 0], df, "n_points", lb(df.n_points.min(), df.n_points.max()), True, "GPS points per trip")
    hist_by_part(axs[0, 1], df, "path_len_km", lb(df.path_len_km.min(), df.path_len_km.max()), True,
                 "GPS path length (km)")
    hist_by_part(axs[0, 2], df, "duration_s", lb(df.duration_s.min(), df.duration_s.max()), True,
                 "trip duration t_end - t_start (s)")
    hist_by_part(axs[1, 0], df, "n_nodes", lb(df.n_nodes.min(), df.n_nodes.max()), True, "road-network nodes")
    hist_by_part(axs[1, 1], df, "route_len_arcs", 25, False, "reference route length (arcs)")
    hist_by_part(axs[1, 2], df, "speed_median", 25, False, "median speed (m/s)")
    fig.suptitle("Per-trip size distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "01_trip_size_distributions.png")
    plt.close(fig)


def fig_sampling(df, pooled):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    dt = np.concatenate([pooled[p]["dt"] for p in pooled])
    vals, cnt = np.unique(np.clip(dt, None, 11), return_counts=True)
    lab = [f"{v:g}" if v <= 10 else ">10" for v in vals]
    axs[0].bar(range(len(vals)), cnt, color="#2a6fdb")
    axs[0].set_xticks(range(len(vals)), lab)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("sampling interval dt (s)")
    axs[0].set_ylabel("steps (log)")
    axs[0].set_title(f"dt == 1 s for {np.mean(dt == 1):.1%} of {len(dt):,} steps")
    big = np.sort(dt[dt > 10])
    axs[1].hist(big, bins=np.logspace(1, np.log10(big.max() + 1), 30), color="#c0392b")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("gap length (s), gaps > 10 s only")
    axs[1].set_ylabel("count")
    axs[1].set_title(f"{len(big)} gaps > 10 s, longest {big.max():,.0f} s")
    d = df.sort_values("max_gap_s", ascending=False)
    axs[2].bar(range(len(d)), d.max_gap_s, color=[PART_COLORS[p] for p in d.partition])
    axs[2].set_yscale("log")
    axs[2].axhline(M.LARGE_GAP_S, ls="--", c="k", lw=1, label=f"large-gap threshold {M.LARGE_GAP_S:g}s")
    axs[2].set_xlabel("trips (sorted)")
    axs[2].set_ylabel("max gap (s, log)")
    axs[2].legend(fontsize=8)
    axs[2].set_title("Longest gap per trip")
    fig.tight_layout()
    fig.savefig(FIG / "02_sampling_intervals_and_gaps.png")
    plt.close(fig)


def fig_geography(df):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for p, g in df.groupby("partition"):
        axs[0].scatter(g.lon_c, g.lat_c, s=np.clip(g.n_points / 50, 8, 200), alpha=0.7, label=p,
                       color=PART_COLORS.get(p), edgecolor="k", lw=0.3)
    axs[0].set_xlabel("longitude (deg)")
    axs[0].set_ylabel("latitude (deg)")
    axs[0].set_title("Trip centroids (marker size ~ #GPS points)")
    axs[0].legend()
    axs[0].set_aspect("equal")  # plate carree: data spans several continents
    for p, g in df.groupby("partition"):
        axs[1].scatter(g.extent_x_km, g.extent_y_km, alpha=0.7, label=p, color=PART_COLORS.get(p))
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("E-W extent of trajectory (km, UTM)")
    axs[1].set_ylabel("N-S extent (km, UTM)")
    axs[1].set_title("Geographic extent of each trajectory")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "03_geography.png")
    plt.close(fig)


def fig_speed_distance(pooled):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for p, a in pooled.items():
        sp = a["speed"][np.isfinite(a["speed"])]
        axs[0].hist(np.clip(sp, 0, 100), bins=100, alpha=0.6, density=True, label=p, color=PART_COLORS.get(p))
        dn = a["d_net"][np.isfinite(a["d_net"])]
        axs[1].hist(np.clip(dn, 0, 60), bins=120, alpha=0.6, density=True, label=p, color=PART_COLORS.get(p))
        dr = a["d_route"][np.isfinite(a["d_route"])]
        axs[2].hist(np.clip(dr, 0, 100), bins=100, alpha=0.6, density=True, label=p, color=PART_COLORS.get(p))
    axs[0].axvline(M.MAX_PLAUSIBLE_SPEED, c="r", ls="--", lw=1, label="jump threshold")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("step speed (m/s, clipped at 100)")
    axs[0].set_title("Point-to-point speed")
    axs[1].set_xlabel("distance GPS point -> nearest arc (m, clipped 60)")
    axs[1].set_title("GPS noise vs. whole network")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("distance GPS point -> nearest REFERENCE-ROUTE arc (m, clipped 100)")
    axs[2].set_title("GPS vs. reference route")
    for a in axs:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "04_speed_and_gps_to_road_distance.png")
    plt.close(fig)


def fig_network(df):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for p, g in df.groupby("partition"):
        axs[0].scatter(g.n_nodes, g.n_arcs, color=PART_COLORS.get(p), label=p, alpha=0.7)
        axs[1].scatter(g.n_nodes, g.largest_scc_frac, color=PART_COLORS.get(p), label=p, alpha=0.7)
        axs[2].scatter(g.n_nodes, g.frac_arcs_with_reverse, color=PART_COLORS.get(p), label=p, alpha=0.7)
    x = np.array([df.n_nodes.min(), df.n_nodes.max()])
    axs[0].plot(x, 2 * x, "k--", lw=1, label="arcs = 2 x nodes")
    for a, yl, t in [(axs[0], "arcs", "Network size"),
                     (axs[1], "share of nodes in largest SCC", "Strong connectivity"),
                     (axs[2], "share of arcs whose reverse also exists", "Two-way vs one-way arcs")]:
        a.set_xscale("log")
        a.set_xlabel("nodes")
        a.set_ylabel(yl)
        a.set_title(t)
        a.legend(fontsize=8)
    axs[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIG / "05_network_size_connectivity.png")
    plt.close(fig)


def fig_route_quality(df):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for p, g in df.groupby("partition"):
        c = PART_COLORS.get(p)
        axs[0].scatter(g.path_len_km, g.route_len_km, color=c, label=p, alpha=0.7)
        axs[1].scatter(g.d_route_median_m, g.d_route_p95_m, color=c, label=p, alpha=0.7)
        axs[2].scatter(g.start_gap_m.clip(0.5), g.end_gap_m.clip(0.5), color=c, label=p, alpha=0.7)
    m = max(df.path_len_km.max(), df.route_len_km.max())
    axs[0].plot([0, m], [0, m], "k--", lw=1)
    axs[0].set_xlabel("GPS path length (km)")
    axs[0].set_ylabel("reference route length (km)")
    axs[0].set_title("Route length vs GPS length")
    axs[1].set_xlabel("median GPS->route dist (m)")
    axs[1].set_ylabel("95th pct GPS->route dist (m)")
    axs[1].set_yscale("log")
    axs[1].set_title("How well GPS follows the route")
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("first fix -> first route arc (m)")
    axs[2].set_ylabel("last fix -> last route arc (m)")
    axs[2].set_title("Route endpoint alignment")
    for a in axs:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "06_reference_route_quality.png")
    plt.close(fig)


def _local_geometry(trip, buffer_m=400, corridor=False):
    """Project a trip and keep only arcs near the track (bbox, or a corridor around the fixes)."""
    lon, lat = trip.track[:, 0], trip.track[:, 1]
    pj = Projector(float(np.median(lon)), float(np.median(lat)))
    tx, ty = pj.to_xy(lon, lat)
    nx_, ny_ = pj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
    segs = np.stack([np.column_stack([nx_, ny_])[trip.arcs[:, 0]], np.column_stack([nx_, ny_])[trip.arcs[:, 1]]], 1)
    if corridor:
        # keep arcs whose endpoints or midpoint lie within buffer_m of some GPS fix (keeps html maps small)
        tree = cKDTree(np.column_stack([tx, ty]))
        near = [tree.query(p, distance_upper_bound=buffer_m)[0] < np.inf
                for p in (segs[:, 0], segs[:, 1], segs.mean(1))]
        keep = near[0] | near[1] | near[2]
    else:
        lo = np.array([tx.min(), ty.min()]) - buffer_m
        hi = np.array([tx.max(), ty.max()]) + buffer_m
        keep = np.all((segs.min(1) >= lo) & (segs.max(1) <= hi), axis=1)
    return pj, tx, ty, segs, keep


def fig_trajectory_grid(df, partition="train", n=12):
    # deterministic, spread across sizes: evenly spaced quantiles of n_points
    d = df[df.partition == partition].sort_values("n_points").reset_index(drop=True)
    pick = d.iloc[np.linspace(0, len(d) - 1, n).round().astype(int)]
    cols = 4
    fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(18, 4.4 * np.ceil(n / cols)))
    for ax, (_, r) in zip(axs.ravel(), pick.iterrows()):
        trip = load_trip(r.partition, r.trip_id)
        pj, tx, ty, segs, keep = _local_geometry(trip)
        ax.add_collection(LineCollection(segs[keep], colors="#bbbbbb", linewidths=0.4))
        if len(trip.route):
            ax.add_collection(LineCollection(segs[trip.route], colors="#2a6fdb", linewidths=2.2, label="route"))
        ax.plot(tx, ty, ".", ms=1.2, color="#c0392b", label="GPS")
        ax.plot(tx[0], ty[0], "g^", ms=8)
        ax.plot(tx[-1], ty[-1], "ks", ms=6)
        ax.set_xlim(tx.min() - 300, tx.max() + 300)
        ax.set_ylim(ty.min() - 300, ty.max() + 300)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{r.trip_id}: {r.n_points} pts, {r.path_len_km:.1f} km\n{r['flags'] or 'no flags'}"[:90],
                     fontsize=8)
    fig.suptitle(f"{partition}: GPS (red) vs reference route (blue) vs network (grey); start ^ end ■", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / f"07_trajectories_{partition}.png")
    plt.close(fig)


def fig_lonlat_distribution(df):
    root = data_root()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for p in df.partition.unique():
        lons, lats = [], []
        for tid in df[df.partition == p].trip_id:
            tr = load_trip(p, tid, root).track
            lons.append(tr[:, 0])
            lats.append(tr[:, 1])
        axs[0].hist(np.concatenate(lons), bins=150, alpha=0.6, color=PART_COLORS.get(p), label=p)
        axs[1].hist(np.concatenate(lats), bins=150, alpha=0.6, color=PART_COLORS.get(p), label=p)
    axs[0].set_xlabel("longitude (deg) - column 1 of .track")
    axs[1].set_xlabel("latitude (deg) - column 2 of .track")
    for a in axs:
        a.set_ylabel("GPS points")
        a.legend()
    fig.suptitle("Distribution of raw GPS coordinates (confirms column order lon, lat)")
    fig.tight_layout()
    fig.savefig(FIG / "08_gps_coordinate_distribution.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Folium maps
# ----------------------------------------------------------------------------
def folium_map(partition, trip_id, why):
    trip = load_trip(partition, trip_id)
    pj, tx, ty, segs, keep = _local_geometry(trip, buffer_m=200, corridor=True)
    to_ll = lambda xy: np.round(np.column_stack(pj.to_lonlat(xy[..., 0].ravel(), xy[..., 1].ravel())), 6)  # noqa: E731
    lon, lat, t = trip.track.T
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], zoom_start=14, tiles="OpenStreetMap")

    net = to_ll(segs[keep]).reshape(-1, 2, 2)
    folium.GeoJson({"type": "MultiLineString", "coordinates": net.tolist()}, name="road network (clipped)",
                   style_function=lambda _: {"color": "#888", "weight": 1, "opacity": 0.6}).add_to(m)
    if len(trip.route):
        r = to_ll(segs[trip.route]).reshape(-1, 2, 2)
        folium.GeoJson({"type": "MultiLineString", "coordinates": r.tolist()}, name="reference route",
                       style_function=lambda _: {"color": "#2a6fdb", "weight": 5, "opacity": 0.7}).add_to(m)
    gps = folium.FeatureGroup(name="GPS track")
    folium.PolyLine(np.column_stack([lat, lon]).tolist(), color="#c0392b", weight=2, opacity=0.8).add_to(gps)
    step = max(1, len(lat) // 400)  # thin the dots so the html stays small
    for i in range(0, len(lat), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color="#c0392b", fill=True,
                            tooltip=f"i={i} t={t[i]:.0f}s").add_to(gps)
    gps.add_to(m)
    gaps = folium.FeatureGroup(name=f"gaps > {M.LARGE_GAP_S:g}s")
    dt = np.diff(t)
    for i in np.where(dt > M.LARGE_GAP_S)[0]:
        folium.PolyLine([[lat[i], lon[i]], [lat[i + 1], lon[i + 1]]], color="black", dash_array="6",
                        weight=3, tooltip=f"gap {dt[i]:.0f}s").add_to(gaps)
        folium.Marker([lat[i], lon[i]], tooltip=f"gap starts, {dt[i]:.0f}s",
                      icon=folium.Icon(color="black", icon="pause")).add_to(gaps)
    gaps.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl().add_to(m)
    title = f"<h4 style='position:fixed;top:8px;left:50px;z-index:9999;background:white;padding:4px 8px'>" \
            f"PathSnap {partition}/{trip_id} - {why}</h4>"
    m.get_root().html.add_child(folium.Element(title))
    fn = MAP / f"{partition}_{trip_id}.html"
    m.save(fn)
    return fn.name


def select_map_trips(df):
    """Pick trips automatically by rule (no hand-picking), train partition only."""
    d = df[df.partition == "train"]
    med = d.n_points.median()
    picks = {
        "typical (median #points)": d.iloc[(d.n_points - med).abs().argsort().iloc[0]].trip_id,
        "longest gap": d.loc[d.max_gap_s.idxmax()].trip_id,
        "most route breaks": d.loc[d.route_n_breaks.idxmax()].trip_id,
        "worst GPS-to-route fit (p95)": d.loc[d.d_route_p95_m.idxmax()].trip_id,
        "most speed jumps": d.loc[d.n_speed_jumps.idxmax()].trip_id,
        "fewest GPS points": d.loc[d.n_points.idxmin()].trip_id,
    }
    out, seen = [], set()
    for why, tid in picks.items():
        if tid in seen:
            continue
        seen.add(tid)
        out.append((why, tid, folium_map("train", tid, why)))
    return out


# ----------------------------------------------------------------------------
# 5. markdown report (numbers only - interpretation lives in docs/)
# ----------------------------------------------------------------------------
def md_table(df: pd.DataFrame, floatfmt=".3g") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return format(v, floatfmt)
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report(df, pooled, maps):
    dt = np.concatenate([pooled[p]["dt"] for p in pooled])
    sp = np.concatenate([pooled[p]["speed"] for p in pooled])
    dn = np.concatenate([pooled[p]["d_net"] for p in pooled])
    dr = np.concatenate([pooled[p]["d_route"] for p in pooled])
    part = pd.read_csv(TAB / "partition_summary.csv")
    countries = df.assign(utm=df.utm_epsg.astype(str)).groupby("utm").size().rename("trips").reset_index()

    q = lambda a, p: np.nanpercentile(a, p)  # noqa: E731
    L = []
    L.append("# PathSnap - Dataset Exploration Report (auto-generated)\n")
    L.append(f"Generated by `scripts/run_eda.py` on {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
             "Every number below is recomputed from the raw files on each run; per-trip values are in "
             "`tables/trip_summary.csv`. Interpretation and decisions: `docs/DATA_QUALITY_NOTES.md`.\n")
    L.append("## 1. Inventory\n")
    L.append(md_table(part, ".2f"))
    L.append(f"\n* Partitions on disk: {', '.join(sorted(df.partition.unique()))}. The evaluator-test partition "
             "is not available locally and is never read.")
    L.append(f"* Every trip folder holds exactly the 4 files `<id>.track/.nodes/.arcs/.route`: "
             f"{bool(df[[c for c in df if c.endswith('_exists')]].all().all())}.")
    L.append(f"* Column counts per file (all rows, all trips): track={sorted(df.track_colcounts.unique())}, "
             f"nodes={sorted(df.nodes_colcounts.unique())}, arcs={sorted(df.arcs_colcounts.unique())}, "
             f"route={sorted(df.route_colcounts.unique())}.")
    L.append(f"* Malformed rows (blank / wrong column count / non-numeric / NaN / inf): "
             f"{int(df[[c for c in df if c.endswith('_bad_rows')]].sum().sum())}.")
    L.append("* Inferred schema: `.track` float lon, float lat, float t (s); `.nodes` float lon, float lat; "
             "`.arcs` int from_node, int to_node; `.route` int arc_id. No headers anywhere.\n")

    L.append("## 2. Trajectories\n")
    L.append(f"* GPS points per trip: min {df.n_points.min()}, median {df.n_points.median():.0f}, "
             f"max {df.n_points.max()} (total {df.n_points.sum():,}).")
    L.append(f"* Coordinates out of range: lon {df.n_lon_out_of_range.sum()}, lat {df.n_lat_out_of_range.sum()}, "
             f"(0,0) points {df.n_zero_coords.sum()}. Share of GPS points inside their own network's bbox: "
             f"min over trips {df.frac_track_in_net_bbox.min():.3f} -> column order (lon, lat) confirmed.")
    L.append(f"* Trips span {len(countries)} UTM zones: " +
             ", ".join(f"EPSG:{r.utm} ({r.trips})" for r in countries.itertuples()) +
             f"; lon {df.lon_min.min():.2f}..{df.lon_max.max():.2f}, lat {df.lat_min.min():.2f}.."
             f"{df.lat_max.max():.2f}.")
    L.append(f"* Trajectory extent (UTM): median {df.extent_x_km.median():.1f} km E-W x "
             f"{df.extent_y_km.median():.1f} km N-S; GPS path length median {df.path_len_km.median():.1f} km, "
             f"max {df.path_len_km.max():.1f} km.")
    L.append(f"* UTM vs haversine step length: max relative difference {df.utm_vs_haversine_max_rel_err.max():.2%} "
             "(sphere-vs-ellipsoid + UTM scale) - negligible vs GPS noise.\n")

    L.append("### Timestamps and sampling\n")
    L.append(f"* Steps with dt == 1 s: {np.mean(dt == 1):.2%} of {len(dt):,}. dt == 2 s: {np.mean(dt == 2):.2%}. "
             f"dt > 10 s: {int(np.sum(dt > 10))} steps. dt <= 0 (out of order / duplicate t): "
             f"{int(df.n_dt_nonpositive.sum())}.")
    L.append(f"* Non-integer timestamps (beyond 1e-6 float noise): {int(df.n_non_integer_t.sum())}.")
    L.append(f"* Trips not starting at t = 1: {int((df.t_start != 1).sum())} "
             f"(t_start up to {df.t_start.max():.0f} s) -> t is relative, not wall-clock.")
    L.append(f"* Trips with at least one gap > {M.LARGE_GAP_S:g} s: {int((df.n_large_gaps > 0).sum())}; "
             f"longest gap {df.max_gap_s.max():,.0f} s ({df.max_gap_s.max() / 3600:.1f} h, trip "
             f"{df.loc[df.max_gap_s.idxmax()].trip_id}).\n")

    L.append("### Motion / noise\n")
    L.append(f"* Step speed: median {np.nanmedian(sp):.1f} m/s, p99 {q(sp, 99):.1f} m/s, max {np.nanmax(sp):.0f} m/s. "
             f"Steps faster than {M.MAX_PLAUSIBLE_SPEED:g} m/s: {int(np.nansum(sp > M.MAX_PLAUSIBLE_SPEED))} "
             f"in {int((df.n_speed_jumps > 0).sum())} trips.")
    L.append(f"* Stationary steps (< {M.STATIONARY_SPEED} m/s): {np.nanmean(sp < M.STATIONARY_SPEED):.1%}; "
             f"exact duplicate consecutive positions: {int(df.n_duplicate_positions.sum())}. "
             f"Window test (moved < {M.PARK_RADIUS_M:g} m over {M.PARK_WINDOW} fixes): median "
             f"{df.frac_parked.median():.1%} of a trip is parked; {int((df.frac_parked > 0.2).sum())} trips are "
             f"parked > 20 % of the time (max {df.frac_parked.max():.0%}, trip "
             f"{df.loc[df.frac_parked.idxmax()].trip_id}).")
    L.append(f"* Distance GPS -> nearest arc of the network: median {np.nanmedian(dn):.1f} m, p95 {q(dn, 95):.1f} m, "
             f"p99 {q(dn, 99):.1f} m, max {np.nanmax(dn):.0f} m.")
    L.append(f"* Distance GPS -> nearest arc of the REFERENCE ROUTE: median {np.nanmedian(dr):.1f} m, "
             f"p95 {q(dr, 95):.1f} m, p99 {q(dr, 99):.1f} m, max {np.nanmax(dr):.0f} m.\n")

    L.append("## 3. Road networks\n")
    L.append(f"* Each trip has its own network extract (no two `.nodes` files identical). Nodes per trip: "
             f"min {df.n_nodes.min():,}, median {df.n_nodes.median():,.0f}, max {df.n_nodes.max():,}; arcs / nodes "
             f"median {np.median(df.n_arcs / df.n_nodes):.2f}.")
    L.append(f"* All arc endpoints are valid node indices: {bool(df.arc_ids_valid.all())}. Self loops: "
             f"{df.n_self_loops.sum()} total; duplicate (from,to) arcs: {df.n_duplicate_arcs.sum()} total; "
             f"zero-length arcs: {df.n_zero_len_arcs.sum()}; duplicate node coordinates: "
             f"{df.n_duplicate_node_coords.sum()}.")
    L.append(f"* Arcs are DIRECTED. Share with a reverse arc: median {df.frac_arcs_with_reverse.median():.2f} "
             f"(min {df.frac_arcs_with_reverse.min():.2f}) -> two-way roads are stored as two arcs, the rest are one-way.")
    L.append(f"* Arc length: median of per-trip medians {df.arc_len_median_m.median():.1f} m; longest arc "
             f"{df.arc_len_max_m.max():,.0f} m.")
    L.append(f"* Isolated nodes (degree 0): {df.n_isolated_nodes.sum()} total. Weakly connected components per trip: "
             f"median {df.n_weak_components.median():.0f}, max {df.n_weak_components.max()}; largest WCC covers "
             f"median {df.largest_wcc_frac.median():.1%} (min {df.largest_wcc_frac.min():.1%}) of nodes.")
    L.append(f"* Strongly connected components: median {df.n_strong_components.median():.0f} per trip; largest SCC "
             f"covers median {df.largest_scc_frac.median():.1%} (min {df.largest_scc_frac.min():.1%}). "
             f"Dead-end nodes (in>0, out=0): median {df.n_dead_end_nodes.median():.0f} per trip.\n")

    L.append("## 4. Reference routes\n")
    L.append(f"* Route present for {int((df.route_len_arcs > 0).sum())}/{len(df)} trips; all arc ids valid: "
             f"{bool(df.route_ids_valid.all())}. Length: median {df.route_len_arcs.median():.0f} arcs "
             f"({df.route_len_km.median():.1f} km).")
    L.append(f"* Route length / GPS path length: median {df.route_to_gps_len_ratio.median():.2f}, "
             f"range {df.route_to_gps_len_ratio.min():.2f}..{df.route_to_gps_len_ratio.max():.2f}.")
    L.append(f"* Contiguity: {int((df.route_n_breaks > 0).sum())} trips have >=1 break (arc[i].to != arc[i+1].from), "
             f"{df.route_n_breaks.sum()} breaks total. Consecutive duplicate arcs: "
             f"{df.route_n_consecutive_repeats.sum()}. Immediate U-turns: {df.route_n_uturns.sum()}. "
             f"Arcs revisited later in the same route: {df.route_n_revisited_arcs.sum()}.")
    L.append(f"* Route arcs inside the network's largest SCC: min share over trips "
             f"{df.route_frac_in_largest_scc.min():.3f}.")
    L.append(f"* First GPS fix -> first route arc: median {df.start_gap_m.median():.1f} m, max {df.start_gap_m.max():.0f} m. "
             f"Last fix -> last arc: median {df.end_gap_m.median():.1f} m, max {df.end_gap_m.max():.0f} m.")
    L.append(f"* Trips with > 5 % of GPS points more than {M.FAR_FROM_NET_M:g} m from the route: "
             f"{int((df.frac_far_from_route > 0.05).sum())}.\n")

    L.append("## 5. Anomalous trips\n")
    an = pd.read_csv(TAB / "anomalies.csv", dtype={"trip_id": str})
    L.append(f"{len(an)} of {len(df)} trips raise at least one flag (thresholds in `src/pathsnap/eda_metrics.py`). "
             "Full list: `tables/anomalies.csv`. Trips with >= 2 flags:\n")
    multi = df[df.n_flags >= 2][["partition", "trip_id", "n_points", "flags"]]
    L.append(md_table(multi) if len(multi) else "_none_")
    L.append("\n## 6. Figures and maps\n")
    for f in sorted(FIG.glob("*.png")):
        L.append(f"![{f.stem}](figures/{f.name})")
    L.append("\nInteractive Folium maps (open in a browser; layers: clipped network, reference route, GPS, gaps):\n")
    for why, tid, fn in maps:
        L.append(f"* [`{fn}`](maps/{fn}) - train/{tid}: {why}")
    (OUT / "EDA_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    t0 = time.time()
    print(f"data root: {data_root()}")
    df, pooled = collect()
    write_tables(df)
    fig_trip_overview(df)
    fig_sampling(df, pooled)
    fig_geography(df)
    fig_speed_distance(pooled)
    fig_network(df)
    fig_route_quality(df)
    fig_trajectory_grid(df, "train", 12)
    fig_trajectory_grid(df, "student_test", 8)
    fig_lonlat_distribution(df)
    maps = select_map_trips(df)
    write_report(df, pooled, maps)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
