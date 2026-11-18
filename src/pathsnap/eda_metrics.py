"""Per-trip exploratory statistics (Dataset Exploration brief, section 4).

`trip_stats(trip)` returns (summary_dict, arrays_dict):
  * summary_dict -> one row of outputs/eda/tables/trip_summary.csv
  * arrays_dict  -> per-point / per-step arrays pooled across trips for plots
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import shapely

from .data_io import Trip
from .geo import Projector, haversine_m

# Thresholds used to flag anomalies (documented in docs/DATA_QUALITY_NOTES.md)
GAP_S = 1.5              # any step longer than this is a "gap" (nominal rate is 1 Hz)
LARGE_GAP_S = 30.0       # gap long enough that the vehicle can change road several times
MAX_PLAUSIBLE_SPEED = 70.0   # m/s (~250 km/h); faster consecutive steps = position jump
STATIONARY_SPEED = 0.5   # m/s
FAR_FROM_NET_M = 50.0    # GPS point farther than this from any arc is suspicious
PARK_WINDOW = 60         # fixes (~60 s at 1 Hz)
PARK_RADIUS_M = 30.0     # net displacement below this over PARK_WINDOW fixes = parked


def _segments(xy_nodes: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    return np.stack([xy_nodes[arcs[:, 0]], xy_nodes[arcs[:, 1]]], axis=1)  # (A, 2, 2)


def _point_to_lines_dist(pts_xy: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance (m) from each point to its nearest segment."""
    if len(segs) == 0 or len(pts_xy) == 0:
        return np.full(len(pts_xy), np.nan)
    lines = shapely.linestrings(segs)
    tree = shapely.STRtree(lines)
    pts = shapely.points(pts_xy)
    (pi, _), d = tree.query_nearest(pts, return_distance=True, all_matches=False)
    out = np.full(len(pts_xy), np.nan)
    out[pi] = d
    return out


def trip_stats(trip: Trip) -> tuple[dict, dict]:
    tr, nodes, arcs, route = trip.track, trip.nodes, trip.arcs, trip.route
    s: dict = {"partition": trip.partition, "trip_id": trip.trip_id}

    # ---------------- file inventory / parsing ----------------
    for kind, rep in trip.reports.items():
        s[f"{kind}_exists"] = rep.exists
        s[f"{kind}_bytes"] = rep.size_bytes
        s[f"{kind}_rows"] = rep.n_rows
        s[f"{kind}_colcounts"] = "/".join(map(str, rep.col_counts))
        s[f"{kind}_bad_rows"] = rep.n_blank_rows + rep.n_bad_colcount + rep.n_non_numeric + rep.n_non_finite

    lon, lat, t = tr[:, 0], tr[:, 1], tr[:, 2]
    n = len(tr)
    s["n_points"] = n

    # ---------------- coordinate plausibility ----------------
    s["n_lon_out_of_range"] = int(np.sum(np.abs(lon) > 180))
    s["n_lat_out_of_range"] = int(np.sum(np.abs(lat) > 90))
    s["n_zero_coords"] = int(np.sum((lon == 0) & (lat == 0)))
    s["lon_min"], s["lon_max"] = float(lon.min()), float(lon.max())
    s["lat_min"], s["lat_max"] = float(lat.min()), float(lat.max())
    s["lon_c"], s["lat_c"] = float(np.median(lon)), float(np.median(lat))
    # coordinate-order check: GPS must fall inside the network's bbox if (lon, lat) is right
    nb = (nodes[:, 0].min(), nodes[:, 0].max(), nodes[:, 1].min(), nodes[:, 1].max())
    inside = (lon >= nb[0]) & (lon <= nb[1]) & (lat >= nb[2]) & (lat <= nb[3])
    s["frac_track_in_net_bbox"] = float(inside.mean())

    proj = Projector(s["lon_c"], s["lat_c"])
    s["utm_epsg"] = proj.epsg
    x, y = proj.to_xy(lon, lat)
    xy = np.column_stack([x, y])
    s["extent_x_km"] = float((x.max() - x.min()) / 1000)
    s["extent_y_km"] = float((y.max() - y.min()) / 1000)

    # ---------------- timestamps ----------------
    dt = np.diff(t)
    dtr = np.round(dt, 6)  # remove float noise like 0.9999999999999432
    s["t_start"], s["t_end"] = float(t[0]), float(t[-1])
    s["duration_s"] = float(t[-1] - t[0])
    s["n_non_integer_t"] = int(np.sum(np.abs(t - np.round(t)) > 1e-6))
    s["n_dt_nonpositive"] = int(np.sum(dtr <= 0))   # out-of-order or duplicate timestamps
    s["frac_dt_eq_1"] = float(np.mean(dtr == 1)) if n > 1 else np.nan
    s["n_gaps"] = int(np.sum(dtr > GAP_S))
    s["n_large_gaps"] = int(np.sum(dtr > LARGE_GAP_S))
    s["max_gap_s"] = float(dtr.max()) if n > 1 else np.nan
    s["missing_seconds"] = float(np.sum(np.clip(dtr - 1, 0, None)))  # seconds with no fix

    # ---------------- motion ----------------
    step = np.hypot(np.diff(x), np.diff(y))
    step_hav = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    s["utm_vs_haversine_max_rel_err"] = float(
        np.max(np.abs(step - step_hav) / np.maximum(step_hav, 1.0))) if n > 1 else np.nan
    speed = np.where(dtr > 0, step / np.where(dtr > 0, dtr, 1), np.nan)
    s["path_len_km"] = float(step.sum() / 1000)
    s["speed_median"] = float(np.nanmedian(speed))
    s["speed_p95"] = float(np.nanpercentile(speed, 95))
    s["speed_max"] = float(np.nanmax(speed))
    s["n_speed_jumps"] = int(np.nansum(speed > MAX_PLAUSIBLE_SPEED))
    s["frac_stationary"] = float(np.nanmean(speed < STATIONARY_SPEED))
    s["n_duplicate_positions"] = int(np.sum(step == 0))
    # GPS jitter makes a parked car look like it moves several m/s, so instantaneous speed
    # under-detects stops. Window test: net displacement over the next PARK_WINDOW fixes.
    if n > PARK_WINDOW:
        disp = np.hypot(x[PARK_WINDOW:] - x[:-PARK_WINDOW], y[PARK_WINDOW:] - y[:-PARK_WINDOW])
        s["frac_parked"] = float(np.mean(disp < PARK_RADIUS_M))
    else:
        s["frac_parked"] = np.nan
    gap_idx = np.where(dtr > LARGE_GAP_S)[0]
    s["max_jump_across_gap_m"] = float(step[gap_idx].max()) if len(gap_idx) else 0.0

    # ---------------- road network ----------------
    A, N = len(arcs), len(nodes)
    s["n_nodes"], s["n_arcs"] = N, A
    s["arc_ids_valid"] = bool(A == 0 or (arcs.min() >= 0 and arcs.max() < N))
    s["n_self_loops"] = int(np.sum(arcs[:, 0] == arcs[:, 1]))
    s["n_duplicate_arcs"] = int(A - len(np.unique(arcs, axis=0)))
    pairs = set(map(tuple, arcs.tolist()))
    s["frac_arcs_with_reverse"] = float(np.mean([(b, a) in pairs for a, b in arcs.tolist()]))
    s["n_duplicate_node_coords"] = int(N - len(np.unique(nodes, axis=0)))
    nx_, ny_ = proj.to_xy(nodes[:, 0], nodes[:, 1])
    xy_nodes = np.column_stack([nx_, ny_])
    segs = _segments(xy_nodes, arcs)
    arc_len = np.hypot(*(segs[:, 1] - segs[:, 0]).T)
    s["arc_len_median_m"] = float(np.median(arc_len))
    s["arc_len_p99_m"] = float(np.percentile(arc_len, 99))
    s["arc_len_max_m"] = float(arc_len.max())
    s["n_zero_len_arcs"] = int(np.sum(arc_len == 0))
    s["net_total_km"] = float(arc_len.sum() / 1000)
    s["net_extent_x_km"] = float((nx_.max() - nx_.min()) / 1000)
    s["net_extent_y_km"] = float((ny_.max() - ny_.min()) / 1000)

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    G.add_edges_from(arcs.tolist())
    deg = np.bincount(arcs.ravel(), minlength=N)
    s["n_isolated_nodes"] = int(np.sum(deg == 0))
    wcc = list(nx.weakly_connected_components(G))
    scc = list(nx.strongly_connected_components(G))
    big_scc = max(scc, key=len)
    s["n_weak_components"] = len(wcc)
    s["n_weak_components_ex_isolated"] = int(sum(len(c) > 1 for c in wcc))
    s["largest_wcc_frac"] = max(len(c) for c in wcc) / N
    s["n_strong_components"] = len(scc)
    s["largest_scc_frac"] = len(big_scc) / N
    s["n_dead_end_nodes"] = int(sum(1 for v in range(N) if G.out_degree(v) == 0 and G.in_degree(v) > 0))

    # ---------------- GPS vs network ----------------
    d_net = _point_to_lines_dist(xy, segs)
    s["d_net_median_m"] = float(np.nanmedian(d_net))
    s["d_net_p95_m"] = float(np.nanpercentile(d_net, 95))
    s["d_net_max_m"] = float(np.nanmax(d_net))
    s["frac_far_from_net"] = float(np.nanmean(d_net > FAR_FROM_NET_M))

    # ---------------- reference route ----------------
    R = len(route)
    s["route_len_arcs"] = R
    s["route_ids_valid"] = bool(R == 0 or (route.min() >= 0 and route.max() < A))
    d_route = np.full(n, np.nan)
    if R and s["route_ids_valid"]:
        ra = arcs[route]
        s["route_n_breaks"] = int(np.sum(ra[:-1, 1] != ra[1:, 0]))   # consecutive arcs not sharing a node
        s["route_n_consecutive_repeats"] = int(np.sum(route[1:] == route[:-1]))
        s["route_n_revisited_arcs"] = int(R - len(np.unique(route)))
        s["route_n_uturns"] = int(np.sum((ra[:-1, 0] == ra[1:, 1]) & (ra[:-1, 1] == ra[1:, 0])))
        s["route_len_km"] = float(arc_len[route].sum() / 1000)
        s["route_to_gps_len_ratio"] = s["route_len_km"] / max(s["path_len_km"], 1e-9)
        s["route_frac_in_largest_scc"] = float(np.mean([a in big_scc and b in big_scc for a, b in ra.tolist()]))
        d_route = _point_to_lines_dist(xy, segs[route])
        s["d_route_median_m"] = float(np.nanmedian(d_route))
        s["d_route_p95_m"] = float(np.nanpercentile(d_route, 95))
        s["d_route_max_m"] = float(np.nanmax(d_route))
        s["frac_far_from_route"] = float(np.nanmean(d_route > FAR_FROM_NET_M))
        # do the route endpoints line up with the first / last GPS fix?
        first_seg, last_seg = segs[route[0]], segs[route[-1]]
        s["start_gap_m"] = float(shapely.distance(shapely.points(xy[0]), shapely.linestrings(first_seg)))
        s["end_gap_m"] = float(shapely.distance(shapely.points(xy[-1]), shapely.linestrings(last_seg)))

    # ---------------- anomaly flags ----------------
    flags = []
    if s["n_large_gaps"]:
        flags.append(f"large_gaps({s['n_large_gaps']}, max {s['max_gap_s']:.0f}s)")
    if s["t_start"] != 1:
        flags.append(f"t_starts_at_{s['t_start']:.0f}")
    if s["n_dt_nonpositive"]:
        flags.append("non_monotonic_t")
    if s["n_speed_jumps"]:
        flags.append(f"speed_jumps({s['n_speed_jumps']})")
    if s["frac_far_from_net"] > 0.05:
        flags.append(f"far_from_net({s['frac_far_from_net']:.0%})")
    if s.get("route_n_breaks", 0):
        flags.append(f"route_breaks({s['route_n_breaks']})")
    if s.get("frac_far_from_route", 0) > 0.05:
        flags.append(f"far_from_route({s['frac_far_from_route']:.0%})")
    if s.get("route_n_uturns", 0):
        flags.append(f"route_uturns({s['route_n_uturns']})")
    if s["frac_parked"] > 0.2:
        flags.append(f"long_parked({s['frac_parked']:.0%})")
    if any(r.n_rows == 0 for r in trip.reports.values()):
        flags.append("missing_or_empty_file")
    s["flags"] = "; ".join(flags)
    s["n_flags"] = len(flags)

    arrays = {"dt": dtr, "speed": speed, "d_net": d_net, "d_route": d_route, "arc_len": arc_len}
    return s, arrays
