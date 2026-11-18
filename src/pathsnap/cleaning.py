"""Cleaning the raw PathSnap trips (decisions from docs/DATA_QUALITY_NOTES.md).

Guiding rule: metrics are computed per GPS fix, so every fix still needs a
prediction. Cleaning therefore NEVER deletes rows or renumbers ids. It adds
metres, segment / stop ids and quality flags. Later stages decide what to skip.

Two kinds of cleaning, kept apart to avoid leakage:

* label-free (`clean_track`, `clean_network`): uses only .track / .nodes / .arcs.
  It runs the same way on every partition, including evaluator-test trips at
  inference time.
* label-dependent (`clean_route`, `label_quality`): uses .route. It is only for
  training labels and for evaluation, never as an input to prediction.

All thresholds are fixed by hand from physical reasoning. None are tuned on
student-test or evaluator-test data.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .data_io import REPO_ROOT, Trip, load_trip
from .eda_metrics import _point_to_lines_dist, _segments
from .geo import Projector

PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

# ---- thresholds (hand-set, see docs/DATA_QUALITY_NOTES.md) ----
SEGMENT_GAP_S = 120.0     # gap longer than this starts a new segment: up to ~2 km of unseen
                          # driving, so "shortest path between fixes" is no longer a safe assumption
MAX_SPEED_MPS = 70.0      # ~250 km/h; faster steps are flagged as unreliable speed
STOP_WINDOW = 60          # fixes (~60 s at 1 Hz)
STOP_RADIUS_M = 30.0      # net movement below this over STOP_WINDOW fixes = stopped
OFF_NETWORK_M = 50.0      # fix farther than this from every road
FAR_FROM_ROUTE_M = 50.0   # fix farther than this from the reference route (label check)
EXCLUDE_TRIP_FAR_FRAC = 0.25  # trip dropped from training if more fixes than this are far from its route


def _runs_to_ids(mask: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Number consecutive True runs 0,1,2,... (-1 where False); `breaks` forces a new run."""
    ids = np.full(len(mask), -1, dtype=np.int64)
    cur, prev = -1, False
    for i, m in enumerate(mask):
        if m and (not prev or breaks[i]):
            cur += 1
        if m:
            ids[i] = cur
        prev = m
    return ids


def _spikes(xy: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Out-and-back glitches: fast step in, fast step out, but the neighbours
    on each side are close together. (EDA showed most fast steps are NOT
    like this: they stay on the road and keep moving forward.)"""
    n = len(xy)
    out = np.zeros(n, dtype=bool)
    if n < 3:
        return out
    step = np.hypot(*np.diff(xy, axis=0).T)
    sp = step / np.maximum(dt, 1e-9)
    skip = np.hypot(*(xy[2:] - xy[:-2]).T) / np.maximum(dt[:-1] + dt[1:], 1e-9)
    out[1:-1] = (sp[:-1] > MAX_SPEED_MPS) & (sp[1:] > MAX_SPEED_MPS) & (skip <= MAX_SPEED_MPS)
    return out


def clean_network(trip: Trip, proj: Projector) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes, arcs = trip.nodes, trip.arcs
    nx_, ny_ = proj.to_xy(nodes[:, 0], nodes[:, 1])
    nodes_df = pd.DataFrame({"node_id": np.arange(len(nodes)), "lon": nodes[:, 0], "lat": nodes[:, 1],
                             "x": nx_, "y": ny_})
    xy_nodes = np.column_stack([nx_, ny_])
    segs = _segments(xy_nodes, arcs)
    length = np.hypot(*(segs[:, 1] - segs[:, 0]).T)

    # Arcs are straight node-to-node lines, so two arcs with the same end
    # coordinates are the same road and no GPS fix can tell them apart.
    # Map each one to the lowest arc_id with that geometry.
    key = np.column_stack([nodes[arcs[:, 0]], nodes[arcs[:, 1]]])
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    canonical = first[inv.ravel()]

    G = nx.DiGraph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(arcs.tolist())
    big = max(nx.strongly_connected_components(G), key=len)
    in_big = np.fromiter((a in big and b in big for a, b in arcs.tolist()), dtype=bool, count=len(arcs))

    arcs_df = pd.DataFrame({
        "arc_id": np.arange(len(arcs)),
        "from_node": arcs[:, 0],
        "to_node": arcs[:, 1],
        "length_m": length,
        "is_self_loop": arcs[:, 0] == arcs[:, 1],
        "is_zero_length": length == 0,
        "canonical_arc_id": canonical,
        "is_duplicate": canonical != np.arange(len(arcs)),
        "in_largest_scc": in_big,
    })
    # a zero-length arc has no direction or extent, so it can never be a match candidate
    arcs_df["is_candidate"] = ~arcs_df["is_zero_length"]
    return nodes_df, arcs_df


def clean_track(trip: Trip, proj: Projector, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> pd.DataFrame:
    tr = trip.track
    n = len(tr)
    t = tr[:, 2]
    x, y = proj.to_xy(tr[:, 0], tr[:, 1])
    xy = np.column_stack([x, y])

    dt = np.round(np.diff(t), 6)  # strip float noise like 0.9999999999999432
    step = np.hypot(*np.diff(xy, axis=0).T)
    speed = step / np.maximum(dt, 1e-9)

    new_seg = np.zeros(n, dtype=bool)
    new_seg[1:] = dt > SEGMENT_GAP_S
    segment_id = np.cumsum(new_seg)

    # stopped: net movement < STOP_RADIUS_M over the next STOP_WINDOW fixes (same segment only);
    # instantaneous speed misses stops because parked GPS jitter looks like several m/s
    stop = np.zeros(n, dtype=bool)
    if n > STOP_WINDOW:
        disp = np.hypot(x[STOP_WINDOW:] - x[:-STOP_WINDOW], y[STOP_WINDOW:] - y[:-STOP_WINDOW])
        same_seg = segment_id[STOP_WINDOW:] == segment_id[:-STOP_WINDOW]
        starts = np.where((disp < STOP_RADIUS_M) & same_seg)[0]
        cover = np.zeros(n + 1, dtype=np.int64)
        np.add.at(cover, starts, 1)
        np.add.at(cover, starts + STOP_WINDOW + 1, -1)
        stop = np.cumsum(cover)[:n] > 0
    stop_id = _runs_to_ids(stop, new_seg)

    cand = arcs_df["is_candidate"].to_numpy()
    xy_nodes = nodes_df[["x", "y"]].to_numpy()
    d_net = _point_to_lines_dist(xy, _segments(xy_nodes, trip.arcs[cand]))

    df = pd.DataFrame({
        "fix_idx": np.arange(n),
        "lon": tr[:, 0], "lat": tr[:, 1], "t": t,
        "t_rel": t - t[0],                       # timestamps are relative, only differences matter
        "dt_prev": np.r_[np.nan, dt],
        "x": x, "y": y,
        "step_m": np.r_[np.nan, step],
        "speed_mps": np.r_[np.nan, speed],
        "segment_id": segment_id,
        "is_spike": _spikes(xy, dt),
        "is_fast_step": np.r_[False, speed > MAX_SPEED_MPS],
        "is_stop": stop,
        "stop_id": stop_id,
        "d_net_m": d_net,
        "is_off_network": d_net > OFF_NETWORK_M,
    })
    return df


def clean_route(trip: Trip, arcs_df: pd.DataFrame) -> pd.DataFrame:
    r = trip.route
    a = trip.arcs[r] if len(r) else np.empty((0, 2), dtype=np.int64)
    brk = np.r_[False, a[:-1, 1] != a[1:, 0]] if len(r) else np.empty(0, dtype=bool)
    uturn = np.r_[False, (a[:-1, 0] == a[1:, 1]) & (a[:-1, 1] == a[1:, 0])] if len(r) else brk
    seen, revisit = set(), []
    for arc in r.tolist():
        revisit.append(arc in seen)
        seen.add(arc)
    return pd.DataFrame({
        "seq": np.arange(len(r)),
        "arc_id": r,
        "canonical_arc_id": arcs_df["canonical_arc_id"].to_numpy()[r] if len(r) else r,
        "break_before": brk,          # not connected to the previous arc (labels are not always a path)
        "is_uturn": uturn,
        "is_revisit": np.array(revisit, dtype=bool),
    })


def label_quality(track_df: pd.DataFrame, route_df: pd.DataFrame, nodes_df: pd.DataFrame,
                  arcs_df: pd.DataFrame, arcs: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Label-dependent checks. Uses the reference route, so only for training and evaluation."""
    if route_df.empty:
        track_df["d_route_m"] = np.nan
        track_df["is_far_from_route"] = False
        track_df["label_reliable"] = False
        return track_df, {"frac_far_from_route": np.nan, "exclude_from_training": True,
                          "exclude_reason": "no reference route"}
    xy = track_df[["x", "y"]].to_numpy()
    segs = _segments(nodes_df[["x", "y"]].to_numpy(), arcs[np.unique(route_df["arc_id"].to_numpy())])
    d_route = _point_to_lines_dist(xy, segs)
    far = d_route > FAR_FROM_ROUTE_M
    frac_far = float(far.mean())
    exclude = frac_far > EXCLUDE_TRIP_FAR_FRAC
    track_df["d_route_m"] = d_route
    track_df["is_far_from_route"] = far
    # a fix is a trustworthy training / scoring example only if it is near a road, near its
    # reference route, not a glitch, and its trip is not excluded
    track_df["label_reliable"] = (~far & ~track_df["is_off_network"] & ~track_df["is_spike"]) & (not exclude)
    reason = f"{frac_far:.0%} of fixes > {FAR_FROM_ROUTE_M:g} m from the reference route" if exclude else ""
    return track_df, {"frac_far_from_route": frac_far, "exclude_from_training": bool(exclude),
                      "exclude_reason": reason}


def clean_trip(trip: Trip) -> dict:
    lon_c, lat_c = float(np.median(trip.track[:, 0])), float(np.median(trip.track[:, 1]))
    proj = Projector(lon_c, lat_c)
    nodes_df, arcs_df = clean_network(trip, proj)
    track_df = clean_track(trip, proj, nodes_df, arcs_df)
    route_df = clean_route(trip, arcs_df)
    track_df, lq = label_quality(track_df, route_df, nodes_df, arcs_df, trip.arcs)
    meta = {
        "partition": trip.partition, "trip_id": trip.trip_id, "utm_epsg": proj.epsg,
        "n_fixes": len(track_df),
        "n_segments": int(track_df["segment_id"].max() + 1),
        "n_spikes": int(track_df["is_spike"].sum()),
        "n_fast_steps": int(track_df["is_fast_step"].sum()),
        "frac_stop": float(track_df["is_stop"].mean()),
        "n_stops": int(track_df["stop_id"].max() + 1),
        "frac_off_network": float(track_df["is_off_network"].mean()),
        "n_arcs": len(arcs_df),
        "n_zero_length_arcs": int(arcs_df["is_zero_length"].sum()),
        "n_duplicate_arcs": int(arcs_df["is_duplicate"].sum()),
        "frac_arcs_in_largest_scc": float(arcs_df["in_largest_scc"].mean()),
        "route_len": len(route_df),
        "route_breaks": int(route_df["break_before"].sum()),
        "route_uses_duplicate_arc": int((route_df["arc_id"] != route_df["canonical_arc_id"]).sum()),
        "frac_label_reliable": float(track_df["label_reliable"].mean()),
        **lq,
        "thresholds": {
            "SEGMENT_GAP_S": SEGMENT_GAP_S, "MAX_SPEED_MPS": MAX_SPEED_MPS, "STOP_WINDOW": STOP_WINDOW,
            "STOP_RADIUS_M": STOP_RADIUS_M, "OFF_NETWORK_M": OFF_NETWORK_M,
            "FAR_FROM_ROUTE_M": FAR_FROM_ROUTE_M, "EXCLUDE_TRIP_FAR_FRAC": EXCLUDE_TRIP_FAR_FRAC,
        },
    }
    return {"track": track_df, "nodes": nodes_df, "arcs": arcs_df, "route": route_df, "meta": meta}


def save_clean_trip(c: dict, root: Path = PROCESSED_ROOT) -> Path:
    d = root / c["meta"]["partition"] / c["meta"]["trip_id"]
    d.mkdir(parents=True, exist_ok=True)
    for k in ("track", "nodes", "arcs", "route"):
        c[k].to_csv(d / f"{k}.csv.gz", index=False)
    (d / "meta.json").write_text(json.dumps(c["meta"], indent=1))
    return d


def load_clean_trip(partition: str, trip_id: str, root: Path = PROCESSED_ROOT) -> dict:
    d = root / partition / trip_id
    out = {k: pd.read_csv(d / f"{k}.csv.gz") for k in ("track", "nodes", "arcs", "route")}
    out["meta"] = json.loads((d / "meta.json").read_text())
    return out


def clean_and_save(partition: str, trip_id: str) -> dict:
    c = clean_trip(load_trip(partition, trip_id))
    save_clean_trip(c)
    return c["meta"]
