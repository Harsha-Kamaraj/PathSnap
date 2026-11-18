"""Step 2 + 3: candidate generation, per-fix reference labels and candidate features.

Pipeline per cleaned trip (output of cleaning.py):

1. Candidates (label-free): for every GPS fix, the candidate arcs (non-zero-length,
   canonical, i.e. geometric duplicates removed) within CAND_RADIUS_M, keeping the
   MAX_CANDS nearest. A fix with nothing in range gets its single nearest arc
   (`is_fallback`) so that every fix still has a prediction.
2. Reference label (uses .route, training / scoring only): the route is an ordered
   arc sequence with no fix-to-arc link, so each fix is aligned to a route POSITION
   with a monotone dynamic programme (positions never go back, except with a large
   penalty to survive route breaks). This handles revisits and U-turns, which a
   plain "nearest route arc" rule gets wrong. Label = canonical arc at that position.
3. Features (label-free) for every (fix, candidate) row: geometry, heading agreement,
   window consistency (how close the neighbouring fixes are to the same arc),
   ranks within the fix, motion and arc topology.

Columns in LABEL_COLS use the route and must never be model inputs.
All thresholds are hand-set (not tuned on student-test / evaluator data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

# ---- hand-set parameters ----
CAND_RADIUS_M = 100.0   # EDA: GPS-to-road p95 ~ 81 m incl. parked trips, typical sigma 5-10 m
MAX_CANDS = 16          # ~8 roads (two-way roads give 2 arcs each)
HEADING_HALF_WIN = 2    # fixes each side used for the centred heading estimate
CONTEXT_HALF_WIN = 5    # fixes each side used for window-consistency features
ALIGN_RADIUS_M = 100.0  # route positions considered for a fix in the label alignment
ALIGN_BACK_PENALTY_M = 300.0  # cost of moving backwards along the route (route breaks / noise)

KEY_COLS = ["partition", "trip_id", "fix_idx", "segment_id", "arc_id"]
LABEL_COLS = ["label", "true_arc", "has_label", "label_reliable", "exclude_trip", "true_in_cands"]
FEATURE_COLS = [
    # geometry
    "dist_m", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands", "is_fallback",
    "proj_t", "off_end_m", "arc_length_m",
    # direction
    "heading_diff_deg", "heading_cos", "heading_disp_m",
    # temporal consistency
    "d_prev_m", "d_next_m", "d_win_mean_m", "d_win_max_m", "win_mean_rank", "win_mean_minus_min",
    # motion
    "speed_med_mps", "dt_prev_s", "is_stop", "is_fast_step",
    # topology
    "is_two_way", "from_in_deg", "to_out_deg", "in_largest_scc",
]


def _seg_bounds(segment_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last fix index of the GPS segment each fix belongs to."""
    n = len(segment_id)
    starts = np.r_[0, np.flatnonzero(np.diff(segment_id)) + 1]
    ends = np.r_[starts[1:] - 1, n - 1]
    rep = np.diff(np.r_[starts, n])
    return np.repeat(starts, rep), np.repeat(ends, rep)


def _point_seg(px, py, ax, ay, bx, by):
    """Distance from points to segments A->B, clamped position t in [0,1] and overshoot."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t_raw = np.where(L2 > 0, ((px - ax) * vx + (py - ay) * vy) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t_raw, 0.0, 1.0)
    d = np.hypot(px - (ax + t * vx), py - (ay + t * vy))
    off_end = np.abs(t_raw - t) * np.sqrt(L2)
    return d, t, off_end


def _rank_within(group: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank (0 = smallest) of value within each group, and group minimum. group must be sorted."""
    order = np.lexsort((value, group))
    g = group[order]
    first = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    starts = np.repeat(first, np.diff(np.r_[first, len(g)]))
    rank = np.empty(len(g), dtype=np.int64)
    rank[order] = np.arange(len(g)) - starts
    gmin = np.empty(len(g))
    gmin[order] = value[order][starts]
    return rank, gmin


# ---------------------------------------------------------------- candidates
def generate_candidates(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                        radius: float = CAND_RADIUS_M, k: int = MAX_CANDS) -> pd.DataFrame:
    """(fix_idx, arc_id, dist_m, proj_t, off_end_m, is_fallback) rows. Label-free."""
    ok = (arcs["is_candidate"] & ~arcs["is_duplicate"]).to_numpy()
    cand_arcs = arcs["arc_id"].to_numpy()[ok]
    nxy = nodes[["x", "y"]].to_numpy()
    A = nxy[arcs["from_node"].to_numpy()[ok]]
    B = nxy[arcs["to_node"].to_numpy()[ok]]
    tree = shapely.STRtree(shapely.linestrings(np.stack([A, B], axis=1)))
    pxy = track[["x", "y"]].to_numpy()
    pts = shapely.points(pxy)

    fi, ai = tree.query(pts, predicate="dwithin", distance=radius)
    fallback = np.setdiff1d(np.arange(len(pxy)), fi)
    fb = np.zeros(len(fi), dtype=bool)
    if len(fallback):
        (ffi, fai) = tree.query_nearest(pts[fallback], all_matches=False)
        fi = np.r_[fi, fallback[ffi]]
        ai = np.r_[ai, fai]
        fb = np.r_[fb, np.ones(len(ffi), dtype=bool)]

    d, t, off = _point_seg(pxy[fi, 0], pxy[fi, 1], A[ai, 0], A[ai, 1], B[ai, 0], B[ai, 1])
    order = np.lexsort((d, fi))
    fi, ai, d, t, off, fb = fi[order], ai[order], d[order], t[order], off[order], fb[order]
    first = np.r_[0, np.flatnonzero(np.diff(fi)) + 1]
    rank = np.arange(len(fi)) - np.repeat(first, np.diff(np.r_[first, len(fi)]))
    keep = rank < k
    return pd.DataFrame({"fix_idx": fi[keep], "arc_id": cand_arcs[ai[keep]], "dist_m": d[keep],
                         "proj_t": t[keep], "off_end_m": off[keep], "is_fallback": fb[keep]})


# ---------------------------------------------------------------- labels
def align_labels(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                 route: pd.DataFrame) -> np.ndarray:
    """Canonical reference arc for every fix (-1 = could not be aligned). USES THE ROUTE.

    Monotone DP over route positions: state of fix i = route position k_i, cost = distance
    from the fix to route arc k_i, moving forward is free, moving back costs
    ALIGN_BACK_PENALTY_M. Only positions within ALIGN_RADIUS_M are considered (sparse).
    """
    n = len(track)
    out = np.full(n, -1, dtype=np.int64)
    if route.empty:
        return out
    r_arc = route["arc_id"].to_numpy()
    r_can = route["canonical_arc_id"].to_numpy()
    nxy = nodes[["x", "y"]].to_numpy()
    A = nxy[arcs["from_node"].to_numpy()[r_arc]]
    B = nxy[arcs["to_node"].to_numpy()[r_arc]]
    tree = shapely.STRtree(shapely.linestrings(np.stack([A, B], axis=1)))
    pxy = track[["x", "y"]].to_numpy()
    fi, ki = tree.query(shapely.points(pxy), predicate="dwithin", distance=ALIGN_RADIUS_M)
    if len(fi) == 0:
        return out
    d, _, _ = _point_seg(pxy[fi, 0], pxy[fi, 1], A[ki, 0], A[ki, 1], B[ki, 0], B[ki, 1])
    order = np.lexsort((ki, fi))
    fi, ki, d = fi[order], ki[order], d[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(fi)) + 1, len(fi)]

    fixes, states, back = [], [], []
    prev_k = prev_cost = None
    for s, e in zip(bounds[:-1], bounds[1:]):
        k, c = ki[s:e], d[s:e]
        if prev_k is None:
            cost, bp = c.copy(), np.full(len(k), -1)
        else:
            # best predecessor with k' <= k (prefix min over prev states sorted by k)
            pm_idx = np.zeros(len(prev_cost), dtype=np.int64)
            best = 0
            for j in range(1, len(prev_cost)):
                if prev_cost[j] < prev_cost[best]:
                    best = j
                pm_idx[j] = best
            pos = np.searchsorted(prev_k, k, side="right") - 1
            fwd = np.where(pos >= 0, prev_cost[pm_idx[np.maximum(pos, 0)]], np.inf)
            fwd_bp = np.where(pos >= 0, pm_idx[np.maximum(pos, 0)], -1)
            gbest = int(np.argmin(prev_cost))
            bwd = prev_cost[gbest] + ALIGN_BACK_PENALTY_M
            use_bwd = bwd < fwd
            cost = c + np.where(use_bwd, bwd, fwd)
            bp = np.where(use_bwd, gbest, fwd_bp)
        fixes.append(fi[s]); states.append(k); back.append(bp)
        prev_k, prev_cost = k, cost

    j = int(np.argmin(prev_cost))
    for step in range(len(fixes) - 1, -1, -1):
        out[fixes[step]] = r_can[states[step][j]]
        j = back[step][j]
    return out


# ---------------------------------------------------------------- features
def build_trip_features(c: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidate-row features + labels for one cleaned trip, and a per-fix summary table."""
    track, nodes, arcs, route, meta = c["track"], c["nodes"], c["arcs"], c["route"], c["meta"]
    n = len(track)
    cand = generate_candidates(track, nodes, arcs)
    fi = cand["fix_idx"].to_numpy()
    ai = cand["arc_id"].to_numpy()

    x, y = track["x"].to_numpy(), track["y"].to_numpy()
    seg = track["segment_id"].to_numpy()
    s0, s1 = _seg_bounds(seg)
    idx = np.arange(n)

    # centred heading from displacement, within the GPS segment only
    lo = np.maximum(idx - HEADING_HALF_WIN, s0)
    hi = np.minimum(idx + HEADING_HALF_WIN, s1)
    hdx, hdy = x[hi] - x[lo], y[hi] - y[lo]
    heading = np.arctan2(hdx, hdy)
    hdisp = np.hypot(hdx, hdy)

    # smoothed speed: median of step speeds in the context window (robust to fast-step runs)
    sp = track["speed_mps"].fillna(0).to_numpy()
    sp_med = pd.Series(sp).rolling(2 * CONTEXT_HALF_WIN + 1, center=True, min_periods=1).median().to_numpy()

    nxy = nodes[["x", "y"]].to_numpy()
    fr, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    A, B = nxy[fr[ai]], nxy[to[ai]]
    arc_bear = np.arctan2(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1])
    dh = np.abs((heading[fi] - arc_bear + np.pi) % (2 * np.pi) - np.pi)

    # neighbouring fixes' distance to the same candidate arc
    def d_at(off: int) -> np.ndarray:
        j = np.clip(fi + off, s0[fi], s1[fi])
        return _point_seg(x[j], y[j], A[:, 0], A[:, 1], B[:, 0], B[:, 1])[0]

    win = np.stack([d_at(o) for o in range(-CONTEXT_HALF_WIN, CONTEXT_HALF_WIN + 1)])
    win_mean = win.mean(axis=0)

    dist = cand["dist_m"].to_numpy()
    rank, dmin = _rank_within(fi, dist)
    wrank, wmin = _rank_within(fi, win_mean)
    n_c = np.bincount(fi, minlength=n)[fi]

    pairs = set(zip(fr.tolist(), to.tolist()))
    two_way = np.fromiter(((b, a) in pairs for a, b in zip(fr.tolist(), to.tolist())), bool, len(fr))
    in_deg = np.bincount(to, minlength=len(nodes))
    out_deg = np.bincount(fr, minlength=len(nodes))

    df = pd.DataFrame({
        "partition": meta["partition"], "trip_id": meta["trip_id"],
        "fix_idx": fi, "segment_id": seg[fi], "arc_id": ai,
        "dist_m": dist, "dist_rank": rank, "dist_minus_min": dist - dmin,
        "dist_ratio": (dist + 1.0) / (dmin + 1.0), "n_cands": n_c,
        "is_fallback": cand["is_fallback"].to_numpy(),
        "proj_t": cand["proj_t"].to_numpy(), "off_end_m": cand["off_end_m"].to_numpy(),
        "arc_length_m": arcs["length_m"].to_numpy()[ai],
        "heading_diff_deg": np.degrees(dh), "heading_cos": np.cos(dh), "heading_disp_m": hdisp[fi],
        "d_prev_m": win[CONTEXT_HALF_WIN - 1], "d_next_m": win[CONTEXT_HALF_WIN + 1],
        "d_win_mean_m": win_mean, "d_win_max_m": win.max(axis=0),
        "win_mean_rank": wrank, "win_mean_minus_min": win_mean - wmin,
        "speed_med_mps": sp_med[fi], "dt_prev_s": track["dt_prev"].fillna(1.0).to_numpy()[fi],
        "is_stop": track["is_stop"].to_numpy()[fi], "is_fast_step": track["is_fast_step"].to_numpy()[fi],
        "is_two_way": two_way[ai], "from_in_deg": in_deg[fr[ai]], "to_out_deg": out_deg[to[ai]],
        "in_largest_scc": arcs["in_largest_scc"].to_numpy()[ai],
    })

    # ---- labels (route-based; never model inputs) ----
    true_arc = align_labels(track, nodes, arcs, route)
    label = ai == true_arc[fi]                    # candidates are canonical arcs already
    in_c = np.zeros(n, dtype=bool)
    in_c[fi[label]] = True
    reliable = track["label_reliable"].to_numpy() & (true_arc >= 0)
    df["label"] = label.astype(np.int8)
    df["true_arc"] = true_arc[fi]
    df["has_label"] = true_arc[fi] >= 0
    df["label_reliable"] = reliable[fi]
    df["exclude_trip"] = bool(meta.get("exclude_from_training", False))
    df["true_in_cands"] = in_c[fi]

    fixes = pd.DataFrame({
        "partition": meta["partition"], "trip_id": meta["trip_id"], "fix_idx": idx,
        "segment_id": seg, "n_cands": np.bincount(fi, minlength=n),
        "true_arc": true_arc, "label_reliable": reliable, "true_in_cands": in_c,
        "nearest_is_true": np.zeros(n, dtype=bool),
    })
    near = rank == 0
    fixes.loc[fi[near], "nearest_is_true"] = label[near]
    return df, fixes


def process_trip(job: tuple[str, str]):
    """Worker for parallel runs (lives in the module so Windows subprocesses can import it)."""
    import time
    from .cleaning import load_clean_trip
    part, tid = job
    t0 = time.perf_counter()
    rows, fixes = build_trip_features(load_clean_trip(part, tid))
    rel = fixes[fixes["label_reliable"]]
    summ = {
        "partition": part, "trip_id": tid, "n_fixes": len(fixes), "n_rows": len(rows),
        "mean_cands": len(rows) / len(fixes),
        "frac_fallback": float(rows.groupby("fix_idx")["is_fallback"].any().mean()),
        "frac_labelled": float((fixes["true_arc"] >= 0).mean()),
        "frac_reliable": float(fixes["label_reliable"].mean()),
        "miss_rate_reliable": float(1 - rel["true_in_cands"].mean()) if len(rel) else float("nan"),
        "nearest_acc_reliable": float(rel["nearest_is_true"].mean()) if len(rel) else float("nan"),
        "exclude_trip": bool(rows["exclude_trip"].iloc[0]),
        "seconds": time.perf_counter() - t0,
    }
    return part, rows, fixes, summ
