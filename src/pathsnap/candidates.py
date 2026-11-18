"""Candidate generation (label-free): which road arcs could each GPS fix be on?

For every fix: the candidate arcs (non-zero-length, canonical, i.e. geometric duplicates
removed) within CAND_RADIUS_M, keeping the MAX_CANDS nearest. A fix with nothing in range
gets its single nearest arc (`is_fallback`) so every fix still gets a prediction.

CAND_RADIUS_M and MAX_CANDS were chosen with a miss-rate sweep on the TRAIN partition
only (notebooks/03_features.ipynb, outputs/features/candidate_sweep.csv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

CAND_RADIUS_M = 50.0   # chosen from the train-only sweep (see module docstring)
MAX_CANDS = 16


def point_seg(px, py, ax, ay, bx, by):
    """Distance from points P to segments A->B (all arrays, metres).

    Returns (distance, t, off_end): t in [0, 1] is the position of the closest point along
    the arc, and off_end is how far the perpendicular foot falls beyond the arc's end.
    """
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    safe = np.where(L2 > 0, L2, 1.0)
    t_raw = np.where(L2 > 0, ((px - ax) * vx + (py - ay) * vy) / safe, 0.0)
    t = np.clip(t_raw, 0.0, 1.0)
    d = np.hypot(px - (ax + t * vx), py - (ay + t * vy))
    off_end = np.abs(t_raw - t) * np.sqrt(L2)
    return d, t, off_end


def rank_within(group: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank (0 = smallest) of `value` inside each `group`, and the group minimum."""
    order = np.lexsort((value, group))
    g = group[order]
    first = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    starts = np.repeat(first, np.diff(np.r_[first, len(g)]))
    rank = np.empty(len(g), dtype=np.int64)
    rank[order] = np.arange(len(g)) - starts
    gmin = np.empty(len(g))
    gmin[order] = value[order][starts]
    return rank, gmin


def generate_candidates(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                        radius: float = CAND_RADIUS_M, k: int | None = MAX_CANDS) -> pd.DataFrame:
    """Rows (fix_idx, arc_id, dist_m, proj_t, off_end_m, is_fallback, dist_rank). k=None keeps all."""
    ok = (arcs["is_candidate"] & ~arcs["is_duplicate"]).to_numpy()
    cand_arcs = arcs["arc_id"].to_numpy()[ok]
    nxy = nodes[["x", "y"]].to_numpy()
    A = nxy[arcs["from_node"].to_numpy()[ok]]
    B = nxy[arcs["to_node"].to_numpy()[ok]]
    tree = shapely.STRtree(shapely.linestrings(np.stack([A, B], axis=1)))   # spatial index per trip
    pxy = track[["x", "y"]].to_numpy()
    pts = shapely.points(pxy)

    fi, ai = tree.query(pts, predicate="dwithin", distance=radius)
    fb = np.zeros(len(fi), dtype=bool)
    lonely = np.setdiff1d(np.arange(len(pxy)), fi)       # fixes with nothing within the radius
    if len(lonely):
        lfi, lai = tree.query_nearest(pts[lonely], all_matches=False)
        fi, ai = np.r_[fi, lonely[lfi]], np.r_[ai, lai]
        fb = np.r_[fb, np.ones(len(lfi), dtype=bool)]

    d, t, off = point_seg(pxy[fi, 0], pxy[fi, 1], A[ai, 0], A[ai, 1], B[ai, 0], B[ai, 1])
    rank, _ = rank_within(fi, d)
    keep = rank < k if k is not None else np.ones(len(fi), dtype=bool)
    out = pd.DataFrame({"fix_idx": fi, "arc_id": cand_arcs[ai], "dist_m": d, "proj_t": t,
                        "off_end_m": off, "is_fallback": fb, "dist_rank": rank})[keep]
    return out.sort_values(["fix_idx", "dist_rank"], ignore_index=True)


def sweep_counts(cand_all: pd.DataFrame, true_arc: np.ndarray, use: np.ndarray,
                 radii, ks) -> list[dict]:
    """Miss-rate sweep for one trip. `cand_all` comes from ONE wide query (radius >= max(radii),
    k=None); for each (radius, K) we count, over the fixes in `use`, how many have their true
    arc outside the candidate set and how many candidates the set would hold.

    A fix's true arc is in the (R, K) set if it is within R and among the K nearest,
    or if nothing is within R and it is the nearest arc (the fallback rule)."""
    n = len(true_arc)
    fi = cand_all["fix_idx"].to_numpy()
    d = cand_all["dist_m"].to_numpy()
    r = cand_all["dist_rank"].to_numpy()
    is_true = cand_all["arc_id"].to_numpy() == true_arc[fi]
    t_rank = np.full(n, -1); t_dist = np.full(n, np.inf)
    t_rank[fi[is_true]] = r[is_true]; t_dist[fi[is_true]] = d[is_true]
    out = []
    for R in radii:
        within = np.bincount(fi[d <= R], minlength=n)            # arcs within R per fix
        for K in ks:
            hit = ((t_dist <= R) & (t_rank >= 0) & (t_rank < K)) | ((within == 0) & (t_rank == 0))
            size = np.where(within == 0, 1, np.minimum(within, K))
            out.append({"radius_m": R, "k": K, "fixes": int(use.sum()),
                        "missed": int((use & ~hit).sum()), "cands": int(size[use].sum())})
    return out
