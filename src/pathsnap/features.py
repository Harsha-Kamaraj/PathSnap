"""Features for every (GPS fix, candidate arc) row, plus the training target.

Features use NO route information (they are computed the same way at inference time).
The target `label` = 1 if the candidate is the fix's reference arc (canonical ids), else 0;
it and the other LABEL_COLS come from the route and must never be model inputs.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .candidates import generate_candidates, point_seg, rank_within
from .labels import align_labels

HEADING_HALF_WIN = 2    # heading from the displacement between fix i-2 and i+2
CONTEXT_HALF_WIN = 5    # window-consistency features look at 5 fixes on each side

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
    # motion / fix quality
    "speed_med_mps", "dt_prev_s", "is_stop", "is_fast_step", "is_off_network",
    # topology
    "is_two_way", "from_in_deg", "to_out_deg", "in_largest_scc",
]


def _seg_bounds(segment_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last fix index of the GPS segment (split at long gaps) of every fix."""
    n = len(segment_id)
    starts = np.r_[0, np.flatnonzero(np.diff(segment_id)) + 1]
    ends = np.r_[starts[1:] - 1, n - 1]
    rep = np.diff(np.r_[starts, n])
    return np.repeat(starts, rep), np.repeat(ends, rep)


def build_trip_features(c: dict, **cand_kw) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidate rows with features + labels for one cleaned trip, and a per-fix table."""
    track, nodes, arcs, route, meta = c["track"], c["nodes"], c["arcs"], c["route"], c["meta"]
    n = len(track)
    cand = generate_candidates(track, nodes, arcs, **cand_kw)
    fi = cand["fix_idx"].to_numpy()
    ai = cand["arc_id"].to_numpy()

    x, y = track["x"].to_numpy(), track["y"].to_numpy()
    seg = track["segment_id"].to_numpy()
    s0, s1 = _seg_bounds(seg)
    idx = np.arange(n)

    # movement heading (compass, radians) from a centred displacement inside the GPS segment
    lo = np.maximum(idx - HEADING_HALF_WIN, s0)
    hi = np.minimum(idx + HEADING_HALF_WIN, s1)
    hdx, hdy = x[hi] - x[lo], y[hi] - y[lo]
    heading = np.arctan2(hdx, hdy)
    hdisp = np.hypot(hdx, hdy)          # small = heading is noise (car stopped)

    # smoothed speed: rolling median, robust to the fast-step runs found in the EDA
    sp = track["speed_mps"].fillna(0).to_numpy()
    sp_med = pd.Series(sp).rolling(2 * CONTEXT_HALF_WIN + 1, center=True, min_periods=1).median().to_numpy()

    nxy = nodes[["x", "y"]].to_numpy()
    fr, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    A, B = nxy[fr[ai]], nxy[to[ai]]
    arc_bear = np.arctan2(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1])
    dh = np.abs((heading[fi] - arc_bear + np.pi) % (2 * np.pi) - np.pi)   # 0..pi, directed

    # distance of the neighbouring fixes (same GPS segment) to the same candidate arc
    def d_at(off: int) -> np.ndarray:
        j = np.clip(fi + off, s0[fi], s1[fi])
        return point_seg(x[j], y[j], A[:, 0], A[:, 1], B[:, 0], B[:, 1])[0]

    win = np.stack([d_at(o) for o in range(-CONTEXT_HALF_WIN, CONTEXT_HALF_WIN + 1)])
    win_mean = win.mean(axis=0)

    dist = cand["dist_m"].to_numpy()
    _, dmin = rank_within(fi, dist)
    wrank, wmin = rank_within(fi, win_mean)
    n_c = np.bincount(fi, minlength=n)[fi]

    pairs = set(zip(fr.tolist(), to.tolist()))
    two_way = np.fromiter(((b, a) in pairs for a, b in zip(fr.tolist(), to.tolist())), bool, len(fr))
    in_deg = np.bincount(to, minlength=len(nodes))
    out_deg = np.bincount(fr, minlength=len(nodes))

    df = pd.DataFrame({
        "partition": meta["partition"], "trip_id": meta["trip_id"],
        "fix_idx": fi, "segment_id": seg[fi], "arc_id": ai,
        "dist_m": dist, "dist_rank": cand["dist_rank"].to_numpy(), "dist_minus_min": dist - dmin,
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
        "is_off_network": track["is_off_network"].to_numpy()[fi],
        "is_two_way": two_way[ai], "from_in_deg": in_deg[fr[ai]], "to_out_deg": out_deg[to[ai]],
        "in_largest_scc": arcs["in_largest_scc"].to_numpy()[ai],
    })

    # ---- labels (route-based; never model inputs) ----
    true_arc = align_labels(track, nodes, arcs, route)
    label = ai == true_arc[fi]                     # candidates are canonical arcs already
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
        "true_arc": true_arc, "has_label": true_arc >= 0, "label_reliable": reliable,
        "exclude_trip": bool(meta.get("exclude_from_training", False)), "true_in_cands": in_c,
    })
    near = df["dist_rank"].to_numpy() == 0
    fixes["nearest_is_true"] = False
    fixes.loc[fi[near], "nearest_is_true"] = label[near]
    return df, fixes


def process_trip(job: tuple[str, str]):
    """Worker for parallel runs (in the module so Windows subprocesses can import it)."""
    from .cleaning import load_clean_trip
    part, tid = job
    t0 = time.perf_counter()
    rows, fixes = build_trip_features(load_clean_trip(part, tid))
    rel = fixes[fixes["label_reliable"]]
    summ = {
        "partition": part, "trip_id": tid, "n_fixes": len(fixes), "n_rows": len(rows),
        "mean_cands": len(rows) / len(fixes),
        "frac_fallback": float(rows.groupby("fix_idx")["is_fallback"].any().mean()),
        "frac_labelled": float(fixes["has_label"].mean()),
        "frac_reliable": float(fixes["label_reliable"].mean()),
        "miss_rate_reliable": float(1 - rel["true_in_cands"].mean()) if len(rel) else float("nan"),
        "nearest_acc_reliable": float(rel["nearest_is_true"].mean()) if len(rel) else float("nan"),
        "exclude_trip": bool(fixes["exclude_trip"].iloc[0]),
        "seconds": time.perf_counter() - t0,
    }
    return part, rows, fixes, summ


def sweep_trip(job: tuple[str, str, tuple, tuple]) -> list[dict]:
    """Candidate miss-rate sweep for one trip (train only). Fixes counted: label_reliable."""
    from .candidates import sweep_counts
    from .cleaning import load_clean_trip
    part, tid, radii, ks = job
    c = load_clean_trip(part, tid)
    true_arc = align_labels(c["track"], c["nodes"], c["arcs"], c["route"])
    use = c["track"]["label_reliable"].to_numpy() & (true_arc >= 0)
    cand_all = generate_candidates(c["track"], c["nodes"], c["arcs"], radius=max(radii), k=None)
    out = sweep_counts(cand_all, true_arc, use, radii, ks)
    for o in out:
        o["trip_id"] = tid
    return out
