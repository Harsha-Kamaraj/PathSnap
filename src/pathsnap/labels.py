"""Per-fix reference labels from the .route file. USES THE ROUTE: training and scoring only.

Problem: `.route` is the ordered list of arcs the car drove. It does not say which GPS fix
was on which arc. "Label = nearest route arc" is wrong on U-turns (the reverse arc is equally
near) and on revisits (the same arc appears twice in the route).

Method: order-respecting alignment by dynamic programming (a Viterbi over route positions).
* State of fix i = a route POSITION k_i (index into the route list, not an arc id), so the
  two visits of a revisited arc are different states.
* Only positions whose arc is within ALIGN_RADIUS_M of the fix are considered (sparse DP).
* Cost of a state = distance from the fix to that route arc.
* Transition from fix i-1 to fix i: staying or moving FORWARD (k_i >= k_{i-1}) is free;
  moving BACKWARD costs ALIGN_BACK_PENALTY_M. The route has 14 known breaks and GPS is noisy,
  so backward moves are allowed but expensive instead of forbidden.
* We keep the cheapest total path and backtrack. Label = canonical arc at the chosen position.
Fixes with no route arc within ALIGN_RADIUS_M get no label (-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from .candidates import point_seg

ALIGN_RADIUS_M = 100.0
ALIGN_BACK_PENALTY_M = 300.0


def _prefix_argmin(cost: np.ndarray) -> np.ndarray:
    """out[j] = index of the smallest cost among cost[0..j]."""
    out = np.zeros(len(cost), dtype=np.int64)
    best = 0
    for j in range(1, len(cost)):
        if cost[j] < cost[best]:
            best = j
        out[j] = best
    return out


def align_labels(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                 route: pd.DataFrame) -> np.ndarray:
    """Canonical reference arc id for every fix (-1 = no label)."""
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
    d, _, _ = point_seg(pxy[fi, 0], pxy[fi, 1], A[ki, 0], A[ki, 1], B[ki, 0], B[ki, 1])
    order = np.lexsort((ki, fi))                       # by fix, then by route position
    fi, ki, d = fi[order], ki[order], d[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(fi)) + 1, len(fi)]

    fixes, states, back = [], [], []
    prev_k = prev_cost = None
    for s, e in zip(bounds[:-1], bounds[1:]):
        k, c = ki[s:e], d[s:e]
        if prev_k is None:
            cost, bp = c.copy(), np.full(len(k), -1)
        else:
            # forward / stay: best previous state with position <= k
            pm = _prefix_argmin(prev_cost)
            pos = np.searchsorted(prev_k, k, side="right") - 1
            fwd_bp = np.where(pos >= 0, pm[np.maximum(pos, 0)], -1)
            fwd = np.where(pos >= 0, prev_cost[np.maximum(fwd_bp, 0)], np.inf)
            # backward: best previous state anywhere, plus the penalty
            g = int(np.argmin(prev_cost))
            bwd = prev_cost[g] + ALIGN_BACK_PENALTY_M
            use_bwd = bwd < fwd
            cost = c + np.where(use_bwd, bwd, fwd)
            bp = np.where(use_bwd, g, fwd_bp)
        fixes.append(fi[s]); states.append(k); back.append(bp)
        prev_k, prev_cost = k, cost

    j = int(np.argmin(prev_cost))                      # backtrack from the cheapest end state
    for step in range(len(fixes) - 1, -1, -1):
        out[fixes[step]] = r_can[states[step][j]]
        j = back[step][j]
    return out
