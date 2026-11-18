"""HMM + Viterbi route reconstruction (step 6).

The classifier scores every (GPS fix, candidate arc) pair on its own. Neighbouring fixes can then
be matched to arcs that are not connected, e.g. a jump to the reverse direction of a two-way
street or to a parallel road. The HMM picks ONE arc per fix so that the whole sequence is both
likely under the classifier AND drivable on the directed road network.

Model
-----
* Hidden state of fix t: one of its candidate arcs (candidates.py; canonical arc ids, so the
  state IS the road segment that is scored, nothing is re-mapped).
* Emission  log E_t(j) = log max(p_tj, p_floor)  [ - (d_tj / sigma)^2 / 2  if use_dist ]
  p_tj = classifier probability that candidate j is the true arc of fix t, d_tj = its distance.
  (Normalising p over the candidates of a fix would add the same constant to every state of
  that fix, which does not change the Viterbi path, so the raw probabilities are used.)
* Transition  log A_t(i, j) = -cost(i -> j) between candidate i of fix t-1 and j of fix t:
      route = driving distance along DIRECTED arcs from the point of i to the point of j
                (same arc: the forward distance along it; reverse arc: a U-turn on the spot;
                 otherwise: rest of arc i + shortest path + start of arc j)
      gps   = straight-line distance between the two GPS fixes
      cost  = min(|route - gps| / beta, no_path_pen)      (no path within the bound = no_path_pen)
            + uturn_pen   if j is the reverse arc of i
            + stop_pen    if both fixes are in the same stop and the arc changes
            + speed_pen   if route / dt > V_MAX_MPS and the GPS step itself is not a known
                          "fast step" (raw speeds are not trusted, see cleaning.py)
  A transition is never -inf, so a chain can always continue (no crash, no "all impossible").
* Viterbi (log space), for one chain of fixes 0..T-1:
      delta_0(j) = log E_0(j)
      delta_t(j) = log E_t(j) + max_i [ delta_{t-1}(i) + log A_t(i, j) ]
      psi_t(j)   = argmax_i [ delta_{t-1}(i) + log A_t(i, j) ]
      path: j_{T-1} = argmax_j delta_{T-1}(j),  j_{t-1} = psi_t(j_t)
* Chains: a new chain starts at every `segment_id` change (GPS gap > 120 s). A fix with no
  candidate within the search radius only has its "fallback" nearest arc: it is left out of the
  chain (the chain is carried over it), it is predicted as that nearest arc and flagged.

The emission weight is fixed at 1: Viterbi only depends on how emission and transition costs
compare, and beta (with the penalties) already sets that balance.

Everything here uses label-free inputs only. Labels are used in notebooks/07 for scoring.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_DETOUR_M = 200.0     # search the network up to gps + max(MIN_DETOUR_M, gps) metres; beyond = "no path"
V_MAX_MPS = 50.0         # 180 km/h; a faster implied driving speed gets speed_pen

# starting point (trial 0 of the decoder search); tuned values go to models/hmm_config.json
DEFAULT_PARAMS = dict(beta_m=5.0, no_path_pen=20.0, uturn_pen=5.0, stop_pen=5.0, speed_pen=5.0,
                      p_floor=1e-4, use_dist=False, sigma_m=10.0)

# decoder search space (Optuna TPE, train out-of-fold probabilities only)
SEARCH_SPACE = {
    "beta_m":      {"type": "float", "low": 1.0, "high": 100.0, "log": True},
    "no_path_pen": {"type": "float", "low": 2.0, "high": 50.0, "log": True},
    "uturn_pen":   {"type": "float", "low": 0.0, "high": 20.0},
    "stop_pen":    {"type": "float", "low": 0.0, "high": 20.0},
    "speed_pen":   {"type": "float", "low": 0.0, "high": 20.0},
    "p_floor":     {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
    "use_dist":    {"type": "categorical", "choices": [False, True]},
    "sigma_m":     {"type": "float", "low": 3.0, "high": 50.0, "log": True},
}


# =====================================================================================
#                         ROAD NETWORK + DISTANCE-BOUNDED DIJKSTRA
# =====================================================================================
@dataclass
class Network:
    """Directed road network of one trip, with an adjacency list for Dijkstra."""
    from_node: np.ndarray
    to_node: np.ndarray
    length: np.ndarray
    canonical: np.ndarray       # arc id -> canonical arc id (geometric duplicates merged)
    node_lon: np.ndarray
    node_lat: np.ndarray
    out_ptr: list               # CSR: arcs leaving node u are out_arc[out_ptr[u]:out_ptr[u+1]]
    out_arc: list

    @classmethod
    def from_frames(cls, nodes: pd.DataFrame, arcs: pd.DataFrame) -> "Network":
        fr, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
        order = np.argsort(fr, kind="stable")
        ptr = np.searchsorted(fr[order], np.arange(len(nodes) + 1))
        return cls(fr, to, arcs["length_m"].to_numpy(), arcs["canonical_arc_id"].to_numpy(),
                   nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), ptr.tolist(), order.tolist())


def bounded_dijkstra(net: Network, src: int, limit: float) -> tuple[dict, dict]:
    """Shortest driving distance (metres, along arc directions) from node `src` to every node
    reachable within `limit`. Returns {node: distance} and {node: last arc on the path}."""
    to, length = net.to_node, net.length
    ptr, arcs_out = net.out_ptr, net.out_arc
    dist, pred = {src: 0.0}, {src: -1}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for k in range(ptr[u], ptr[u + 1]):
            a = arcs_out[k]
            v, nd = int(to[a]), d + float(length[a])
            if nd <= limit and nd < dist.get(v, np.inf):
                dist[v], pred[v] = nd, a
                heapq.heappush(heap, (nd, v))
    return dist, pred


def arc_path(net: Network, a: int, b: int, limit: float) -> list[int] | None:
    """Canonical arcs driven strictly between arc a and arc b (shortest path from the end of a
    to the start of b). [] if b follows a directly or is its reverse; None if not reachable."""
    if a == b or net.to_node[a] == net.from_node[b]:      # same arc, next arc, or the reverse arc
        return []
    dist, pred = bounded_dijkstra(net, int(net.to_node[a]), limit)
    tgt = int(net.from_node[b])
    if tgt not in dist:
        return None
    path, u = [], tgt
    while pred[u] != -1:
        path.append(int(net.canonical[pred[u]]))
        u = int(net.from_node[pred[u]])
    return path[::-1]


# =====================================================================================
#                 PER-TRIP STATE SPACE (everything that does not depend on params)
# =====================================================================================
@dataclass
class TripHMM:
    """Hidden states and pairwise route distances of one trip, computed once per trip.

    Fixes (n): fix_idx, cand_ptr (candidates of fix f are cand_*[cand_ptr[f]:cand_ptr[f+1]]).
    Chain order (m fixes that take part in Viterbi): chain_fix (positions into the fix arrays),
    new_chain (True = first fix of a chain). Step k joins chain_fix[k-1] -> chain_fix[k]; its
    candidate pairs (i, j), row-major, are pair_*[pair_ptr[k]:pair_ptr[k+1]] (empty if new_chain).
    """
    trip_id: str
    fix_idx: np.ndarray
    cand_ptr: np.ndarray
    cand_arc: np.ndarray
    cand_dist: np.ndarray
    cand_row: np.ndarray        # position of the candidate in the trip's rows (-> probability)
    in_chain: np.ndarray        # False = carried over (no candidate within the radius)
    chain_fix: np.ndarray
    new_chain: np.ndarray
    pair_ptr: np.ndarray
    pair_route: np.ndarray      # driving distance (m), inf = no path within the bound
    pair_same: np.ndarray       # same arc
    pair_uturn: np.ndarray      # j is the reverse arc of i
    step_gps: np.ndarray        # straight-line GPS distance of the step (m)
    step_dt: np.ndarray
    step_stop: np.ndarray       # both fixes in the same stop
    step_fast: np.ndarray       # the GPS step is flagged is_fast_step (speed not trustworthy)
    build_s: float = 0.0

    @property
    def n_pairs(self) -> int:
        return len(self.pair_route)


def build_trip_hmm(trip_id: str, track: pd.DataFrame, net: Network, rows: pd.DataFrame) -> TripHMM:
    """`rows`: candidate rows of this trip (fix_idx, arc_id, dist_m, proj_t, is_fallback),
    in any order; probabilities are later looked up by row position."""
    t0 = time.perf_counter()
    order = np.lexsort((rows["dist_m"].to_numpy(), rows["fix_idx"].to_numpy()))
    r_fix = rows["fix_idx"].to_numpy()[order]
    fix_idx = np.unique(r_fix)
    cand_ptr = np.r_[np.searchsorted(r_fix, fix_idx), len(r_fix)]
    cand_arc = rows["arc_id"].to_numpy()[order]
    cand_t = rows["proj_t"].to_numpy()[order]
    fallback = rows["is_fallback"].to_numpy()[order]

    tr = track.set_index("fix_idx").loc[fix_idx]
    x, y, t = tr["x"].to_numpy(), tr["y"].to_numpy(), tr["t"].to_numpy()
    seg, stop, stop_id = tr["segment_id"].to_numpy(), tr["is_stop"].to_numpy(), tr["stop_id"].to_numpy()
    fast = tr["is_fast_step"].to_numpy()

    in_chain = ~fallback[cand_ptr[:-1]]          # fallback fixes have exactly one (fallback) candidate
    chain_fix = np.flatnonzero(in_chain)
    new_chain = np.ones(len(chain_fix), dtype=bool)
    new_chain[1:] = seg[chain_fix[1:]] != seg[chain_fix[:-1]]

    u, v = chain_fix[:-1], chain_fix[1:]
    gps = np.r_[0.0, np.hypot(x[v] - x[u], y[v] - y[u])]
    dt = np.r_[1.0, t[v] - t[u]]
    same_stop = np.r_[False, stop[u] & stop[v] & (stop_id[u] == stop_id[v]) & (stop_id[u] >= 0)]
    # the step is "fast" if any GPS step between the two chain fixes is flagged
    fast_cum = np.cumsum(fast)
    step_fast = np.r_[False, (fast_cum[v] - fast_cum[u]) > 0]
    limit = gps + np.maximum(MIN_DETOUR_M, gps)

    fr, to, L = net.from_node, net.to_node, net.length
    # pass 1: how far each source node must be searched
    need: dict[int, float] = {}
    for k in np.flatnonzero(~new_chain):
        a = cand_arc[cand_ptr[chain_fix[k - 1]]:cand_ptr[chain_fix[k - 1] + 1]]
        for s in to[a].tolist():
            if limit[k] > need.get(s, -1.0):
                need[s] = limit[k]
    sp = {s: bounded_dijkstra(net, s, lim)[0] for s, lim in need.items()}

    # pass 2: route distance for every candidate pair of every step
    ptr = [0]
    routes, sames, uturns = [], [], []
    for k in range(len(chain_fix)):
        if new_chain[k]:
            ptr.append(ptr[-1])
            continue
        fu, fv = chain_fix[k - 1], chain_fix[k]
        ia = slice(cand_ptr[fu], cand_ptr[fu + 1])
        ib = slice(cand_ptr[fv], cand_ptr[fv + 1])
        a, b = cand_arc[ia], cand_arc[ib]
        ta, tb = cand_t[ia][:, None], cand_t[ib][None, :]
        La, Lb = L[a][:, None], L[b][None, :]
        same = a[:, None] == b[None, :]
        uturn = (fr[b][None, :] == to[a][:, None]) & (to[b][None, :] == fr[a][:, None]) & ~same
        spm = np.array([[sp[s].get(e, np.inf) for e in fr[b].tolist()] for s in to[a].tolist()])
        route = (1 - ta) * La + spm + tb * Lb
        route = np.where(same, np.maximum(tb - ta, 0.0) * La, route)      # backward jitter = standing still
        route = np.where(uturn, np.minimum(np.abs(ta * La - (1 - tb) * Lb), route), route)  # U-turn on the spot
        route = np.where(route > limit[k], np.inf, route)
        routes.append(route.ravel()); sames.append(same.ravel()); uturns.append(uturn.ravel())
        ptr.append(ptr[-1] + route.size)

    cat = lambda xs, dt_: np.concatenate(xs).astype(dt_) if xs else np.empty(0, dt_)
    return TripHMM(trip_id, fix_idx, cand_ptr, cand_arc, rows["dist_m"].to_numpy()[order], order, in_chain,
                   chain_fix, new_chain, np.asarray(ptr), cat(routes, np.float32), cat(sames, bool),
                   cat(uturns, bool), gps, dt, same_stop, step_fast, time.perf_counter() - t0)


# =====================================================================================
#                          EMISSION, TRANSITION AND VITERBI
# =====================================================================================
def log_emission(h: TripHMM, p_rows: np.ndarray, params: dict) -> np.ndarray:
    """log E for every candidate (flat, in TripHMM candidate order)."""
    le = np.log(np.maximum(p_rows[h.cand_row], params["p_floor"]))
    if params.get("use_dist", False):
        le = le - 0.5 * (h.cand_dist / params["sigma_m"]) ** 2
    return le


def log_transition(h: TripHMM, params: dict) -> np.ndarray:
    """log A for every candidate pair of every step (flat). Always finite."""
    steps = np.repeat(np.arange(len(h.chain_fix)), np.diff(h.pair_ptr))
    gps, dt = h.step_gps[steps], h.step_dt[steps]
    r = h.pair_route.astype(np.float64)
    reach = np.isfinite(r)
    excess = np.where(reach, np.abs(r - gps), np.inf)
    cost = np.minimum(excess / params["beta_m"], params["no_path_pen"])
    cost += params["uturn_pen"] * h.pair_uturn
    cost += params["stop_pen"] * (h.step_stop[steps] & ~h.pair_same)
    too_fast = reach & (np.where(reach, r, 0.0) / np.maximum(dt, 1e-6) > V_MAX_MPS) & ~h.step_fast[steps]
    cost += params["speed_pen"] * too_fast
    return -cost


def viterbi(h: TripHMM, le: np.ndarray, la: np.ndarray) -> np.ndarray:
    """Chosen candidate (flat index) for every fix. Recurrence in the module docstring."""
    choice = np.empty(len(h.fix_idx), dtype=np.int64)
    choice[~h.in_chain] = h.cand_ptr[:-1][~h.in_chain]          # carried-over fix: its only (nearest) arc
    cp, pp = h.cand_ptr, h.pair_ptr
    m = len(h.chain_fix)
    psi = [None] * m
    k = 0
    while k < m:                                                # one chain at a time
        f = h.chain_fix[k]
        delta = le[cp[f]:cp[f + 1]]
        start = k
        k += 1
        while k < m and not h.new_chain[k]:
            f = h.chain_fix[k]
            n_prev, n_cur = len(delta), cp[f + 1] - cp[f]
            score = delta[:, None] + la[pp[k]:pp[k + 1]].reshape(n_prev, n_cur)
            best_i = score.argmax(axis=0)
            psi[k] = best_i
            delta = score[best_i, np.arange(n_cur)] + le[cp[f]:cp[f + 1]]
            k += 1
        j = int(delta.argmax())                                 # backtrack this chain
        for q in range(k - 1, start - 1, -1):
            choice[h.chain_fix[q]] = cp[h.chain_fix[q]] + j
            if q > start:
                j = int(psi[q][j])
    return choice


def decode(h: TripHMM, p_rows: np.ndarray, params: dict) -> np.ndarray:
    """HMM-predicted canonical arc for every fix in h.fix_idx."""
    return h.cand_arc[viterbi(h, log_emission(h, p_rows, params), log_transition(h, params))]


def classifier_only(h: TripHMM, p_rows: np.ndarray) -> np.ndarray:
    """Highest-probability candidate of every fix (no sequence information)."""
    p = p_rows[h.cand_row]
    out = np.empty(len(h.fix_idx), dtype=np.int64)
    for f in range(len(h.fix_idx)):
        s, e = h.cand_ptr[f], h.cand_ptr[f + 1]
        out[f] = h.cand_arc[s + int(np.argmax(p[s:e]))]
    return out


# =====================================================================================
#                              CONNECTED PATH FOR THE MAP
# =====================================================================================
def fill_path(h: TripHMM, net: Network, pred: np.ndarray) -> list[list[int]]:
    """Turn per-fix arcs into connected pieces of arcs: between consecutive chain fixes whose arcs
    differ, insert the shortest driving path. A new piece starts at each chain start and wherever
    no path exists within the search bound."""
    pieces, cur = [], []
    cache: dict = {}
    for k, f in enumerate(h.chain_fix):
        a = int(pred[f])
        if h.new_chain[k] or not cur:
            if cur:
                pieces.append(cur)
            cur = [a]
            continue
        prev = cur[-1]
        if a == prev:
            continue
        key = (prev, a)
        if key not in cache:
            g = h.step_gps[k]
            cache[key] = arc_path(net, prev, a, g + max(MIN_DETOUR_M, g))
        mid = cache[key]
        if mid is None:
            pieces.append(cur)
            cur = [a]
        else:
            cur.extend(mid + [a])
    if cur:
        pieces.append(cur)
    return pieces


# =====================================================================================
#                      INFERENCE FOR ONE TRIP (label-free; for the API)
# =====================================================================================
def build_trip_hmm_job(job):
    """Parallel worker: (partition, trip_id, rows) -> TripHMM (in the module for Windows)."""
    from .cleaning import load_clean_trip
    part, tid, rows = job
    c = load_clean_trip(part, tid)
    return build_trip_hmm(tid, c["track"], Network.from_frames(c["nodes"], c["arcs"]), rows)


def predict_trip(partition: str, trip_id: str, model, params: dict, root=None) -> dict:
    """Full pipeline for one trip from the RAW files, using only .track/.nodes/.arcs:
    clean -> candidates + features -> classifier -> HMM/Viterbi -> connected path.
    The .route file is never read here (ground truth stays with the scorer)."""
    from .cleaning import clean_trip
    from .data_io import load_trip
    from .features import FEATURE_COLS, build_trip_features

    timing = {}
    t0 = time.perf_counter()
    raw = load_trip(partition, trip_id, root)
    raw.route = np.empty(0, dtype=np.int64)                     # label-free by construction
    c = clean_trip(raw)
    rows, _ = build_trip_features(c)
    timing["clean_features_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    p = model.predict_proba(rows[FEATURE_COLS].astype("float64"))[:, 1]
    timing["classifier_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    net = Network.from_frames(c["nodes"], c["arcs"])
    h = build_trip_hmm(trip_id, c["track"], net, rows)
    timing["hmm_build_s"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    hmm_arc = decode(h, p, params)
    timing["viterbi_s"] = time.perf_counter() - t3
    clf_arc = classifier_only(h, p)
    pieces = fill_path(h, net, hmm_arc)
    timing["total_s"] = time.perf_counter() - t0

    tr = c["track"].set_index("fix_idx").loc[h.fix_idx]
    fixes = pd.DataFrame({"fix_idx": h.fix_idx, "lon": tr["lon"].to_numpy(), "lat": tr["lat"].to_numpy(),
                          "t": tr["t"].to_numpy(), "segment_id": tr["segment_id"].to_numpy(),
                          "pred_arc_classifier": clf_arc, "pred_arc_hmm": hmm_arc,
                          "carried_over": ~h.in_chain})
    path = [{"arcs": pc, "coords": arcs_to_coords(net, pc)} for pc in pieces]
    return {"trip_id": trip_id, "fixes": fixes, "path": path, "timing": timing}


def arcs_to_coords(net: Network, arcs: list[int]) -> list[list[float]]:
    """[lon, lat] polyline through a list of connected arcs."""
    if not arcs:
        return []
    nodes = [int(net.from_node[arcs[0]])] + [int(net.to_node[a]) for a in arcs]
    return [[float(net.node_lon[n]), float(net.node_lat[n])] for n in nodes]


# =====================================================================================
#           SCORING (uses the reference labels; notebooks only, never predict_trip)
# =====================================================================================
def score_predictions(pred: pd.DataFrame, fixes: pd.DataFrame, rows: pd.DataFrame, arcs_by_trip: dict,
                      ref_arcs: dict, arc_len: dict, filled: dict | None = None) -> dict:
    """Metrics for one prediction (one row per fix: trip_id, fix_idx, pred_arc).

    * top1: share of fixes whose predicted arc is the reference arc ("all" = labelled fixes of
      non-excluded trips, "reliable" = label_reliable fixes), same sets as training.evaluate.
    * sel_macro_f1: candidate-row macro F1 when the chosen candidate of each fix is "positive" and
      all others "negative". Same definition for the classifier (argmax) and the HMM (Viterbi), so
      the two are comparable. (training.evaluate's macro F1 thresholds p at 0.5 instead.)
    * route_continuity: as in training.route_continuity.
    * seq_*: sequence level. Set of predicted arcs of a trip vs the set of reference route arcs,
      weighted by arc length: precision = predicted length on the route / predicted length,
      recall = reference length that was predicted / reference length. Pooled over trips.
      "filled" uses the connected path (fill_path) instead of the per-fix arcs.
    """
    from sklearn.metrics import f1_score
    from .training import route_continuity

    t = pred.merge(fixes[["trip_id", "fix_idx", "segment_id", "true_arc", "has_label", "label_reliable",
                          "exclude_trip"]], on=["trip_id", "fix_idx"], how="left")
    t["correct"] = t["pred_arc"] == t["true_arc"]
    out = {"route_continuity": route_continuity(t, arcs_by_trip)}
    r = rows[["trip_id", "fix_idx", "arc_id", "label", "has_label", "label_reliable", "exclude_trip"]].merge(
        pred[["trip_id", "fix_idx", "pred_arc"]], on=["trip_id", "fix_idx"], how="left")
    yhat = (r["arc_id"] == r["pred_arc"]).to_numpy().astype(int)
    for name, fm, rm in [("all", t["has_label"] & ~t["exclude_trip"], r["has_label"] & ~r["exclude_trip"]),
                         ("reliable", t["label_reliable"], r["label_reliable"])]:
        out[f"{name}_top1"] = float(t.loc[fm, "correct"].mean())
        out[f"{name}_n_fixes"] = int(fm.sum())
        rm = rm.to_numpy()
        out[f"{name}_sel_macro_f1"] = float(f1_score(r["label"].to_numpy()[rm], yhat[rm], average="macro"))
    keep = ~t["exclude_trip"]
    sets = {"all": t[keep].groupby("trip_id")["pred_arc"].agg(set),
            "reliable": t[t["label_reliable"]].groupby("trip_id")["pred_arc"].agg(set)}
    if filled is not None:
        sets["filled"] = pd.Series({tid: {a for pc in pcs for a in pc} for tid, pcs in filled.items()
                                    if tid in set(t.loc[keep, "trip_id"])})
    for name, s in sets.items():
        inter = plen = rlen = 0.0
        for tid in t.loc[keep, "trip_id"].unique():
            L, R = arc_len[tid], ref_arcs[tid]
            P = s.get(tid, set())
            inter += L[list(P & R)].sum(); plen += L[list(P)].sum(); rlen += L[list(R)].sum()
        prec, rec = inter / plen if plen else np.nan, inter / rlen if rlen else np.nan
        out[f"seq_{name}_precision"], out[f"seq_{name}_recall"] = prec, rec
        out[f"seq_{name}_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else np.nan
    return out
