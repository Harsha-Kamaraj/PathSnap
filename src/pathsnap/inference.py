"""End-to-end prediction for one trip (used by the API in api/main.py).

Pipeline: raw trip -> cleaning -> candidates + features -> classifier probabilities
-> route decoder -> one predicted road segment per GPS fix -> polylines for the map.

Keeping ground truth out of prediction:
* `predict_trip` gets a copy of the trip with the reference route REMOVED, so no step of
  the prediction can see it, even by accident.
* `score_trip` is the only function that reads the route. It runs after prediction and
  returns only numbers (accuracy, etc.), never the reference arcs.
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import clean_route, clean_trip
from .data_io import Trip, load_trip
from .features import build_trip_features
from .labels import align_labels
from .training import X_of

DECODER_NAME = "none (classifier only)"


def list_trip_ids(trips_dir: Path) -> list[str]:
    return sorted(p.name for p in trips_dir.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def load_raw_trip(trips_dir: Path, trip_id: str) -> Trip:
    """`trips_dir` holds one folder per trip (<id>/<id>.track/.nodes/.arcs/.route)."""
    return load_trip(trips_dir.name, trip_id, trips_dir.parent)


# ---------------------------------------------------------------------------- prediction
def decode(c: dict, rows: pd.DataFrame, p: np.ndarray, pred_classifier: np.ndarray) -> np.ndarray:
    """Route decoder: turns per-candidate probabilities into one arc per fix.

    For now this returns the classifier-only prediction (highest-probability candidate).
    When the HMM/Viterbi step (src/pathsnap/hmm.py) is ready, call it here, set
    DECODER_NAME, and the API and UI pick it up with no other change.
    """
    return pred_classifier


def predict_trip(trip: Trip, model) -> dict:
    """Predict a road segment for every GPS fix. Never sees the reference route."""
    blind = dataclasses.replace(trip, route=np.empty(0, dtype=np.int64))
    t = {}

    t0 = time.perf_counter()
    c = clean_trip(blind)
    t["clean"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rows, _ = build_trip_features(c)        # label columns are empty here: there is no route
    t["features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    p = model.predict_proba(X_of(rows))[:, 1]
    t["model"] = time.perf_counter() - t0

    # classifier only: highest-probability candidate per fix (every fix has >= 1 candidate)
    n = len(c["track"])
    best = pd.Series(p).groupby(rows["fix_idx"].to_numpy()).idxmax()
    pred_classifier = np.full(n, -1, dtype=np.int64)
    pred_classifier[best.index.to_numpy()] = rows["arc_id"].to_numpy()[best.to_numpy()]

    t0 = time.perf_counter()
    pred_final = decode(c, rows, p, pred_classifier)
    t["decode"] = time.perf_counter() - t0

    return {"clean": c, "pred_classifier": pred_classifier, "pred_final": pred_final,
            "n_candidates": len(rows), "timings_s": t}


def route_continuity(pred: np.ndarray, segment_id: np.ndarray, arcs: pd.DataFrame) -> float | None:
    """Share of consecutive fixes (same GPS segment) whose arcs are the same or directly connected."""
    fr, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    pair = segment_id[1:] == segment_id[:-1]
    if not pair.any():
        return None
    ok = (pred[1:] == pred[:-1]) | (to[pred[:-1]] == fr[pred[1:]])
    return float(ok[pair].mean())


def route_polylines(pred: np.ndarray, c: dict) -> list[list[list[float]]]:
    """Predicted route as [lon, lat] polylines along the predicted arcs.

    A new polyline starts at every GPS segment (long gap) and wherever two consecutive
    predicted arcs are not connected, so the map never draws a fake straight jump.
    """
    nodes, arcs, seg = c["nodes"], c["arcs"], c["track"]["segment_id"].to_numpy()
    lonlat = nodes[["lon", "lat"]].to_numpy().round(6)
    fr, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    lines: list[list[list[float]]] = []
    for s in np.unique(seg):
        seq = pred[seg == s]
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]]          # drop repeats of the same arc
        cur, last = [], None
        for a in seq:
            u, v = fr[a], to[a]
            if cur and last == u:
                cur.append(lonlat[v].tolist())
            else:
                if len(cur) >= 2:
                    lines.append(cur)
                cur = [lonlat[u].tolist(), lonlat[v].tolist()]
            last = v
        if len(cur) >= 2:
            lines.append(cur)
    return lines


# ---------------------------------------------------------------------------- scoring
def score_trip(trip: Trip, c: dict, pred_final: np.ndarray, pred_classifier: np.ndarray) -> dict:
    """Server-side scoring against the reference route. Returns numbers only."""
    route_df = clean_route(trip, c["arcs"])
    labels = align_labels(c["track"], c["nodes"], c["arcs"], route_df)   # canonical arc per fix, -1 = none
    has = labels >= 0
    n = int(has.sum())
    seg = c["track"]["segment_id"].to_numpy()
    return {
        "top1_accuracy": float(np.mean(pred_final[has] == labels[has])) if n else None,
        "top1_accuracy_classifier_only": float(np.mean(pred_classifier[has] == labels[has])) if n else None,
        "route_continuity": route_continuity(pred_final, seg, c["arcs"]),
        "n_scored_points": n,
    }
