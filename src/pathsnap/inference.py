"""End-to-end prediction for one trip (used by the API in api/main.py).

Final frozen pipeline (models/hmm_config.json): RandomForest_tuned + HMM/Viterbi.
raw trip -> cleaning -> candidates + features -> classifier probabilities -> HMM/Viterbi
-> one predicted road segment per GPS fix + a connected route for the map.
The prediction itself is `hmm.predict_trip` (notebook 07), which reads only
.track/.nodes/.arcs.

Keeping ground truth out of prediction:
* `hmm.predict_trip` drops the reference route before doing anything else.
* `score_trip` is the only function that reads the route. It runs after prediction and
  returns only numbers (accuracy, etc.), never the reference arcs.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import hmm
from .cleaning import clean_route, clean_trip
from .data_io import REPO_ROOT, Trip, load_trip
from .labels import align_labels

HMM_CONFIG = REPO_ROOT / "models" / "hmm_config.json"
DECODER_NAME = "HMM-Viterbi"


def list_trip_ids(trips_dir: Path) -> list[str]:
    return sorted(p.name for p in trips_dir.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def load_raw_trip(trips_dir: Path, trip_id: str) -> Trip:
    """`trips_dir` holds one folder per trip (<id>/<id>.track/.nodes/.arcs/.route)."""
    return load_trip(trips_dir.name, trip_id, trips_dir.parent)


def load_pipeline(classifier: str | None = None) -> dict:
    """Classifier + its tuned HMM decoder parameters from models/hmm_config.json.
    Default = the frozen final choice (RandomForest_tuned)."""
    cfg = json.loads(HMM_CONFIG.read_text())
    name = classifier or cfg["classifier"]
    if name == cfg["classifier"]:
        params = cfg["decoder_params"]
    else:
        params = cfg["decoder_params_other_classifier"][name]   # KeyError = no tuned decoder for it
    model = joblib.load(REPO_ROOT / "models" / f"{name}.joblib")
    return {"name": name, "model": model, "params": params}


# ---------------------------------------------------------------------------- prediction
def predict_trip(trips_dir: Path, trip_id: str, pipe: dict) -> dict:
    """Predict a road segment for every GPS fix. Never sees the reference route."""
    out = hmm.predict_trip(trips_dir.name, trip_id, pipe["model"], pipe["params"], root=trips_dir.parent)
    fx = out["fixes"].sort_values("fix_idx")
    return {
        "fix_idx": fx["fix_idx"].to_numpy(),
        "segment_id": fx["segment_id"].to_numpy(),
        "pred_final": fx["pred_arc_hmm"].to_numpy(),
        "pred_classifier": fx["pred_arc_classifier"].to_numpy(),
        "n_carried_over": int(fx["carried_over"].sum()),
        "route": [p["coords"] for p in out["path"] if len(p["coords"]) >= 2],
        "timings_s": out["timing"],
    }


def route_continuity(pred: np.ndarray, segment_id: np.ndarray, arcs: pd.DataFrame) -> float | None:
    """Share of consecutive fixes (same GPS segment) whose arcs are the same or directly connected."""
    fr, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    pair = segment_id[1:] == segment_id[:-1]
    if not pair.any():
        return None
    ok = (pred[1:] == pred[:-1]) | (to[pred[:-1]] == fr[pred[1:]])
    return float(ok[pair].mean())


# ---------------------------------------------------------------------------- scoring
def score_trip(trip: Trip, pred: dict) -> dict:
    """Server-side scoring against the reference route. Returns numbers only."""
    c = clean_trip(trip)                                     # same label-free cleaning, plus the route
    n = len(c["track"])
    final = np.full(n, -1, dtype=np.int64)
    clf = np.full(n, -1, dtype=np.int64)
    final[pred["fix_idx"]] = pred["pred_final"]
    clf[pred["fix_idx"]] = pred["pred_classifier"]
    labels = align_labels(c["track"], c["nodes"], c["arcs"], clean_route(trip, c["arcs"]))  # canonical, -1 = none
    has = labels >= 0
    k = int(has.sum())
    return {
        "top1_accuracy": float(np.mean(final[has] == labels[has])) if k else None,
        "top1_accuracy_classifier_only": float(np.mean(clf[has] == labels[has])) if k else None,
        "route_continuity": route_continuity(final, c["track"]["segment_id"].to_numpy(), c["arcs"]),
        "n_scored_points": k,
    }
