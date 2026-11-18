"""MOCK API for building and testing the UI before the real API exists.

It follows docs/API_CONTRACT.md exactly, but it is NOT our model. It serves the 10
student_test trips and "predicts" each GPS fix as its NEAREST road segment, the simple
no-model baseline. Every response has "mock": true, so the UI shows a banner.

Run (from the repo root, inside the venv):
    python -m uvicorn ui.mock_api:app --host 0.0.0.0 --port 8000

Needs data/processed/ (run scripts/run_cleaning.py once).
"""
from __future__ import annotations

import logging
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pathsnap.cleaning import PROCESSED_ROOT, load_clean_trip  # noqa: E402
from pathsnap.labels import align_labels  # noqa: E402

PARTITION = "student_test"   # the mock never reads evaluator-test data

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("pathsnap.mock_api")
app = FastAPI(title="PathSnap MOCK API", version="mock")


class PredictRequest(BaseModel):
    trip_id: str


def trip_ids() -> list[str]:
    d = PROCESSED_ROOT / PARTITION
    if not d.is_dir():
        raise HTTPException(500, f"{d} not found: run scripts/run_cleaning.py first")
    return sorted(p.name for p in d.iterdir() if p.is_dir())


@lru_cache(maxsize=16)
def load(trip_id: str) -> dict:
    return load_clean_trip(PARTITION, trip_id)


@app.get("/health")
def health():
    return {"status": "ok", "classifier": "nearest-road baseline (mock)", "decoder": "none", "mock": True}


@app.get("/trips")
def trips():
    out = []
    for tid in trip_ids():
        t = load(tid)["track"]
        out.append({"trip_id": tid, "n_points": len(t), "duration_s": float(t["t_rel"].iloc[-1])})
    log.info("GET /trips -> %d trips", len(out))
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    if req.trip_id not in trip_ids():
        raise HTTPException(404, f"unknown trip_id {req.trip_id}")
    c = load(req.trip_id)
    track, nodes, arcs = c["track"], c["nodes"], c["arcs"]

    # "prediction" = nearest usable road segment for every fix (no model)
    usable = arcs[arcs["is_candidate"]]
    nxy = nodes[["x", "y"]].to_numpy()
    a_xy = nxy[usable["from_node"].to_numpy()]
    b_xy = nxy[usable["to_node"].to_numpy()]
    lines = shapely.linestrings(np.stack([a_xy, b_xy], axis=1))
    pts = shapely.points(track[["x", "y"]].to_numpy())
    (pi, li) = shapely.STRtree(lines).query_nearest(pts, all_matches=False)
    nearest = np.empty(len(track), dtype=np.int64)
    nearest[pi] = li
    pred_arc = usable["arc_id"].to_numpy()[nearest]
    pred_can = arcs["canonical_arc_id"].to_numpy()[pred_arc]

    # draw the snapped points (projection of each fix onto its predicted segment)
    snapped = shapely.line_interpolate_point(lines[nearest], shapely.line_locate_point(lines[nearest], pts))
    s_lon, s_lat = _to_lonlat(c, shapely.get_x(snapped), shapely.get_y(snapped))
    route = []
    for seg in np.unique(track["segment_id"]):
        m = track["segment_id"].to_numpy() == seg
        route.append(np.column_stack([s_lon[m], s_lat[m]]).round(6).tolist())

    # scoring stays here on the server: the reference labels are never sent to the UI
    labels = align_labels(track, nodes, arcs, c["route"])
    has = labels >= 0
    top1 = float(np.mean(pred_can[has] == labels[has])) if has.any() else None
    frm, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()
    same = pred_arc[1:] == pred_arc[:-1]
    joined = to[pred_arc[:-1]] == frm[pred_arc[1:]]
    continuity = float(np.mean(same | joined)) if len(pred_arc) > 1 else None

    ms = (time.perf_counter() - t0) * 1000
    log.info("POST /predict trip=%s points=%d top1=%s time=%.0f ms", req.trip_id, len(track),
             f"{top1:.4f}" if top1 is not None else "-", ms)
    return {
        "trip_id": req.trip_id,
        "n_points": len(track),
        "gps": track[["lon", "lat"]].to_numpy().round(6).tolist(),
        "predicted_route": route,
        "metrics": {"top1_accuracy": top1, "top1_accuracy_classifier_only": None,
                    "route_continuity": continuity, "n_scored_points": int(has.sum())},
        "timing_ms": {"total": round(ms, 1)},
        "classifier": "nearest-road baseline (mock)",
        "decoder": "none",
        "mock": True,
    }


def _to_lonlat(c: dict, x: np.ndarray, y: np.ndarray):
    """Trip metres (its saved UTM zone) -> lon/lat."""
    return Transformer.from_crs(c["meta"]["utm_epsg"], 4326, always_xy=True).transform(x, y)
