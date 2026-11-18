"""PathSnap FastAPI backend (runs on laptop 1). Contract: docs/API_CONTRACT.md.

Endpoints
    GET  /health    is the API up, which model and decoder it runs
    GET  /trips     the trips the evaluator can pick (no ground truth)
    POST /predict   run the full pipeline on one trip and score it server-side

Run from the repo root, inside the venv:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Settings (environment variables):
    PATHSNAP_TRIPS_DIR   folder with one sub-folder per trip
                         (default: "Student folder/student_test"; demo day: "evaluator_test")
    PATHSNAP_CLASSIFIER  classifier to use with its tuned HMM decoder (default: the frozen choice in
                         models/hmm_config.json, RandomForest_tuned; alternative: LightGBM_tuned)

The reference routes stay on this laptop. They are only read by `score_trip`, after
the prediction is finished, and only the resulting numbers are sent to the UI.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pathsnap import inference as I  # noqa: E402


def _repo_path(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, default))
    return p if p.is_absolute() else (REPO / p).resolve()


TRIPS_DIR = _repo_path("PATHSNAP_TRIPS_DIR", "Student folder/student_test")
CLASSIFIER = os.environ.get("PATHSNAP_CLASSIFIER") or None

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not TRIPS_DIR.is_dir():
        raise RuntimeError(f"Trips folder not found: {TRIPS_DIR} (set PATHSNAP_TRIPS_DIR)")
    STATE["pipe"] = I.load_pipeline(CLASSIFIER)
    STATE["trip_ids"] = I.list_trip_ids(TRIPS_DIR)
    log.info("PathSnap API ready | classifier=%s | decoder=%s | trips=%d from %s",
             STATE["pipe"]["name"], I.DECODER_NAME, len(STATE["trip_ids"]), TRIPS_DIR.name)
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "classifier": STATE["pipe"]["name"], "decoder": I.DECODER_NAME,
            "n_trips": len(STATE["trip_ids"]), "mock": False}


@app.get("/trips")
def trips():
    out = []
    for tid in STATE["trip_ids"]:
        t = I.load_raw_trip(TRIPS_DIR, tid).track            # GPS only: no route is read here
        out.append({"trip_id": tid, "n_points": len(t), "duration_s": float(t[-1, 2] - t[0, 2])})
    log.info("GET  /trips    -> %d trips", len(out))
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id
    if tid not in STATE["trip_ids"]:           # also blocks path tricks like "../x"
        log.warning("POST /predict  trip=%s -> 404 unknown trip", tid)
        raise HTTPException(404, f"unknown trip_id {tid}")
    t0 = time.perf_counter()
    try:
        out = I.predict_trip(TRIPS_DIR, tid, STATE["pipe"])        # prediction: no route
        trip = I.load_raw_trip(TRIPS_DIR, tid)
        t_score = time.perf_counter()
        metrics = I.score_trip(trip, out)                          # scoring: route, numbers only
        t_score = time.perf_counter() - t_score
    except Exception as e:  # report the failure to the UI instead of a bare 500
        log.exception("POST /predict  trip=%s -> 500", tid)
        raise HTTPException(500, f"prediction failed for trip {tid}: {type(e).__name__}: {e}") from e
    total_ms = (time.perf_counter() - t0) * 1000

    ts = out["timings_s"]
    log.info("POST /predict  trip=%s points=%d | clean+features %.2fs classifier %.2fs hmm build %.2fs "
             "viterbi %.2fs score %.2fs | total %.0f ms",
             tid, len(trip.track), ts["clean_features_s"], ts["classifier_s"], ts["hmm_build_s"],
             ts["viterbi_s"], t_score, total_ms)
    log.info("               trip=%s Top-1 %s (classifier only %s) | route continuity %s | scored %d points",
             tid, _pct(metrics["top1_accuracy"]), _pct(metrics["top1_accuracy_classifier_only"]),
             _pct(metrics["route_continuity"]), metrics["n_scored_points"])

    return {
        "trip_id": tid,
        "n_points": len(trip.track),
        "gps": np.round(trip.track[:, :2], 6).tolist(),
        "predicted_route": out["route"],
        "metrics": metrics,
        "timing_ms": {"total": round(total_ms, 1)},
        "classifier": STATE["pipe"]["name"],
        "decoder": I.DECODER_NAME,
        "mock": False,
    }


def _pct(v) -> str:
    return "-" if v is None else f"{v:.1%}"
