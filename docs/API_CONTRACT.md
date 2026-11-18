# PathSnap: API contract (laptop 1 API ↔ laptop 2 UI)

The UI (`ui/app.py`, laptop 2) talks to the API (laptop 1) **only** through these three
endpoints. The real FastAPI backend must return exactly these shapes, so it can replace
the mock (`ui/mock_api.py`) without any UI change.

* All coordinates are **`[lon, lat]`** (GeoJSON order, the same as the raw files).
* **Ground truth never leaves the API.** Responses contain GPS points, the predicted route
  and the metrics the API computed. They never contain the reference route or reference labels.

Default address: `http://<laptop-1-ip>:8000`.

---

## `GET /health`

Is the API up, and what is it running?

```json
{
  "status": "ok",
  "classifier": "LightGBM_tuned",
  "decoder": "HMM-Viterbi",
  "mock": false
}
```

## `GET /trips`

The trips the evaluator can pick (the 10 reserved evaluation trips on demo day).

```json
{
  "trips": [
    {"trip_id": "00000040", "n_points": 1420, "duration_s": 1812.0},
    ...
  ]
}
```

## `POST /predict`

Request body:

```json
{"trip_id": "00000040"}
```

Response:

```json
{
  "trip_id": "00000040",
  "n_points": 1420,
  "gps": [[lon, lat], ...],
  "predicted_route": [
    [[lon, lat], [lon, lat], ...],
    ...
  ],
  "metrics": {
    "top1_accuracy": 0.912,
    "top1_accuracy_classifier_only": 0.894,
    "route_continuity": 0.981,
    "n_scored_points": 1398
  },
  "timing_ms": {"total": 830.5},
  "classifier": "LightGBM_tuned",
  "decoder": "HMM-Viterbi",
  "mock": false
}
```

| field | meaning |
|---|---|
| `gps` | raw GPS fixes in time order, one `[lon, lat]` per fix (`len == n_points`) |
| `predicted_route` | the predicted route as a **list of polylines**. A new polyline starts after a long GPS gap (`segment_id` change), so the map does not draw a straight line across a gap |
| `metrics.top1_accuracy` | share of scored GPS fixes whose final predicted road segment equals the reference segment (scored on `canonical_arc_id`) |
| `metrics.top1_accuracy_classifier_only` | the same score before HMM decoding. `null` if not available |
| `metrics.route_continuity` | share of consecutive fixes whose predicted segments are the same or directly connected |
| `metrics.n_scored_points` | fixes that have a reference label and so count in the scores |
| `timing_ms.total` | server-side time to process the request |
| `mock` | `true` only for `ui/mock_api.py`; the UI then shows a "mock data" banner |

Any metric can be `null` (the UI shows "–").

Errors: HTTP 404 for an unknown `trip_id` and HTTP 500 for a processing failure, with
`{"detail": "<message>"}`. The UI shows `detail` to the evaluator.
