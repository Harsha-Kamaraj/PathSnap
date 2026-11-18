# PathSnap: GPS Map-Matching

PES University UE24CS352A hackathon. Match noisy GPS trajectories to arcs of a road network.

## Repository layout

```
Student folder/      raw dataset (train/, student_test/), as provided
PathSnap_student_materials/  hackathon brief PDFs
src/pathsnap/        reusable code
  data_io.py         robust loaders for .track/.nodes/.arcs/.route + parse-quality report
  geo.py             per-trip UTM projection and haversine
  eda_metrics.py     per-trip exploratory statistics and anomaly flags
  cleaning.py        per-trip cleaning: metres, segments, stops, quality flags
  labels.py          per-fix reference arc (route alignment)
  candidates.py      candidate arcs per fix
  features.py        candidate features
  training.py        model registry, CV, metrics
  tuning.py          LightGBM + Random Forest search spaces, Optuna objective
scripts/run_eda.py   dataset exploration: tables, figures, Folium maps, report
scripts/run_cleaning.py  cleans every trip into data/processed/
notebooks/03_features.ipynb  labels, candidates and features -> data/features/
notebooks/04_model_training.ipynb  5 models, CV + student_test comparison -> outputs/models/, models/
notebooks/05_bayesian_tuning.ipynb  Optuna TPE tuning of LightGBM -> outputs/tuning/, models/LightGBM_tuned.joblib
notebooks/06_bayesian_tuning_rf.ipynb  Optuna TPE tuning of Random Forest -> outputs/tuning_rf/, models/RandomForest_tuned.joblib
outputs/eda/         generated EDA outputs (committed as evidence)
outputs/cleaning/    cleaning summary + report (committed as evidence)
data/processed/      cleaned trips (git-ignored, regenerate with run_cleaning.py)
docs/DATA_QUALITY_NOTES.md   interpretation, assumptions and decisions
docs/API_CONTRACT.md request/response format between the API (laptop 1) and the UI (laptop 2)
ui/app.py            Streamlit evaluator UI (laptop 2)
ui/mock_api.py       mock API that follows the contract, for testing the UI
ui/requirements.txt  packages needed on laptop 2 (UI only)
```

## Setup

```bash
# Windows
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The raw data (train + student_test) is committed under `Student folder/`, and
the code reads it from there by default. To use another location, set
`PATHSNAP_DATA` to a folder that contains `train/` and `student_test/`.

## Dataset exploration

```bash
.venv/Scripts/python scripts/run_eda.py
```

It takes about 3–5 minutes and is deterministic. Results:

* [`outputs/eda/EDA_REPORT.md`](outputs/eda/EDA_REPORT.md): key numbers and embedded figures
* [`outputs/eda/tables/`](outputs/eda/tables): `inventory.csv`, `trip_summary.csv` (one row per trip, ~100 columns), partition/trajectory/network/route summaries, `anomalies.csv`
* [`outputs/eda/maps/`](outputs/eda/maps): interactive Folium maps of six trips chosen by rule (open them in a browser)
* [`docs/DATA_QUALITY_NOTES.md`](docs/DATA_QUALITY_NOTES.md): what the numbers mean for the pipeline

## Data cleaning

```bash
.venv/Scripts/python scripts/run_cleaning.py   # Windows
.venv/bin/python scripts/run_cleaning.py       # macOS / Linux
```

It takes about 1 minute and is deterministic. **No rows are deleted**, because every GPS fix
still needs a prediction. Each fix gets metres (per-trip UTM), a `segment_id` (split at gaps
> 120 s), a `stop_id` (parked periods) and quality flags. Arcs get a length, a
`canonical_arc_id` (geometric duplicates) and an `is_candidate` flag. Route-based columns
(`d_route_m`, `is_far_from_route`, `label_reliable`) are only for training and scoring, and
are never model inputs. Trips 33 and 54 are excluded from training (most fixes are far from
their reference route).

* [`outputs/cleaning/CLEANING_REPORT.md`](outputs/cleaning/CLEANING_REPORT.md): what was flagged and excluded
* [`outputs/cleaning/cleaning_summary.csv`](outputs/cleaning/cleaning_summary.csv): one row per trip
* Load a cleaned trip: `from pathsnap.cleaning import load_clean_trip`

## Labels, candidates and features (notebook 03)

Run `notebooks/03_features.ipynb` (kernel "PathSnap (.venv)"; needs `data/processed` from `run_cleaning.py`; about 2 min).

```bash
.venv/Scripts/python -m ipykernel install --user --name pathsnap --display-name "PathSnap (.venv)"   # once
.venv/Scripts/python -m jupyter lab notebooks/
```

* `src/pathsnap/labels.py`: per-fix reference arc from an order-respecting DP alignment of the fixes to `.route`
  (handles U-turns and revisits). Training and scoring only.
* `src/pathsnap/candidates.py`: arcs within 50 m, at most 16 per fix. This was chosen by a train-only miss-rate sweep
  (`outputs/features/candidate_sweep.csv`). Train miss rate 0.11 %.
* `src/pathsnap/features.py`: 27 label-free features per (fix, candidate) row (`FEATURE_COLS`).
* Outputs: `data/features/*.parquet` (git-ignored) and [`outputs/features/FEATURES_REPORT.md`](outputs/features/FEATURES_REPORT.md).

## Model training and comparison (notebook 04)

Run `notebooks/04_model_training.ipynb` after notebook 03 (about 25 min on CPU). It trains Logistic Regression, KNN,
Random Forest, LightGBM and an MLP with 5-fold GroupKFold on `trip_id`, refits each on all usable train trips and scores
it once on student_test. Code: `src/pathsnap/training.py`.

**Where the model code is:** all 5 models are defined in `model_specs()` in
[`src/pathsnap/training.py`](src/pathsnap/training.py), each under its own `MODEL n` banner:

| # | model | family | code |
|---|---|---|---|
| 1 | Logistic Regression | Linear | [`training.py` line 69](src/pathsnap/training.py#L69) |
| 2 | K-Nearest Neighbours | Instance-based | [`training.py` line 79](src/pathsnap/training.py#L79) |
| 3 | Random Forest | Ensemble (bagging) | [`training.py` line 91](src/pathsnap/training.py#L91) |
| 4 | LightGBM | Boosting | [`training.py` line 102](src/pathsnap/training.py#L102) |
| 5 | MLP | Neural network | [`training.py` line 113](src/pathsnap/training.py#L113) |

The same file has the training loop (`fit_model`, including the MLP's per-epoch loop) and the metrics (`evaluate`).
The notebook runs the experiment and makes the tables and plots.

* [`outputs/models/MODEL_COMPARISON.md`](outputs/models/MODEL_COMPARISON.md): results tables and figures
* `outputs/models/`: `model_comparison.csv`, `per_fold.csv`, `configs.json` (all settings and the seed), `experiment_log.jsonl`,
  `mlp_history.csv`, `confusion_matrices.json`, `figures/`
* `models/*.joblib`: fitted models. Files over 50 MB are git-ignored, so regenerate them with the notebook.

## Bayesian hyperparameter optimisation of LightGBM (notebook 05)

Run `notebooks/05_bayesian_tuning.ipynb` after notebook 04 (about 80 min on CPU).

* Optuna TPE sampler (seed 42) with a MedianPruner. Objective: mean CV Top-1 over the same 5 GroupKFold folds as notebook 04.
  At most 60 trials or 75 minutes. student_test is scored once, after the search.
* Code: [`src/pathsnap/tuning.py`](src/pathsnap/tuning.py) holds the search space, objective and study setup.
* [`outputs/tuning/TUNING_REPORT.md`](outputs/tuning/TUNING_REPORT.md) answers the report questions. Also in `outputs/tuning/`:
  `trials.csv` (every trial with per-fold scores), `best_params.json`, `search_space.json`, `tuned_vs_untuned.csv`,
  `per_fold_top1.csv`, `study.db` (the Optuna study) and `figures/`.
* Model: `models/LightGBM_tuned.joblib`. The untuned `models/LightGBM.joblib` is unchanged.

## Bayesian hyperparameter optimisation of Random Forest (notebook 06)

Run `notebooks/06_bayesian_tuning_rf.ipynb` after notebook 05 (about 80 min on CPU). It uses the same protocol as notebook 05:
the same folds, the Top-1 objective, TPE (seed 42) and MedianPruner, with at most 40 trials or 75 minutes. Random Forest was
tuned too because it was the most accurate untuned model, only a few tenths of a point ahead of LightGBM.

* Search space: `RF_SEARCH_SPACE` in [`src/pathsnap/tuning.py`](src/pathsnap/tuning.py).
* [`outputs/tuning_rf/TUNING_REPORT_RF.md`](outputs/tuning_rf/TUNING_REPORT_RF.md) answers the report questions and compares
  RF and LightGBM, untuned and tuned (all four models re-timed in one session). Also in `outputs/tuning_rf/`: `trials.csv`,
  `best_params.json`, `search_space.json`, `rf_lgbm_tuned_vs_untuned.csv`, `per_fold_top1.csv`, `study.db`, `figures/`.
* Model: `models/RandomForest_tuned.joblib`. The untuned `models/RandomForest.joblib` is unchanged.


## Evaluator UI (Streamlit, laptop 2)

The UI talks to the API **only over HTTP**. It never loads the model, the data or any ground truth.
The request and response format is fixed in [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md)
(`GET /health`, `GET /trips`, `POST /predict`).

```bash
# laptop 2: only the UI packages are needed
python3 -m venv .venv && .venv/bin/python -m pip install -r ui/requirements.txt
PATHSNAP_API_URL=http://<laptop-1-ip>:8000 .venv/bin/streamlit run ui/app.py
```

You can also type the API address in the sidebar. The page shows a connection light, the list of
trips (click a row), a **Run prediction** button, the metrics computed by the API (Top-1, classifier-only
Top-1, route continuity) and a Folium map of the raw GPS points and the predicted route, with a legend
and layer control. The reference route is never sent to the UI.

**Testing without the real API:** `ui/mock_api.py` follows the same contract, using the 10
student_test trips and a nearest-road baseline (no model). The UI shows a "MOCK API" banner while
it is connected.

```bash
.venv/bin/python -m uvicorn ui.mock_api:app --host 0.0.0.0 --port 8000   # needs data/processed
.venv/bin/streamlit run ui/app.py
```
