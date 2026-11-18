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
scripts/run_eda.py   dataset exploration: tables, figures, Folium maps, report
scripts/run_cleaning.py  cleans every trip into data/processed/
outputs/eda/         generated EDA outputs (committed as evidence)
outputs/cleaning/    cleaning summary + report (committed as evidence)
data/processed/      cleaned trips (git-ignored, regenerate with run_cleaning.py)
docs/DATA_QUALITY_NOTES.md   interpretation, assumptions and decisions
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
