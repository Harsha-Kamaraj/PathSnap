# PathSnap: Data Quality Issues, Assumptions and Decisions

This is the interpretation of the dataset exploration. All numbers come from
`outputs/eda/EDA_REPORT.md` and `outputs/eda/tables/*.csv`. Regenerate them with
`python scripts/run_eda.py`. Scope: the `train` (80 trips) and `student_test`
(10 trips) partitions. The evaluator-test partition is not on disk and was not used.

## 1. How the raw dataset is organised

```
<partition>/<trip_id>/<trip_id>.track   lon  lat  t        GPS fixes, chronological
                      <trip_id>.nodes   lon  lat           row index = node_id
                      <trip_id>.arcs    from to            row index = arc_id, from/to are node_ids
                      <trip_id>.route   arc_id             ground-truth arc sequence
```

* All 90 trips contain exactly these 4 files, with no headers and no malformed,
  blank, NaN or inf rows. The column counts are constant (3 / 2 / 2 / 1).
* **Every trip has its own road-network extract** (all 90 `.nodes` files differ).
  A `node_id` or `arc_id` only means something inside its own trip, so features
  and lookups must always be built per trip.
* How the parts relate: `route[k]` is an index into `.arcs`, and each arc's
  `from`/`to` are row indices into `.nodes`. **There is no row-level link between
  `.track` and the network.** Assigning GPS fixes to arcs is exactly the
  map-matching task, and the route only gives the ordered arc sequence, not which
  fix belongs to which arc.

## 2. Geography and distance computations

* The trips are worldwide, not one city: 25 UTM zones on 5+ continents
  (longitude -100 to 137, latitude -35 to 62). Examples: Amsterdam, Toulouse,
  central Germany, North America and Asia.
* **Decision:** project each trip into the UTM zone of its own median GPS
  position (`src/pathsnap/geo.py`) and do all geometry in metres there. One trip
  covers at most a few tens of km, so it stays inside one zone, and the UTM scale
  error (<0.1 %) is negligible next to GPS noise. As a check, UTM step lengths
  agree with haversine to within 0.55 % (most of that is the sphere-vs-ellipsoid
  difference in haversine). Euclidean distance in degrees is never used: a degree
  of longitude is 111 km at the equator but only about 68 km at 52°N.
* **Column order** is confirmed as (lon, lat). 100 % of GPS fixes fall inside
  their trip's network bounding box, and swapping the columns would put every
  trip in the wrong place. No coordinates are out of range and there are no (0,0) fixes.

## 3. Trajectory / timestamp issues

| Issue | Evidence | Decision for the pipeline |
|---|---|---|
| t is relative seconds, not wall-clock | 16 trips start at t > 1 (up to 693 s) | Use only differences `dt`; never use absolute t as a feature |
| Sampling is nominally 1 Hz | 96.7 % of steps have dt = 1 s, 2.0 % have dt = 2 s | Treat dt as a feature / transition-probability input, not a constant |
| Long gaps | 30 trips have a gap > 30 s; the longest is 153,064 s (42.5 h, trip 90), with others at 52,810 s, 2,658 s and 2,161 s | In the HMM, scale the transition model by dt / gap distance; consider breaking a trip into segments at very long gaps |
| Non-integer timestamps | Trips 2, 71 and 97 have fractional t | Do not cast t to int; keep it as float |
| Timestamps always increase | 0 steps with dt ≤ 0 | No re-sorting needed, but the loader keeps the check |
| Implausible speed runs | 151 steps > 70 m/s in 11 trips, mostly trips 13 and 32. They come in **runs** right after small gaps (for example 8 s for 29 m, then 87, 72, 83, 86 m/s), which looks like timestamp/position desync rather than single outliers | Do not trust instantaneous speed. Use smoothed or median speed features and cap the speed used in transition costs |
| Parked periods with GPS jitter | 7 trips are parked > 20 % of the time. Trip 54 (Amsterdam, 5.4 h) is parked 79 % of the time, with long stops between t ≈ 2,900 s and 19,100 s at a place ~85 m from any road. Jitter makes it look like it moves at ~7 m/s, giving ~140 km of fake path length (169.6 km GPS path vs a 31.4 km route). Trip 33 is similar (79 %) | Detect stops with a window test (net displacement < 30 m over 60 fixes), not with speed < 0.5 m/s. Down-weight or collapse stop segments before matching |
| GPS noise level | Distance from fix to nearest arc: median 5.0 m, p95 81 m (inflated by parked trips) | Typical σ is ~5–10 m. Search for candidate arcs within ~50–100 m and fit the emission σ on **train only** |

## 4. Road-network issues

* Arcs are **directed**. For most arcs (median 95 % per trip) the reverse arc also
  exists, so two-way streets appear as two arcs. In some trips many arcs are
  one-way only (min 35 %, trip 80). Transitions must respect direction.
* There are minor defects: 24 self loops, 1,903 duplicate (from, to) pairs
  (parallel roads between the same two nodes, which are separate `arc_id`s and so
  separate labels), 106 zero-length arcs, and 92 duplicate node coordinates.
  None of them break indexing; all arc endpoints are valid node ids.
* There are no isolated nodes. Each extract has several weakly connected
  components (median 19). Its largest strongly connected component covers a
  median of 96 % of nodes, with a minimum of 64 % in trip 5. The rest are pieces
  cut off at the edge of the extract, dead ends and one-way fragments.
  **Decision:** shortest-path transitions can fail when two candidates lie in
  different SCCs. Handle "no path" with a large finite cost, not a crash.
* Network size varies 500-fold (299 to 155,003 nodes). **Decision:** use a
  spatial index (STRtree / KD-tree) per trip for candidate search, and run
  Dijkstra with a distance bound.

## 5. Reference-route limitations (labels are not perfect ground truth)

* A route exists for all 90 trips, and every arc id in it is valid.
* **Route breaks:** 11 trips (14 breaks) contain consecutive arcs that share no
  node. Many line up with GPS gaps (trips 69, 42, 89, 2, 41, 80), but some do not
  (trips 19, 86, 75 have breaks without a gap). So the ground truth is not always
  a connected path. **Decision:** do not assume label paths are contiguous when
  computing metrics, and do not train a "transition must be adjacent" constraint
  as if it were absolute.
* **U-turns and revisits:** there are 54 immediate U-turns (arc followed by its
  reverse) and 322 revisited arcs. These are real driving patterns, so the
  matcher must allow U-turns and loops.
* **Incomplete coverage:** route length / GPS length has a median of 0.85, but
  some trips are far lower: 0.13 (trip 33), 0.14 (trip 2), 0.19 (trip 54). In trip 2
  the vehicle travels mostly on paths that are **not in the network** (likely
  tracks), and the route covers only the part on mapped roads. In trips 33 and 54
  the gap is mostly parked jitter. 4 trips have > 5 % of fixes more than 50 m
  from the route.
* In trip 5 only 46 % of the route arcs lie in the largest SCC: the route uses a
  part of the extract that is poorly connected.
* Endpoints: the first fix is close to the first route arc (median 7 m, max 90 m),
  but the last fix can be up to 548 m from the last arc. The route may start or
  end before or after the recorded track.
* **Implications for evaluation:** per-fix accuracy against these labels is an
  upper-bounded, noisy measure. We will report metrics on all trips **and**
  excluding fixes far (> 50 m) from any arc, with the flagged trips listed, and we
  will not overstate certainty.

## 6. Leakage and partition hygiene

* Trips do not share networks, and every trip sits in exactly one partition.
  Cross-validation must split **by trip** (GroupKFold on `trip_id`), never by fix,
  because neighbouring fixes are almost duplicates.
* This EDA fits nothing. Any learned preprocessing (scalers, σ estimates,
  thresholds that are tuned rather than set by hand) will be fitted on train folds only.
* The thresholds used in this EDA (30 s gap, 70 m/s, 50 m, 30 m / 60 fixes) are
  fixed by hand from physical reasoning, **not** tuned on any test labels.

## 7. Assumptions that shape the map-matching approach

1. GPS fixes are noisy observations of a vehicle on the directed network, with
   roughly Gaussian distance noise (σ around 5–10 m) plus outliers. → HMM emission
   = f(distance to arc, heading agreement), refined by a learned candidate classifier.
2. Between consecutive fixes the vehicle follows a short path in the network.
   → transition = f(|route distance − great-circle distance|, dt), with
   "no path" handled as above.
3. Some fixes cannot be matched reliably (off-network, parked, long gaps).
   → the pipeline needs gap segmentation and stop handling, and evaluation must
   account for imperfect labels.
