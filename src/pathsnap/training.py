"""Model registry, cross-validation and metrics for the candidate classifier (step 4).

Task: binary classification of (fix, candidate arc) rows: is this candidate the true arc?
At prediction time each fix takes its highest-probability candidate (Top-1).

Rules implemented here:
* GroupKFold by trip_id (5 folds, fixed seed): rows of one trip never sit on both sides.
* Training rows = label_reliable & not an excluded trip (33, 54). Route-derived columns never enter X.
* Any scaler / transform is part of the model Pipeline, so it is fitted on the training fold only.
* Evaluation sets: "all" = every labelled fix of non-excluded trips; "reliable" = label_reliable fixes.
  A fix whose true arc is not among its candidates counts as wrong in Top-1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import FEATURE_COLS

SEED = 42
N_FOLDS = 5
KNN_SAMPLE_ROWS = 150_000      # KNN stores every training row and compares each query to all of them
MLP_EPOCHS = 25
MLP_TRAIN_LOSS_SAMPLE = 200_000  # train-loss per epoch is measured on a fixed sample (speed)

# heavy-tailed, non-negative features: log1p makes them less dominated by a few huge values
# (matters for distance-based KNN and for the linear / neural models, not for trees)
SKEWED = ["dist_m", "dist_minus_min", "dist_ratio", "off_end_m", "arc_length_m", "heading_disp_m",
          "d_prev_m", "d_next_m", "d_win_mean_m", "d_win_max_m", "win_mean_minus_min", "speed_med_mps", "dt_prev_s"]


def scaled(est) -> Pipeline:
    """log1p on skewed columns, then standardise everything, then the estimator."""
    pre = ColumnTransformer([("log", FunctionTransformer(np.log1p), SKEWED)], remainder="passthrough")
    return Pipeline([("log1p", pre), ("scale", StandardScaler()), ("model", est)])


@dataclass
class ModelSpec:
    name: str
    family: str
    config: dict
    make: callable
    sample_rows: int | None = None      # train on a random sample of this many rows
    epochs: int | None = None           # trained epoch by epoch (MLP only)
    notes: str = ""


def model_specs() -> dict[str, ModelSpec]:
    lr_cfg = dict(C=1.0, solver="lbfgs", max_iter=1000)   # default penalty = L2
    knn_cfg = dict(n_neighbors=50, weights="uniform", algorithm="auto")
    rf_cfg = dict(n_estimators=150, max_features="sqrt", min_samples_leaf=20, max_samples=0.3,
                  bootstrap=True, n_jobs=-1)
    lgb_cfg = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    mlp_cfg = dict(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=1e-4,
                   batch_size=1024, learning_rate_init=1e-3)
    return {
        "LogisticRegression": ModelSpec("LogisticRegression", "Linear", lr_cfg,
                                        lambda: scaled(LogisticRegression(**lr_cfg, random_state=SEED)),
                                        notes="log1p + StandardScaler"),
        "KNN": ModelSpec("KNN", "Instance-based", knn_cfg,
                         lambda: scaled(KNeighborsClassifier(**knn_cfg, n_jobs=-1)),
                         sample_rows=KNN_SAMPLE_ROWS,
                         notes=f"log1p + StandardScaler; trained on a random {KNN_SAMPLE_ROWS:,}-row sample"),
        "RandomForest": ModelSpec("RandomForest", "Ensemble (bagging)", rf_cfg,
                                  lambda: RandomForestClassifier(**rf_cfg, random_state=SEED),
                                  notes="raw features (trees do not need scaling)"),
        "LightGBM": ModelSpec("LightGBM", "Boosting", lgb_cfg,
                              lambda: LGBMClassifier(**lgb_cfg, random_state=SEED),
                              notes="raw features"),
        "MLP": ModelSpec("MLP", "Neural network", {**mlp_cfg, "epochs": MLP_EPOCHS},
                         lambda: scaled(MLPClassifier(**mlp_cfg, random_state=SEED)),
                         epochs=MLP_EPOCHS,
                         notes="log1p + StandardScaler; trained with partial_fit one epoch at a time"),
    }


# ------------------------------------------------------------------ data
def training_mask(rows: pd.DataFrame) -> np.ndarray:
    return (rows["label_reliable"] & ~rows["exclude_trip"]).to_numpy()


def X_of(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[FEATURE_COLS].astype("float64")


# ------------------------------------------------------------------ fitting
def fit_model(spec: ModelSpec, rows_tr: pd.DataFrame, epochs: int | None = None,
              rows_val: pd.DataFrame | None = None, rng_seed: int = SEED):
    """Fit one model. For the MLP, returns per-epoch train / validation log-loss as well."""
    X, y = X_of(rows_tr), rows_tr["label"].to_numpy()
    if spec.sample_rows and len(X) > spec.sample_rows:
        idx = np.random.default_rng(rng_seed).choice(len(X), spec.sample_rows, replace=False)
        X, y = X.iloc[idx], y[idx]
    model = spec.make()
    history = []
    t0 = time.perf_counter()
    if spec.epochs is None:
        model.fit(X, y)
    else:
        n_ep = epochs or spec.epochs
        # fit the preprocessing on the training fold once, then train the network epoch by epoch
        pre = Pipeline(model.steps[:-1]).fit(X)
        net = model.steps[-1][1]
        Xt = pre.transform(X)
        rng = np.random.default_rng(rng_seed)
        mon = rng.choice(len(Xt), min(MLP_TRAIN_LOSS_SAMPLE, len(Xt)), replace=False)
        Xv = pre.transform(X_of(rows_val)) if rows_val is not None else None
        yv = rows_val["label"].to_numpy() if rows_val is not None else None
        for ep in range(1, n_ep + 1):
            perm = rng.permutation(len(Xt))
            net.partial_fit(Xt[perm], y[perm], classes=np.array([0, 1]))
            rec = {"epoch": ep, "train_logloss": log_loss(y[mon], net.predict_proba(Xt[mon])[:, 1], labels=[0, 1])}
            if Xv is not None:
                rec["val_logloss"] = log_loss(yv, net.predict_proba(Xv)[:, 1], labels=[0, 1])
            history.append(rec)
    return model, time.perf_counter() - t0, history


def predict(model, rows: pd.DataFrame) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    p = model.predict_proba(X_of(rows))[:, 1]
    return p, time.perf_counter() - t0


# ------------------------------------------------------------------ metrics
def top1_predictions(rows: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Highest-probability candidate per fix: one row per fix with its predicted arc and correctness."""
    r = rows[["trip_id", "fix_idx", "segment_id", "arc_id", "label"]].assign(p=p)
    best = r.loc[r.groupby(["trip_id", "fix_idx"], sort=False)["p"].idxmax()]
    return best.rename(columns={"arc_id": "pred_arc", "label": "correct"}).reset_index(drop=True)


def route_continuity(top1: pd.DataFrame, arcs_by_trip: dict) -> float:
    """Share of consecutive fix pairs (same GPS segment) whose predicted arcs are the same arc or
    directly connected (end node of the first = start node of the second). Uses the network only."""
    ok = tot = 0
    for tid, g in top1.sort_values(["trip_id", "fix_idx"]).groupby("trip_id", sort=False):
        fr, to = arcs_by_trip[tid]
        a = g["pred_arc"].to_numpy()
        same_seg = g["segment_id"].to_numpy()[1:] == g["segment_id"].to_numpy()[:-1]
        consec = np.diff(g["fix_idx"].to_numpy()) == 1
        pair = same_seg & consec
        cont = (a[1:] == a[:-1]) | (to[a[:-1]] == fr[a[1:]])
        ok += int((cont & pair).sum()); tot += int(pair.sum())
    return ok / tot if tot else float("nan")


def evaluate(rows: pd.DataFrame, fixes: pd.DataFrame, p: np.ndarray, arcs_by_trip: dict) -> dict:
    """All metrics on two fix sets: 'all' (labelled, non-excluded trips) and 'reliable'."""
    out = {}
    top1 = top1_predictions(rows, p)
    fx = fixes.set_index(["trip_id", "fix_idx"])
    t = top1.join(fx[["has_label", "label_reliable", "exclude_trip", "true_in_cands"]], on=["trip_id", "fix_idx"])
    out["route_continuity"] = route_continuity(top1, arcs_by_trip)
    for name, fmask, rmask in [
        ("all", t["has_label"] & ~t["exclude_trip"], rows["has_label"] & ~rows["exclude_trip"]),
        ("reliable", t["label_reliable"], rows["label_reliable"]),
    ]:
        rm = rmask.to_numpy()
        y, pp = rows["label"].to_numpy()[rm], p[rm]
        yhat = (pp >= 0.5).astype(int)
        out[f"{name}_top1"] = float(t.loc[fmask, "correct"].mean())
        out[f"{name}_macro_f1"] = float(f1_score(y, yhat, average="macro"))
        out[f"{name}_precision"] = float(precision_score(y, yhat, zero_division=0))
        out[f"{name}_recall"] = float(recall_score(y, yhat, zero_division=0))
        out[f"{name}_logloss"] = float(log_loss(y, np.clip(pp, 1e-7, 1 - 1e-7), labels=[0, 1]))
        out[f"{name}_miss_rate"] = float(1 - t.loc[fmask, "true_in_cands"].mean())
        out[f"{name}_n_fixes"] = int(fmask.sum())
        out[f"{name}_cm"] = confusion_matrix(y, yhat, labels=[0, 1]).tolist()
    return out
