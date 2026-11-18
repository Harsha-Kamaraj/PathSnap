"""Bayesian hyperparameter optimisation of LightGBM (step 5).

* Sampler: Optuna TPE (Tree-structured Parzen Estimator), seed 42. TPE builds two density
  models, one of good past trials and one of bad ones, and proposes parameters that are likely
  under the "good" model. That is what makes the search Bayesian and not random.
* Objective: maximise mean CV **observation-level Top-1 accuracy** (all labelled fixes of
  non-excluded trips) over the SAME 5 GroupKFold folds, features and training rows as
  notebook 04. Top-1 is the task itself (pick the right road for each GPS fix); macro F1 and
  log loss judge single candidate rows at a 0.5 threshold, which is not how a road is chosen.
* Pruner: MedianPruner. After each fold the running mean Top-1 is reported; from the first
  fold on, a trial that is below the median of earlier trials at the same fold is stopped.
* student_test is NEVER used here. It is scored once, after the search, in the notebook.

Metrics and fitting are reused from training.py (fit_model, predict, evaluate).
"""
from __future__ import annotations

import time

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier

from . import training as T

N_TRIALS = 60
TIMEOUT_S = 75 * 60           # time budget: the search must finish in ~75 min (hackathon schedule)
PRUNER_STARTUP_TRIALS = 5     # no pruning until 5 trials have finished (needs a fair median)

# Untuned baseline from notebook 04 (unchanged). It is queued as trial 0 so the search starts
# from a known point. reg_alpha=1e-8 and min_split_gain=0 are effectively LightGBM's defaults.
BASELINE = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, max_depth=-1, min_child_samples=50,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=1e-8, reg_lambda=1.0, min_split_gain=0.0)

# Search space. log=True where a value matters by its order of magnitude.
SEARCH_SPACE = {
    "learning_rate":     {"type": "float", "low": 0.01, "high": 0.2, "log": True},
    "n_estimators":      {"type": "int", "low": 100, "high": 1500, "step": 50},
    "num_leaves":        {"type": "int", "low": 15, "high": 255, "log": True},
    "max_depth":         {"type": "categorical", "choices": [-1, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
    "min_child_samples": {"type": "int", "low": 10, "high": 300, "log": True},
    "subsample":         {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree":  {"type": "float", "low": 0.4, "high": 1.0},
    "reg_alpha":         {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    "reg_lambda":        {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    "min_split_gain":    {"type": "float", "low": 0.0, "high": 1.0},
}
FIXED = dict(subsample_freq=1, verbose=-1)   # subsample_freq=1 so `subsample` is actually applied


def suggest(trial: optuna.Trial) -> dict:
    """Draw one configuration from SEARCH_SPACE."""
    p = {}
    for name, s in SEARCH_SPACE.items():
        if s["type"] == "float":
            p[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
        elif s["type"] == "int":
            if "step" in s:
                p[name] = trial.suggest_int(name, s["low"], s["high"], step=s["step"])
            else:
                p[name] = trial.suggest_int(name, s["low"], s["high"], log=s.get("log", False))
        else:
            p[name] = trial.suggest_categorical(name, s["choices"])
    return p


def lgbm_spec(params: dict, name: str = "LightGBM_tuned") -> T.ModelSpec:
    cfg = {**params, **FIXED}
    return T.ModelSpec(name, "Boosting", cfg, lambda: LGBMClassifier(**cfg, random_state=T.SEED),
                       notes="raw features")


def cv_score(params: dict, train: pd.DataFrame, train_fx: pd.DataFrame, folds: list, arcs: dict,
             trial: optuna.Trial | None = None) -> list[dict]:
    """Fit + evaluate one configuration fold by fold (same protocol as notebook 04).
    With a trial, reports the running mean Top-1 after every fold so the pruner can stop it."""
    spec = lgbm_spec(params)
    usable = T.training_mask(train)
    out = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        tr_rows = train.iloc[tr_idx][usable[tr_idx]]
        va_rows = train.iloc[va_idx]
        model, fit_s, _ = T.fit_model(spec, tr_rows)
        p, pred_s = T.predict(model, va_rows)
        m = T.evaluate(va_rows, train_fx[train_fx.trip_id.isin(set(va_rows.trip_id))], p, arcs)
        out.append({"fold": k, "fit_s": fit_s, "predict_s": pred_s,
                    **{key: v for key, v in m.items() if not key.endswith("_cm")}})
        if trial is not None:
            for key in ("all_top1", "all_macro_f1", "all_logloss", "route_continuity", "fit_s"):
                trial.set_user_attr(f"fold{k}_{key}", out[-1][key])
            trial.report(float(np.mean([o["all_top1"] for o in out])), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return out


def make_objective(train, train_fx, folds, arcs):
    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        res = cv_score(suggest(trial), train, train_fx, folds, arcs, trial)
        trial.set_user_attr("trial_s", time.perf_counter() - t0)
        for key in ("all_top1", "all_macro_f1", "all_logloss", "route_continuity", "reliable_top1"):
            vals = [r[key] for r in res]
            trial.set_user_attr(f"cv_{key}_mean", float(np.mean(vals)))
            trial.set_user_attr(f"cv_{key}_std", float(np.std(vals, ddof=1)))
        return float(np.mean([r["all_top1"] for r in res]))
    return objective


def make_study(storage: str | None = None, name: str = "lightgbm_top1") -> optuna.Study:
    return optuna.create_study(
        study_name=name, storage=storage, load_if_exists=False, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=T.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP_TRIALS, n_warmup_steps=0),
    )
