"""
src/models/tune.py — Hyperparameter Tuning with Optuna
=======================================================

LAYMAN EXPLANATION:
    Imagine you are baking a cake. The recipe says "bake at 350°F for 30 minutes",
    but maybe 375°F for 25 minutes gives a better result. Hyperparameter tuning
    does exactly this for ML models — it automatically tries hundreds of different
    "recipe settings" (hyperparameters) and finds the combination that makes the
    model most accurate.

    Optuna is the search engine that intelligently picks WHICH settings to try
    next, based on what worked before. It is smarter than random guessing because
    it uses Bayesian optimization (TPE sampler) — it learns from past trials.

DATA SCIENTIST EXPLANATION:
    MEMORY-SAFE DESIGN:
    - n_jobs=1 everywhere → single process, no memory-exploding parallelism
    - cross_val_score(n_jobs=1) → sequential CV folds, no subprocess spawning
    - 3-fold CV (not 5) → 40% fewer model fits per trial
    - 8,000-row subsample for tuning → each trial uses only 8k rows
    - n_estimators capped at 300 → prevents very slow/large trees per trial
    - gc_after_trial=True → free memory between trials

    Why n_jobs=-1 kills your machine:
      n_jobs=-1 tells sklearn/XGB/LGBM to spawn 1 worker per CPU core.
      With 3-fold CV × 8 cores = 24 model copies in RAM simultaneously.
      On 20k rows, each copy ~500MB → 12GB just for CV data copies.
      Multiply by 2 models (XGB+LGBM) = crash.

    SEARCH SPACES:
       XGBoost: max_depth, learning_rate, n_estimators, subsample,
                colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda
       LightGBM: num_leaves, learning_rate, n_estimators, subsample,
                 colsample_bytree, min_child_samples, reg_alpha, reg_lambda

    USAGE:
        python run.py --mode tune --trials 30
        python run.py --mode tune --trials 50 --model xgb
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import gc
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.model_selection import StratifiedKFold, cross_val_score

import xgboost as xgb
import lightgbm as lgb

from config import cfg, MODELS_DIR
from src.data.ingestion import load_raw_data, validate_data
from src.features.pipeline import build_full_pipeline, transform_data

# Silence noisy output during tuning
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Memory safety constants ────────────────────────────────────────────────
TUNE_SAMPLE_SIZE = 8_000   # rows used per trial (enough signal, small footprint)
CV_FOLDS         = 3        # 3-fold is fast and reliable; 5-fold is overkill here
N_JOBS           = 1        # NEVER use -1: it spawns 1 process per CPU core
MAX_ESTIMATORS   = 300      # cap trees per trial to keep fit time reasonable
# ──────────────────────────────────────────────────────────────────────────


# ============================================================
# OPTUNA OBJECTIVE FUNCTIONS
# ============================================================

def xgb_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """
    Optuna objective for XGBoost.
    Returns mean AUC-ROC from 3-fold CV. Single-threaded — memory-safe.
    """
    params = {
        # Tree structure
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma":            trial.suggest_float("gamma", 0.0, 1.0),

        # Sampling
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),

        # Regularisation
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),

        # Learning
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 80, MAX_ESTIMATORS),

        # Fixed — n_jobs=1 is CRITICAL for memory safety
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "random_state":     cfg.data.random_state,
        "n_jobs":           N_JOBS,
        "verbosity":        0,
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=cfg.data.random_state
    )
    model = xgb.XGBClassifier(**params)

    # n_jobs=1 here → sequential folds, zero subprocess spawning
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=N_JOBS)
    mean_auc = float(scores.mean())

    trial.report(mean_auc, step=0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return mean_auc


def lgbm_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """
    Optuna objective for LightGBM.
    Returns mean AUC-ROC from 3-fold CV. Single-threaded — memory-safe.
    """
    params = {
        # Tree structure
        "num_leaves":        trial.suggest_int("num_leaves", 16, 128),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),

        # Sampling
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq":    trial.suggest_int("subsample_freq", 1, 5),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),

        # Regularisation
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),

        # Learning
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 80, MAX_ESTIMATORS),

        # Fixed — n_jobs=1 is CRITICAL for memory safety
        "objective":         "binary",
        "metric":            "auc",
        "verbosity":         -1,
        "random_state":      cfg.data.random_state,
        "n_jobs":            N_JOBS,
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=cfg.data.random_state
    )
    model = lgb.LGBMClassifier(**params)

    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=N_JOBS)
    mean_auc = float(scores.mean())

    trial.report(mean_auc, step=0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return mean_auc


# ============================================================
# MAIN TUNING PIPELINE
# ============================================================

def run_tuning_pipeline(
    n_trials: int = 30,
    model_type: str = "both",
    save_results: bool = True,
) -> dict:
    """
    Master tuning pipeline: load data → preprocess → Optuna → save best params.

    Memory-safe: data is subsampled to TUNE_SAMPLE_SIZE rows before tuning.
    Best params are saved to JSON and auto-loaded by the next `--mode train`.

    Args:
        n_trials:     Optuna trials per model (30 is a good default)
        model_type:   "xgb", "lgbm", or "both"
        save_results: Save best params JSON files

    Returns:
        dict: Best params for each tuned model
    """
    logger.info("=" * 60)
    logger.info(f"HYPERPARAMETER TUNING — {n_trials} trials per model")
    logger.info(f"Memory-safe mode: {TUNE_SAMPLE_SIZE} rows, {CV_FOLDS}-fold CV, n_jobs=1")
    logger.info("=" * 60)

    # ── Step 1: Load + preprocess ──────────────────────────────
    logger.info("Loading and preprocessing data...")
    df_raw       = load_raw_data()
    df_validated = validate_data(df_raw)

    feature_engineer, preprocessor, feature_names = build_full_pipeline(df_validated)

    target = cfg.data.target_column
    X_full, y_full = transform_data(df_validated, feature_engineer, preprocessor, target)

    # Convert string labels → binary int
    if hasattr(y_full, "values"):
        y_full = y_full.values
    if y_full.dtype == object:
        y_full = (y_full == cfg.data.default_label).astype(int)

    # ── Step 2: Subsample for tuning (memory safety) ───────────
    n_total = len(y_full)
    if n_total > TUNE_SAMPLE_SIZE:
        rng = np.random.RandomState(cfg.data.random_state)
        # Stratified subsample: keep class balance
        pos_idx = np.where(y_full == 1)[0]
        neg_idx = np.where(y_full == 0)[0]
        n_pos   = int(TUNE_SAMPLE_SIZE * y_full.mean())
        n_neg   = TUNE_SAMPLE_SIZE - n_pos

        chosen_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
        chosen_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
        idx        = np.concatenate([chosen_pos, chosen_neg])
        rng.shuffle(idx)

        X = X_full[idx]
        y = y_full[idx]
        logger.info(
            f"Subsampled {n_total:,} → {len(y):,} rows for tuning "
            f"(default_rate={y.mean():.1%})"
        )
    else:
        X, y = X_full, y_full
        logger.info(f"Using full dataset: {len(y):,} rows")

    # Free the full arrays — we only need the subsample during tuning
    del X_full, y_full, df_raw, df_validated
    gc.collect()

    logger.info(f"Tuning feature matrix: {X.shape} | RAM used by X: "
                f"{X.nbytes / 1e6:.1f} MB")

    # ── Step 3: Run Optuna ─────────────────────────────────────
    results = {}

    if model_type in ("xgb", "both"):
        results["xgb"] = _tune_model(
            model_name="XGBoost",
            objective_fn=xgb_objective,
            X=X, y=y,
            n_trials=n_trials,
            save_results=save_results,
            output_path=MODELS_DIR / "best_params_xgb.json",
        )
        gc.collect()

    if model_type in ("lgbm", "both"):
        results["lgbm"] = _tune_model(
            model_name="LightGBM",
            objective_fn=lgbm_objective,
            X=X, y=y,
            n_trials=n_trials,
            save_results=save_results,
            output_path=MODELS_DIR / "best_params_lgbm.json",
        )
        gc.collect()

    # ── Step 4: Summary ────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TUNING COMPLETE — Best Results:")
    for name, result in results.items():
        logger.info(f"  {name.upper()}: Best CV AUC = {result['best_value']:.4f}")
    logger.info("=" * 60)
    logger.info("Next step: python run.py --mode train")

    return results


def _tune_model(
    model_name: str,
    objective_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    save_results: bool,
    output_path: Path,
) -> dict:
    """Run Optuna for one model, log progress every 5 trials, save best params."""
    logger.info(f"\nTuning {model_name} ({n_trials} trials) ...")

    def objective(trial):
        return objective_fn(trial, X, y)

    sampler = TPESampler(seed=cfg.data.random_state, n_startup_trials=8)
    pruner  = MedianPruner(n_startup_trials=8, n_warmup_steps=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{model_name.lower()}_tuning",
    )

    completed = [0]

    def _cb(study, trial):
        completed[0] += 1
        if completed[0] % 5 == 0:
            logger.info(
                f"  [{model_name}] Trial {completed[0]:3d}/{n_trials} | "
                f"Best AUC: {study.best_value:.4f}"
            )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_cb],
        show_progress_bar=False,
        gc_after_trial=True,   # free each trial's memory before next
    )

    best_params = study.best_params
    best_value  = study.best_value

    logger.info(f"\n{model_name} BEST AUC: {best_value:.4f}")
    logger.info(f"{model_name} Best Hyperparameters:")
    for k, v in sorted(best_params.items()):
        logger.info(f"    {k:30s} = {v}")

    if save_results:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "model":       model_name,
                    "best_auc":    best_value,
                    "best_params": best_params,
                    "n_trials":    n_trials,
                    "n_completed": len(study.trials),
                },
                f,
                indent=2,
            )
        logger.info(f"Best params saved → {output_path}")

    return {"best_params": best_params, "best_value": best_value, "study": study}


# ============================================================
# LOAD TUNED PARAMS (called by train.py automatically)
# ============================================================

def load_best_params(model_type: str) -> dict:
    """
    Load previously saved best hyperparameters from JSON.
    Called by train.py before constructing models.
    Returns {} if no tuning has been run yet (falls back to config defaults).
    """
    param_file = MODELS_DIR / f"best_params_{model_type}.json"

    if not param_file.exists():
        logger.debug(
            f"No tuned params at {param_file}. "
            "Using config.py defaults."
        )
        return {}

    with open(param_file) as f:
        data = json.load(f)

    best_params = data.get("best_params", {})
    best_auc    = data.get("best_auc", 0.0)
    logger.info(f"Loaded tuned {model_type.upper()} params (tuning CV AUC: {best_auc:.4f})")
    return best_params


# ============================================================
# CLI (direct invocation)
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--model",  choices=["xgb", "lgbm", "both"], default="both")
    args = parser.parse_args()

    run_tuning_pipeline(n_trials=args.trials, model_type=args.model)
