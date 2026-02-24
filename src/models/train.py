"""
src/models/train.py — Model Training
======================================

LAYMAN EXPLANATION:
    This is where the actual LEARNING happens.

    We train THREE models:
    1. XGBoost — learns patterns from the training data
    2. LightGBM — learns the same patterns, slightly differently
    3. Logistic Regression (meta-learner) — learns how to COMBINE
       the two models above for the best final prediction

    This approach (using multiple models) is called "Stacking."
    Think of it like getting a second medical opinion — one doctor
    (XGBoost) gives a diagnosis, another (LightGBM) gives a different
    one, and a specialist (Logistic Regression) weighs both opinions
    to give the final verdict.

    WHY STACKING?
    Each model has different strengths and weaknesses. Their errors
    are somewhat uncorrelated. Combining them reduces the total error.
    This typically improves AUC by 2-5% over any single model.

DATA SCIENTIST EXPLANATION:
    Implements a 2-level stacking ensemble:

    Level 1:
    - XGBoostClassifier: depth-6 trees, early stopping, AUC metric
    - LGBMClassifier: leaf-wise growth, 63 leaves, AUC metric

    Level 2:
    - LogisticRegression meta-learner trained on out-of-fold (OOF)
      predictions from Level 1 models

    OOF approach prevents leakage in the stacking layer —
    the meta-learner never sees Level 1 predictions made on the
    same data the Level 1 models were trained on.

    All models are tracked via MLflow for experiment comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from loguru import logger
from typing import Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import cfg
from src.models.tune import load_best_params


def train_xgboost(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
) -> XGBClassifier:
    """
    Train the XGBoost model with early stopping.

    LAYMAN: XGBoost builds many decision trees in sequence.
    Each new tree tries to fix the mistakes of the previous ones.
    "Early stopping" means we stop adding trees when performance
    on the validation set stops improving — this prevents overfitting
    (memorizing training data without learning generalizable patterns).

    DATA SCIENTIST NOTE:
    Early stopping monitors val AUC and stops if it doesn't improve
    for 50 consecutive rounds. This prevents overfitting while
    automatically finding the optimal n_estimators.

    scale_pos_weight compensates for class imbalance — since defaults
    are rare (~10-15%), we upweight them so the model takes them seriously.
    The weight is approximately: count(negative) / count(positive).
    """
    logger.info("Training XGBoost model...")

    # Calculate class weight for imbalance handling
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 10
    logger.info(
        f"  Class balance: {n_neg:,} negatives, {n_pos:,} positives "
        f"(scale_pos_weight={scale_pos_weight:.1f})"
    )

    # Start from config defaults, then overlay any tuned params
    params = cfg.model.xgb_params.copy()
    tuned = load_best_params("xgb")
    if tuned:
        params.update(tuned)
        logger.info("  Using Optuna-tuned XGBoost hyperparameters.")
    params["scale_pos_weight"] = scale_pos_weight

    # Remove non-sklearn params that XGBClassifier doesn't accept via constructor
    params.pop("use_label_encoder", None)
    params.pop("eval_metric", None)

    model = XGBClassifier(
        **params,
        early_stopping_rounds=50,  # Stop if val AUC doesn't improve for 50 rounds
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  # Monitor validation AUC
        verbose=100,                 # Print progress every 100 trees
    )

    best_iter = model.best_iteration
    logger.info(
        f"XGBoost training complete. "
        f"Best iteration: {best_iter}, "
        f"Best val AUC: {model.best_score:.4f}"
    )

    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
) -> LGBMClassifier:
    """
    Train the LightGBM model.

    LAYMAN: LightGBM is similar to XGBoost but:
    - Faster (uses "histogram-based" splitting)
    - Better at handling categorical features natively
    - Grows trees leaf-by-leaf (vs XGBoost's level-by-level)
      which means it can focus on the most important splits first

    DATA SCIENTIST NOTE:
    Key differences from XGBoost:
    - num_leaves controls complexity (not max_depth)
    - min_child_samples prevents tiny leaf nodes
    - class_weight='balanced' auto-computes class weights
    - callbacks replace early_stopping_rounds parameter style
    """
    logger.info("Training LightGBM model...")

    import lightgbm as lgb

    params = cfg.model.lgbm_params.copy()
    tuned = load_best_params("lgbm")
    if tuned:
        params.update(tuned)
        logger.info("  Using Optuna-tuned LightGBM hyperparameters.")

    # LightGBM callbacks for early stopping and logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model = LGBMClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=callbacks,
    )

    logger.info(
        f"LightGBM training complete. "
        f"Best iteration: {model.best_iteration_}"
    )

    return model


def generate_oof_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    n_folds: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Out-of-Fold (OOF) predictions for stacking.

    LAYMAN: Imagine you have 5 students (folds) taking an exam.
    Each time, 4 students study (train) while 1 student is tested.
    We rotate which student is tested. By the end, every student
    has been tested exactly once without studying from test questions.

    This is cross-validation. OOF predictions are the test answers
    collected from each student's "test" turn. The meta-learner
    uses these to learn how to combine XGBoost and LightGBM.

    WHY NOT JUST TRAIN ON ALL DATA?
    If we trained XGBoost on all training data and then made predictions
    on the same data to train the meta-learner, the meta-learner would
    learn from PERFECT predictions (the models memorized the data).
    It would overfit catastrophically.

    OOF predictions solve this: each prediction is made on data the
    model NEVER saw during training → honest predictions.

    DATA SCIENTIST NOTE:
    5-fold stratified CV. OOF predictions are stacked as a 2-column
    matrix [xgb_prob, lgbm_prob] which becomes the input to the
    meta-learner.
    """
    logger.info(f"Generating {n_folds}-fold OOF predictions for stacking...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.data.random_state)

    # Arrays to store OOF predictions
    oof_xgb = np.zeros(len(X_train))   # XGBoost OOF predictions
    oof_lgbm = np.zeros(len(X_train))  # LightGBM OOF predictions

    # For validation set: average predictions across all folds
    val_xgb_preds = np.zeros(len(X_val))
    val_lgbm_preds = np.zeros(len(X_val))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info(f"  Fold {fold + 1}/{n_folds}...")

        # Split this fold's train/validation
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # --- Train XGBoost on this fold ---
        n_pos = int(y_fold_train.sum())
        n_neg = len(y_fold_train) - n_pos
        spw = n_neg / max(n_pos, 1)

        xgb_params = cfg.model.xgb_params.copy()
        tuned_xgb = load_best_params("xgb")
        if tuned_xgb:
            xgb_params.update(tuned_xgb)
        xgb_params.pop("use_label_encoder", None)
        xgb_params.pop("eval_metric", None)
        xgb_params["scale_pos_weight"] = spw
        xgb_params["n_estimators"] = 200  # Fewer trees for speed in CV
        xgb_params.pop("early_stopping_rounds", None)

        fold_xgb = XGBClassifier(**xgb_params)
        fold_xgb.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False,
        )

        # --- Train LightGBM on this fold ---
        lgbm_params = cfg.model.lgbm_params.copy()
        tuned_lgbm = load_best_params("lgbm")
        if tuned_lgbm:
            lgbm_params.update(tuned_lgbm)
        lgbm_params["n_estimators"] = 200  # Fewer trees for speed in CV

        import lightgbm as lgb
        fold_lgbm = LGBMClassifier(**lgbm_params)
        fold_lgbm.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            eval_metric="auc",
            callbacks=[lgb.log_evaluation(period=-1)],  # Suppress output
        )

        # --- Store OOF predictions ---
        oof_xgb[val_idx] = fold_xgb.predict_proba(X_fold_val)[:, 1]
        oof_lgbm[val_idx] = fold_lgbm.predict_proba(X_fold_val)[:, 1]

        # --- Accumulate validation predictions ---
        val_xgb_preds += fold_xgb.predict_proba(X_val)[:, 1] / n_folds
        val_lgbm_preds += fold_lgbm.predict_proba(X_val)[:, 1] / n_folds

    logger.info("OOF generation complete.")

    return oof_xgb, oof_lgbm, val_xgb_preds, val_lgbm_preds


def train_meta_learner(
    oof_xgb: np.ndarray,
    oof_lgbm: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """
    Train the Logistic Regression meta-learner on OOF predictions.

    LAYMAN: The meta-learner's job is to answer:
    "Given that XGBoost says 30% probability and LightGBM says 40%,
    what is the BEST combined estimate?"

    It learns the optimal weighting of the two models.
    Sometimes XGBoost is more accurate for certain types of borrowers,
    sometimes LightGBM is better. The meta-learner figures this out.

    DATA SCIENTIST NOTE:
    Input: (n_samples, 2) matrix of [xgb_prob, lgbm_prob]
    Output: Final probability (scalar 0-1)
    Logistic regression is ideal here — it's interpretable, can't
    overfit a 2-feature problem, and outputs calibrated probabilities.
    """
    logger.info("Training meta-learner (Logistic Regression)...")

    # Stack OOF predictions as 2-column matrix
    oof_features = np.column_stack([oof_xgb, oof_lgbm])

    meta_model = LogisticRegression(**cfg.model.meta_params)
    meta_model.fit(oof_features, y_train)

    # Log the learned weights (shows which model the meta-learner trusts more)
    xgb_weight, lgbm_weight = meta_model.coef_[0]
    logger.info(
        f"Meta-learner trained. "
        f"Weights — XGBoost: {xgb_weight:.3f}, LightGBM: {lgbm_weight:.3f}"
    )

    return meta_model


def save_models(xgb_model, lgbm_model, meta_model):
    """
    Save all trained models to disk.

    LAYMAN: Like saving a trained athlete's "muscle memory" to a file.
    We save the trained models so the API can load them quickly without
    retraining. XGBoost with 500 trees would take 5 minutes to retrain
    on every API restart — loading from disk takes <1 second.
    """
    Path(cfg.model.xgb_model_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb_model, cfg.model.xgb_model_path)
    joblib.dump(lgbm_model, cfg.model.lgbm_model_path)
    joblib.dump(meta_model, cfg.model.meta_model_path)

    logger.info(f"Models saved to {Path(cfg.model.xgb_model_path).parent}")


def load_models():
    """
    Load saved models from disk.

    Returns:
        (xgb_model, lgbm_model, meta_model)
    """
    xgb_model = joblib.load(cfg.model.xgb_model_path)
    lgbm_model = joblib.load(cfg.model.lgbm_model_path)
    meta_model = joblib.load(cfg.model.meta_model_path)
    logger.info("All models loaded from disk.")
    return xgb_model, lgbm_model, meta_model


def run_training_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    use_mlflow: bool = True,
) -> tuple:
    """
    Master function: run the complete training pipeline.

    LAYMAN: This is the big button that trains everything end-to-end:
    1. Train XGBoost
    2. Train LightGBM
    3. Generate OOF predictions for stacking
    4. Train the meta-learner
    5. Save everything to disk
    6. Log all metrics to MLflow (experiment tracking)

    After running this, the models are ready for scoring.

    Args:
        X_train, y_train: Training features and labels
        X_val, y_val: Validation features and labels
        use_mlflow: Whether to log to MLflow experiment tracker

    Returns:
        (xgb_model, lgbm_model, meta_model): Trained models
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Convert Series to arrays if needed
    y_train_arr = np.array(y_train)
    y_val_arr = np.array(y_val)

    run_context = mlflow.start_run() if use_mlflow else None

    try:
        if use_mlflow:
            mlflow.log_params({
                "xgb_n_estimators": cfg.model.xgb_params.get("n_estimators"),
                "xgb_max_depth": cfg.model.xgb_params.get("max_depth"),
                "xgb_learning_rate": cfg.model.xgb_params.get("learning_rate"),
                "lgbm_n_estimators": cfg.model.lgbm_params.get("n_estimators"),
                "lgbm_num_leaves": cfg.model.lgbm_params.get("num_leaves"),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "default_rate_train": float(y_train_arr.mean()),
            })

        # Step 1: Train XGBoost
        xgb_model = train_xgboost(X_train, y_train_arr, X_val, y_val_arr)

        # Step 2: Train LightGBM
        lgbm_model = train_lightgbm(X_train, y_train_arr, X_val, y_val_arr)

        # Step 3: Generate OOF predictions for stacking
        oof_xgb, oof_lgbm, val_xgb, val_lgbm = generate_oof_predictions(
            X_train, y_train_arr, X_val, n_folds=5
        )

        # Step 4: Train meta-learner
        meta_model = train_meta_learner(oof_xgb, oof_lgbm, y_train_arr)

        # Step 5: Save models
        save_models(xgb_model, lgbm_model, meta_model)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)

    finally:
        if use_mlflow and run_context:
            mlflow.end_run()

    return xgb_model, lgbm_model, meta_model
