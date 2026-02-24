"""
src/models/predict.py — Inference / Prediction
================================================

LAYMAN EXPLANATION:
    Once the model is trained, this module handles making predictions
    on NEW, unseen loan applications. It's the "runtime" component —
    it runs 24/7 in the API server.

    Think of this like a doctor who studied for years (training)
    and now uses that knowledge to diagnose new patients (inference).

    Input: A loan application (dictionary of applicant data)
    Output: Default probability, risk grade, decision, and explanation

DATA SCIENTIST EXPLANATION:
    Implements the full inference pipeline:
    1. Accept raw applicant dict or DataFrame
    2. Apply the same feature engineering + preprocessing as training
       (using the SAME fitted pipeline objects — critical for consistency)
    3. Get predictions from XGBoost + LightGBM
    4. Combine with meta-learner for final probability
    5. Apply business rules (threshold → approve/reject)
    6. Return structured prediction with grade and top risk factors

    Thread-safe singleton pattern for model loading ensures the API
    doesn't reload models on every request.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from typing import Union

from config import cfg
from src.models.evaluate import assign_risk_grade


# ============================================================
# MODEL SINGLETON — Load once, reuse forever
# ============================================================

class ModelStore:
    """
    Singleton class that holds loaded models in memory.

    LAYMAN: Loading model files from disk takes 1-2 seconds.
    If the API loaded models on EVERY request, it would be very slow.
    Instead, we load them ONCE when the API starts, then reuse the
    in-memory models for all predictions.

    This is the "singleton" pattern — only one instance exists.

    DATA SCIENTIST NOTE:
    In production, you'd use a proper model serving layer (e.g., BentoML,
    Triton, or SageMaker endpoints). Here we keep it simple with a
    module-level singleton for the FastAPI server.
    """
    _instance = None
    _models_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.xgb_model = None
            cls._instance.lgbm_model = None
            cls._instance.meta_model = None
            cls._instance.feature_engineer = None
            cls._instance.preprocessor = None
            cls._instance.feature_names = None
        return cls._instance

    def load(self):
        """Load all models and pipeline from disk."""
        if self._models_loaded:
            return  # Already loaded, skip

        logger.info("Loading models and pipeline into memory...")

        # Load models
        self.xgb_model  = joblib.load(cfg.model.xgb_model_path)
        self.lgbm_model = joblib.load(cfg.model.lgbm_model_path)
        self.meta_model = joblib.load(cfg.model.meta_model_path)

        # Load feature pipeline
        pipeline_data = joblib.load(cfg.model.feature_pipeline_path)
        self.feature_engineer = pipeline_data["feature_engineer"]
        self.preprocessor     = pipeline_data["preprocessor"]
        self.feature_names    = joblib.load(cfg.model.feature_names_path)

        self._models_loaded = True
        logger.info("All models loaded successfully.")

    def is_loaded(self) -> bool:
        return self._models_loaded


# Global model store instance
model_store = ModelStore()


# ============================================================
# CORE PREDICTION FUNCTIONS
# ============================================================

def predict_single(applicant: dict) -> dict:
    """
    Score a single loan applicant.

    LAYMAN: This is the main function called by the API for each
    loan application. It takes the applicant's data, runs it through
    the model, and returns a score + decision + explanation.

    Example input:
    {
        "loan_amnt": 15000,
        "annual_inc": 65000,
        "dti": 18.5,
        "fico_range_low": 690,
        ...
    }

    Example output:
    {
        "applicant_id": "APP_001",
        "default_probability": 0.23,
        "risk_grade": "B",
        "decision": "APPROVE",
        "top_risk_factors": ["High DTI ratio (35%)", ...]
    }

    Args:
        applicant: Dictionary of applicant features

    Returns:
        dict: Prediction result with score, grade, decision
    """
    if not model_store.is_loaded():
        model_store.load()

    # Convert single applicant dict to 1-row DataFrame
    df = pd.DataFrame([applicant])

    # Get prediction
    probabilities = _get_ensemble_probability(df)
    prob = float(probabilities[0])

    # Apply business rules
    grade = assign_risk_grade(prob)
    decision = "REJECT" if prob >= cfg.scoring.rejection_threshold else "APPROVE"

    return {
        "default_probability": round(prob, 4),
        "risk_grade": grade,
        "decision": decision,
        "model_version": "1.0.0",
    }


def predict_batch(applicants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score multiple loan applicants at once.

    LAYMAN: Instead of scoring one applicant at a time, this processes
    thousands simultaneously — much more efficient for batch jobs
    like scoring all applications received overnight.

    DATA SCIENTIST NOTE:
    Batch prediction is critical for operational use cases:
    - Nightly batch scoring of all active loan applications
    - Scoring historical portfolios for audit
    - A/B testing new model versions on existing portfolios

    Args:
        applicants_df: DataFrame with one row per applicant

    Returns:
        pd.DataFrame: Original data + prediction columns appended
    """
    if not model_store.is_loaded():
        model_store.load()

    logger.info(f"Batch scoring {len(applicants_df):,} applicants...")

    # Get ensemble probabilities for all applicants
    probabilities = _get_ensemble_probability(applicants_df)

    # Create results DataFrame
    results = applicants_df.copy()
    results["default_probability"] = np.round(probabilities, 4)
    results["risk_grade"] = results["default_probability"].apply(assign_risk_grade)
    results["decision"] = results["default_probability"].apply(
        lambda p: "REJECT" if p >= cfg.scoring.rejection_threshold else "APPROVE"
    )

    n_approved = (results["decision"] == "APPROVE").sum()
    n_rejected = (results["decision"] == "REJECT").sum()
    logger.info(
        f"Batch scoring complete. "
        f"Approved: {n_approved:,} ({n_approved/len(results):.1%}), "
        f"Rejected: {n_rejected:,} ({n_rejected/len(results):.1%})"
    )

    return results


def _get_ensemble_probability(df: pd.DataFrame) -> np.ndarray:
    """
    Internal function: get final ensemble probability for a DataFrame.

    LAYMAN: This is the "engine room" — it applies the pipeline and
    runs the data through all three models (XGBoost + LightGBM +
    meta-learner) to produce the final probability.

    DATA SCIENTIST NOTE:
    Pipeline:
    1. Feature engineering (creates new ratio features)
    2. Preprocessing (impute NaN, scale, encode)
    3. XGBoost → probability
    4. LightGBM → probability
    5. [XGB_prob, LGBM_prob] → meta-learner → final probability
    """
    # Step 1: Feature engineering
    df_engineered = model_store.feature_engineer.transform(df)

    # Remove target column if present
    target = cfg.data.target_column
    if target in df_engineered.columns:
        df_engineered = df_engineered.drop(columns=[target])

    # Step 2: Fill missing columns that were present during training.
    # At inference the input may not have all training columns (e.g. funded_amnt).
    # The SimpleImputer inside the preprocessor will fill NaN with training medians.
    expected_num_cols = model_store.preprocessor.transformers_[0][2]
    expected_cat_cols = model_store.preprocessor.transformers_[1][2]
    for col in list(expected_num_cols) + list(expected_cat_cols):
        if col not in df_engineered.columns:
            df_engineered[col] = np.nan

    # Step 3: Preprocessing (impute, scale, encode)
    X = model_store.preprocessor.transform(df_engineered)

    # Step 3: Get XGBoost probability
    xgb_prob = model_store.xgb_model.predict_proba(X)[:, 1]

    # Step 4: Get LightGBM probability
    lgbm_prob = model_store.lgbm_model.predict_proba(X)[:, 1]

    # Step 5: Stack and apply meta-learner
    stacked = np.column_stack([xgb_prob, lgbm_prob])
    final_prob = model_store.meta_model.predict_proba(stacked)[:, 1]

    return final_prob


def get_model_info() -> dict:
    """
    Return metadata about the currently loaded model.

    LAYMAN: Like a label on a medicine bottle — shows which version
    of the model is running, when it was trained, and basic stats.

    Returns:
        dict: Model metadata
    """
    return {
        "model_version": "1.0.0",
        "models_loaded": model_store.is_loaded(),
        "ensemble": ["XGBoost", "LightGBM", "LogisticRegression (meta)"],
        "rejection_threshold": cfg.scoring.rejection_threshold,
        "risk_grades": list(cfg.scoring.grade_thresholds.keys()),
    }
