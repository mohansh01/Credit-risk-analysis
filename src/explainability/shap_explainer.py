"""
src/explainability/shap_explainer.py — SHAP Explainability
============================================================

LAYMAN EXPLANATION:
    Imagine you apply for a loan and get rejected. You ask the bank:
    "WHY was I rejected?" The bank is legally required to tell you.

    But a black-box ML model can't explain itself. SHAP (SHapley
    Additive exPlanations) solves this:

    It says: "You were rejected because:
    ✗ Your debt-to-income ratio of 45% pushed your risk score UP by 0.23
    ✗ You had 3 late payments in the past 2 years (pushed UP by 0.18)
    ✓ Your FICO score of 710 pulled your risk score DOWN by 0.12
    ✓ You've had credit for 12 years (pulled DOWN by 0.08)
    → Final score: 0.62 (above 0.50 rejection threshold)"

    This explanation satisfies:
    - The customer ("I understand why I was rejected")
    - The regulator ("The bank can justify its decision")
    - The data scientist ("The model is using sensible features")

DATA SCIENTIST EXPLANATION:
    SHAP values are based on Shapley values from cooperative game theory.
    They satisfy three key axioms:
    1. Efficiency: SHAP values sum to the prediction (minus baseline)
    2. Symmetry: Identical features get equal SHAP values
    3. Null player: Features with no effect get SHAP value = 0

    TreeExplainer is used for XGBoost/LightGBM — it computes exact
    SHAP values in O(TLD^2) time (T=trees, L=leaves, D=depth),
    much faster than the model-agnostic KernelExplainer.

    SHAP values are computed in the original feature space (before
    the preprocessor transforms), making them more interpretable.
    We compute them on the XGBoost model (slightly more interpretable
    than LightGBM for SHAP visualization due to tree structure).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from typing import Optional

from config import cfg, REPORTS_DIR


class SHAPExplainer:
    """
    Generates SHAP explanations for individual predictions and global analysis.

    LAYMAN: Think of this as the model's "translator" — it converts
    the model's complex mathematical reasoning into plain English
    explanations that humans can understand.

    Usage:
        explainer = SHAPExplainer(xgb_model, feature_names)
        explainer.fit(X_background)  # Learn the "baseline" prediction

        # Explain a single prediction
        explanation = explainer.explain_single(X_applicant)
        print(explanation["top_reasons"])
    """

    def __init__(self, model, feature_names: list):
        """
        Initialize the SHAP explainer.

        Args:
            model: Trained XGBoost or LightGBM model
            feature_names: List of feature names in the same order
                           as the model's input features
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.expected_value = None  # Baseline prediction (average prediction)

    def fit(self, X_background: np.ndarray, sample_size: int = 500):
        """
        Fit the SHAP explainer using background data.

        LAYMAN: The "background" is a representative sample of the training data.
        SHAP uses this to compute the "baseline" — what the model would predict
        on average with no specific information about the applicant.

        The difference between the baseline and the actual prediction is what
        SHAP explains: "How did each feature move the prediction from baseline?"

        DATA SCIENTIST NOTE:
        TreeExplainer is deterministic — it doesn't need background data for
        exact SHAP values. We use a background sample as the reference point
        for force plots. For TreeExplainer, fit is mainly setting expected_value.

        Args:
            X_background: Background dataset (representative sample of training data)
            sample_size: How many background samples to use (more = more accurate
                        but slower for KernelExplainer; doesn't affect TreeExplainer)
        """
        logger.info("Fitting SHAP TreeExplainer...")

        # Sample background data if too large
        if len(X_background) > sample_size:
            idx = np.random.choice(len(X_background), sample_size, replace=False)
            X_background = X_background[idx]

        # TreeExplainer: fast, exact SHAP for tree-based models (XGBoost/LightGBM)
        # model_output='probability' ensures SHAP values are in probability space
        self.explainer = shap.TreeExplainer(
            self.model,
            data=X_background,
            model_output='raw',  # 'raw' = log-odds space; transform to probability after
        )

        # Expected value = model's baseline prediction (average prediction)
        self.expected_value = self.explainer.expected_value
        if isinstance(self.expected_value, list):
            self.expected_value = self.expected_value[1]  # Binary classification: class 1

        logger.info(
            f"SHAP explainer fitted. "
            f"Baseline prediction (expected value): {self.expected_value:.4f}"
        )

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a dataset.

        LAYMAN: For each applicant and each feature, compute:
        "How much did THIS feature contribute to THIS applicant's score?"

        A positive SHAP value means the feature INCREASED the default risk.
        A negative SHAP value means it DECREASED the default risk.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            np.ndarray: SHAP values (n_samples, n_features)
        """
        if self.explainer is None:
            raise RuntimeError("Call fit() before computing SHAP values!")

        logger.info(f"Computing SHAP values for {len(X)} samples...")
        shap_values = self.explainer.shap_values(X)

        # For binary classifiers, shap_values may be a list [class0, class1]
        # We want class 1 (default probability)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        logger.info(f"SHAP values computed. Shape: {shap_values.shape}")
        return shap_values

    def explain_single(
        self,
        X_single: np.ndarray,
        applicant_id: str = "Applicant",
    ) -> dict:
        """
        Generate a human-readable explanation for a single prediction.

        LAYMAN: Takes one applicant's features and returns:
        1. Top risk factors (reasons for high score / rejection)
        2. Top protective factors (reasons for low score)
        3. The magnitude of each factor's impact

        The output is formatted for both:
        - The rejection letter the customer receives
        - The regulatory audit trail the bank must maintain

        DATA SCIENTIST NOTE:
        SHAP values are in log-odds space (from TreeExplainer with raw output).
        We convert to probability space for display by applying sigmoid.
        The ranking is by absolute SHAP value — we show both direction and magnitude.

        Args:
            X_single: Feature vector for one applicant (shape: (1, n_features)
                      or (n_features,))
            applicant_id: Name/ID for the explanation (for display)

        Returns:
            dict: {
                "top_risk_factors": [...],      # Features increasing default risk
                "top_protective_factors": [...], # Features decreasing default risk
                "shap_values_dict": {...},       # All SHAP values
                "baseline_probability": float,  # Model's average prediction
                "explained_prediction": float,  # SHAP-explained probability
            }
        """
        if self.explainer is None:
            raise RuntimeError("Call fit() before explaining predictions!")

        # Ensure correct shape (2D array)
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)

        # Compute SHAP values
        shap_vals = self.explainer.shap_values(X_single)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = shap_vals[0]  # Get first (only) row

        # Pair feature names with their SHAP values
        feature_shap = list(zip(self.feature_names[:len(shap_vals)], shap_vals))

        # Sort by absolute SHAP value (most impactful first)
        feature_shap_sorted = sorted(feature_shap, key=lambda x: abs(x[1]), reverse=True)

        # Get raw feature values for display
        feature_values = dict(zip(self.feature_names[:X_single.shape[1]], X_single[0]))

        # Top N to show (from config)
        n_show = cfg.scoring.top_features_to_explain

        # --- Format risk factors (positive SHAP = increases risk) ---
        risk_factors = []
        for feature, shap_val in feature_shap_sorted[:n_show * 2]:
            if shap_val > 0:  # Increases risk
                raw_val = feature_values.get(feature, "N/A")
                factor = self._format_factor_description(
                    feature, raw_val, shap_val, is_risk=True
                )
                risk_factors.append(factor)
                if len(risk_factors) >= n_show:
                    break

        # --- Format protective factors (negative SHAP = decreases risk) ---
        protective_factors = []
        for feature, shap_val in feature_shap_sorted[:n_show * 2]:
            if shap_val < 0:  # Decreases risk
                raw_val = feature_values.get(feature, "N/A")
                factor = self._format_factor_description(
                    feature, raw_val, shap_val, is_risk=False
                )
                protective_factors.append(factor)
                if len(protective_factors) >= n_show:
                    break

        # All SHAP values as dictionary (for API response)
        shap_dict = {
            feat: round(float(val), 6)
            for feat, val in feature_shap_sorted
        }

        return {
            "applicant_id": applicant_id,
            "top_risk_factors": risk_factors[:n_show],
            "top_protective_factors": protective_factors[:n_show],
            "shap_values": shap_dict,
            "baseline_log_odds": round(float(self.expected_value), 4),
            "n_features_explained": len(shap_vals),
        }

    def _format_factor_description(
        self,
        feature: str,
        raw_value,
        shap_value: float,
        is_risk: bool,
    ) -> dict:
        """
        Format a single SHAP factor into a human-readable description.

        LAYMAN: Converts technical numbers into plain English.
        "dti=35.2, shap=+0.23" → "High debt-to-income ratio (35.2%) (+0.23 impact)"

        Returns:
            dict: {
                "feature": feature name,
                "value": raw value,
                "impact": SHAP value,
                "description": human-readable explanation
            }
        """
        # Feature name → human readable label mapping
        feature_labels = {
            "dti": "Debt-to-income ratio",
            "fico_score": "FICO credit score",
            "revol_util": "Credit card utilization",
            "delinq_2yrs": "Late payments (past 2 years)",
            "annual_inc": "Annual income",
            "loan_amnt": "Requested loan amount",
            "int_rate": "Interest rate",
            "open_acc": "Open credit accounts",
            "pub_rec": "Public derogatory records",
            "credit_age_years": "Credit history age",
            "payment_to_income": "Payment-to-income ratio",
            "loan_to_income": "Loan-to-income ratio",
            "delinquency_severity": "Delinquency severity score",
            "high_utilization_flag": "High credit utilization flag",
            "derogatory_flag": "Derogatory history flag",
            "inq_last_6mths": "Recent credit inquiries",
        }

        label = feature_labels.get(feature, feature.replace("_", " ").title())
        direction = "↑ increases risk" if is_risk else "↓ decreases risk"
        impact_str = f"+{shap_value:.4f}" if shap_value > 0 else f"{shap_value:.4f}"

        # Format value based on feature type
        if isinstance(raw_value, float) and not np.isnan(raw_value):
            if "rate" in feature or "util" in feature or "dti" in feature:
                val_str = f"{raw_value:.1f}%"
            elif "inc" in feature or "amnt" in feature or "bal" in feature:
                val_str = f"${raw_value:,.0f}"
            elif raw_value == int(raw_value):
                val_str = f"{int(raw_value)}"
            else:
                val_str = f"{raw_value:.2f}"
        else:
            val_str = str(raw_value)

        description = f"{label}: {val_str} ({direction}, SHAP impact: {impact_str})"

        return {
            "feature": feature,
            "label": label,
            "value": val_str,
            "shap_impact": round(float(shap_value), 6),
            "direction": "increases_risk" if is_risk else "decreases_risk",
            "description": description,
        }

    def plot_summary(
        self,
        X: np.ndarray,
        max_display: int = 20,
        save_path: str = None,
    ) -> None:
        """
        Generate a global SHAP summary plot (beeswarm plot).

        LAYMAN: This plot shows the BIG PICTURE:
        - Which features matter most overall? (y-axis position)
        - For high values of that feature, does it increase or decrease risk?
          (color: red = high value, blue = low value; x-axis = SHAP impact)

        Example: "High DTI (red dots on the right) increases default risk."
                 "High FICO score (red dots on the left) decreases default risk."

        DATA SCIENTIST NOTE:
        The beeswarm plot is the most information-rich SHAP visualization:
        - Feature importance AND direction AND value correlation in one plot
        - Preferred over bar charts which lose direction information
        - More informative than partial dependence plots for multi-feature overview
        """
        logger.info("Generating SHAP summary plot...")

        shap_values = self.compute_shap_values(X)

        plt.figure(figsize=(10, max(6, max_display * 0.35)))

        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names[:X.shape[1]],
            max_display=max_display,
            show=False,
            plot_type="dot",  # Beeswarm plot
        )

        plt.title(
            "SHAP Feature Impact — Credit Risk Model",
            fontsize=13, fontweight='bold', pad=15
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        plt.close()

    def plot_waterfall(
        self,
        X_single: np.ndarray,
        predicted_probability: float,
        applicant_id: str = "Applicant",
        save_path: str = None,
    ) -> None:
        """
        Generate a waterfall plot for a single prediction.

        LAYMAN: A waterfall chart shows HOW the model got from
        the baseline prediction to the final score, step by step.

        It looks like a staircase:
        - Start: Baseline prediction (average borrower)
        - Each step: One feature pushes the score up (red) or down (blue)
        - End: Final predicted probability for THIS applicant

        This is the most intuitive explanation for non-technical audiences
        (bank managers, customers, regulators).

        DATA SCIENTIST NOTE:
        Uses shap.plots.waterfall which requires SHAP Explanation objects.
        We construct this manually for compatibility across SHAP versions.
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)

        shap_vals = self.explainer.shap_values(X_single)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = shap_vals[0]

        # Create SHAP Explanation object for waterfall plot
        feature_names_used = self.feature_names[:len(shap_vals)]
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=float(self.expected_value),
            data=X_single[0][:len(shap_vals)],
            feature_names=feature_names_used,
        )

        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.title(
            f"Prediction Explanation — {applicant_id}\n"
            f"Final Probability: {predicted_probability:.1%}",
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Waterfall plot saved to {save_path}")
        plt.close()

    def get_global_feature_importance(self, X: np.ndarray) -> pd.DataFrame:
        """
        Compute global feature importance as mean absolute SHAP values.

        LAYMAN: Answers "Which features matter most OVERALL across all applicants?"
        (Unlike individual explanations which show per-applicant impacts.)

        DATA SCIENTIST NOTE:
        Mean |SHAP| is preferred over model's native feature_importances_
        because it's model-agnostic, accounts for feature interactions,
        and is consistent across model types. It's the gold standard for
        feature importance in explainable AI.

        Returns:
            pd.DataFrame: Features sorted by mean absolute SHAP value
        """
        shap_values = self.compute_shap_values(X)

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_names[:len(mean_abs_shap)],
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        importance_df["rank"] = importance_df.index + 1
        importance_df["importance_pct"] = (
            importance_df["mean_abs_shap"] / importance_df["mean_abs_shap"].sum() * 100
        ).round(2)

        return importance_df
