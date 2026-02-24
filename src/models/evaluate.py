"""
src/models/evaluate.py — Model Evaluation & Metrics
=====================================================

LAYMAN EXPLANATION:
    After training, we need to measure HOW GOOD the model is.
    But "accuracy" alone is misleading for credit risk:

    If only 10% of loans default, a model that ALWAYS predicts
    "no default" would be 90% accurate — yet completely useless!

    Instead, we use metrics designed for imbalanced classification:

    - AUC-ROC: "How well does the model RANK borrowers by risk?"
      (1.0 = perfect, 0.5 = random, >0.85 = industry standard)

    - KS Statistic: "What's the maximum separation between default
      and non-default score distributions?" (>0.40 is good)

    - Gini Coefficient: AUC × 2 - 1. Used widely in credit scoring.
      (>0.60 is considered good by banks)

    - Precision-Recall: For a given rejection threshold, how many
      of our "predicted defaults" actually defaulted?

DATA SCIENTIST EXPLANATION:
    Comprehensive evaluation suite including:
    - AUROC, Gini, KS statistic (standard credit risk metrics)
    - Precision, recall, F1 at configurable thresholds
    - Calibration analysis (are probabilities well-calibrated?)
    - Feature importance from XGBoost and LightGBM
    - Confusion matrix with business-relevant framing
    - Score distribution analysis by risk grade
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
from typing import Optional

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

from config import cfg, REPORTS_DIR


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the Kolmogorov-Smirnov (KS) statistic.

    LAYMAN: The KS statistic measures how well the model SEPARATES
    good borrowers (who repay) from bad borrowers (who default).

    Imagine two overlapping bell curves:
    - One curve = score distribution of people who REPAID
    - One curve = score distribution of people who DEFAULTED

    KS is the maximum gap between these two curves.
    A large gap means the model is great at telling them apart.

    Industry benchmark:
    KS < 0.20: Poor model
    KS 0.20-0.40: Adequate
    KS > 0.40: Good
    KS > 0.60: Very good (rare in consumer lending)

    DATA SCIENTIST NOTE:
    KS = max|F_0(t) - F_1(t)| where F_0 is the CDF of scores for
    negatives and F_1 is the CDF for positives.
    """
    # Sort by predicted probability
    sorted_idx = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_idx]
    y_prob_sorted = y_prob[sorted_idx]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Cumulative distribution functions
    cum_pos = np.cumsum(y_true_sorted) / n_pos       # CDF of positives (defaults)
    cum_neg = np.cumsum(1 - y_true_sorted) / n_neg   # CDF of negatives (non-defaults)

    ks = np.max(np.abs(cum_pos - cum_neg))
    return float(ks)


def compute_gini_coefficient(auc_roc: float) -> float:
    """
    Compute the Gini coefficient from AUC-ROC.

    LAYMAN: The Gini coefficient is just AUC × 2 - 1.
    AUC of 0.85 → Gini of 0.70.

    Banks often prefer Gini over AUC because it's centered at 0
    (random model = 0 Gini) rather than 0.5 (random model = 0.5 AUC).
    It's easier to talk about: "our model has 70% Gini" vs "0.85 AUC."

    Industry standards:
    Gini > 0.40: Minimum acceptable
    Gini > 0.60: Good
    Gini > 0.70: Excellent
    """
    return 2 * auc_roc - 1


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = None,
    dataset_name: str = "Test",
) -> dict:
    """
    Compute all model evaluation metrics.

    LAYMAN: This function calculates all the "report card" metrics
    for the model — AUC, KS, Gini, precision, recall, and more.

    Args:
        y_true: Actual outcomes (0=paid, 1=defaulted)
        y_prob: Predicted probabilities (0-1)
        threshold: Decision threshold for binary classification
        dataset_name: Name for logging (e.g., "Validation", "Test")

    Returns:
        dict: All computed metrics
    """
    if threshold is None:
        threshold = cfg.scoring.rejection_threshold

    # Convert probabilities to binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    # --- Core Credit Risk Metrics ---
    auc = roc_auc_score(y_true, y_prob)
    ks = compute_ks_statistic(np.array(y_true), np.array(y_prob))
    gini = compute_gini_coefficient(auc)
    avg_precision = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)  # Lower is better (0=perfect)

    # --- Classification Metrics at Threshold ---
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # --- Business Metrics ---
    # At this threshold, what % of actual defaults do we catch?
    # (True Positive Rate / Recall / Sensitivity)
    default_capture_rate = recall

    # What % of our "approve" decisions will eventually default?
    # This is the false negative rate among approvals
    approved_default_rate = fn / (tn + fn) if (tn + fn) > 0 else 0

    metrics = {
        # Core ML metrics
        "auc_roc": round(float(auc), 4),
        "ks_statistic": round(float(ks), 4),
        "gini_coefficient": round(float(gini), 4),
        "avg_precision": round(float(avg_precision), 4),
        "brier_score": round(float(brier), 4),

        # Classification metrics
        "threshold": threshold,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),

        # Confusion matrix
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),

        # Business metrics
        "default_capture_rate": round(float(default_capture_rate), 4),
        "approved_default_rate": round(float(approved_default_rate), 4),
        "rejection_rate": round(float((y_pred == 1).mean()), 4),

        # Dataset info
        "n_samples": len(y_true),
        "n_defaults": int(y_true.sum()),
        "actual_default_rate": round(float(y_true.mean()), 4),
    }

    # --- Log Summary ---
    logger.info(f"\n{'='*50}")
    logger.info(f"MODEL EVALUATION — {dataset_name} Set")
    logger.info(f"{'='*50}")
    logger.info(f"  AUC-ROC:         {metrics['auc_roc']:.4f} (industry target: >0.85)")
    logger.info(f"  KS Statistic:    {metrics['ks_statistic']:.4f} (industry target: >0.40)")
    logger.info(f"  Gini Coefficient:{metrics['gini_coefficient']:.4f} (industry target: >0.60)")
    logger.info(f"  Brier Score:     {metrics['brier_score']:.4f} (lower is better)")
    logger.info(f"  Precision:       {metrics['precision']:.4f}")
    logger.info(f"  Recall:          {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:        {metrics['f1_score']:.4f}")
    logger.info(f"  Default Capture: {metrics['default_capture_rate']:.1%}")
    logger.info(f"  Rejection Rate:  {metrics['rejection_rate']:.1%}")
    logger.info(f"{'='*50}\n")

    return metrics


def assign_risk_grade(probability: float) -> str:
    """
    Assign a risk grade letter based on default probability.

    LAYMAN: Converts the raw probability (e.g., 0.23) into a
    human-readable risk grade:
    A = Safest (low probability of default)
    B = Good
    C = Moderate
    D = Risky
    F = Very risky (would almost certainly default)

    This matches how LendingClub and most banks grade their loans.

    Args:
        probability: Default probability between 0 and 1

    Returns:
        str: Risk grade letter (A, B, C, D, or F)
    """
    thresholds = cfg.scoring.grade_thresholds
    for grade, threshold in thresholds.items():
        if probability <= threshold:
            return grade
    return "F"


def score_distribution_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """
    Analyze model performance by risk grade/score bucket.

    LAYMAN: This breaks down model performance by score range.
    We want to see: in each grade bucket, how many loans defaulted?
    If Grade A has 3% default rate and Grade F has 70% default rate,
    the model is doing its job (discriminating well between grades).

    DATA SCIENTIST NOTE:
    This is the "score card" analysis. The key property we want is
    "monotonicity" — higher grade = higher default rate, with no
    reversals. Non-monotonic buckets indicate model issues.

    Returns:
        pd.DataFrame: Default rates by score bucket
    """
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["grade"] = df["y_prob"].apply(assign_risk_grade)

    grade_stats = df.groupby("grade").agg(
        n_loans=("y_true", "count"),
        n_defaults=("y_true", "sum"),
        avg_score=("y_prob", "mean"),
        min_score=("y_prob", "min"),
        max_score=("y_prob", "max"),
    ).reset_index()

    grade_stats["default_rate"] = grade_stats["n_defaults"] / grade_stats["n_loans"]
    grade_stats["pct_of_portfolio"] = grade_stats["n_loans"] / grade_stats["n_loans"].sum()

    # Sort by grade
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    grade_stats["grade_order"] = grade_stats["grade"].map(grade_order)
    grade_stats = grade_stats.sort_values("grade_order").drop(columns=["grade_order"])

    logger.info(f"\nScore Distribution by Risk Grade:\n{grade_stats.to_string(index=False)}")
    return grade_stats


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_path: str = None,
) -> None:
    """
    Plot the ROC (Receiver Operating Characteristic) curve.

    LAYMAN: The ROC curve shows the tradeoff between:
    - True Positive Rate: "What fraction of actual defaults do we catch?"
    - False Positive Rate: "What fraction of good borrowers do we falsely reject?"

    The curve bows toward the top-left corner. The further it bows
    (larger AUC), the better the model. A random model follows the
    diagonal line (AUC = 0.5). A perfect model hits the top-left
    corner (AUC = 1.0).

    DATA SCIENTIST NOTE:
    For credit risk, we often care more about the left part of the
    ROC curve (low FPR region) because we can't afford to reject
    too many good borrowers (revenue loss).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2196F3', linewidth=2.5,
            label=f'Model (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#9E9E9E', linestyle='--',
            linewidth=1.5, label='Random Model (AUC = 0.50)')

    ax.fill_between(fpr, tpr, alpha=0.15, color='#2196F3')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate (Bad borrowers incorrectly approved)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Defaults correctly identified)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    plt.close()


def plot_score_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Score Distribution by Outcome",
    save_path: str = None,
) -> None:
    """
    Plot score distributions for defaulters vs non-defaulters.

    LAYMAN: This shows two overlapping histograms:
    - Blue histogram: distribution of scores for people who REPAID
    - Red histogram: distribution of scores for people who DEFAULTED

    We WANT these histograms to be far apart — that means the model
    gives very different scores to defaulters vs repayers (good separation).
    Lots of overlap = poor discrimination.
    """
    df = pd.DataFrame({"score": y_prob, "outcome": y_true})

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot distributions for each class
    for outcome, label, color in [(0, "Fully Paid (Good)", "#4CAF50"),
                                    (1, "Charged Off (Default)", "#F44336")]:
        subset = df[df["outcome"] == outcome]["score"]
        ax.hist(subset, bins=50, alpha=0.6, color=color, label=label,
                density=True, edgecolor='white', linewidth=0.5)

    ax.axvline(
        cfg.scoring.rejection_threshold,
        color='#FF9800', linestyle='--', linewidth=2,
        label=f'Rejection Threshold ({cfg.scoring.rejection_threshold:.0%})'
    )

    ax.set_xlabel("Predicted Default Probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Score distribution saved to {save_path}")
    plt.close()


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: str = None,
) -> None:
    """
    Plot top-N most important features.

    LAYMAN: This bar chart shows WHICH features the model found most
    useful for predicting defaults. The longer the bar, the more
    important the feature.

    Typically you'd see things like:
    - DTI ratio (debt burden) → top importance
    - FICO score → very important
    - Credit utilization → important
    - Delinquency history → important

    This helps verify the model is using sensible features and
    not learning spurious patterns.

    DATA SCIENTIST NOTE:
    XGBoost uses "gain" importance by default — the average gain
    across all splits using that feature. This is more reliable than
    split count importance which biases high-cardinality features.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        logger.warning("Model doesn't have feature_importances_")
        return

    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        "feature": feature_names[:len(importances)],
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    bars = ax.barh(
        range(len(importance_df)),
        importance_df["importance"],
        color='#2196F3',
        alpha=0.8,
        edgecolor='white'
    )
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    plt.close()


def generate_evaluation_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    feature_names: list = None,
    xgb_model=None,
    lgbm_model=None,
    save_dir: str = None,
) -> dict:
    """
    Generate a complete model evaluation report with all plots.

    LAYMAN: This is the "full report card" — runs all evaluations,
    creates all charts, and saves them to the reports folder.
    The final output is a dictionary of all metrics that can be
    logged to MLflow or printed as a summary.

    Returns:
        dict: All evaluation metrics
    """
    if save_dir is None:
        save_dir = str(REPORTS_DIR)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Generating full evaluation report...")

    # Compute all metrics
    metrics = evaluate_model(y_true, y_prob, dataset_name="Test")

    # Score distribution by grade
    grade_stats = score_distribution_analysis(y_true, y_prob)
    metrics["grade_distribution"] = grade_stats.to_dict(orient="records")

    # Generate plots
    plot_roc_curve(
        y_true, y_prob,
        title="Credit Risk Model — ROC Curve",
        save_path=f"{save_dir}/roc_curve.png"
    )
    plot_score_distribution(
        y_true, y_prob,
        title="Score Distribution: Defaults vs Non-Defaults",
        save_path=f"{save_dir}/score_distribution.png"
    )

    if xgb_model is not None and feature_names is not None:
        plot_feature_importance(
            xgb_model, feature_names,
            title="XGBoost Feature Importance (Top 20)",
            save_path=f"{save_dir}/xgb_feature_importance.png"
        )

    if lgbm_model is not None and feature_names is not None:
        plot_feature_importance(
            lgbm_model, feature_names,
            title="LightGBM Feature Importance (Top 20)",
            save_path=f"{save_dir}/lgbm_feature_importance.png"
        )

    logger.info(f"Evaluation report saved to {save_dir}/")
    return metrics
