"""
src/monitoring/drift_detector.py — Model Monitoring & Drift Detection
=======================================================================

LAYMAN EXPLANATION:
    A model trained on 2022 loan data might perform poorly on 2025 data.
    Why? Because the world changes:
    - Inflation → people's income-to-debt ratios change
    - Recessions → default rates spike suddenly
    - New loan products → different applicant profiles
    - Demographic shifts → different mix of borrowers

    When this happens, the model's predictions become unreliable.
    "Model drift" is the term for this degradation.

    This module detects drift BEFORE the model causes bad lending decisions.
    Think of it like a car dashboard warning light — it alerts you when
    something needs attention, before the engine breaks down completely.

    TWO TYPES OF DRIFT:
    1. DATA DRIFT (Covariate Shift): The inputs (applicant profiles) changed
       "We're suddenly seeing much riskier applicants than before"
    2. CONCEPT DRIFT: The relationship between inputs and outputs changed
       "High DTI used to mean high default risk, but not anymore"

DATA SCIENTIST EXPLANATION:
    Implements drift detection using three statistical methods:

    1. PSI (Population Stability Index):
       PSI = Σ (actual% - expected%) × ln(actual%/expected%)
       Industry standard for credit score monitoring.
       <0.10: stable, 0.10-0.25: investigate, >0.25: retrain

    2. KS Test (Kolmogorov-Smirnov):
       Compares CDFs of reference and current distributions.
       p-value < 0.05 → distributions are significantly different.
       Sensitive to both location and scale shifts.

    3. Chi-Square Test (for categorical features):
       Tests if category proportions changed significantly.
       p-value < 0.05 → category distribution has shifted.

    Monitoring is applied to:
    - Input features (data drift)
    - Score distribution (output drift)
    - Performance metrics over time (concept drift via rolling AUC)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from loguru import logger
from typing import Optional
import json

from config import cfg, REPORTS_DIR


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute the Population Stability Index (PSI).

    LAYMAN: PSI measures "how different is the current data from
    the training data?" It's like comparing two histograms:
    the original (training) and the new (current).

    If the current applicants look very similar to training applicants,
    PSI will be low (stable). If they look very different (e.g., suddenly
    many more high-DTI applicants), PSI will be high (unstable).

    PSI Interpretation:
    < 0.10: Stable — model is fine, no action needed
    0.10 - 0.25: Slight shift — monitor closely
    > 0.25: Significant shift — RETRAIN THE MODEL

    This is the industry standard metric used by banks worldwide
    for monitoring credit score cards.

    DATA SCIENTIST NOTE:
    PSI is computed on the predicted score distribution, not raw features.
    It's sensitive to the tails of the distribution (which is good for
    detecting regime changes in high-risk/low-risk populations).

    PSI formula:
    PSI = Σ (A_i - E_i) × ln(A_i / E_i)
    where A_i = actual% in bucket i, E_i = expected% in bucket i

    Args:
        expected: Reference distribution (training/validation data scores)
        actual: Current distribution (new data scores to compare)
        n_bins: Number of bins for histogram comparison

    Returns:
        float: PSI value (0 = identical distributions, higher = more drift)
    """
    # Create bins based on the expected (reference) distribution
    # Using expected quantiles ensures each bin has equal expected mass
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-6   # Slightly expand left edge to include minimum
    bins[-1] += 1e-6  # Slightly expand right edge to include maximum
    bins = np.unique(bins)  # Remove duplicate edges

    if len(bins) < 3:
        logger.warning("Not enough unique bins for PSI computation. Returning 0.")
        return 0.0

    # Count samples in each bin
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _   = np.histogram(actual, bins=bins)

    # Convert to fractions (percentages)
    expected_pct = expected_counts / len(expected)
    actual_pct   = actual_counts / len(actual)

    # Avoid division by zero or log(0) by adding small epsilon
    epsilon = 1e-6
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct   = np.where(actual_pct == 0, epsilon, actual_pct)

    # PSI formula
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = float(np.sum(psi_values))

    return psi


def compute_ks_test(
    reference: np.ndarray,
    current: np.ndarray,
) -> dict:
    """
    Compute the Kolmogorov-Smirnov two-sample test.

    LAYMAN: The KS test asks: "Could these two sets of numbers have come
    from the same underlying distribution?"

    Think of it as: "Are the new applicants statistically similar to the
    applicants we trained on, or are they different?"

    If the p-value is very small (< 0.05), the distributions are
    significantly different → potential drift detected.

    DATA SCIENTIST NOTE:
    Two-sample KS test: D = max|F_n(x) - F_m(x)|
    where F_n and F_m are empirical CDFs of the two samples.
    KS test is sensitive to BOTH location shifts (mean changes) and
    scale shifts (variance changes), making it more powerful than
    simple mean comparison for detecting distribution changes.

    For large samples, even tiny shifts become statistically significant.
    Combine statistical significance with effect size (KS statistic) for
    practical significance assessment.

    Args:
        reference: Reference distribution (training data)
        current: Current distribution (new data)

    Returns:
        dict: {ks_statistic, p_value, drift_detected}
    """
    ks_statistic, p_value = stats.ks_2samp(reference, current)

    drift_detected = (
        ks_statistic > cfg.monitoring.ks_threshold and
        p_value < 0.05
    )

    return {
        "ks_statistic": round(float(ks_statistic), 4),
        "p_value": round(float(p_value), 6),
        "drift_detected": bool(drift_detected),
        "interpretation": (
            "DRIFT DETECTED" if drift_detected
            else "No significant drift"
        )
    }


def compute_chi_square_test(
    reference_categories: np.ndarray,
    current_categories: np.ndarray,
) -> dict:
    """
    Chi-square test for categorical feature drift.

    LAYMAN: For categorical features (like loan purpose or home ownership),
    we check if the proportions have changed.

    Example: If "debt_consolidation" loans were 45% of applications in training
    but are now 20% of applications, that's a big shift in the borrower mix
    that might affect model performance.

    DATA SCIENTIST NOTE:
    Chi-square test on observed vs expected category counts.
    Sensitive to category frequency changes. Small categories (<5 expected)
    should be combined or excluded for valid chi-square approximation.

    Args:
        reference_categories: Category labels from reference period
        current_categories: Category labels from current period

    Returns:
        dict: {chi2_statistic, p_value, drift_detected}
    """
    # Get all unique categories (union of both periods)
    all_categories = list(set(reference_categories) | set(current_categories))
    all_categories.sort()

    # Count occurrences of each category
    ref_counts = pd.Series(reference_categories).value_counts()
    cur_counts = pd.Series(current_categories).value_counts()

    # Fill 0 for categories not present in one period
    ref_expected = np.array([ref_counts.get(cat, 0) for cat in all_categories], dtype=float)
    cur_observed = np.array([cur_counts.get(cat, 0) for cat in all_categories], dtype=float)

    # Scale expected to match current sample size
    if ref_expected.sum() > 0:
        ref_expected = ref_expected / ref_expected.sum() * cur_observed.sum()

    # Replace zeros to avoid division errors
    ref_expected = np.maximum(ref_expected, 0.5)

    try:
        chi2, p_value = stats.chisquare(cur_observed, f_exp=ref_expected)
        drift_detected = p_value < 0.05
    except Exception as e:
        logger.warning(f"Chi-square test failed: {e}")
        chi2, p_value, drift_detected = 0.0, 1.0, False

    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "drift_detected": bool(drift_detected),
        "interpretation": "DRIFT DETECTED" if drift_detected else "No significant drift"
    }


def monitor_score_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    save_path: str = None,
) -> dict:
    """
    Monitor the model's output score distribution for drift.

    LAYMAN: Even if individual features don't change, the OVERALL SCORE
    distribution might shift. This could happen if the relationships
    between features and default risk changed (concept drift).

    We compare the distribution of scores for recent applicants vs
    the distribution we saw during model validation.

    Example alarm: "The average default probability in our model output
    jumped from 15% to 35% over the past month. Investigation needed!"

    DATA SCIENTIST NOTE:
    Score drift is often the first observable signal of concept drift.
    Monitoring score distributions catches regime changes faster than
    waiting for actual defaults to accumulate (which takes months in lending).

    Args:
        reference_scores: Scores from training/validation period
        current_scores: Scores from the monitoring period
        save_path: Path to save the comparison plot

    Returns:
        dict: Drift metrics and alert status
    """
    psi = compute_psi(reference_scores, current_scores)
    ks_result = compute_ks_test(reference_scores, current_scores)

    # Determine alert level
    if psi > cfg.monitoring.psi_threshold_critical:
        alert_level = "CRITICAL"
        alert_message = f"Score distribution has shifted significantly (PSI={psi:.3f}). RETRAIN MODEL."
    elif psi > cfg.monitoring.psi_threshold_warning:
        alert_level = "WARNING"
        alert_message = f"Score distribution is shifting (PSI={psi:.3f}). Monitor closely."
    else:
        alert_level = "OK"
        alert_message = f"Score distribution is stable (PSI={psi:.3f})."

    logger.info(f"\n[SCORE MONITORING] {alert_level}: {alert_message}")
    logger.info(f"  PSI: {psi:.4f} | KS: {ks_result['ks_statistic']:.4f}")

    # Summary statistics comparison
    stats_comparison = {
        "reference_mean": round(float(np.mean(reference_scores)), 4),
        "current_mean": round(float(np.mean(current_scores)), 4),
        "reference_std": round(float(np.std(reference_scores)), 4),
        "current_std": round(float(np.std(current_scores)), 4),
        "reference_median": round(float(np.median(reference_scores)), 4),
        "current_median": round(float(np.median(current_scores)), 4),
        "mean_shift": round(float(np.mean(current_scores) - np.mean(reference_scores)), 4),
    }

    # Generate comparison plot
    if save_path:
        _plot_score_drift(reference_scores, current_scores, psi, save_path)

    return {
        "psi": round(float(psi), 4),
        "psi_alert_level": alert_level,
        "alert_message": alert_message,
        "ks_test": ks_result,
        "stats_comparison": stats_comparison,
        "action_required": alert_level in ["CRITICAL", "WARNING"],
        "n_reference": len(reference_scores),
        "n_current": len(current_scores),
    }


def monitor_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    top_n_features: int = 10,
) -> pd.DataFrame:
    """
    Monitor individual feature distributions for drift.

    LAYMAN: Instead of just monitoring the final score, this looks at
    each individual input feature to pinpoint WHERE the drift is coming from.

    Example: "PSI is high. Why? Let's check each feature:
    - annual_income: PSI=0.05 (stable)
    - dti: PSI=0.35 (DRIFTING!) ← the culprit"

    This narrows down which features to investigate.

    DATA SCIENTIST NOTE:
    Feature-level PSI is used in bank model validations to identify
    which input variables have changed. This guides:
    1. Feature recalibration (update encoding/scaling for drifted features)
    2. Model retraining trigger assessment (is drift severe enough?)
    3. Data quality investigation (is it real drift or a data pipeline issue?)

    Args:
        reference_df: Feature DataFrame from reference period
        current_df: Feature DataFrame from current period
        top_n_features: How many features to report on

    Returns:
        pd.DataFrame: Per-feature PSI and drift status
    """
    logger.info("Running feature-level drift analysis...")

    # Get numeric features (PSI works on continuous distributions)
    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target if present
    numeric_cols = [c for c in numeric_cols if c != cfg.data.target_column]

    results = []
    for col in numeric_cols[:top_n_features]:  # Limit to avoid excessive computation
        if col not in current_df.columns:
            continue

        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        psi = compute_psi(ref_vals, cur_vals)
        ks  = compute_ks_test(ref_vals, cur_vals)

        # Alert level based on PSI
        if psi > cfg.monitoring.psi_threshold_critical:
            status = "CRITICAL"
        elif psi > cfg.monitoring.psi_threshold_warning:
            status = "WARNING"
        else:
            status = "OK"

        results.append({
            "feature": col,
            "psi": round(float(psi), 4),
            "ks_statistic": ks["ks_statistic"],
            "ks_p_value": ks["p_value"],
            "drift_status": status,
            "ref_mean": round(float(np.mean(ref_vals)), 4),
            "cur_mean": round(float(np.mean(cur_vals)), 4),
            "mean_shift": round(float(np.mean(cur_vals) - np.mean(ref_vals)), 4),
        })

    results_df = pd.DataFrame(results).sort_values("psi", ascending=False)

    n_critical = (results_df["drift_status"] == "CRITICAL").sum()
    n_warning  = (results_df["drift_status"] == "WARNING").sum()

    logger.info(
        f"Feature drift summary: "
        f"{n_critical} CRITICAL, {n_warning} WARNING, "
        f"{len(results_df) - n_critical - n_warning} OK features"
    )
    logger.info(f"\nTop drifted features:\n{results_df.head(5).to_string(index=False)}")

    return results_df


def _plot_score_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    psi: float,
    save_path: str,
) -> None:
    """
    Plot comparison of reference vs current score distributions.

    LAYMAN: Two overlapping histograms showing:
    - Blue: How scores looked when model was trained (reference)
    - Orange: How scores look now (current)

    If they overlap a lot → stable model (good)
    If they diverge significantly → drifted model (needs attention)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Overlapping histograms ---
    ax1 = axes[0]
    ax1.hist(reference_scores, bins=40, alpha=0.6, color='#2196F3',
             label='Reference (Training)', density=True)
    ax1.hist(current_scores, bins=40, alpha=0.6, color='#FF9800',
             label='Current (Monitoring)', density=True)
    ax1.set_xlabel("Default Probability Score", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(f"Score Distribution Comparison\nPSI = {psi:.4f}", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add PSI alert level annotation
    if psi > cfg.monitoring.psi_threshold_critical:
        ax1.set_facecolor('#FFEBEE')
        ax1.text(0.5, 0.95, "⚠ CRITICAL DRIFT", transform=ax1.transAxes,
                fontsize=11, color='#F44336', ha='center', va='top', fontweight='bold')
    elif psi > cfg.monitoring.psi_threshold_warning:
        ax1.set_facecolor('#FFF8E1')
        ax1.text(0.5, 0.95, "⚠ WARNING: DRIFT DETECTED", transform=ax1.transAxes,
                fontsize=11, color='#FF9800', ha='center', va='top', fontweight='bold')

    # --- Plot 2: Quantile-Quantile comparison ---
    ax2 = axes[1]
    ref_percentiles = np.percentile(reference_scores, np.linspace(0, 100, 100))
    cur_percentiles = np.percentile(current_scores, np.linspace(0, 100, 100))

    ax2.scatter(ref_percentiles, cur_percentiles, alpha=0.6, s=20, color='#2196F3')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect agreement')
    ax2.set_xlabel("Reference Quantiles", fontsize=11)
    ax2.set_ylabel("Current Quantiles", fontsize=11)
    ax2.set_title("Q-Q Plot: Reference vs Current", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Score drift plot saved to {save_path}")


def run_monitoring_report(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    reference_df: pd.DataFrame = None,
    current_df: pd.DataFrame = None,
    save_dir: str = None,
) -> dict:
    """
    Run the complete monitoring pipeline and generate a report.

    LAYMAN: This is the daily/weekly health check for the model.
    It runs all drift tests and outputs a summary report with:
    - Overall model health: GREEN / YELLOW / RED
    - Which features have drifted
    - Whether retraining is recommended
    - All detailed statistics

    DATA SCIENTIST NOTE:
    In production, this would be triggered by:
    1. A daily cron job (e.g., runs every morning at 6am)
    2. After accumulating N new predictions (e.g., every 1,000 applications)
    3. On-demand by a data scientist or MLOps engineer

    Results are saved as JSON + HTML report for team review.

    Returns:
        dict: Complete monitoring report
    """
    if save_dir is None:
        save_dir = str(REPORTS_DIR)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RUNNING MODEL MONITORING REPORT")
    logger.info("=" * 60)

    # 1. Score-level drift
    score_drift = monitor_score_drift(
        reference_scores, current_scores,
        save_path=f"{save_dir}/score_drift.png"
    )

    # 2. Feature-level drift (if DataFrames provided)
    feature_drift = None
    if reference_df is not None and current_df is not None:
        feature_drift_df = monitor_feature_drift(reference_df, current_df)
        feature_drift = feature_drift_df.to_dict(orient="records")

    # 3. Overall health assessment
    overall_psi = score_drift["psi"]
    if overall_psi > cfg.monitoring.psi_threshold_critical:
        health = "RED"
        recommendation = "URGENT: Model is significantly drifted. Retrain immediately."
    elif overall_psi > cfg.monitoring.psi_threshold_warning:
        health = "YELLOW"
        recommendation = "CAUTION: Model drift detected. Schedule retraining soon."
    else:
        health = "GREEN"
        recommendation = "Model is stable. No immediate action required."

    report = {
        "health_status": health,
        "recommendation": recommendation,
        "score_drift": score_drift,
        "feature_drift": feature_drift,
        "timestamp": pd.Timestamp.now().isoformat(),
        "monitoring_config": {
            "psi_warning_threshold": cfg.monitoring.psi_threshold_warning,
            "psi_critical_threshold": cfg.monitoring.psi_threshold_critical,
            "ks_threshold": cfg.monitoring.ks_threshold,
        }
    }

    # Save report as JSON
    report_path = f"{save_dir}/monitoring_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Monitoring report saved to {report_path}")

    logger.info(f"\n{'='*50}")
    logger.info(f"MONITORING SUMMARY: {health}")
    logger.info(f"Recommendation: {recommendation}")
    logger.info(f"{'='*50}\n")

    return report
