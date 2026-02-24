"""
src/fairness/bias_detector.py — Fairness & Bias Detection
===========================================================

LAYMAN EXPLANATION:
    Imagine a loan model that approves 90% of men but only 55% of women
    with IDENTICAL financial profiles. That's discrimination — and it's
    illegal under the Equal Credit Opportunity Act (ECOA).

    This module checks whether our model treats different groups fairly.
    It answers questions like:
    - "Do people in different states get approved at similar rates?"
    - "Does the model perform equally well across loan grades?"
    - "Are we inadvertently using proxies for protected attributes?"

    KEY LEGAL STANDARD — The 4/5ths Rule:
    If Group B's approval rate is less than 80% of Group A's approval rate,
    there's evidence of "disparate impact" (illegal discrimination).

    Example:
    - Group A (Grade A loans): 95% approval rate
    - Group B (Grade D loans): 40% approval rate
    - Ratio: 40% / 95% = 42% — FAR below the 80% threshold
    → This would need investigation

DATA SCIENTIST EXPLANATION:
    Implements three fairness frameworks:
    1. Demographic Parity: Equal approval rates across groups
       (Disparate Impact: approval_rate_group / approval_rate_baseline)
    2. Equal Opportunity: Equal True Positive Rates (TPR) across groups
       (Are we detecting defaults equally well in all groups?)
    3. Equalized Odds: Equal TPR AND FPR across groups
       (The most stringent fairness criterion)

    References:
    - EEOC 4/5ths rule (29 CFR § 1607.4)
    - Hardt et al. (2016) "Equality of Opportunity in Supervised Learning"
    - Barocas & Hardt (2017) "Fairness in Machine Learning"
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
from loguru import logger
from pathlib import Path

from config import cfg, REPORTS_DIR


def compute_group_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_labels: np.ndarray,
    threshold: float = None,
) -> pd.DataFrame:
    """
    Compute fairness metrics for each subgroup.

    LAYMAN: For each group (e.g., each loan grade: A, B, C, D, E, F),
    compute the approval rate, default capture rate, and false rejection rate.
    Then compare groups to detect unfairness.

    DATA SCIENTIST NOTE:
    Computes per-group:
    - Approval rate (1 - rejection_rate)
    - TPR (True Positive Rate): Recall for default class
    - FPR (False Positive Rate): Fraction of non-defaults incorrectly rejected
    - Precision: Among rejected, what fraction actually defaulted?
    - AUC per group (if enough samples)

    Args:
        y_true: True labels (0/1)
        y_prob: Predicted probabilities
        group_labels: Group membership for each sample
        threshold: Decision threshold

    Returns:
        pd.DataFrame: One row per group with all fairness metrics
    """
    if threshold is None:
        threshold = cfg.scoring.rejection_threshold

    y_pred = (y_prob >= threshold).astype(int)
    groups = np.unique(group_labels)

    results = []
    for group in sorted(groups):
        mask = group_labels == group
        n = mask.sum()

        if n < 10:  # Skip groups with too few samples
            continue

        g_true = y_true[mask]
        g_prob = y_prob[mask]
        g_pred = y_pred[mask]

        # Basic counts
        tp = int(((g_true == 1) & (g_pred == 1)).sum())
        tn = int(((g_true == 0) & (g_pred == 0)).sum())
        fp = int(((g_true == 0) & (g_pred == 1)).sum())
        fn = int(((g_true == 1) & (g_pred == 0)).sum())

        # Rates
        approval_rate   = (g_pred == 0).mean()  # Predicted non-default = approved
        default_rate    = g_true.mean()           # Actual default rate in group
        rejection_rate  = (g_pred == 1).mean()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate

        # AUC per group (only if both classes present)
        if len(np.unique(g_true)) > 1:
            from sklearn.metrics import roc_auc_score
            try:
                group_auc = roc_auc_score(g_true, g_prob)
            except Exception:
                group_auc = np.nan
        else:
            group_auc = np.nan

        results.append({
            "group": group,
            "n_samples": int(n),
            "n_defaults": int(g_true.sum()),
            "actual_default_rate": round(float(default_rate), 4),
            "approval_rate": round(float(approval_rate), 4),
            "rejection_rate": round(float(rejection_rate), 4),
            "tpr_recall": round(float(tpr), 4),
            "fpr": round(float(fpr), 4),
            "tnr_specificity": round(float(tnr), 4),
            "group_auc": round(float(group_auc), 4) if not np.isnan(group_auc) else None,
        })

    df = pd.DataFrame(results)
    return df


def compute_disparate_impact(
    group_metrics: pd.DataFrame,
    baseline_group: str = None,
) -> pd.DataFrame:
    """
    Compute the Disparate Impact Ratio for each group.

    LAYMAN: Disparate Impact Ratio = (Group's approval rate) / (Best group's approval rate)

    If the ratio is below 0.80 (the "4/5ths rule"), there's evidence
    of illegal discrimination against that group.

    Example:
    - Grade A loans: 92% approval rate (baseline — best group)
    - Grade C loans: 75% approval rate
    - Grade D loans: 40% approval rate
    - Grade C ratio: 75/92 = 0.82 ✓ (above 0.80 threshold)
    - Grade D ratio: 40/92 = 0.43 ✗ (below 0.80 threshold — investigate!)

    DATA SCIENTIST NOTE:
    Disparate Impact uses approval rate, NOT default rate.
    A group having more defaults is NOT discrimination — lending decisions
    SHOULD reflect creditworthiness. DI compares outcomes for similar risk profiles.
    In practice, you'd control for creditworthiness before computing DI.

    Args:
        group_metrics: DataFrame from compute_group_metrics()
        baseline_group: Reference group for comparison (defaults to highest approval rate)

    Returns:
        pd.DataFrame: Group metrics + disparate_impact_ratio column
    """
    df = group_metrics.copy()

    if baseline_group is None:
        # Use group with highest approval rate as baseline
        baseline_approval = df["approval_rate"].max()
    else:
        baseline_approval = df.loc[df["group"] == baseline_group, "approval_rate"].values[0]

    # Disparate Impact Ratio: group_approval / baseline_approval
    df["disparate_impact_ratio"] = (df["approval_rate"] / baseline_approval).round(4)

    # Flag groups that fail the 4/5ths rule
    threshold = cfg.fairness.disparate_impact_threshold  # 0.80
    df["passes_4fifths_rule"] = df["disparate_impact_ratio"] >= threshold
    df["fairness_concern"] = ~df["passes_4fifths_rule"]

    n_failing = (~df["passes_4fifths_rule"]).sum()
    if n_failing > 0:
        failing_groups = df[df["fairness_concern"]]["group"].tolist()
        logger.warning(
            f"FAIRNESS ALERT: {n_failing} group(s) fail the 4/5ths rule: {failing_groups}\n"
            f"Threshold: {threshold:.0%} | Baseline approval rate: {baseline_approval:.1%}"
        )
    else:
        logger.info(
            f"All groups pass the 4/5ths rule "
            f"(threshold: {threshold:.0%})"
        )

    return df


def compute_equal_opportunity(
    group_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Equal Opportunity scores across groups.

    LAYMAN: Equal Opportunity means: "For borrowers who ACTUALLY default,
    does the model detect them equally well across all groups?"

    If the model catches 80% of defaults in one group but only 40% in another,
    it's not treating those groups equally — some defaulters are "let through"
    just because of their group membership.

    DATA SCIENTIST NOTE:
    Equal Opportunity = Equal TPR across protected groups.
    Defined by Hardt et al. (2016): "the algorithm should satisfy equal
    opportunity with respect to a protected attribute A and outcome Y if
    P(Ŷ=1|A=a, Y=1) = P(Ŷ=1|A=b, Y=1) for all a,b."

    In credit: this means equal default detection rates across groups.

    Returns:
        pd.DataFrame: Group metrics + equal_opportunity_gap column
    """
    df = group_metrics.copy()

    mean_tpr = df["tpr_recall"].mean()
    df["equal_opportunity_gap"] = (df["tpr_recall"] - mean_tpr).round(4)
    df["equal_opportunity_gap_abs"] = df["equal_opportunity_gap"].abs()

    # Flag groups with large TPR deviation (>10% from mean)
    df["tpr_concern"] = df["equal_opportunity_gap_abs"] > 0.10

    max_gap = df["equal_opportunity_gap_abs"].max()
    logger.info(
        f"Equal Opportunity — Max TPR gap across groups: {max_gap:.1%} "
        f"({'CONCERN' if max_gap > 0.10 else 'OK'})"
    )

    return df


def run_fairness_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    df_original: pd.DataFrame,
    save_dir: str = None,
) -> dict:
    """
    Run the complete fairness analysis pipeline.

    LAYMAN: This is the master fairness report. It checks the model
    for bias across all protected attributes defined in the config,
    generates charts, and returns a structured fairness report.

    The report answers:
    1. Does the model have disparate impact on any group? (4/5ths rule)
    2. Does the model have equal opportunity across groups?
    3. What's the TPR and FPR breakdown by group?

    DATA SCIENTIST NOTE:
    In a real production system, you'd run this:
    - After every model retraining
    - Monthly on live production predictions
    - Whenever demographics of the applicant pool shift significantly

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        df_original: Original DataFrame with demographic columns
        save_dir: Directory to save fairness report plots

    Returns:
        dict: Complete fairness report
    """
    if save_dir is None:
        save_dir = str(REPORTS_DIR)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RUNNING FAIRNESS ANALYSIS")
    logger.info("=" * 60)

    fairness_report = {}

    for attr in cfg.fairness.protected_attributes:
        if attr not in df_original.columns:
            logger.warning(f"Protected attribute '{attr}' not found in data. Skipping.")
            continue

        logger.info(f"\nAnalyzing fairness for attribute: '{attr}'")

        group_labels = df_original[attr].values

        # 1. Compute group metrics
        group_metrics = compute_group_metrics(y_true, y_prob, group_labels)

        # 2. Compute Disparate Impact
        group_metrics = compute_disparate_impact(group_metrics)

        # 3. Compute Equal Opportunity
        group_metrics = compute_equal_opportunity(group_metrics)

        logger.info(f"\nFairness Report for '{attr}':\n{group_metrics.to_string(index=False)}")

        # 4. Generate fairness visualization
        _plot_fairness_chart(group_metrics, attr, save_dir)

        # Store in report
        fairness_report[attr] = {
            "group_metrics": group_metrics.to_dict(orient="records"),
            "overall_concern": bool(group_metrics["fairness_concern"].any()),
            "n_failing_4fifths": int(group_metrics["fairness_concern"].sum()),
            "max_disparate_impact": float(group_metrics["disparate_impact_ratio"].min()),
            "mean_tpr": float(group_metrics["tpr_recall"].mean()),
            "max_tpr_gap": float(group_metrics["equal_opportunity_gap_abs"].max()),
        }

    return fairness_report


def _plot_fairness_chart(
    group_metrics: pd.DataFrame,
    attribute: str,
    save_dir: str,
) -> None:
    """
    Generate a visual fairness comparison chart.

    LAYMAN: A side-by-side bar chart showing:
    - Left bars: Approval rates per group
    - Right bars: Disparate Impact Ratio per group (with 0.80 threshold line)

    If any bar on the right is below the red threshold line (0.80),
    there's a fairness concern.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    groups = group_metrics["group"].astype(str)
    x = range(len(groups))
    colors = [
        "#F44336" if concern else "#4CAF50"
        for concern in group_metrics["fairness_concern"]
    ]

    # --- Plot 1: Approval Rates ---
    ax1 = axes[0]
    bars = ax1.bar(x, group_metrics["approval_rate"], color=colors, alpha=0.8, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, rotation=45, ha='right')
    ax1.set_ylabel("Approval Rate", fontsize=11)
    ax1.set_title(f"Approval Rate by {attribute}", fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, group_metrics["approval_rate"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}", ha='center', va='bottom', fontsize=9
        )

    # --- Plot 2: Disparate Impact Ratio ---
    ax2 = axes[1]
    bars2 = ax2.bar(
        x, group_metrics["disparate_impact_ratio"],
        color=colors, alpha=0.8, edgecolor='white'
    )
    ax2.axhline(
        y=cfg.fairness.disparate_impact_threshold,
        color='#FF5722', linestyle='--', linewidth=2,
        label=f'4/5ths Rule Threshold ({cfg.fairness.disparate_impact_threshold:.0%})'
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, rotation=45, ha='right')
    ax2.set_ylabel("Disparate Impact Ratio", fontsize=11)
    ax2.set_title(f"Disparate Impact by {attribute}", fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.2])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars2, group_metrics["disparate_impact_ratio"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}", ha='center', va='bottom', fontsize=9
        )

    plt.suptitle(
        f"Fairness Analysis — Protected Attribute: {attribute}\n"
        f"(Red bars indicate fairness concerns)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    save_path = f"{save_dir}/fairness_{attribute}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Fairness chart saved to {save_path}")
