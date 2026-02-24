"""
src/features/engineer.py — Feature Engineering
================================================

LAYMAN EXPLANATION:
    Imagine you're a bank loan officer. You don't just look at raw numbers.
    You COMBINE them to ask smarter questions:

    "Can this person actually afford this loan?"
    → monthly_payment / monthly_income  (payment burden ratio)

    "Are they stretched thin on existing debt?"
    → total_debt / annual_income  (debt-to-income ratio)

    "Do they live beyond their means on credit cards?"
    → credit_card_balance / credit_card_limit  (utilization ratio)

    Each of these COMBINED features is FAR more predictive than the
    raw numbers alone. This is what feature engineering does —
    it creates smarter inputs for the model.

    We go from ~30 raw columns to ~60 engineered features.

DATA SCIENTIST EXPLANATION:
    This module creates domain-specific features grounded in credit
    risk theory (Basel II, FICO scoring methodology, Altman Z-score):

    1. Debt ratios (capacity indicators)
    2. Credit utilization signals
    3. Delinquency and payment history features
    4. Credit mix and age features
    5. Loan structure features
    6. Interaction and polynomial features

    All engineering logic is implemented as pure functions operating
    on DataFrames, making them easy to test and compose.
    The pipeline.py module wraps these in sklearn transformers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from loguru import logger


def engineer_debt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create debt capacity and affordability features.

    LAYMAN: These features answer the core question banks ask:
    "Can this person realistically pay back this loan?"

    High debt relative to income → higher default risk
    High monthly payment relative to income → higher default risk

    DATA SCIENTIST NOTE:
    DTI (debt-to-income) is the #1 feature in most credit models.
    CFPB (Consumer Financial Protection Bureau) caps QM loans at 43% DTI.
    """
    logger.debug("Engineering debt features...")

    # Monthly income (annual_inc is provided yearly)
    df["monthly_income"] = df["annual_inc"] / 12

    # Payment-to-income ratio: What fraction of monthly income goes to THIS loan
    # >30% is generally considered high burden
    df["payment_to_income"] = df["installment"] / (df["monthly_income"] + 1)

    # Total debt burden: DTI × income gives total monthly debt payments
    # Then we add THIS loan's payment on top
    df["total_monthly_debt"] = (df["dti"] / 100 * df["annual_inc"]) / 12
    df["total_debt_with_loan"] = df["total_monthly_debt"] + df["installment"]

    # Affordability ratio: (total debt + new payment) / monthly income
    # >50% is a serious red flag
    df["total_payment_burden"] = df["total_debt_with_loan"] / (df["monthly_income"] + 1)

    # Loan-to-income ratio: How large is this loan relative to annual income?
    # Borrowing 3× annual income is very different from borrowing 10%
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)

    # Annual payment vs income
    df["annual_payment_to_income"] = (df["installment"] * 12) / (df["annual_inc"] + 1)

    return df


def engineer_credit_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create credit utilization and revolving credit features.

    LAYMAN: Your credit card limit is $10,000 but you've spent $8,000.
    That's 80% utilization — a major red flag for lenders. It means
    you're close to maxing out your credit, suggesting financial stress.

    FICO scoring penalizes utilization above 30%.
    Utilization above 70% is considered "credit overextension."

    DATA SCIENTIST NOTE:
    Credit utilization is ~30% of FICO score weight. It's one of the
    most responsive features — paying down a card can improve FICO by
    50+ points within a month.
    """
    logger.debug("Engineering credit utilization features...")

    # Raw utilization is already a percentage (0-100)
    # Create binary flags for dangerous thresholds
    df["high_utilization_flag"] = (df["revol_util"] > 70).astype(int)
    df["moderate_utilization_flag"] = (
        (df["revol_util"] > 30) & (df["revol_util"] <= 70)
    ).astype(int)

    # Revolving balance per open account
    # Very high balance per account suggests maxed-out cards
    df["revol_bal_per_account"] = df["revol_bal"] / (df["open_acc"] + 1)

    # Utilization tier (0=low, 1=medium, 2=high, 3=maxed)
    df["util_tier"] = pd.cut(
        df["revol_util"].fillna(50),
        bins=[-1, 30, 60, 80, 101],
        labels=[0, 1, 2, 3]
    ).astype(float)

    return df


def engineer_delinquency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create payment history and delinquency features.

    LAYMAN: "Delinquency" means you were late on a payment.
    Someone who was late 3 times in the past 2 years is much more
    likely to default than someone with a perfect payment history.

    FICO scoring weights payment history at 35% — the single largest factor.

    DATA SCIENTIST NOTE:
    Delinquency features are strong predictors but are very sparse
    (most borrowers have 0 delinquencies). We create interaction terms
    and recency-weighted features to extract maximum signal.
    """
    logger.debug("Engineering delinquency features...")

    # Delinquency rate (normalized by number of accounts)
    df["delinquency_rate"] = df["delinq_2yrs"] / (df["total_acc"] + 1)

    # Binary: has ANY delinquency in past 2 years
    df["has_delinquency"] = (df["delinq_2yrs"] > 0).astype(int)

    # Multiple delinquencies flag (serial late-payer)
    df["multiple_delinquencies"] = (df["delinq_2yrs"] > 1).astype(int)

    # Public records flag (bankruptcies, tax liens, judgments)
    # Even 1 public record is a serious negative indicator
    df["has_public_record"] = (df["pub_rec"] > 0).astype(int)

    # Combined derogatory flag: delinquency OR public record
    df["derogatory_flag"] = (
        (df["delinq_2yrs"] > 0) | (df["pub_rec"] > 0)
    ).astype(int)

    # Months since last delinquency (recency matters)
    # Recent delinquency = higher risk than delinquency 5 years ago
    if "mths_since_last_delinq" in df.columns:
        # Fill NaN with a large number (meaning "never delinquent")
        df["mths_since_last_delinq_filled"] = df["mths_since_last_delinq"].fillna(999)
        df["recent_delinquency"] = (df["mths_since_last_delinq_filled"] < 24).astype(int)
    else:
        df["mths_since_last_delinq_filled"] = 999
        df["recent_delinquency"] = 0

    # Inquiries in last 6 months (applying for lots of credit = financial stress)
    df["high_inquiries"] = (df["inq_last_6mths"] > 3).astype(int)

    # Delinquency severity score (weighted combination)
    df["delinquency_severity"] = (
        df["delinq_2yrs"] * 2.0 +
        df["pub_rec"] * 3.0 +
        df["high_inquiries"] * 1.0
    )

    return df


def engineer_credit_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create credit history length and depth features.

    LAYMAN: Someone who's had a credit card for 20 years without issues
    is a fundamentally different borrower than someone who opened their
    first credit card 6 months ago. Age and diversity of credit history
    are strong trust signals.

    FICO weights "length of credit history" at 15%.

    DATA SCIENTIST NOTE:
    Credit age interacts importantly with other features. A 25-year-old
    with 5 years of credit history is very different from a 50-year-old
    with 5 years of history (the latter is more concerning).
    """
    logger.debug("Engineering credit history features...")

    # Average age per account (credit_age_years / total_acc)
    if "credit_age_years" in df.columns:
        df["avg_account_age"] = df["credit_age_years"] / (df["total_acc"] + 1)
        df["young_credit_flag"] = (df["credit_age_years"] < 3).astype(int)
        df["mature_credit_flag"] = (df["credit_age_years"] > 10).astype(int)
    else:
        df["avg_account_age"] = 5.0  # Median imputation
        df["young_credit_flag"] = 0
        df["mature_credit_flag"] = 0

    # Account diversity: ratio of open to total accounts
    # High ratio = still actively using many of their credit lines
    df["account_openness_ratio"] = df["open_acc"] / (df["total_acc"] + 1)

    # Credit breadth score (more accounts = more credit experience)
    df["credit_breadth"] = np.log1p(df["total_acc"])  # Log to reduce skew

    return df


def engineer_loan_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create loan-specific structural features.

    LAYMAN: The STRUCTURE of a loan matters:
    - A 60-month loan is riskier than a 36-month loan (more time to default)
    - A higher interest rate means the bank already priced in more risk
    - A large loan relative to what was requested might mean the applicant
      got a smaller loan than wanted (bank wasn't fully confident)

    DATA SCIENTIST NOTE:
    Interest rate is a proxy for bank-assessed risk (banks price risk into rates).
    Including it as a feature is sometimes considered "leakage" because it
    reflects model-like scoring by the bank. However, it IS known at application
    time (at least as a range), so we include it with this caveat noted.
    """
    logger.debug("Engineering loan structure features...")

    # Monthly interest cost as fraction of loan amount
    df["monthly_interest_rate"] = df["int_rate"] / 100 / 12

    # Total interest paid over life of loan (approximate)
    if "term" in df.columns:
        # Ensure term is numeric — at inference time it may still be "36 months" string
        term_numeric = pd.to_numeric(
            df["term"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce"
        ).fillna(36)
        df["total_interest_cost"] = df["installment"] * term_numeric - df["loan_amnt"]
        df["total_interest_cost"] = df["total_interest_cost"].clip(lower=0)

        # Interest-to-principal ratio (how expensive is this loan?)
        df["interest_to_principal"] = df["total_interest_cost"] / (df["loan_amnt"] + 1)

    # Loan funding gap: Did the bank fund the full requested amount?
    # If funded_amnt < loan_amnt, the bank was conservative
    if "funded_amnt" in df.columns:
        df["funding_gap"] = df["loan_amnt"] - df["funded_amnt"]
        df["funding_gap"] = df["funding_gap"].clip(lower=0)
        df["partial_funding_flag"] = (df["funding_gap"] > 0).astype(int)

    # High interest rate flag (>15% is typically subprime)
    df["high_interest_flag"] = (df["int_rate"] > 15).astype(int)
    df["very_high_interest_flag"] = (df["int_rate"] > 20).astype(int)

    return df


def engineer_fico_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create FICO score-based features.

    LAYMAN: FICO score is the most widely used credit score (range 300-850).
    Lenders use it as a quick summary of creditworthiness.
    - 750+ = Excellent (very unlikely to default)
    - 700-749 = Good
    - 650-699 = Fair
    - 600-649 = Poor
    - <600 = Very Poor (very likely to default)

    DATA SCIENTIST NOTE:
    We use the midpoint of the range and create categorical bins
    because the relationship between FICO and default is non-linear
    (below 620, risk rises sharply; above 750, it plateaus).
    """
    logger.debug("Engineering FICO features...")

    # FICO midpoint (LendingClub provides a range, not exact score)
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # FICO bins (standard credit risk categories)
    df["fico_tier"] = pd.cut(
        df["fico_score"].fillna(650),
        bins=[0, 580, 620, 660, 700, 740, 780, 850],
        labels=[0, 1, 2, 3, 4, 5, 6]  # 0=worst, 6=best
    ).astype(float)

    # Subprime flag: FICO below 640 is considered "subprime"
    df["subprime_flag"] = (df["fico_score"] < 640).astype(int)

    # Prime flag: FICO above 720 is considered "prime"
    df["prime_flag"] = (df["fico_score"] > 720).astype(int)

    return df


def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between key risk signals.

    LAYMAN: Sometimes two bad things together are much worse than
    each bad thing alone. For example:
    - High DTI AND low FICO = VERY high risk (both struggling financially
      AND has bad credit history)
    - High utilization AND recent delinquency = maxed out AND late = crisis

    These "interaction features" capture such combined signals.

    DATA SCIENTIST NOTE:
    Tree-based models (XGBoost, LightGBM) can discover interactions
    automatically, but explicitly adding the most important ones often
    improves AUC by 1-3 percentage points. These are grounded in
    credit risk domain knowledge.
    """
    logger.debug("Engineering interaction features...")

    # DTI × FICO interaction: normalized risk composite
    # High DTI (bad) × Low FICO (bad) → very high risk signal
    df["dti_fico_interaction"] = df["dti"] * (850 - df["fico_score"].fillna(680)) / 100

    # Utilization × Delinquency interaction
    # High utilization AND delinquencies = serious distress signal
    df["util_delinq_interaction"] = (
        df["revol_util"].fillna(50) / 100 * (df["delinq_2yrs"] + 1)
    )

    # Payment burden × FICO: struggling AND bad credit = highest risk
    if "payment_to_income" in df.columns:
        df["burden_fico_interaction"] = (
            df["payment_to_income"] * (850 - df["fico_score"].fillna(680)) / 850
        )

    # Income volatility proxy: high loan relative to income despite low FICO
    if "loan_to_income" in df.columns:
        df["aspirational_borrower"] = (
            df["loan_to_income"] * (1 - df["fico_score"].fillna(680) / 850)
        )

    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: apply ALL feature engineering in sequence.

    LAYMAN: This runs all the feature creation functions above,
    one after another, building up from ~30 raw columns to ~70+
    engineered features that give the model the richest possible
    signal to learn from.

    DATA SCIENTIST NOTE:
    Order matters here — later functions may use columns created
    by earlier functions (e.g., engineer_interaction_features uses
    columns created by engineer_debt_features).

    Args:
        df: Preprocessed DataFrame with clean raw columns.

    Returns:
        pd.DataFrame: DataFrame with all engineered features added.
    """
    logger.info(f"Starting feature engineering. Input shape: {df.shape}")
    initial_cols = df.shape[1]

    df = engineer_debt_features(df)
    df = engineer_credit_utilization_features(df)
    df = engineer_delinquency_features(df)
    df = engineer_credit_history_features(df)
    df = engineer_loan_structure_features(df)
    df = engineer_fico_features(df)
    df = engineer_interaction_features(df)

    new_cols = df.shape[1] - initial_cols
    logger.info(
        f"Feature engineering complete. "
        f"Added {new_cols} new features. "
        f"Output shape: {df.shape}"
    )

    return df
