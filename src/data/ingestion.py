"""
src/data/ingestion.py — Data Ingestion & Validation
=====================================================

LAYMAN EXPLANATION:
    Before you can train a model, you need DATA. This file handles:
    1. Loading the raw loan dataset from a CSV file
    2. Validating that the data looks correct (right columns, right types)
    3. Generating a sample dataset if no real data is available
       (so you can run the project without downloading anything)

    Think of this as the "front door" of the project — all data
    enters the system through here.

DATA SCIENTIST EXPLANATION:
    This module provides a robust data loading pipeline with:
    - Schema validation against expected columns
    - Basic sanity checks (shape, dtypes, target distribution)
    - Synthetic data generation via numpy for testing/demo
    - Structured logging via loguru for observability
    - Clean separation between raw load and downstream processing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from config import cfg


# ============================================================
# CORE COLUMNS — Minimum required columns in the dataset
# ============================================================
# The LendingClub dataset has 150+ columns. We only REQUIRE these
# core ones to function. The rest are optional/engineered.
REQUIRED_COLUMNS = [
    "loan_amnt",        # How much the borrower wants to borrow
    "funded_amnt",      # How much was actually funded
    "int_rate",         # Interest rate on the loan (%)
    "installment",      # Monthly payment amount ($)
    "annual_inc",       # Borrower's self-reported annual income
    "dti",              # Debt-to-income ratio (total debt / annual income)
    "delinq_2yrs",      # Times borrower was 30+ days late in past 2 years
    "fico_range_low",   # Lower bound of FICO credit score
    "fico_range_high",  # Upper bound of FICO credit score
    "open_acc",         # Number of open credit lines
    "pub_rec",          # Number of derogatory public records
    "revol_bal",        # Total revolving credit balance
    "revol_util",       # Revolving line utilization rate (%)
    "total_acc",        # Total number of credit lines ever
    "loan_status",      # TARGET: "Fully Paid" or "Charged Off"
    "term",             # Loan term: "36 months" or "60 months"
    "grade",            # LC assigned loan grade (A-G)
    "home_ownership",   # RENT, OWN, MORTGAGE, OTHER
    "purpose",          # Loan purpose (debt_consolidation, credit_card, etc.)
    "verification_status",  # Whether income was verified
]


def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """
    Load raw loan data from a CSV file.

    LAYMAN: Opens the CSV file containing loan data and loads it
    into a pandas DataFrame (like an in-memory spreadsheet).

    Args:
        filepath: Path to the CSV file. Defaults to config path.

    Returns:
        pd.DataFrame: Raw loan data with all original columns.
    """
    if filepath is None:
        filepath = cfg.data.raw_filename
        filepath = str(cfg.data.raw_filename)
        # Build the full path
        from config import RAW_DATA_DIR
        filepath = RAW_DATA_DIR / cfg.data.raw_filename

    filepath = Path(filepath)

    if not filepath.exists():
        logger.warning(
            f"Data file not found at {filepath}. "
            "Generating synthetic dataset for demonstration..."
        )
        return generate_synthetic_data()

    logger.info(f"Loading data from {filepath}...")
    # low_memory=False prevents pandas from guessing dtypes incorrectly
    # on large files with mixed types in the same column
    df = pd.read_csv(filepath, low_memory=False)

    logger.info(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the loaded DataFrame has the expected structure.

    LAYMAN: Like a quality control check at a factory. Before the data
    goes into the ML pipeline, we verify it has the right columns and
    the right types of values. If something is wrong, we raise an error
    with a clear explanation of what's missing.

    DATA SCIENTIST NOTE:
    In production, you'd use Great Expectations or Pandera for
    schema validation. Here we keep it simple for clarity.

    Args:
        df: Raw DataFrame to validate.

    Returns:
        pd.DataFrame: Validated DataFrame (same data, just confirmed valid).

    Raises:
        ValueError: If required columns are missing.
    """
    logger.info("Validating dataset schema...")

    # --- Check 1: Required columns present ---
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)[:20]}..."
        )

    # --- Check 2: Dataset is not empty ---
    if len(df) == 0:
        raise ValueError("Dataset is empty! Check the CSV file.")

    # --- Check 3: Target column has expected values ---
    target_col = cfg.data.target_column
    if target_col in df.columns:
        unique_statuses = df[target_col].unique()
        logger.info(f"Loan status distribution:\n{df[target_col].value_counts()}")

        # Filter to only the two classes we care about
        valid_statuses = [cfg.data.default_label, cfg.data.paid_label]
        df = df[df[target_col].isin(valid_statuses)].copy()
        logger.info(
            f"After filtering to {valid_statuses}: "
            f"{len(df):,} rows remain"
        )

    # --- Check 4: Report missing value summary ---
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 20]
    if len(high_missing) > 0:
        logger.warning(
            f"Columns with >20% missing values (will be handled in preprocessing):\n"
            f"{high_missing.head(10)}"
        )

    logger.info(f"Validation passed. Dataset shape: {df.shape}")
    return df


def generate_synthetic_data(n_samples: int = 20_000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic loan dataset for demo/testing.

    LAYMAN: If you don't have the real LendingClub dataset, this function
    creates fake but realistic loan data so you can still run and test
    the entire project. The fake data follows the same statistical patterns
    as real loan data (e.g., most loans are repaid, FICO scores cluster
    around 680-750, etc.)

    DATA SCIENTIST NOTE:
    This uses carefully calibrated numpy distributions to mimic real
    LendingClub data. Default rate is ~15% to reflect real-world base rates.
    Features are correlated with the target in realistic ways:
    - Higher DTI → higher default probability
    - Higher FICO → lower default probability
    - Grade A/B → lower default probability

    Args:
        n_samples: Number of synthetic loan records to generate.
        random_state: Seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic loan dataset with all required columns.
    """
    logger.info(f"Generating {n_samples:,} synthetic loan records...")
    rng = np.random.RandomState(random_state)

    # --- Generate correlated risk signal ---
    # This hidden "risk score" drives correlations between features and target
    # High risk_score = more likely to default
    risk_score = rng.randn(n_samples)  # Standard normal distribution

    # --- Loan Amount: $1,000 to $40,000 ---
    loan_amnt = rng.choice(
        [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
         12000, 15000, 20000, 25000, 30000, 35000, 40000],
        size=n_samples
    )

    # --- Annual Income: Log-normal distribution (realistic income distribution) ---
    # Most people earn $30k-$100k, some earn more, very few earn <$20k
    annual_inc = np.exp(rng.normal(11.0, 0.5, n_samples))  # ~$60k median
    annual_inc = np.clip(annual_inc, 12000, 500000)

    # --- Interest Rate: Higher for riskier borrowers ---
    # Risk score increases interest rate (lenders charge more for risk)
    int_rate = 10 + 3 * risk_score + rng.normal(0, 2, n_samples)
    int_rate = np.clip(int_rate, 5.0, 30.99)

    # --- DTI (Debt-to-Income Ratio): Higher for riskier borrowers ---
    dti = 15 + 8 * risk_score + rng.normal(0, 5, n_samples)
    dti = np.clip(dti, 0, 50)

    # --- FICO Score: Lower for riskier borrowers ---
    # FICO scores range 300-850. Most LendingClub borrowers: 660-780
    fico_low = 680 - 30 * risk_score + rng.normal(0, 20, n_samples)
    fico_low = np.clip(fico_low, 600, 840).astype(int)
    fico_high = fico_low + 4  # FICO range is typically 4 points wide

    # --- Delinquencies: More for riskier borrowers ---
    delinq_2yrs = np.maximum(0, rng.poisson(0.3 + 0.5 * np.maximum(0, risk_score), n_samples))

    # --- Revolving Utilization: Higher for riskier borrowers ---
    revol_util = 30 + 20 * risk_score + rng.normal(0, 15, n_samples)
    revol_util = np.clip(revol_util, 0, 100)

    # --- Revolving Balance ---
    revol_bal = rng.lognormal(8, 1, n_samples)
    revol_bal = np.clip(revol_bal, 0, 100000)

    # --- Open Accounts ---
    open_acc = rng.poisson(10, n_samples)
    open_acc = np.clip(open_acc, 1, 40)

    # --- Total Accounts ---
    total_acc = open_acc + rng.poisson(8, n_samples)

    # --- Public Records ---
    pub_rec = np.maximum(0, rng.poisson(0.1 + 0.3 * np.maximum(0, risk_score), n_samples))

    # --- Installment (monthly payment) ---
    installment = (loan_amnt * int_rate / 100 / 12) + rng.normal(0, 20, n_samples)
    installment = np.clip(installment, 50, 2000)

    # --- Loan Term ---
    # 60% take 36-month loans, 40% take 60-month loans
    term = rng.choice(["36 months", "60 months"], n_samples, p=[0.60, 0.40])

    # --- Loan Grade (A=safest, G=riskiest) ---
    # Grade is strongly correlated with risk_score
    grade_probs_by_risk = np.column_stack([
        1 / (1 + np.exp(risk_score + 2)),   # P(A)
        1 / (1 + np.exp(risk_score + 1)),   # P(B) - simplified
        0.2 * np.ones(n_samples),            # P(C)
        0.15 * np.ones(n_samples),           # P(D)
        0.10 * np.ones(n_samples),           # P(E)
        0.05 * np.ones(n_samples),           # P(F)
        0.05 * np.ones(n_samples),           # P(G)
    ])
    # Normalize to proper probabilities
    grade_probs_by_risk = grade_probs_by_risk / grade_probs_by_risk.sum(axis=1, keepdims=True)
    grades = np.array(["A", "B", "C", "D", "E", "F", "G"])
    grade = np.array([
        rng.choice(grades, p=grade_probs_by_risk[i])
        for i in range(n_samples)
    ])

    # --- Home Ownership ---
    home_ownership = rng.choice(
        ["MORTGAGE", "RENT", "OWN", "OTHER"],
        n_samples,
        p=[0.48, 0.38, 0.12, 0.02]
    )

    # --- Loan Purpose ---
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement",
         "other", "major_purchase", "medical", "small_business",
         "car", "vacation", "moving", "house", "wedding"],
        n_samples,
        p=[0.45, 0.20, 0.08, 0.07, 0.05, 0.04, 0.03,
           0.03, 0.02, 0.01, 0.01, 0.01]
    )

    # --- Verification Status ---
    verification_status = rng.choice(
        ["Not Verified", "Verified", "Source Verified"],
        n_samples,
        p=[0.35, 0.33, 0.32]
    )

    # --- Credit History Age (months since earliest credit line) ---
    mths_since_earliest = rng.randint(24, 360, n_samples)

    # --- Inquiries in last 6 months ---
    inq_last_6mths = np.maximum(0, rng.poisson(1 + np.maximum(0, risk_score), n_samples))
    inq_last_6mths = np.clip(inq_last_6mths, 0, 10)

    # --- Collections in last 12 months ---
    collections_12_mths_ex_med = np.maximum(
        0, rng.poisson(0.05 + 0.2 * np.maximum(0, risk_score), n_samples)
    )

    # === TARGET VARIABLE: Loan Status ===
    # Probability of default = risk_score + delinquency penalties + irreducible noise.
    # The irreducible noise (std=1.2) simulates real-world uncertainty — not every
    # high-risk borrower defaults, not every low-risk one repays. This keeps
    # AUC bounded to ~0.73-0.78 on any platform/Python version.
    delinq_effect = np.minimum(delinq_2yrs * 0.3, 1.5)   # each delinquency adds ~0.3 risk
    pub_rec_effect = np.minimum(pub_rec * 0.4, 1.5)       # each public record adds ~0.4 risk
    irreducible_noise = rng.normal(0, 1.2, n_samples)     # caps achievable AUC ~0.73-0.78
    adjusted_risk = risk_score + delinq_effect + pub_rec_effect + irreducible_noise
    default_prob = 1 / (1 + np.exp(-1.0 * adjusted_risk + 1.73))
    default_prob = np.clip(default_prob, 0.01, 0.99)

    loan_status = np.where(
        rng.random(n_samples) < default_prob,
        cfg.data.default_label,   # "Charged Off"
        cfg.data.paid_label       # "Fully Paid"
    )

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "loan_amnt": loan_amnt.astype(float),
        "funded_amnt": loan_amnt * rng.uniform(0.95, 1.0, n_samples),
        "funded_amnt_inv": loan_amnt * rng.uniform(0.90, 1.0, n_samples),
        "term": term,
        "int_rate": np.round(int_rate, 2),
        "installment": np.round(installment, 2),
        "grade": grade,
        "sub_grade": [g + n for g, n in zip(grade, rng.choice(["1","2","3","4","5"], n_samples))],
        "emp_length": rng.choice(
            ["< 1 year","1 year","2 years","3 years","4 years",
             "5 years","6 years","7 years","8 years","9 years","10+ years"],
            n_samples
        ),
        "home_ownership": home_ownership,
        "annual_inc": np.round(annual_inc, 2),
        "verification_status": verification_status,
        "loan_status": loan_status,
        "purpose": purpose,
        "dti": np.round(dti, 2),
        "delinq_2yrs": delinq_2yrs.astype(float),
        "fico_range_low": fico_low.astype(float),
        "fico_range_high": fico_high.astype(float),
        "inq_last_6mths": inq_last_6mths.astype(float),
        "mths_since_last_delinq": np.where(
            delinq_2yrs > 0,
            rng.randint(1, 24, n_samples),
            np.nan
        ),
        "mths_since_last_record": np.where(
            pub_rec > 0,
            rng.randint(1, 120, n_samples),
            np.nan
        ),
        "open_acc": open_acc.astype(float),
        "pub_rec": pub_rec.astype(float),
        "revol_bal": np.round(revol_bal, 2),
        "revol_util": np.round(revol_util, 1),
        "total_acc": total_acc.astype(float),
        "initial_list_status": rng.choice(["w", "f"], n_samples, p=[0.7, 0.3]),
        "collections_12_mths_ex_med": collections_12_mths_ex_med.astype(float),
        "mths_since_last_major_derog": np.where(
            rng.random(n_samples) < 0.1,
            rng.randint(1, 60, n_samples),
            np.nan
        ),
        "application_type": rng.choice(
            ["Individual", "Joint App"],
            n_samples,
            p=[0.90, 0.10]
        ),
        "acc_now_delinq": np.maximum(0, rng.poisson(0.02, n_samples)).astype(float),
        "tot_coll_amt": np.maximum(0, rng.exponential(50, n_samples)),
        "tot_cur_bal": np.maximum(0, rng.lognormal(10, 1.5, n_samples)),
        "total_rev_hi_lim": np.maximum(1000, rng.lognormal(10, 1, n_samples)),
        "issue_d": pd.date_range("2015-01-01", periods=n_samples, freq="H").strftime("%b-%Y"),
        "earliest_cr_line": [
            f"Jan-{2023 - m // 12}" for m in mths_since_earliest
        ],
        "addr_state": rng.choice(
            ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"],
            n_samples
        ),
    })

    # Log the synthetic dataset stats
    default_rate = (df["loan_status"] == cfg.data.default_label).mean()
    logger.info(
        f"Synthetic dataset generated: {len(df):,} rows, "
        f"default rate: {default_rate:.1%}"
    )

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset for logging/reporting.

    LAYMAN: Creates a "report card" for the dataset — how many rows,
    how many defaults, how many missing values, etc.

    Returns:
        dict: Summary statistics dictionary.
    """
    target_col = cfg.data.target_column
    summary = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_missing_total": int(df.isnull().sum().sum()),
        "missing_pct": float(df.isnull().mean().mean() * 100),
    }

    if target_col in df.columns:
        value_counts = df[target_col].value_counts()
        summary["class_distribution"] = value_counts.to_dict()
        summary["default_rate"] = float(
            (df[target_col] == cfg.data.default_label).mean()
        )

    return summary


if __name__ == "__main__":
    # Quick test — run this file directly to verify it works
    df = load_raw_data()
    df = validate_data(df)
    summary = get_data_summary(df)
    print("\n=== Dataset Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
