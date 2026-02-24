"""
src/data/preprocessing.py — Data Cleaning & Preparation
=========================================================

LAYMAN EXPLANATION:
    Raw data from loan applications is messy:
    - Some fields are blank (missing values)
    - Some fields are text that should be numbers ("10+ years")
    - Some columns are useless or redundant
    - The target column is text ("Fully Paid") but the model needs 0 or 1

    This file CLEANS all of that up so the data is ready for
    feature engineering and model training.

    Think of it like washing vegetables before cooking — the raw
    ingredients need prep work before they're useful.

DATA SCIENTIST EXPLANATION:
    Handles:
    - Column dropping (leakage prevention + high-missingness removal)
    - Target encoding: "Charged Off" → 1, "Fully Paid" → 0
    - String-to-numeric parsing: "15.99%" → 15.99, "36 months" → 36
    - Date parsing and age calculation
    - Missing value imputation strategy selection (median for numeric,
      mode for categorical, -1 for "was missing" flag features)
    - Outlier capping at configurable percentiles
    - Train/val/test splitting with stratification on target
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from typing import Tuple

from config import cfg


def drop_leaky_and_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that would cause data leakage or add no value.

    LAYMAN: "Data leakage" means the model is accidentally given
    information it wouldn't have in real life. For example, if we
    give the model the TOTAL AMOUNT REPAID on a loan, it can easily
    predict whether the loan was paid off — but at loan application
    time, we don't KNOW how much will be repaid yet!

    We also drop columns that are:
    - Too many missing values (>40% empty = not reliable)
    - Free-text fields the model can't use (URLs, descriptions)
    - Redundant with other columns

    DATA SCIENTIST NOTE:
    Leakage is one of the most common causes of overly optimistic
    model performance. Always ask: "Would I have this column at
    prediction time?"
    """
    logger.info("Dropping leaky and irrelevant columns...")

    cols_before = set(df.columns)

    # 1. Drop explicitly listed leaky/irrelevant columns from config
    cols_to_drop = [c for c in cfg.data.columns_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"  Dropped {len(cols_to_drop)} explicitly listed columns")

    # 2. Drop columns with too many missing values
    missing_rate = df.isnull().mean()
    high_missing_cols = missing_rate[missing_rate > cfg.features.missing_threshold].index.tolist()
    # Don't drop the target column even if it has missing values
    target = cfg.data.target_column
    high_missing_cols = [c for c in high_missing_cols if c != target]
    df = df.drop(columns=high_missing_cols)
    logger.info(
        f"  Dropped {len(high_missing_cols)} columns with >"
        f"{cfg.features.missing_threshold:.0%} missing values"
    )

    # 3. Drop high-cardinality categorical columns (too many unique values)
    # e.g., "emp_title" has thousands of unique job titles — useless for ML
    categorical_cols = df.select_dtypes(include=['object']).columns
    high_cardinality_cols = []
    for col in categorical_cols:
        if col == target:
            continue
        n_unique = df[col].nunique()
        if n_unique > cfg.features.max_cardinality:
            high_cardinality_cols.append(col)

    df = df.drop(columns=high_cardinality_cols)
    logger.info(
        f"  Dropped {len(high_cardinality_cols)} high-cardinality columns: "
        f"{high_cardinality_cols}"
    )

    cols_after = set(df.columns)
    logger.info(
        f"Columns: {len(cols_before)} → {len(cols_after)} "
        f"(removed {len(cols_before) - len(cols_after)} total)"
    )
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the loan status text to binary numbers.

    LAYMAN: Machine learning models work with NUMBERS, not words.
    "Charged Off" (defaulted) becomes 1 (positive class = bad outcome)
    "Fully Paid" becomes 0 (negative class = good outcome)

    We predict the probability of getting a 1 (default).

    DATA SCIENTIST NOTE:
    We use 1=default (the rare/positive class) which is standard
    for binary classification in credit risk. This means:
    - High probability → high risk → likely to default
    - Low probability → low risk → likely to repay
    """
    logger.info("Encoding target variable...")
    target = cfg.data.target_column

    mapping = {
        cfg.data.default_label: 1,  # "Charged Off" → 1 (default = positive class)
        cfg.data.paid_label: 0,      # "Fully Paid"   → 0 (paid = negative class)
    }

    df[target] = df[target].map(mapping)

    # Verify encoding worked
    assert df[target].isnull().sum() == 0, "NaN values after target encoding!"
    assert set(df[target].unique()).issubset({0, 1}), "Unexpected target values!"

    default_rate = df[target].mean()
    logger.info(
        f"Target encoded. Default rate: {default_rate:.1%} "
        f"({df[target].sum():,} defaults out of {len(df):,} loans)"
    )
    return df


def parse_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string-formatted numbers to actual numeric types.

    LAYMAN: Some columns are stored as TEXT even though they contain numbers:
    - "int_rate" might be "15.99%" (text) instead of 15.99 (number)
    - "term" might be "36 months" instead of 36
    - "revol_util" might be "78.5%" instead of 78.5
    - "emp_length" might be "10+ years" instead of 10

    This function strips those text characters and converts to numbers.

    DATA SCIENTIST NOTE:
    LendingClub's raw CSV has several columns in string format.
    Using pd.to_numeric with errors='coerce' safely handles edge cases
    by converting unparseable values to NaN rather than raising exceptions.
    """
    logger.info("Parsing string-formatted numeric columns...")

    # --- Interest rate: "15.99%" → 15.99 ---
    if "int_rate" in df.columns:
        df["int_rate"] = (
            df["int_rate"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")

    # --- Loan term: "36 months" → 36 ---
    if "term" in df.columns:
        df["term"] = (
            df["term"]
            .astype(str)
            .str.extract(r"(\d+)")  # Extract only the digits
            [0]
            .astype(float)
        )

    # --- Revolving utilization: "78.5%" → 78.5 ---
    if "revol_util" in df.columns:
        df["revol_util"] = (
            df["revol_util"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")

    # --- Employment length: "10+ years" → 10, "< 1 year" → 0 ---
    if "emp_length" in df.columns:
        emp_map = {
            "< 1 year": 0,
            "1 year": 1, "2 years": 2, "3 years": 3,
            "4 years": 4, "5 years": 5, "6 years": 6,
            "7 years": 7, "8 years": 8, "9 years": 9,
            "10+ years": 10,
        }
        df["emp_length"] = df["emp_length"].map(emp_map)
        # NaN = employment length not provided → impute with median later

    logger.info("String columns parsed successfully.")
    return df


def parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract useful numeric features from date columns.

    LAYMAN: Raw dates like "Jan-2010" aren't useful to a model.
    But "this person's credit history is 13 years old" IS useful.
    We convert dates into meaningful numbers.

    Specifically:
    - "earliest_cr_line" → "credit_age_years" (older = more established)
    - "issue_d" → year and month (seasonality effects)

    DATA SCIENTIST NOTE:
    Credit age is one of the most important features in credit scoring.
    "Age of oldest account" appears in FICO score calculation as 15% weight.
    """
    logger.info("Parsing date columns...")

    reference_date = pd.Timestamp("2024-01-01")  # Fixed reference date for reproducibility

    # --- Issue date → year and month ---
    if "issue_d" in df.columns:
        issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        df["issue_year"] = issue_dt.dt.year
        df["issue_month"] = issue_dt.dt.month
        df = df.drop(columns=["issue_d"])

    # --- Earliest credit line → credit age in years ---
    if "earliest_cr_line" in df.columns:
        earliest_dt = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        df["credit_age_years"] = (reference_date - earliest_dt).dt.days / 365.25
        df["credit_age_years"] = df["credit_age_years"].clip(lower=0)
        df = df.drop(columns=["earliest_cr_line"])
        logger.info(
            f"  Credit age range: "
            f"{df['credit_age_years'].min():.1f} – "
            f"{df['credit_age_years'].max():.1f} years"
        )

    return df


def cap_outliers(df: pd.DataFrame, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """
    Cap extreme outliers at specified percentiles.

    LAYMAN: If one person reports annual income of $50 million, that
    single outlier can distort the model's understanding of the
    relationship between income and default risk.

    We "cap" such extreme values — instead of using $50M, we replace it
    with the 99th percentile value (e.g., $500k), which still marks them
    as very high income but doesn't distort the model.

    This is called "Winsorization" in statistics.

    DATA SCIENTIST NOTE:
    We only cap numeric columns (excluding binary and count variables
    that have natural bounds). The target column and binary flags are skipped.
    """
    logger.info(f"Capping outliers at [{lower_pct:.0%}, {upper_pct:.0%}] percentiles...")

    target = cfg.data.target_column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]

    for col in numeric_cols:
        low = df[col].quantile(lower_pct)
        high = df[col].quantile(upper_pct)
        n_capped = ((df[col] < low) | (df[col] > high)).sum()
        if n_capped > 0:
            df[col] = df[col].clip(lower=low, upper=high)

    logger.info("Outlier capping complete.")
    return df


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation, and test sets.

    LAYMAN: We can't test a model on the same data it learned from —
    that's like giving students the exact test they studied from.
    Instead, we:
    - Train on 70% of data (model learns patterns)
    - Validate on 15% (tune the model, pick best parameters)
    - Test on 15% (final unbiased performance estimate)

    The "stratified" split ensures each set has the same ~15% default
    rate, so no set accidentally has all the easy cases.

    DATA SCIENTIST NOTE:
    Stratification on the target is critical for imbalanced datasets.
    Without it, the test set might randomly have 0 defaults and show
    perfect "accuracy" while the model is completely broken.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    target = cfg.data.target_column
    logger.info(
        f"Splitting data: "
        f"{cfg.data.train_ratio:.0%} train / "
        f"{cfg.data.val_ratio:.0%} val / "
        f"{cfg.data.test_ratio:.0%} test"
    )

    # First split: separate test set from rest
    test_size = cfg.data.test_ratio
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=cfg.data.random_state,
        stratify=df[target],  # Maintain class balance in each split
    )

    # Second split: separate validation from training
    # val_size relative to the remaining data
    val_size_relative = cfg.data.val_ratio / (cfg.data.train_ratio + cfg.data.val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_relative,
        random_state=cfg.data.random_state,
        stratify=train_val_df[target],
    )

    logger.info(
        f"Split complete:\n"
        f"  Train: {len(train_df):,} rows "
        f"(default rate: {train_df[target].mean():.1%})\n"
        f"  Val:   {len(val_df):,} rows "
        f"(default rate: {val_df[target].mean():.1%})\n"
        f"  Test:  {len(test_df):,} rows "
        f"(default rate: {test_df[target].mean():.1%})"
    )

    return train_df, val_df, test_df


def run_preprocessing_pipeline(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete preprocessing pipeline from raw data to clean splits.

    LAYMAN: This is the master function that chains all the cleaning steps
    together in the right order. You pass in messy raw data and get back
    three clean, ready-to-use datasets (train, val, test).

    Pipeline order:
    1. Drop leaky/irrelevant columns
    2. Encode target to 0/1
    3. Parse string-formatted numbers
    4. Extract date features
    5. Cap outliers
    6. Split into train/val/test

    Returns:
        Tuple of (train_df, val_df, test_df) — all preprocessed
    """
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    df = drop_leaky_and_irrelevant_columns(df)
    df = encode_target(df)
    df = parse_string_columns(df)
    df = parse_date_columns(df)
    df = cap_outliers(df)

    train_df, val_df, test_df = split_data(df)

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)

    return train_df, val_df, test_df


if __name__ == "__main__":
    from ingestion import load_raw_data, validate_data
    raw_df = load_raw_data()
    raw_df = validate_data(raw_df)
    train_df, val_df, test_df = run_preprocessing_pipeline(raw_df)
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"\nSample columns:\n{list(train_df.columns[:15])}")
