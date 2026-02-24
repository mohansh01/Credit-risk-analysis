"""
tests/test_features.py — Unit Tests for Feature Engineering
=============================================================

LAYMAN EXPLANATION:
    Tests verify that our feature engineering code works correctly.
    Think of them as quality checks on a factory assembly line —
    we test each step to make sure it produces the right output.

    For example:
    - If annual_inc=60000 and installment=500, then
      payment_to_income should equal 500 / (60000/12) = 0.10 (10%)

    If the code produces a different answer, the test FAILS and alerts us.
    This catches bugs BEFORE they reach production and affect real loans.

RUN TESTS:
    pytest tests/test_features.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    engineer_debt_features,
    engineer_credit_utilization_features,
    engineer_delinquency_features,
    engineer_fico_features,
    engineer_all_features,
)
from src.data.ingestion import generate_synthetic_data
from src.data.preprocessing import (
    parse_string_columns,
    encode_target,
    cap_outliers,
)


# ============================================================
# FIXTURES — Reusable test data
# ============================================================

@pytest.fixture
def sample_df():
    """Generate a small synthetic dataset for testing."""
    return generate_synthetic_data(n_samples=100, random_state=42)


@pytest.fixture
def clean_numeric_df():
    """A minimal clean DataFrame for testing specific features."""
    return pd.DataFrame({
        "annual_inc": [60000.0, 40000.0, 120000.0],
        "installment": [500.0, 800.0, 200.0],
        "dti": [15.0, 35.0, 8.0],
        "loan_amnt": [15000.0, 20000.0, 10000.0],
        "funded_amnt": [15000.0, 19500.0, 10000.0],
        "int_rate": [10.0, 20.0, 6.0],
        "term": [36.0, 60.0, 36.0],
        "revol_util": [30.0, 75.0, 10.0],
        "open_acc": [8.0, 4.0, 15.0],
        "total_acc": [20.0, 10.0, 30.0],
        "delinq_2yrs": [0.0, 3.0, 0.0],
        "pub_rec": [0.0, 1.0, 0.0],
        "revol_bal": [5000.0, 15000.0, 2000.0],
        "inq_last_6mths": [1.0, 5.0, 0.0],
        "fico_range_low": [720.0, 620.0, 780.0],
        "fico_range_high": [724.0, 624.0, 784.0],
        "credit_age_years": [10.0, 3.0, 20.0],
    })


# ============================================================
# TESTS: Data Generation
# ============================================================

class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_generates_correct_number_of_rows(self, sample_df):
        """Dataset should have exactly 100 rows."""
        assert len(sample_df) == 100

    def test_required_columns_present(self, sample_df):
        """Required columns should be in the dataset."""
        required = ["loan_amnt", "annual_inc", "dti", "fico_range_low", "loan_status"]
        for col in required:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_loan_status_binary(self, sample_df):
        """Loan status should only have two values."""
        assert set(sample_df["loan_status"].unique()) <= {"Fully Paid", "Charged Off"}

    def test_default_rate_reasonable(self, sample_df):
        """Default rate should be between 5% and 40%."""
        default_rate = (sample_df["loan_status"] == "Charged Off").mean()
        assert 0.05 <= default_rate <= 0.40, f"Unrealistic default rate: {default_rate:.1%}"

    def test_loan_amount_in_range(self, sample_df):
        """Loan amounts should be within expected range."""
        assert sample_df["loan_amnt"].min() >= 1000
        assert sample_df["loan_amnt"].max() <= 40000

    def test_fico_scores_valid(self, sample_df):
        """FICO scores should be in valid range (300-850)."""
        assert sample_df["fico_range_low"].min() >= 300
        assert sample_df["fico_range_high"].max() <= 850


# ============================================================
# TESTS: Preprocessing
# ============================================================

class TestPreprocessing:
    """Tests for data preprocessing functions."""

    def test_encode_target_creates_binary(self, sample_df):
        """Target encoding should produce only 0s and 1s."""
        encoded = encode_target(sample_df.copy())
        assert set(encoded["loan_status"].unique()).issubset({0, 1})

    def test_encode_target_default_is_one(self, sample_df):
        """Charged Off should map to 1 (positive class)."""
        original_default_count = (sample_df["loan_status"] == "Charged Off").sum()
        encoded = encode_target(sample_df.copy())
        assert encoded["loan_status"].sum() == original_default_count

    def test_parse_string_columns_int_rate(self):
        """Interest rate strings should be parsed to floats."""
        df = pd.DataFrame({"int_rate": ["15.99%", "8.50%", "22.00%"]})
        result = parse_string_columns(df)
        assert result["int_rate"].dtype in [float, np.float64]
        assert abs(result["int_rate"].iloc[0] - 15.99) < 0.01

    def test_parse_string_columns_term(self):
        """Term '36 months' should become 36.0."""
        df = pd.DataFrame({"term": ["36 months", "60 months"]})
        result = parse_string_columns(df)
        assert result["term"].iloc[0] == 36.0
        assert result["term"].iloc[1] == 60.0

    def test_cap_outliers_respects_bounds(self):
        """After capping, no value should exceed the 99th percentile."""
        df = pd.DataFrame({
            "annual_inc": [10000, 50000, 100000, 5000000, 60000, 80000, 75000],
            "loan_status": [0, 0, 0, 0, 1, 0, 1],
        })
        result = cap_outliers(df.copy())
        # The extreme value (5M) should be capped
        assert result["annual_inc"].max() < 5000000

    def test_missing_values_preserved_as_nan(self):
        """Preprocessing should handle NaN without crashing."""
        df = pd.DataFrame({
            "annual_inc": [50000.0, None, 70000.0],
            "dti": [15.0, 20.0, None],
            "loan_status": ["Fully Paid", "Charged Off", "Fully Paid"],
        })
        # Should not raise any exception
        try:
            result = encode_target(df.copy())
            assert result is not None
        except Exception as e:
            pytest.fail(f"Preprocessing raised unexpected exception: {e}")


# ============================================================
# TESTS: Feature Engineering
# ============================================================

class TestDebtFeatures:
    """Tests for debt/affordability feature engineering."""

    def test_monthly_income_calculation(self, clean_numeric_df):
        """Monthly income = annual_inc / 12."""
        result = engineer_debt_features(clean_numeric_df.copy())
        expected = clean_numeric_df["annual_inc"] / 12
        pd.testing.assert_series_equal(result["monthly_income"], expected, check_names=False)

    def test_payment_to_income_positive(self, clean_numeric_df):
        """Payment-to-income ratio should always be positive."""
        result = engineer_debt_features(clean_numeric_df.copy())
        assert (result["payment_to_income"] >= 0).all()

    def test_loan_to_income_positive(self, clean_numeric_df):
        """Loan-to-income ratio should always be positive."""
        result = engineer_debt_features(clean_numeric_df.copy())
        assert (result["loan_to_income"] >= 0).all()

    def test_high_dti_leads_to_high_burden(self, clean_numeric_df):
        """Higher DTI should produce higher total_payment_burden."""
        result = engineer_debt_features(clean_numeric_df.copy())
        # Person 2 has higher DTI (35 vs 15) but similar income
        # They should have higher payment burden (assuming similar income)
        person_0_burden = result["total_payment_burden"].iloc[0]
        person_1_burden = result["total_payment_burden"].iloc[1]
        # Person 1 has higher DTI AND higher installment, so higher burden
        assert person_1_burden > person_0_burden


class TestCreditUtilizationFeatures:
    """Tests for credit utilization features."""

    def test_high_utilization_flag_correct(self, clean_numeric_df):
        """Utilization > 70% should set high_utilization_flag = 1."""
        result = engineer_credit_utilization_features(clean_numeric_df.copy())
        # Person 1 has revol_util=75 (>70) → flag should be 1
        assert result["high_utilization_flag"].iloc[1] == 1
        # Person 0 has revol_util=30 (<70) → flag should be 0
        assert result["high_utilization_flag"].iloc[0] == 0

    def test_revol_bal_per_account_positive(self, clean_numeric_df):
        """Revolving balance per account should be non-negative."""
        result = engineer_credit_utilization_features(clean_numeric_df.copy())
        assert (result["revol_bal_per_account"] >= 0).all()


class TestDelinquencyFeatures:
    """Tests for delinquency and payment history features."""

    def test_has_delinquency_flag(self, clean_numeric_df):
        """has_delinquency should be 1 if delinq_2yrs > 0."""
        result = engineer_delinquency_features(clean_numeric_df.copy())
        # Person 1 has 3 delinquencies → should be 1
        assert result["has_delinquency"].iloc[1] == 1
        # Person 0 has 0 delinquencies → should be 0
        assert result["has_delinquency"].iloc[0] == 0

    def test_derogatory_flag_includes_pub_rec(self, clean_numeric_df):
        """derogatory_flag should be 1 if pub_rec > 0."""
        result = engineer_delinquency_features(clean_numeric_df.copy())
        # Person 1 has pub_rec=1 → derogatory_flag should be 1
        assert result["derogatory_flag"].iloc[1] == 1

    def test_delinquency_rate_non_negative(self, clean_numeric_df):
        """Delinquency rate should always be non-negative."""
        result = engineer_delinquency_features(clean_numeric_df.copy())
        assert (result["delinquency_rate"] >= 0).all()


class TestFICOFeatures:
    """Tests for FICO score features."""

    def test_fico_score_is_midpoint(self, clean_numeric_df):
        """fico_score should be average of fico_range_low and fico_range_high."""
        result = engineer_fico_features(clean_numeric_df.copy())
        expected_fico = (clean_numeric_df["fico_range_low"] + clean_numeric_df["fico_range_high"]) / 2
        pd.testing.assert_series_equal(result["fico_score"], expected_fico, check_names=False)

    def test_subprime_flag_correct(self, clean_numeric_df):
        """Subprime flag should be 1 for FICO < 640."""
        result = engineer_fico_features(clean_numeric_df.copy())
        # Person 1 has FICO ≈ 622 (<640) → subprime
        assert result["subprime_flag"].iloc[1] == 1
        # Person 0 has FICO ≈ 722 (>640) → not subprime
        assert result["subprime_flag"].iloc[0] == 0

    def test_prime_flag_correct(self, clean_numeric_df):
        """Prime flag should be 1 for FICO > 720."""
        result = engineer_fico_features(clean_numeric_df.copy())
        # Person 0 has FICO ≈ 722 (>720) → prime
        assert result["prime_flag"].iloc[0] == 1
        # Person 1 has FICO ≈ 622 (<720) → not prime
        assert result["prime_flag"].iloc[1] == 0


class TestAllFeatures:
    """Integration tests for the complete feature engineering pipeline."""

    def test_engineer_all_increases_column_count(self, clean_numeric_df):
        """Feature engineering should add new columns."""
        initial_count = len(clean_numeric_df.columns)
        result = engineer_all_features(clean_numeric_df.copy())
        assert len(result.columns) > initial_count, "No new features were created!"

    def test_no_infinite_values(self, clean_numeric_df):
        """Feature engineering should not produce infinite values."""
        result = engineer_all_features(clean_numeric_df.copy())
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(result[col]).any(), f"Infinite values in column: {col}"

    def test_works_with_synthetic_data(self, sample_df):
        """Feature engineering should work on full synthetic dataset."""
        from src.data.preprocessing import (
            drop_leaky_and_irrelevant_columns,
            encode_target,
            parse_string_columns,
            parse_date_columns,
        )
        df = drop_leaky_and_irrelevant_columns(sample_df.copy())
        df = encode_target(df)
        df = parse_string_columns(df)
        df = parse_date_columns(df)
        cols_before = len(df.columns)  # snapshot BEFORE engineering
        result = engineer_all_features(df)
        assert len(result) == len(sample_df)
        assert len(result.columns) > cols_before


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
