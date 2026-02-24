"""
tests/test_api.py — API Integration Tests
==========================================

LAYMAN EXPLANATION:
    These tests verify that the API endpoints work correctly.
    They simulate real HTTP requests to the API and check that:
    - Valid loan applications get a proper score response
    - Invalid data is rejected with a clear error message
    - The health endpoint reports correctly
    - Batch scoring handles multiple applicants

    Think of these as "end-to-end tests" — testing the ENTIRE system
    from HTTP request all the way to model prediction and back.

RUN TESTS:
    pytest tests/test_api.py -v
    (Note: Models don't need to be trained — the API handles this gracefully)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

# Create test client
# TestClient from FastAPI/Starlette lets us test without starting a real server
client = TestClient(app)


# ============================================================
# SAMPLE TEST DATA
# ============================================================

VALID_APPLICATION = {
    "applicant_id": "TEST_001",
    "loan_amnt": 15000.0,
    "annual_inc": 65000.0,
    "dti": 18.5,
    "int_rate": 12.99,
    "installment": 350.0,
    "fico_range_low": 690.0,
    "fico_range_high": 694.0,
    "term": "36 months",
    "grade": "B",
    "home_ownership": "RENT",
    "purpose": "debt_consolidation",
    "verification_status": "Verified",
    "delinq_2yrs": 0.0,
    "open_acc": 8.0,
    "pub_rec": 0.0,
    "revol_bal": 8500.0,
    "revol_util": 45.5,
    "total_acc": 18.0,
    "inq_last_6mths": 1.0,
}

HIGH_RISK_APPLICATION = {
    "loan_amnt": 30000.0,
    "annual_inc": 30000.0,
    "dti": 45.0,
    "int_rate": 28.0,
    "installment": 1200.0,
    "fico_range_low": 600.0,
    "fico_range_high": 604.0,
    "term": "60 months",
    "grade": "F",
    "home_ownership": "RENT",
    "purpose": "small_business",
    "verification_status": "Not Verified",
    "delinq_2yrs": 5.0,
    "open_acc": 3.0,
    "pub_rec": 2.0,
    "revol_bal": 25000.0,
    "revol_util": 92.0,
    "total_acc": 6.0,
    "inq_last_6mths": 6.0,
}


# ============================================================
# TESTS: Health & Root Endpoints
# ============================================================

class TestHealthEndpoints:
    """Tests for system health and status endpoints."""

    def test_root_returns_200(self):
        """Root endpoint should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_api_info(self):
        """Root should contain API name and documentation link."""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "documentation" in data

    def test_health_endpoint_returns_200(self):
        """Health endpoint should always return 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_has_status(self):
        """Health response should have a 'status' field."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]

    def test_health_response_has_model_info(self):
        """Health response should include model information."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "model_loaded" in data
        assert "model_version" in data
        assert "rejection_threshold" in data


# ============================================================
# TESTS: Input Validation
# ============================================================

class TestInputValidation:
    """Tests for request validation (Pydantic schema enforcement)."""

    def test_missing_required_field_returns_422(self):
        """Missing required field (loan_amnt) should return 422 Unprocessable Entity."""
        incomplete = VALID_APPLICATION.copy()
        del incomplete["loan_amnt"]  # Remove required field
        response = client.post("/api/v1/score", json=incomplete)
        assert response.status_code == 422

    def test_negative_loan_amount_rejected(self):
        """Negative loan amount should be rejected."""
        invalid = VALID_APPLICATION.copy()
        invalid["loan_amnt"] = -1000.0
        response = client.post("/api/v1/score", json=invalid)
        assert response.status_code == 422

    def test_loan_amount_over_limit_rejected(self):
        """Loan amount > $40,000 should be rejected."""
        invalid = VALID_APPLICATION.copy()
        invalid["loan_amnt"] = 100000.0
        response = client.post("/api/v1/score", json=invalid)
        assert response.status_code == 422

    def test_dti_over_100_rejected(self):
        """DTI > 100 is impossible and should be rejected."""
        invalid = VALID_APPLICATION.copy()
        invalid["dti"] = 150.0
        response = client.post("/api/v1/score", json=invalid)
        assert response.status_code == 422

    def test_fico_below_300_rejected(self):
        """FICO score below 300 is invalid."""
        invalid = VALID_APPLICATION.copy()
        invalid["fico_range_low"] = 200.0
        invalid["fico_range_high"] = 204.0
        response = client.post("/api/v1/score", json=invalid)
        assert response.status_code == 422

    def test_string_for_numeric_rejected(self):
        """String value for numeric field should be rejected."""
        invalid = VALID_APPLICATION.copy()
        invalid["annual_inc"] = "sixty thousand"
        response = client.post("/api/v1/score", json=invalid)
        assert response.status_code == 422

    def test_valid_application_accepted(self):
        """Valid application should return 200 or 503 (if model not trained yet)."""
        response = client.post("/api/v1/score", json=VALID_APPLICATION)
        # 200 = scored successfully, 503 = models not trained yet (both acceptable)
        assert response.status_code in [200, 503]


# ============================================================
# TESTS: Score Response Structure (when model is loaded)
# ============================================================

class TestScoringResponseStructure:
    """Tests for the structure of scoring responses."""

    def test_score_response_has_required_fields(self):
        """Scoring response must include all required fields."""
        response = client.post("/api/v1/score", json=VALID_APPLICATION)
        if response.status_code == 503:
            pytest.skip("Model not trained yet — skip response structure test")
        if response.status_code == 200:
            data = response.json()
            assert "default_probability" in data
            assert "risk_grade" in data
            assert "decision" in data
            assert "model_version" in data

    def test_probability_in_range(self):
        """Default probability should be between 0 and 1."""
        response = client.post("/api/v1/score", json=VALID_APPLICATION)
        if response.status_code == 200:
            data = response.json()
            prob = data["default_probability"]
            assert 0.0 <= prob <= 1.0, f"Invalid probability: {prob}"

    def test_risk_grade_is_valid(self):
        """Risk grade should be one of A, B, C, D, F."""
        response = client.post("/api/v1/score", json=VALID_APPLICATION)
        if response.status_code == 200:
            data = response.json()
            assert data["risk_grade"] in ["A", "B", "C", "D", "F"]

    def test_decision_is_valid(self):
        """Decision should be APPROVE or REJECT."""
        response = client.post("/api/v1/score", json=VALID_APPLICATION)
        if response.status_code == 200:
            data = response.json()
            assert data["decision"] in ["APPROVE", "REJECT"]

    def test_high_risk_more_likely_rejected(self):
        """High risk applicant should have higher default probability."""
        response_good  = client.post("/api/v1/score", json=VALID_APPLICATION)
        response_risky = client.post("/api/v1/score", json=HIGH_RISK_APPLICATION)

        if response_good.status_code == 200 and response_risky.status_code == 200:
            prob_good  = response_good.json()["default_probability"]
            prob_risky = response_risky.json()["default_probability"]
            assert prob_risky > prob_good, (
                f"High risk applicant ({prob_risky:.2f}) should have higher "
                f"probability than good applicant ({prob_good:.2f})"
            )


# ============================================================
# TESTS: Batch Scoring
# ============================================================

class TestBatchScoring:
    """Tests for batch scoring endpoint."""

    def test_batch_empty_list_rejected(self):
        """Empty list should be rejected (min_length=1)."""
        response = client.post("/api/v1/score/batch", json={"applications": []})
        assert response.status_code == 422

    def test_batch_valid_request_accepted(self):
        """Valid batch request should return 200 or 503."""
        batch_request = {
            "applications": [
                VALID_APPLICATION,
                HIGH_RISK_APPLICATION,
            ]
        }
        response = client.post("/api/v1/score/batch", json=batch_request)
        assert response.status_code in [200, 503]

    def test_batch_response_structure(self):
        """Batch response should have correct structure."""
        batch_request = {"applications": [VALID_APPLICATION, HIGH_RISK_APPLICATION]}
        response = client.post("/api/v1/score/batch", json=batch_request)
        if response.status_code == 200:
            data = response.json()
            assert "total_applications" in data
            assert "approved" in data
            assert "rejected" in data
            assert "approval_rate" in data
            assert "results" in data
            assert data["total_applications"] == 2
            assert data["approved"] + data["rejected"] == data["total_applications"]

    def test_batch_approval_rate_in_range(self):
        """Approval rate should be between 0 and 1."""
        batch_request = {"applications": [VALID_APPLICATION]}
        response = client.post("/api/v1/score/batch", json=batch_request)
        if response.status_code == 200:
            data = response.json()
            assert 0.0 <= data["approval_rate"] <= 1.0


# ============================================================
# TESTS: Model Info & Demo Endpoints
# ============================================================

class TestUtilityEndpoints:
    """Tests for utility endpoints."""

    def test_model_info_returns_200(self):
        """Model info endpoint should return 200."""
        response = client.get("/api/v1/model-info")
        assert response.status_code == 200

    def test_model_info_has_version(self):
        """Model info should include version and ensemble info."""
        response = client.get("/api/v1/model-info")
        data = response.json()
        assert "model_version" in data

    def test_demo_endpoint_returns_200_or_503(self):
        """Demo endpoint should work (200) or indicate model not loaded (503)."""
        response = client.get("/api/v1/demo")
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
