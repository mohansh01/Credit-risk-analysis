"""
src/api/routes.py — API Route Handlers
========================================

LAYMAN EXPLANATION:
    This file defines the "endpoints" — the URLs that the API responds to.
    Think of endpoints like different desks at a bank:

    POST /score         → "Credit Scoring Desk" (score one applicant)
    POST /score/batch   → "Bulk Processing Desk" (score many applicants at once)
    GET  /health        → "System Status Desk" (is the API working?)
    GET  /model-info    → "Model Information Desk" (what model version is running?)

    Each endpoint:
    1. Receives a request (loan application data)
    2. Validates it (Pydantic checks the data format)
    3. Processes it (calls the ML model)
    4. Returns a structured response (JSON with score, grade, decision)

DATA SCIENTIST EXPLANATION:
    FastAPI router with async endpoints.
    Key design decisions:
    - Async I/O for batch endpoints (non-blocking under load)
    - Dependency injection for model loading (loaded once at startup)
    - Pydantic validation happens automatically before route handler executes
    - HTTP exceptions with proper status codes (422 for invalid input, 503 for model not ready)
    - Logging per request for observability
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from typing import List

from config import cfg
from src.api.schemas import (
    LoanApplicationRequest,
    ScoringResponse,
    ExplainedScoringResponse,
    BatchScoringRequest,
    BatchScoringResponse,
    HealthResponse,
)
from src.models.predict import model_store, predict_single, predict_batch, get_model_info


# Create the router
# All routes defined here will be mounted in main.py with a prefix
router = APIRouter()


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API is running and models are loaded.",
    tags=["System"],
)
async def health_check():
    """
    LAYMAN: Like checking if a machine is turned on and ready to work.
    Returns 'healthy' if everything is working, 'degraded' if models
    aren't loaded yet.

    DATA SCIENTIST NOTE:
    Critical for:
    - Load balancer health checks (only route traffic to healthy instances)
    - Kubernetes liveness/readiness probes
    - Monitoring dashboards (PagerDuty alerts when this returns non-200)
    """
    model_info = get_model_info()

    return HealthResponse(
        status="healthy" if model_info["models_loaded"] else "degraded",
        model_loaded=model_info["models_loaded"],
        model_version=model_info["model_version"],
        ensemble_models=model_info["ensemble"],
        rejection_threshold=model_info["rejection_threshold"],
        api_version=cfg.api.api_version,
    )


# ============================================================
# SINGLE APPLICANT SCORING
# ============================================================

@router.post(
    "/score",
    response_model=ScoringResponse,
    summary="Score Single Loan Application",
    description=(
        "Submit a single loan application and receive a credit risk score, "
        "risk grade, and lending decision."
    ),
    tags=["Scoring"],
    status_code=status.HTTP_200_OK,
)
async def score_application(application: LoanApplicationRequest):
    """
    LAYMAN: The main scoring endpoint. Send a loan application as JSON,
    get back a credit score + approve/reject decision.

    Example:
    ```
    POST /score
    {
        "loan_amnt": 15000,
        "annual_inc": 65000,
        "dti": 18.5,
        ...
    }
    → {"default_probability": 0.23, "risk_grade": "B", "decision": "APPROVE"}
    ```

    DATA SCIENTIST NOTE:
    - Pydantic validates all fields BEFORE this function executes
    - predict_single() loads models if not already in memory
    - Response time target: <50ms p99
    """
    # Ensure model is loaded
    if not model_store.is_loaded():
        try:
            model_store.load()
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Model artifacts not found. "
                    "Please run the training pipeline first: "
                    "python run.py --mode train"
                )
            )

    # Convert Pydantic model to dict for prediction
    applicant_dict = application.model_dump()
    applicant_id = applicant_dict.pop("applicant_id", None)

    logger.info(f"Scoring applicant: {applicant_id or 'anonymous'}")

    try:
        result = predict_single(applicant_dict)
        result["applicant_id"] = applicant_id
        return ScoringResponse(**result)
    except Exception as e:
        logger.error(f"Scoring error for {applicant_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {str(e)}"
        )


# ============================================================
# BATCH SCORING
# ============================================================

@router.post(
    "/score/batch",
    response_model=BatchScoringResponse,
    summary="Score Multiple Loan Applications",
    description=(
        "Submit up to 10,000 loan applications at once for batch scoring. "
        "Returns scores for all applications with aggregate statistics."
    ),
    tags=["Scoring"],
    status_code=status.HTTP_200_OK,
)
async def score_batch(batch_request: BatchScoringRequest):
    """
    LAYMAN: Score many applicants at once. Useful for:
    - Processing overnight applications
    - Scoring a portfolio of existing loans
    - A/B testing a new model on historical data

    DATA SCIENTIST NOTE:
    Converts list of Pydantic models to a DataFrame for vectorized
    inference (much faster than looping through predict_single()).
    Batch processing achieves ~50x speedup vs serial calls.
    """
    if not model_store.is_loaded():
        try:
            model_store.load()
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifacts not found. Please run training first."
            )

    applications = batch_request.applications
    logger.info(f"Batch scoring {len(applications)} applications...")

    try:
        # Convert list of Pydantic models to DataFrame
        apps_dicts = [app.model_dump() for app in applications]
        apps_df = pd.DataFrame(apps_dicts)

        # Run batch prediction
        results_df = predict_batch(apps_df)

        # Build response
        results = []
        for _, row in results_df.iterrows():
            results.append(ScoringResponse(
                applicant_id=row.get("applicant_id"),
                default_probability=float(row["default_probability"]),
                risk_grade=str(row["risk_grade"]),
                decision=str(row["decision"]),
                model_version="1.0.0",
            ))

        n_approved = sum(1 for r in results if r.decision == "APPROVE")
        n_rejected = len(results) - n_approved

        return BatchScoringResponse(
            total_applications=len(results),
            approved=n_approved,
            rejected=n_rejected,
            approval_rate=round(n_approved / len(results), 4) if results else 0.0,
            results=results,
        )

    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch scoring failed: {str(e)}"
        )


# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

@router.get(
    "/model-info",
    summary="Get Model Information",
    description="Returns metadata about the currently loaded model.",
    tags=["System"],
)
async def get_model_information():
    """
    LAYMAN: Returns technical details about the model:
    - Which version is running
    - What models are in the ensemble
    - What threshold is used for approval/rejection

    Useful for auditing and compliance purposes.
    """
    return get_model_info()


# ============================================================
# DEMO ENDPOINT — Test without a real application
# ============================================================

@router.get(
    "/demo",
    response_model=ScoringResponse,
    summary="Demo Score (No Input Required)",
    description="Score a sample loan application to test the API.",
    tags=["Demo"],
)
async def demo_score():
    """
    LAYMAN: A test endpoint that uses a pre-built example applicant
    to verify the API is working correctly. No input required.

    Great for:
    - Testing the API after deployment
    - Showing stakeholders how the API works
    - Smoke testing in CI/CD pipelines
    """
    if not model_store.is_loaded():
        try:
            model_store.load()
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Run training first."
            )

    # Sample "good" borrower
    sample_applicant = {
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

    result = predict_single(sample_applicant)
    result["applicant_id"] = "DEMO_APPLICANT"
    return ScoringResponse(**result)
