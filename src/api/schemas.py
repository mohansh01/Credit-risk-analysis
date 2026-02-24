"""
src/api/schemas.py — Pydantic Request/Response Models
=======================================================

LAYMAN EXPLANATION:
    These "schemas" define the exact structure of data that goes IN
    and comes OUT of the API.

    Think of them as forms:
    - The "Application Form" defines what fields an applicant must fill in
    - The "Decision Letter" defines what the bank's response includes

    If someone sends incomplete or wrong data (e.g., sends "abc" for
    annual income instead of a number), Pydantic automatically rejects
    it with a clear error message BEFORE it reaches the model.

    This is your first line of defense against bad data.

DATA SCIENTIST EXPLANATION:
    Pydantic v2 models with:
    - Field-level validation (type checking, range constraints)
    - Optional fields with sensible defaults (None for rarely-available data)
    - Custom validators for business rules (e.g., loan_amnt > 0)
    - Example values in Field() for automatic API documentation
    - Nested response models for structured outputs (SHAP explanations, etc.)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum


# ============================================================
# ENUMS — Allowed values for categorical fields
# ============================================================

class LoanTerm(str, Enum):
    """Allowed loan terms."""
    MONTHS_36 = "36 months"
    MONTHS_60 = "60 months"


class HomeOwnership(str, Enum):
    """Allowed home ownership values."""
    MORTGAGE = "MORTGAGE"
    RENT = "RENT"
    OWN = "OWN"
    OTHER = "OTHER"


class VerificationStatus(str, Enum):
    """Allowed verification status values."""
    NOT_VERIFIED = "Not Verified"
    VERIFIED = "Verified"
    SOURCE_VERIFIED = "Source Verified"


class LoanPurpose(str, Enum):
    """Allowed loan purposes."""
    DEBT_CONSOLIDATION = "debt_consolidation"
    CREDIT_CARD = "credit_card"
    HOME_IMPROVEMENT = "home_improvement"
    OTHER = "other"
    MAJOR_PURCHASE = "major_purchase"
    MEDICAL = "medical"
    SMALL_BUSINESS = "small_business"
    CAR = "car"
    VACATION = "vacation"
    MOVING = "moving"
    HOUSE = "house"
    WEDDING = "wedding"


class DecisionEnum(str, Enum):
    """Loan decision outcomes."""
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class RiskGrade(str, Enum):
    """Risk grade letters."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


# ============================================================
# REQUEST SCHEMA — Loan Application Input
# ============================================================

class LoanApplicationRequest(BaseModel):
    """
    Schema for a single loan application scoring request.

    LAYMAN: This is the "form" that must be filled out to request
    a credit score. Each field is validated automatically.

    Required fields are the ones with no default value.
    Optional fields default to None if not provided.
    """

    # --- Core Required Fields ---
    loan_amnt: float = Field(
        ...,  # ... means required
        gt=0, le=40000,
        description="Requested loan amount in dollars",
        example=15000.0,
    )
    annual_inc: float = Field(
        ...,
        gt=0,
        description="Borrower's annual income in dollars",
        example=65000.0,
    )
    dti: float = Field(
        ...,
        ge=0, le=100,
        description="Debt-to-income ratio (%)",
        example=18.5,
    )
    int_rate: float = Field(
        ...,
        gt=0, le=35,
        description="Interest rate on the loan (%)",
        example=12.99,
    )
    installment: float = Field(
        ...,
        gt=0,
        description="Monthly payment amount in dollars",
        example=350.0,
    )
    fico_range_low: float = Field(
        ...,
        ge=300, le=850,
        description="Lower bound of FICO credit score range",
        example=690.0,
    )
    fico_range_high: float = Field(
        ...,
        ge=300, le=850,
        description="Upper bound of FICO credit score range",
        example=694.0,
    )

    # --- Categorical Fields ---
    term: str = Field(
        default="36 months",
        description="Loan term (36 or 60 months)",
        example="36 months",
    )
    grade: str = Field(
        default="B",
        description="LC-assigned loan grade (A-G)",
        example="B",
    )
    home_ownership: str = Field(
        default="RENT",
        description="Borrower's home ownership status",
        example="RENT",
    )
    purpose: str = Field(
        default="debt_consolidation",
        description="Purpose of the loan",
        example="debt_consolidation",
    )
    verification_status: str = Field(
        default="Not Verified",
        description="Income verification status",
        example="Verified",
    )

    # --- Optional Credit History Fields ---
    delinq_2yrs: Optional[float] = Field(
        default=0.0,
        ge=0,
        description="Number of delinquencies in past 2 years",
        example=0.0,
    )
    open_acc: Optional[float] = Field(
        default=10.0,
        ge=0,
        description="Number of open credit lines",
        example=8.0,
    )
    pub_rec: Optional[float] = Field(
        default=0.0,
        ge=0,
        description="Number of derogatory public records",
        example=0.0,
    )
    revol_bal: Optional[float] = Field(
        default=5000.0,
        ge=0,
        description="Total revolving credit balance",
        example=8500.0,
    )
    revol_util: Optional[float] = Field(
        default=30.0,
        ge=0, le=100,
        description="Revolving line utilization rate (%)",
        example=45.5,
    )
    total_acc: Optional[float] = Field(
        default=20.0,
        ge=0,
        description="Total number of credit lines ever",
        example=18.0,
    )
    inq_last_6mths: Optional[float] = Field(
        default=0.0,
        ge=0,
        description="Number of credit inquiries in last 6 months",
        example=1.0,
    )
    mths_since_last_delinq: Optional[float] = Field(
        default=None,
        description="Months since last delinquency (null if never delinquent)",
        example=None,
    )
    emp_length: Optional[str] = Field(
        default="5 years",
        description="Employment length",
        example="5 years",
    )

    # --- Optional Field for API Tracking ---
    applicant_id: Optional[str] = Field(
        default=None,
        description="Optional applicant ID for tracking",
        example="APP_20240224_001",
    )

    @field_validator('fico_range_high')
    @classmethod
    def fico_high_gt_low(cls, v, info):
        """Ensure FICO high >= FICO low."""
        if 'fico_range_low' in info.data and v < info.data['fico_range_low']:
            raise ValueError("fico_range_high must be >= fico_range_low")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "applicant_id": "APP_20240224_001",
                "loan_amnt": 15000,
                "annual_inc": 65000,
                "dti": 18.5,
                "int_rate": 12.99,
                "installment": 350.0,
                "fico_range_low": 690,
                "fico_range_high": 694,
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
        }


# ============================================================
# RESPONSE SCHEMAS — API Outputs
# ============================================================

class RiskFactor(BaseModel):
    """A single SHAP-based risk factor explanation."""
    feature: str
    label: str
    value: str
    shap_impact: float
    direction: str
    description: str


class ScoringResponse(BaseModel):
    """
    Response schema for a single loan scoring request.

    LAYMAN: This is the "decision letter" from the bank.
    It tells you:
    - The probability of default (0-100%)
    - The risk grade (A/B/C/D/F)
    - The decision (APPROVE/REJECT)
    - The TOP reasons why (for regulatory compliance)
    """
    applicant_id: Optional[str] = Field(None, description="Applicant identifier")
    default_probability: float = Field(
        ...,
        description="Predicted probability of loan default (0-1)",
        example=0.23,
    )
    risk_grade: str = Field(
        ...,
        description="Risk grade assigned based on probability",
        example="B",
    )
    decision: str = Field(
        ...,
        description="Lending decision: APPROVE or REJECT",
        example="APPROVE",
    )
    model_version: str = Field(
        default="1.0.0",
        description="Version of the model that produced this score",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "applicant_id": "APP_20240224_001",
                "default_probability": 0.23,
                "risk_grade": "B",
                "decision": "APPROVE",
                "model_version": "1.0.0",
            }
        }


class ExplainedScoringResponse(ScoringResponse):
    """
    Extended response with SHAP explanations.

    LAYMAN: Like ScoringResponse but with the full explanation
    of WHY the model made this decision.
    """
    top_risk_factors: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Top features that INCREASED the default risk",
    )
    top_protective_factors: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Top features that DECREASED the default risk",
    )
    shap_available: bool = Field(
        default=False,
        description="Whether SHAP explanations are available",
    )


class BatchScoringRequest(BaseModel):
    """
    Request schema for batch scoring multiple applicants.

    LAYMAN: Instead of scoring one applicant at a time, send
    a list of applicants to score all at once.
    """
    applications: List[LoanApplicationRequest] = Field(
        ...,
        description="List of loan applications to score",
        min_length=1,
        max_length=10000,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "applications": [
                    {
                        "applicant_id": "APP_001",
                        "loan_amnt": 15000,
                        "annual_inc": 65000,
                        "dti": 18.5,
                        "int_rate": 12.99,
                        "installment": 350.0,
                        "fico_range_low": 690,
                        "fico_range_high": 694,
                    },
                    {
                        "applicant_id": "APP_002",
                        "loan_amnt": 25000,
                        "annual_inc": 40000,
                        "dti": 35.0,
                        "int_rate": 22.5,
                        "installment": 850.0,
                        "fico_range_low": 620,
                        "fico_range_high": 624,
                    }
                ]
            }
        }


class BatchScoringResponse(BaseModel):
    """Response schema for batch scoring."""
    total_applications: int
    approved: int
    rejected: int
    approval_rate: float
    results: List[ScoringResponse]


class HealthResponse(BaseModel):
    """API health check response."""
    status: str = "healthy"
    model_loaded: bool
    model_version: str
    ensemble_models: List[str]
    rejection_threshold: float
    api_version: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    status_code: int
