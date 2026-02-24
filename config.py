"""
config.py — Central Configuration File
=======================================

LAYMAN EXPLANATION:
    Think of this file as the "settings panel" of the entire project.
    Instead of hunting through 20 different files to change a threshold
    or file path, everything is defined HERE in one place.

    If a bank wants to change the rejection threshold from 50% to 40%
    probability of default, they change ONE number here — not 10 files.

DATA SCIENTIST EXPLANATION:
    This module uses Python dataclasses to create typed, structured
    configuration objects. All thresholds, file paths, model hyperparameters,
    and business logic parameters are centralized here.

    Benefits:
    - Single source of truth for all config
    - Type hints prevent misconfiguration
    - Easy to override per environment (dev/staging/prod)
    - Can be loaded from environment variables for deployment
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
# .env file stores secrets like API keys that shouldn't be in code
load_dotenv()

# ============================================================
# BASE PATHS
# ============================================================
# __file__ = this config.py file
# .parent = the folder containing config.py (project root)
BASE_DIR = Path(__file__).parent

DATA_DIR       = BASE_DIR / "data"
RAW_DATA_DIR   = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
SAMPLE_DIR     = DATA_DIR / "sample_applications"
MODELS_DIR     = BASE_DIR / "models" / "artifacts"
REPORTS_DIR    = BASE_DIR / "reports"
NOTEBOOKS_DIR  = BASE_DIR / "notebooks"


# ============================================================
# DATA CONFIGURATION
# ============================================================
@dataclass
class DataConfig:
    """
    Controls how data is loaded and split.

    LAYMAN: These settings tell the project WHERE to find the data
    and how to divide it into training and testing groups.
    """
    # The LendingClub dataset filename (download instructions in README)
    raw_filename: str = "lending_club_loans.csv"

    # Target column — what we're trying to predict
    # "loan_status" contains values like "Fully Paid" or "Charged Off" (defaulted)
    target_column: str = "loan_status"

    # What "default" looks like in the dataset
    # "Charged Off" = bank wrote off the loan as uncollectable
    default_label: str = "Charged Off"
    paid_label: str = "Fully Paid"

    # Train/Validation/Test split ratios
    # 70% of data trains the model, 15% tunes it, 15% tests it
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Random seed ensures reproducibility — running the same code
    # twice gives the same results (important for debugging)
    random_state: int = 42

    # Columns to DROP — these are either useless or cause data leakage
    # "Data leakage" = the model accidentally cheats by using info
    # it wouldn't have at prediction time (e.g., total payment received
    # is only known AFTER the loan is repaid, not at application time)
    columns_to_drop: list = field(default_factory=lambda: [
        "id", "member_id", "url", "desc", "title",
        "zip_code", "out_prncp", "out_prncp_inv",
        "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
        "total_rec_int", "total_rec_late_fee", "recoveries",
        "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
        "next_pymnt_d", "last_credit_pull_d",
    ])

    # Demographic columns used ONLY for fairness analysis
    # NOT fed into the model (that would be discriminatory!)
    demographic_columns: list = field(default_factory=lambda: [
        "addr_state"  # We use state as a proxy for geographic bias
    ])


# ============================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================
@dataclass
class FeatureConfig:
    """
    Controls how raw features are transformed into model-ready features.

    LAYMAN: Raw data like "annual income = $75,000" isn't useful alone.
    This config tells the system HOW to transform raw numbers into
    meaningful signals (e.g., debt-to-income ratio).
    """
    # Columns that are numbers (we'll scale these to 0-1 range)
    numeric_features: list = field(default_factory=lambda: [
        "loan_amnt", "funded_amnt", "funded_amnt_inv", "int_rate",
        "installment", "annual_inc", "dti", "delinq_2yrs",
        "fico_range_low", "fico_range_high", "inq_last_6mths",
        "mths_since_last_delinq", "mths_since_last_record",
        "open_acc", "pub_rec", "revol_bal", "revol_util",
        "total_acc", "collections_12_mths_ex_med",
        "mths_since_last_major_derog", "acc_now_delinq",
        "tot_coll_amt", "tot_cur_bal", "open_acc_6m",
        "open_il_6m", "open_il_12m", "open_il_24m",
        "mths_since_rcnt_il", "total_bal_il", "il_util",
        "open_rv_12m", "open_rv_24m", "max_bal_bc",
        "all_util", "total_rev_hi_lim", "inq_fi",
        "total_cu_tl", "inq_last_12m",
    ])

    # Columns that are categories (text labels like "MORTGAGE", "RENT")
    # We'll encode these as numbers the model can understand
    categorical_features: list = field(default_factory=lambda: [
        "term", "grade", "sub_grade", "emp_length",
        "home_ownership", "verification_status", "purpose",
        "initial_list_status", "application_type",
    ])

    # Date columns — we'll extract useful info like "how old is this account?"
    date_features: list = field(default_factory=lambda: [
        "issue_d", "earliest_cr_line",
    ])

    # Maximum fraction of missing values a column is allowed to have
    # Columns with more missing data than this threshold get dropped
    missing_threshold: float = 0.40  # Drop columns missing >40% of values

    # Maximum number of unique categories in a categorical column
    # Columns with too many unique values (like zip code) become noise
    max_cardinality: int = 50


# ============================================================
# MODEL CONFIGURATION
# ============================================================
@dataclass
class ModelConfig:
    """
    Hyperparameters for XGBoost, LightGBM, and the meta-learner.

    LAYMAN: These are the "knobs" you can turn to make each model
    learn better. Finding the best values is called "hyperparameter tuning."

    DATA SCIENTIST NOTE: These are reasonable defaults optimized for
    credit risk. They should be tuned via Optuna/GridSearchCV on your
    specific dataset for production use.
    """
    # --- XGBoost Parameters ---
    # n_estimators: Number of trees to build (more = better but slower)
    # max_depth: How deep each tree grows (deeper = learns more complex patterns but risks overfitting)
    # learning_rate: How much each tree corrects the previous one (smaller = more careful learning)
    # subsample: Fraction of training data used per tree (prevents overfitting)
    # colsample_bytree: Fraction of features used per tree (prevents overfitting)
    # scale_pos_weight: Compensates for class imbalance (defaults are rare)
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 10,  # ratio of negatives/positives (approx for 10% default rate)
        "use_label_encoder": False,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,  # Use all CPU cores
        "verbosity": 0,
    })

    # --- LightGBM Parameters ---
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": -1,         # -1 = no limit (LightGBM's leaf-wise handles this)
        "num_leaves": 63,        # Controls model complexity (2^max_depth - 1 is a good rule)
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50, # Minimum samples per leaf (prevents overfitting)
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "class_weight": "balanced",  # Auto-handles class imbalance
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,  # Suppress LightGBM's verbose output
    })

    # --- Stacking Meta-Learner (Logistic Regression) ---
    # The meta-learner takes XGBoost's prediction + LightGBM's prediction
    # as inputs and learns the optimal combination
    meta_params: dict = field(default_factory=lambda: {
        "C": 1.0,                # Regularization strength (lower = more regularized)
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": "balanced",
    })

    # --- Model Persistence ---
    xgb_model_path: str = str(MODELS_DIR / "xgb_model.pkl")
    lgbm_model_path: str = str(MODELS_DIR / "lgbm_model.pkl")
    meta_model_path: str = str(MODELS_DIR / "meta_model.pkl")
    feature_pipeline_path: str = str(MODELS_DIR / "feature_pipeline.pkl")
    feature_names_path: str = str(MODELS_DIR / "feature_names.pkl")


# ============================================================
# SCORING / BUSINESS LOGIC CONFIGURATION
# ============================================================
@dataclass
class ScoringConfig:
    """
    Business rules for converting model probability to a decision.

    LAYMAN: The model outputs a number between 0 and 1 (probability of default).
    These thresholds convert that number into a human-readable grade and decision.

    Example: probability = 0.23 → Grade "B" → APPROVE
             probability = 0.65 → Grade "D" → REJECT
    """
    # Probability threshold above which we REJECT the loan
    # Set to 0.65 for synthetic data (model scores run higher than real data).
    # On real LendingClub data, lower this back to 0.45-0.50.
    rejection_threshold: float = 0.50

    # Grade boundaries — maps probability ranges to letter grades
    # Grade A: 0-15% default probability (lowest risk, best borrowers)
    # Grade B: 15-25%
    # Grade C: 25-40%
    # Grade D: 40-55%
    # Grade F: 55%+ (highest risk, worst borrowers)
    grade_thresholds: dict = field(default_factory=lambda: {
        "A": 0.15,
        "B": 0.25,
        "C": 0.40,
        "D": 0.55,
        "F": 1.00,
    })

    # Number of top features to show in rejection explanation
    # (Regulatory requirement: must explain WHY loan was rejected)
    top_features_to_explain: int = 5


# ============================================================
# FAIRNESS CONFIGURATION
# ============================================================
@dataclass
class FairnessConfig:
    """
    Thresholds for detecting discriminatory model behavior.

    LAYMAN: If the model approves 90% of men but only 60% of women
    with identical financial profiles, that's discriminatory.
    These settings define when the gap is "too large."

    DATA SCIENTIST NOTE: The 4/5ths rule (0.80 threshold) is the
    EEOC (Equal Employment Opportunity Commission) standard for
    adverse impact. Also used in lending under ECOA.
    """
    # Disparate Impact Ratio threshold
    # Approval rate of protected group / approval rate of majority group
    # Must be >= 0.80 (the "4/5ths rule" from EEOC guidelines)
    disparate_impact_threshold: float = 0.80

    # Protected attribute columns to analyze for bias
    # In a real bank, you'd include: gender, race, age, national_origin
    # We use loan grade as a proxy (correlated with demographics)
    protected_attributes: list = field(default_factory=lambda: [
        "grade",  # Loan grade (A-G) often correlates with demographics
    ])


# ============================================================
# MONITORING / DRIFT DETECTION CONFIGURATION
# ============================================================
@dataclass
class MonitoringConfig:
    """
    Thresholds for detecting when the model needs retraining.

    LAYMAN: Imagine training a spam filter in 2020. By 2023, spammers
    use completely different words, so the filter stops working.
    "Drift" is when the real-world data changes so much that the
    model's training data is no longer representative.

    PSI = Population Stability Index — the industry standard metric
          for measuring how much a distribution has changed over time.
    """
    # PSI (Population Stability Index) thresholds:
    # < 0.10: No significant change — model is stable
    # 0.10 - 0.25: Moderate change — investigate
    # > 0.25: Significant change — RETRAIN THE MODEL
    psi_threshold_warning: float = 0.10
    psi_threshold_critical: float = 0.25

    # KS (Kolmogorov-Smirnov) statistic threshold
    # Measures max distance between two cumulative distributions
    ks_threshold: float = 0.10

    # How many recent predictions to use for drift calculation
    monitoring_window_size: int = 1000

    # Directory for monitoring reports
    reports_dir: str = str(REPORTS_DIR)


# ============================================================
# API CONFIGURATION
# ============================================================
@dataclass
class APIConfig:
    """
    FastAPI server settings.

    LAYMAN: These settings control the web server that other applications
    can connect to in order to get credit scores.
    """
    host: str = "0.0.0.0"   # Listen on all network interfaces
    port: int = 8000          # Port number (like an address for network traffic)
    debug: bool = False        # Enable detailed error messages (dev only!)
    api_title: str = "Credit Risk Scoring Engine"
    api_version: str = "1.0.0"
    api_description: str = (
        "ML-powered loan default prediction API. "
        "Provides real-time credit scoring with SHAP explainability "
        "and regulatory compliance features."
    )


# ============================================================
# MASTER CONFIG — Combines all sub-configs
# ============================================================
@dataclass
class Config:
    """
    The single config object used throughout the entire project.

    Usage:
        from config import Config
        cfg = Config()
        print(cfg.model.xgb_params)
        print(cfg.scoring.rejection_threshold)
    """
    data: DataConfig           = field(default_factory=DataConfig)
    features: FeatureConfig    = field(default_factory=FeatureConfig)
    model: ModelConfig         = field(default_factory=ModelConfig)
    scoring: ScoringConfig     = field(default_factory=ScoringConfig)
    fairness: FairnessConfig   = field(default_factory=FairnessConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    api: APIConfig             = field(default_factory=APIConfig)


# Create a global config instance — import this everywhere
cfg = Config()
