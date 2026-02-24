"""
src/features/pipeline.py — Sklearn Feature Pipeline
=====================================================

LAYMAN EXPLANATION:
    Imagine a factory assembly line with multiple stations:
    Station 1: Wash the raw materials
    Station 2: Cut them to size
    Station 3: Shape them
    Station 4: Polish them

    This file builds exactly that — an assembly line for data.
    Raw loan data goes in one end and perfectly formatted,
    model-ready numbers come out the other end.

    CRITICALLY: The pipeline learns statistics ONLY from training data
    (e.g., "what is the median income?") and then applies those SAME
    statistics to new data. This prevents the model from "peeking" at
    test data statistics.

DATA SCIENTIST EXPLANATION:
    Builds a full sklearn Pipeline that:
    1. Applies feature engineering (from engineer.py)
    2. Separates numeric and categorical columns
    3. Handles missing values (median for numeric, "MISSING" for categorical)
    4. Scales numeric features (StandardScaler)
    5. Encodes categorical features (OrdinalEncoder for tree models)
    6. Combines everything with ColumnTransformer

    Key design choice: OrdinalEncoder (not OneHotEncoder) for tree-based
    models — trees can natively split on ordinal integers, while one-hot
    encoding creates unnecessary dimensionality without benefit for trees.

    The FunctionTransformer wraps engineer_all_features() to make it
    a proper sklearn step that can be saved/loaded with joblib.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from config import cfg
from src.features.engineer import engineer_all_features


# ============================================================
# CUSTOM TRANSFORMER: Feature Engineering Step
# ============================================================

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps the feature engineering functions as a sklearn transformer.

    LAYMAN: sklearn's Pipeline requires each step to follow a standard
    interface (fit/transform). This wrapper lets us plug our custom
    feature engineering functions into the pipeline seamlessly.

    DATA SCIENTIST NOTE:
    Inheriting from BaseEstimator + TransformerMixin gives us:
    - get_params() / set_params() for hyperparameter search
    - fit_transform() for free (combines fit + transform)
    - clone() support for cross-validation
    """

    def __init__(self, target_col: str = "loan_status"):
        self.target_col = target_col
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        """
        Nothing to learn in this step — just store state.

        LAYMAN: The "fit" step normally learns statistics from data.
        But feature engineering (like computing ratios) doesn't need
        to learn anything — it just transforms. So fit() does nothing.
        """
        return self  # Nothing to learn

    def transform(self, X, y=None):
        """
        Apply all feature engineering transformations.

        LAYMAN: This is where the actual work happens.
        Input: Raw DataFrame with original columns
        Output: DataFrame with 40+ new engineered features added
        """
        # We work on a copy to avoid modifying the original data
        X_copy = X.copy()

        # Remove target column if present (shouldn't be in features)
        if self.target_col in X_copy.columns:
            X_copy = X_copy.drop(columns=[self.target_col])

        # Parse string-formatted numeric columns that may arrive raw at inference
        # (e.g. term="36 months"→36, int_rate="12.5%"→12.5, revol_util="45%"→45)
        from src.data.preprocessing import parse_string_columns
        X_copy = parse_string_columns(X_copy)

        # Apply all feature engineering
        X_engineered = engineer_all_features(X_copy)

        # Store feature names for later use (e.g., SHAP explanations)
        self.feature_names_out_ = list(X_engineered.columns)

        return X_engineered

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_) if self.feature_names_out_ else np.array([])


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selects only the columns that the model was trained on.

    LAYMAN: When making predictions on new loan applications, the input
    data might have different columns than the training data. This step
    ensures we only use the columns the model knows about, filling
    in missing ones with zeros.

    DATA SCIENTIST NOTE:
    Critical for production robustness. New API inputs may not have
    all training columns (e.g., a new field was added after training).
    Also handles column ordering — sklearn models are order-sensitive.
    """

    def __init__(self):
        self.columns_ = None

    def fit(self, X, y=None):
        """Remember which columns exist after engineering."""
        if isinstance(X, pd.DataFrame):
            self.columns_ = list(X.columns)
        return self

    def transform(self, X, y=None):
        """Keep only the trained columns, add missing ones as 0."""
        if not isinstance(X, pd.DataFrame):
            return X

        if self.columns_ is None:
            return X

        # Add missing columns with 0
        for col in self.columns_:
            if col not in X.columns:
                X = X.copy()
                X[col] = 0.0

        # Select only the trained columns in the right order
        return X[self.columns_]


# ============================================================
# PIPELINE BUILDER
# ============================================================

def identify_feature_types(df: pd.DataFrame, target_col: str) -> tuple:
    """
    Automatically detect which columns are numeric vs categorical.

    LAYMAN: Numbers (like income, DTI) and categories (like "RENT", "OWN")
    need different processing. This function automatically separates them.

    Returns:
        (numeric_cols, categorical_cols): Two lists of column names
    """
    # Exclude target
    feature_cols = [c for c in df.columns if c != target_col]

    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            numeric_cols.append(col)
        elif df[col].dtype == object or str(df[col].dtype) == 'category':
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)  # Default to numeric

    logger.info(
        f"Detected {len(numeric_cols)} numeric features, "
        f"{len(categorical_cols)} categorical features"
    )
    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(
    numeric_cols: list,
    categorical_cols: list,
) -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer for numeric + categorical features.

    LAYMAN: This creates two parallel processing tracks:
    - Track 1 (Numeric): Fill missing values with median → Scale to 0-1 range
    - Track 2 (Categorical): Fill missing values with "MISSING" → Encode as numbers
    Then combine both tracks into one feature matrix.

    WHY SCALE NUMERICS?
    Without scaling, a feature like annual_income (values: $10k-$500k)
    would dominate a feature like delinq_2yrs (values: 0-5) just because
    of its larger magnitude. Scaling makes all features comparable.

    WHY ENCODE CATEGORICALS?
    ML models work with numbers. "RENT" → 0, "OWN" → 1, "MORTGAGE" → 2, etc.
    OrdinalEncoder works well for tree models.

    DATA SCIENTIST NOTE:
    For neural networks or logistic regression, you'd use StandardScaler
    (not MinMax) and OneHotEncoder (not Ordinal). For XGBoost/LightGBM,
    tree splits are invariant to monotone transformations, so scaling
    only matters for the stacking meta-learner (LogReg).
    """
    # --- Numeric Pipeline ---
    # Step 1: Fill NaN with column median
    # Step 2: Scale to zero mean, unit variance
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # Fill NaN with column median
        ("scaler", StandardScaler()),                    # Normalize to mean=0, std=1
    ])

    # --- Categorical Pipeline ---
    # Step 1: Fill NaN with special "MISSING" category
    # Step 2: Encode each category as an integer
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",  # Unknown categories → -1
            unknown_value=-1,
        )),
    ])

    # --- Combine both pipelines ---
    # ColumnTransformer applies the right pipeline to the right columns
    # "remainder='drop'" drops any columns not explicitly listed
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",   # Drop any unspecified columns
        verbose_feature_names_out=True,  # Keep track of feature names
    )

    return preprocessor


def build_full_pipeline(
    train_df: pd.DataFrame,
) -> tuple:
    """
    Build and fit the complete feature pipeline on training data.

    LAYMAN: This is the master function that:
    1. Applies feature engineering to create new columns
    2. Figures out which columns are numeric vs categorical
    3. Builds the preprocessing pipeline (scale, encode, impute)
    4. FITS the pipeline on training data (learns medians, scales, etc.)

    After this, we can call pipeline.transform() on any new data
    to prepare it for the model.

    Args:
        train_df: Training DataFrame (with target column included)

    Returns:
        (fitted_pipeline, feature_names): The fitted pipeline and
        list of feature names in the output matrix
    """
    target = cfg.data.target_column
    logger.info("Building and fitting feature pipeline...")

    # Step 1: Apply feature engineering
    feature_engineer = FeatureEngineeringTransformer(target_col=target)
    train_engineered = feature_engineer.fit_transform(train_df)

    # Step 2: Identify feature types
    numeric_cols, categorical_cols = identify_feature_types(train_engineered, target)

    # Step 3: Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)

    # Step 4: Fit on training data (learns stats like median, mean, etc.)
    X_train = train_engineered.drop(columns=[target], errors='ignore')
    preprocessor.fit(X_train)

    # Step 5: Get output feature names
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(numeric_cols) + len(categorical_cols))]

    logger.info(f"Pipeline fitted. Output features: {len(feature_names)}")

    return feature_engineer, preprocessor, feature_names


def transform_data(
    df: pd.DataFrame,
    feature_engineer: FeatureEngineeringTransformer,
    preprocessor: ColumnTransformer,
    target: str = None,
) -> tuple:
    """
    Transform a DataFrame using the fitted pipeline.

    LAYMAN: Takes raw loan data and converts it to a numeric matrix
    that the model can read. Uses the same transformations learned
    during training.

    Args:
        df: DataFrame to transform
        feature_engineer: Fitted FeatureEngineeringTransformer
        preprocessor: Fitted ColumnTransformer
        target: Target column name (if present, will be separated)

    Returns:
        (X_array, y_series): Feature matrix and target series (or None)
    """
    if target is None:
        target = cfg.data.target_column

    # Extract target BEFORE feature engineering (which strips the target column)
    y = df[target] if target in df.columns else None

    # Apply feature engineering (removes target column internally)
    df_engineered = feature_engineer.transform(df)

    # Drop target if still present after engineering (safety guard)
    X = df_engineered.drop(columns=[target], errors='ignore')

    # Apply preprocessing (impute, scale, encode)
    X_array = preprocessor.transform(X)

    return X_array, y


def save_pipeline(
    feature_engineer: FeatureEngineeringTransformer,
    preprocessor: ColumnTransformer,
    feature_names: list,
):
    """
    Save all pipeline components to disk.

    LAYMAN: Once the pipeline is trained, we save it to files so
    the API can load it quickly without retraining every time it starts.
    Think of it like saving a Word document — it's there when you reopen it.
    """
    # Ensure the models directory exists
    Path(cfg.model.feature_pipeline_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"feature_engineer": feature_engineer, "preprocessor": preprocessor},
        cfg.model.feature_pipeline_path
    )
    joblib.dump(feature_names, cfg.model.feature_names_path)
    logger.info(f"Pipeline saved to {cfg.model.feature_pipeline_path}")


def load_pipeline() -> tuple:
    """
    Load the saved pipeline components from disk.

    LAYMAN: When the API starts up, it loads the previously trained
    pipeline from disk so it's ready to process loan applications.

    Returns:
        (feature_engineer, preprocessor, feature_names)
    """
    pipeline_data = joblib.load(cfg.model.feature_pipeline_path)
    feature_names = joblib.load(cfg.model.feature_names_path)

    logger.info("Pipeline loaded from disk.")
    return (
        pipeline_data["feature_engineer"],
        pipeline_data["preprocessor"],
        feature_names,
    )
