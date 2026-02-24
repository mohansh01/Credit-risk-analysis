"""
run.py — Master Entry Point
============================

LAYMAN EXPLANATION:
    This is the ONE file you need to run everything in the project.
    Think of it as the remote control for the entire system.

    Commands:
    python run.py --mode train    → Train the model (takes 5-10 minutes)
    python run.py --mode serve    → Start the API server
    python run.py --mode evaluate → Evaluate model performance
    python run.py --mode monitor  → Run drift detection report
    python run.py --mode demo     → Run a quick end-to-end demo

    You don't need to understand the internals to use it.
    Just pick a mode and run.

DATA SCIENTIST EXPLANATION:
    CLI entry point using argparse. Orchestrates the full ML lifecycle:
    - train: Full pipeline from raw data → trained model → evaluation report
    - serve: Start FastAPI server with uvicorn
    - evaluate: Load trained model, run evaluation on test set
    - monitor: Run drift detection on simulated current vs reference data
    - demo: Quick smoke test to verify the entire system works
"""

import argparse
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

from config import cfg, MODELS_DIR, REPORTS_DIR


def run_training():
    """
    Run the full training pipeline end-to-end.

    LAYMAN: This is the "study" phase. The model reads all the loan data,
    learns patterns, and saves what it learned to disk.

    Steps:
    1. Load data (real or synthetic)
    2. Validate & preprocess
    3. Engineer features (create new useful columns)
    4. Build feature pipeline (fit scalers, encoders)
    5. Train XGBoost + LightGBM ensemble
    6. Evaluate on test set
    7. Save all models and pipeline

    Duration: ~5-15 minutes depending on dataset size
    """
    logger.info("=" * 70)
    logger.info("CREDIT RISK ENGINE — TRAINING PIPELINE")
    logger.info("=" * 70)

    # Ensure model artifacts directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load Data ---
    logger.info("\n[STEP 1/6] Loading data...")
    from src.data.ingestion import load_raw_data, validate_data
    raw_df = load_raw_data()
    raw_df = validate_data(raw_df)

    # --- Step 2: Preprocessing ---
    logger.info("\n[STEP 2/6] Preprocessing data...")
    from src.data.preprocessing import run_preprocessing_pipeline
    train_df, val_df, test_df = run_preprocessing_pipeline(raw_df)

    # --- Step 3: Feature Pipeline ---
    logger.info("\n[STEP 3/6] Building feature pipeline...")
    from src.features.pipeline import build_full_pipeline, transform_data, save_pipeline
    feature_engineer, preprocessor, feature_names = build_full_pipeline(train_df)

    # Transform all splits using the fitted pipeline
    target = cfg.data.target_column
    X_train, y_train = transform_data(train_df, feature_engineer, preprocessor, target)
    X_val, y_val     = transform_data(val_df, feature_engineer, preprocessor, target)
    X_test, y_test   = transform_data(test_df, feature_engineer, preprocessor, target)

    logger.info(
        f"  Feature matrix shapes: "
        f"Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
    )

    # Save the pipeline for use at inference time
    save_pipeline(feature_engineer, preprocessor, feature_names)

    # --- Step 4: Train Models ---
    logger.info("\n[STEP 4/6] Training ensemble models...")
    from src.models.train import run_training_pipeline
    xgb_model, lgbm_model, meta_model = run_training_pipeline(
        X_train, y_train,
        X_val, y_val,
        use_mlflow=False,  # Set True if MLflow server is running
    )

    # --- Step 5: Evaluate on Test Set ---
    logger.info("\n[STEP 5/6] Evaluating on held-out test set...")
    from src.models.evaluate import generate_evaluation_report

    # Get final ensemble predictions on test set
    xgb_prob  = xgb_model.predict_proba(X_test)[:, 1]
    lgbm_prob = lgbm_model.predict_proba(X_test)[:, 1]

    import joblib, numpy as np
    stacked   = np.column_stack([xgb_prob, lgbm_prob])
    final_prob = meta_model.predict_proba(stacked)[:, 1]

    metrics = generate_evaluation_report(
        y_test.values if hasattr(y_test, 'values') else np.array(y_test),
        final_prob,
        feature_names=feature_names,
        xgb_model=xgb_model,
        lgbm_model=lgbm_model,
        save_dir=str(REPORTS_DIR),
    )

    # --- Step 6: Fairness Analysis ---
    logger.info("\n[STEP 6/6] Running fairness analysis...")
    from src.fairness.bias_detector import run_fairness_analysis
    if cfg.fairness.protected_attributes[0] in test_df.columns:
        fairness_report = run_fairness_analysis(
            y_test.values if hasattr(y_test, 'values') else np.array(y_test),
            final_prob,
            test_df,
            save_dir=str(REPORTS_DIR),
        )

    # --- Summary ---
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"  AUC-ROC:          {metrics['auc_roc']:.4f}")
    logger.info(f"  KS Statistic:     {metrics['ks_statistic']:.4f}")
    logger.info(f"  Gini Coefficient: {metrics['gini_coefficient']:.4f}")
    logger.info(f"  Models saved to:  {MODELS_DIR}")
    logger.info(f"  Reports saved to: {REPORTS_DIR}")
    logger.info("=" * 70)
    logger.info("\nNext step: python run.py --mode serve")


def run_serve():
    """
    Start the FastAPI server.

    LAYMAN: Starts the web server that listens for loan scoring requests.
    Once running, visit http://localhost:8000/docs to see the API documentation.

    Press Ctrl+C to stop the server.
    """
    logger.info("Starting Credit Risk Scoring API...")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("Demo endpoint:     http://localhost:8000/api/v1/demo")
    logger.info("Health check:      http://localhost:8000/api/v1/health")

    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=False,
        log_level="info",
    )


def run_demo():
    """
    Run a quick end-to-end demo without starting the full API server.

    LAYMAN: A "dry run" that tests the entire pipeline from raw data
    to a credit score decision, printing results to the console.
    No server needed — great for quick testing.
    """
    logger.info("=" * 70)
    logger.info("CREDIT RISK ENGINE — END-TO-END DEMO")
    logger.info("=" * 70)

    # Check if models exist
    xgb_path = Path(cfg.model.xgb_model_path)
    if not xgb_path.exists():
        logger.info("Models not found. Running training first...")
        run_training()

    # Load models
    from src.models.predict import model_store
    model_store.load()

    # Test applicants
    test_cases = [
        {
            "name": "Safe Borrower (Expected: APPROVE)",
            "data": {
                "loan_amnt": 8000.0, "annual_inc": 90000.0, "dti": 8.0,
                "int_rate": 7.5, "installment": 250.0,
                "fico_range_low": 760.0, "fico_range_high": 764.0,
                "term": "36 months", "grade": "A",
                "home_ownership": "MORTGAGE", "purpose": "debt_consolidation",
                "verification_status": "Verified",
                "delinq_2yrs": 0.0, "open_acc": 10.0, "pub_rec": 0.0,
                "revol_bal": 3000.0, "revol_util": 10.0, "total_acc": 20.0,
                "inq_last_6mths": 0.0,
            }
        },
        {
            "name": "Risky Borrower (Expected: REJECT)",
            "data": {
                "loan_amnt": 30000.0, "annual_inc": 28000.0, "dti": 45.0,
                "int_rate": 28.0, "installment": 1100.0,
                "fico_range_low": 595.0, "fico_range_high": 599.0,
                "term": "60 months", "grade": "G",
                "home_ownership": "RENT", "purpose": "small_business",
                "verification_status": "Not Verified",
                "delinq_2yrs": 5.0, "open_acc": 3.0, "pub_rec": 2.0,
                "revol_bal": 25000.0, "revol_util": 95.0, "total_acc": 6.0,
                "inq_last_6mths": 6.0,
            }
        },
    ]

    from src.models.predict import predict_single

    logger.info("\n" + "─" * 70)
    for case in test_cases:
        result = predict_single(case["data"])
        decision_icon = "✓" if result["decision"] == "APPROVE" else "✗"
        logger.info(f"\n{decision_icon} {case['name']}")
        logger.info(f"  Default Probability: {result['default_probability']:.1%}")
        logger.info(f"  Risk Grade:          {result['risk_grade']}")
        logger.info(f"  Decision:            {result['decision']}")
        logger.info("─" * 70)

    logger.info("\nDemo complete! Run 'python run.py --mode serve' to start the API.")


def run_tune(n_trials: int = 50, model_type: str = "both"):
    """
    Run Optuna hyperparameter tuning for XGBoost and/or LightGBM.

    LAYMAN: This is the "auto-pilot chef" that runs many experiments
    to find the best model settings. After it finishes, the next
    `python run.py --mode train` automatically uses the discovered best settings.

    DATA SCIENTIST:
    Uses Optuna TPE sampler with 5-fold cross-validation per trial.
    Best params saved to models/artifacts/best_params_xgb.json and
    models/artifacts/best_params_lgbm.json.
    `--mode train` will auto-load these if they exist.

    Args:
        n_trials: Number of Optuna trials per model (default: 50)
        model_type: "xgb", "lgbm", or "both"

    Example:
        python run.py --mode tune                   # 50 trials, both models
        python run.py --mode tune --trials 100      # 100 trials
        python run.py --mode tune --model xgb       # XGBoost only
    """
    logger.info("=" * 70)
    logger.info("CREDIT RISK ENGINE — HYPERPARAMETER TUNING (Optuna)")
    logger.info("=" * 70)
    logger.info(f"Trials per model: {n_trials} | Model: {model_type}")
    logger.info("This will take a few minutes. Progress printed every 10 trials.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    from src.models.tune import run_tuning_pipeline
    results = run_tuning_pipeline(
        n_trials=n_trials,
        model_type=model_type,
        save_results=True,
    )

    logger.info("\n" + "=" * 70)
    logger.info("TUNING COMPLETE")
    for model_key, result in results.items():
        logger.info(
            f"  {model_key.upper()} best AUC (CV): {result['best_value']:.4f}"
        )
    logger.info("=" * 70)
    logger.info(
        "\nNext step: Run `python run.py --mode train` to train with best params."
    )


def run_monitor():
    """
    Run drift detection on simulated current vs reference data.

    LAYMAN: Simulates a monthly monitoring check by comparing the
    model's score distribution from training to "current" data.
    Generates an HTML report showing if the model is still reliable.
    """
    logger.info("=" * 70)
    logger.info("CREDIT RISK ENGINE — MONITORING REPORT")
    logger.info("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate simulated reference and current score distributions
    rng = np.random.RandomState(42)
    reference_scores = rng.beta(2, 8, 1000)   # Training-time score distribution

    # Simulate slight drift (scores shifted slightly higher = riskier applicant mix)
    current_scores = rng.beta(2.5, 7, 500)

    from src.monitoring.drift_detector import run_monitoring_report
    report = run_monitoring_report(
        reference_scores=reference_scores,
        current_scores=current_scores,
        save_dir=str(REPORTS_DIR),
    )

    logger.info(f"\nMonitoring Report:")
    logger.info(f"  Health Status:     {report['health_status']}")
    logger.info(f"  Recommendation:    {report['recommendation']}")
    logger.info(f"  PSI Score:         {report['score_drift']['psi']:.4f}")
    logger.info(f"  Report saved to:   {REPORTS_DIR}/monitoring_report.json")


# ============================================================
# CLI ARGUMENT PARSER
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Credit Risk Scoring Engine — Master CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode train                    Train with current/tuned params
  python run.py --mode tune                     Tune hyperparams (50 trials each)
  python run.py --mode tune --trials 100        Tune with 100 trials each
  python run.py --mode tune --model xgb         Tune XGBoost only
  python run.py --mode serve                    Start the REST API server
  python run.py --mode demo                     Quick end-to-end test
  python run.py --mode monitor                  Run drift detection report
        """
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "serve", "demo", "monitor", "tune"],
        default="demo",
        help="Which pipeline to run (default: demo)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model for --mode tune (default: 50)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgb", "lgbm", "both"],
        default="both",
        help="Which model to tune with --mode tune (default: both)"
    )

    args = parser.parse_args()

    logger.info(f"Running mode: {args.mode}")

    if args.mode == "tune":
        run_tune(n_trials=args.trials, model_type=args.model)
    elif args.mode == "train":
        run_training()
    elif args.mode == "serve":
        run_serve()
    elif args.mode == "demo":
        run_demo()
    elif args.mode == "monitor":
        run_monitor()
