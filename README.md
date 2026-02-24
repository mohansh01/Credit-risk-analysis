# Credit Risk Scoring Engine

A production-grade, ML-powered loan default prediction system built with XGBoost, LightGBM, SHAP explainability, fairness detection, and a FastAPI REST API.

---

## What This Does

Banks receive thousands of loan applications daily. This engine predicts — given an applicant's financial profile — **what is the probability they will default within 12 months?**

The output is a **probability score (0–1)** + a **risk grade (A/B/C/D/F)** + **SHAP explanations** of why the score was given.

---

## Quick Start (5 Minutes)

### 1. Install dependencies
```bash
cd credit-risk-engine
pip install -r requirements.txt
```

### 2. Train the model (uses synthetic data automatically)
```bash
python run.py --mode train
```

### 3. Start the API server
```bash
python run.py --mode serve
```

### 4. Visit the interactive API documentation
Open: **http://localhost:8000/docs**

### 5. Score a demo applicant
```bash
curl http://localhost:8000/api/v1/demo
```

---

## All Commands

| Command | What It Does |
|---|---|
| `python run.py --mode train` | Train all models from scratch |
| `python run.py --mode serve` | Start the REST API server |
| `python run.py --mode demo` | Quick end-to-end test (no server needed) |
| `python run.py --mode monitor` | Run drift detection report |
| `pytest tests/ -v` | Run all unit and integration tests |

---

## API Endpoints

Once the server is running (`python run.py --mode serve`):

| Method | URL | Description |
|---|---|---|
| `GET` | `/` | API homepage |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/demo` | Score a demo applicant |
| `POST` | `/api/v1/score` | Score a single applicant |
| `POST` | `/api/v1/score/batch` | Score up to 10,000 applicants |
| `GET` | `/api/v1/model-info` | Model metadata |
| `GET` | `/docs` | Interactive Swagger UI |

### Example: Score a Single Applicant

```bash
curl -X POST "http://localhost:8000/api/v1/score" \
  -H "Content-Type: application/json" \
  -d '{
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
    "revol_util": 45.5
  }'
```

### Expected Response
```json
{
  "applicant_id": null,
  "default_probability": 0.2341,
  "risk_grade": "B",
  "decision": "APPROVE",
  "model_version": "1.0.0"
}
```

---

## Project Structure

```
credit-risk-engine/
│
├── run.py                          ← Master entry point (start here)
├── config.py                       ← All configuration in one place
├── requirements.txt                ← Python dependencies
├── Dockerfile                      ← Docker containerization
│
├── src/
│   ├── data/
│   │   ├── ingestion.py            ← Load & validate raw data
│   │   └── preprocessing.py        ← Clean, encode, split data
│   │
│   ├── features/
│   │   ├── engineer.py             ← Create 40+ engineered features
│   │   └── pipeline.py             ← Sklearn feature pipeline
│   │
│   ├── models/
│   │   ├── train.py                ← XGBoost + LightGBM + stacking
│   │   ├── evaluate.py             ← AUC, KS, Gini, ROC curves
│   │   └── predict.py              ← Inference / scoring
│   │
│   ├── explainability/
│   │   └── shap_explainer.py       ← SHAP values & explanations
│   │
│   ├── fairness/
│   │   └── bias_detector.py        ← Disparate impact analysis
│   │
│   ├── monitoring/
│   │   └── drift_detector.py       ← PSI & KS drift detection
│   │
│   └── api/
│       ├── main.py                 ← FastAPI app
│       ├── routes.py               ← API endpoint handlers
│       └── schemas.py              ← Pydantic request/response models
│
├── data/
│   ├── raw/                        ← Raw LendingClub CSV (or synthetic)
│   ├── processed/                  ← Cleaned datasets
│   └── sample_applications/        ← Test JSON payloads
│
├── models/artifacts/               ← Saved model .pkl files
├── reports/                        ← Evaluation plots & monitoring reports
│
└── tests/
    ├── test_features.py            ← Unit tests for feature engineering
    └── test_api.py                 ← Integration tests for API
```

---

## Model Architecture

```
                    ┌─────────────────────┐
                    │  Loan Application   │
                    │  (JSON Input)       │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │  Feature Pipeline   │  50+ engineered features
                    │  (sklearn Pipeline) │  Scaled, encoded, imputed
                    └────────┬────────────┘
                             │
               ┌─────────────┴──────────────┐
               │                            │
      ┌────────▼────────┐         ┌─────────▼────────┐
      │    XGBoost      │         │    LightGBM      │
      │  (500 trees)    │         │  (500 trees)     │
      └────────┬────────┘         └─────────┬────────┘
               │                            │
               └─────────────┬──────────────┘
                             │
                    ┌────────▼────────────┐
                    │  Logistic Regression │  Meta-learner
                    │  (Meta-Learner)      │  combines both
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │  Final Probability  │  0.0 – 1.0
                    │  + Risk Grade       │  A / B / C / D / F
                    │  + Decision         │  APPROVE / REJECT
                    └─────────────────────┘
```

---

## Performance Targets

| Metric | Target | Description |
|---|---|---|
| AUC-ROC | > 0.85 | Model discrimination ability |
| KS Statistic | > 0.40 | Max separation between good/bad |
| Gini Coefficient | > 0.60 | Normalised AUC (industry standard) |
| API Latency (p99) | < 50ms | Single request scoring time |
| Batch (1000 apps) | < 2s | Bulk processing speed |
| Disparate Impact | > 0.80 | Fairness (4/5ths rule) |

---

## Using Real Data

1. Download the **LendingClub Loan Data** from Kaggle:
   https://www.kaggle.com/datasets/wordsforthewise/lending-club

2. Place `accepted_2007_to_2018Q4.csv` in `data/raw/`

3. Update `config.py`:
   ```python
   raw_filename: str = "accepted_2007_to_2018Q4.csv"
   ```

4. Run training:
   ```bash
   python run.py --mode train
   ```

---

## Docker Deployment

```bash
# Build the image
docker build -t credit-risk-engine .

# Train model inside container (creates artifacts)
docker run -v $(pwd)/models:/app/models credit-risk-engine \
    python run.py --mode train

# Start the API server
docker run -p 8000:8000 -v $(pwd)/models:/app/models credit-risk-engine

# Health check
curl http://localhost:8000/api/v1/health
```

---

## Regulatory Compliance Features

This engine is designed with financial regulation in mind:

| Regulation | Feature | Implementation |
|---|---|---|
| ECOA (Fair Lending) | Bias detection | `bias_detector.py` — 4/5ths rule |
| Basel III | Model explainability | SHAP values per prediction |
| FCRA | Adverse action reasons | Top 3 denial factors in response |
| SR 11-7 | Model governance | MLflow experiment tracking |
| GDPR | Audit trail | Structured logging per request |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only feature engineering tests
pytest tests/test_features.py -v

# Run only API tests
pytest tests/test_api.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Models | XGBoost, LightGBM |
| Meta-Learner | scikit-learn LogisticRegression |
| Feature Engineering | pandas, numpy |
| Explainability | SHAP (TreeExplainer) |
| Fairness | fairlearn |
| API Framework | FastAPI + uvicorn |
| Data Validation | Pydantic v2 |
| Monitoring | Evidently AI, scipy |
| Experiment Tracking | MLflow |
| Testing | pytest |
| Containerization | Docker |

---

## License

For educational and research purposes.
