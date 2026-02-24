"""
pages/4_About.py — Model Info & Architecture
=============================================
Technical overview of the model, training pipeline,
performance metrics, and technology stack.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="About · Credit Risk Engine",
    page_icon="ℹ️",
    layout="wide",
)

from frontend.utils import inject_css, page_header, model_status_sidebar

inject_css()

with st.sidebar:
    st.title("ℹ️ About")
    st.caption("Technical documentation.")

model_status_sidebar()

page_header("ℹ️", "Model Info & Architecture",
            "How the Credit Risk Scoring Engine works under the hood.")

# ─────────────────────────────────────────────────────────────────────────
# Section 1 — Pipeline Overview
# ─────────────────────────────────────────────────────────────────────────
st.subheader("⚙️ ML Pipeline Architecture")

pipeline_data = {
    "Stage": [
        "1. Data Generation",
        "2. Validation",
        "3. Feature Engineering",
        "4. Train / Val / Test Split",
        "5. OOF Stacking (5-fold)",
        "6. Hyperparameter Tuning",
        "7. Final Training",
        "8. Evaluation",
        "9. Inference (API)",
    ],
    "Description": [
        "20,000 synthetic records calibrated to LendingClub statistics (income, FICO, default rate ~15%)",
        "Schema checks, dtype coercion, missing-value imputation, outlier clipping",
        "19 raw → 76 features: debt ratios, credit utilization bins, interaction terms, date features",
        "70% train / 15% validation / 15% test — stratified to preserve default rate",
        "XGBoost + LightGBM each trained 5× to produce clean OOF predictions (prevents leakage)",
        "Optuna TPE (30 trials, 8k subsample, 3-fold CV) — finds best hyperparameters per model",
        "Full 14k training set with tuned hyperparameters; Logistic Regression meta-learner on OOF",
        "AUC-ROC, KS statistic, Gini, PSI, fairness (Disparate Impact Ratio)",
        "FastAPI + Streamlit; models loaded once into memory; SHAP explains every decision",
    ],
}
pipeline_df = pd.DataFrame(pipeline_data)
st.dataframe(pipeline_df, use_container_width=True, hide_index=True, height=360)

# ─────────────────────────────────────────────────────────────────────────
# Section 2 — Model Performance
# ─────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Model Performance")

m1, m2, m3, m4 = st.columns(4)
metrics = [
    ("AUC-ROC",         "0.73",  "0.52 baseline",  "#1a56db"),
    ("KS Statistic",    "0.38",  "Good separation", "#059669"),
    ("Gini Coeff.",     "0.46",  "2×AUC − 1",      "#7c3aed"),
    ("Default Rate",    "~15%",  "Training data",   "#d97706"),
]
for col, (label, val, sub, color) in zip([m1, m2, m3, m4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:{color}">
            <div class="label">{label}</div>
            <div class="value" style="color:{color}">{val}</div>
            <div class="delta">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Section 3 — Model Architecture
# ─────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧠 Stacking Ensemble Architecture")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    #### Level 1 — Base Learners
    Both trained on **14,000 rows** with 5-fold OOF stacking:

    **XGBoost**
    - `max_depth`: 3 *(tuned by Optuna)*
    - `learning_rate`: 0.021
    - `n_estimators`: 129
    - `Best CV AUC`: 0.7238

    **LightGBM**
    - `num_leaves`: 62 *(tuned by Optuna)*
    - `learning_rate`: 0.020
    - `n_estimators`: 80
    - `Best CV AUC`: 0.7227

    #### Level 2 — Meta-Learner
    **Logistic Regression**
    - Input: [XGBoost OOF prob, LightGBM OOF prob]
    - Learns the optimal weighted combination
    - `C`: 1.0, `class_weight`: balanced
    """)

with col2:
    # Stacking diagram using plotly
    fig_arch = go.Figure()

    boxes = [
        # (x, y, text, color)
        (0.5, 0.90, "Raw Application\n(19 features)", "#dbeafe"),
        (0.5, 0.72, "Feature Engineering\n(76 features)", "#e9d5ff"),
        (0.25, 0.52, "XGBoost\n(Level 1)", "#d1fae5"),
        (0.75, 0.52, "LightGBM\n(Level 1)", "#d1fae5"),
        (0.5,  0.30, "Logistic Regression\nMeta-Learner (Level 2)", "#fef3c7"),
        (0.5,  0.10, "Decision\n(APPROVE / REJECT)", "#fee2e2"),
    ]

    for x, y, text, color in boxes:
        fig_arch.add_shape(
            type="rect",
            x0=x-0.19, x1=x+0.19, y0=y-0.07, y1=y+0.07,
            fillcolor=color, line_color="#6b7280", line_width=1,
        )
        fig_arch.add_annotation(x=x, y=y, text=text, showarrow=False,
                                font=dict(size=10), align="center")

    # Arrows
    arrows = [
        (0.5, 0.83, 0.5, 0.79),
        (0.5, 0.65, 0.25, 0.59),
        (0.5, 0.65, 0.75, 0.59),
        (0.25, 0.45, 0.5, 0.37),
        (0.75, 0.45, 0.5, 0.37),
        (0.5, 0.23, 0.5, 0.17),
    ]
    for x0, y0, x1, y1 in arrows:
        fig_arch.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowcolor="#6b7280", arrowwidth=1.5,
        )

    fig_arch.update_layout(
        height=420, margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_arch, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────
# Section 4 — Data & Features
# ─────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📂 Data & Features")

d1, d2 = st.columns(2)
with d1:
    st.markdown("""
    #### Dataset
    | Property | Value |
    |----------|-------|
    | Source | Synthetic (LendingClub schema) |
    | Total records | 20,000 |
    | Default rate | ~14–18% |
    | Train rows | 14,000 (70%) |
    | Val rows | 3,000 (15%) |
    | Test rows | 3,000 (15%) |
    | Random seed | 42 (reproducible) |
    """)

with d2:
    st.markdown("""
    #### Feature Engineering
    | Category | Count | Examples |
    |----------|-------|---------|
    | Raw numeric | 19 | income, DTI, FICO |
    | Engineered ratios | 15 | loan/income, payment/income |
    | Binned features | 10 | FICO bins, DTI buckets |
    | Credit history | 12 | delinq rate, util trend |
    | Categorical encoded | 20 | grade, purpose, term |
    | **Total** | **76** | |
    """)

# ─────────────────────────────────────────────────────────────────────────
# Section 5 — Tech Stack
# ─────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🛠️ Technology Stack")

tech = [
    ("🤖 XGBoost 2.0",           "Gradient boosting — base learner 1"),
    ("⚡ LightGBM 4.3",           "Leaf-wise boosting — base learner 2"),
    ("📐 Scikit-learn 1.4",       "Preprocessing, stacking, metrics"),
    ("🔍 Optuna",                  "Bayesian hyperparameter tuning (TPE)"),
    ("💡 SHAP 0.45",              "Shapley value explainability"),
    ("⚖️ Fairlearn",               "ECOA fairness metrics"),
    ("🚀 FastAPI",                 "REST API backend"),
    ("🎨 Streamlit",              "Interactive frontend"),
    ("☁️ Render",                  "Cloud deployment"),
    ("📊 Plotly",                  "Interactive charts"),
]

cols = st.columns(2)
for i, (name, desc) in enumerate(tech):
    with cols[i % 2]:
        st.markdown(f"""
        <div style="background:white;border-radius:8px;padding:0.8rem 1rem;
                    margin-bottom:0.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.08);
                    border-left:3px solid #1a56db">
            <b>{name}</b><br>
            <span style="color:#6b7280;font-size:0.87rem">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Section 6 — Regulatory Compliance
# ─────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⚖️ Regulatory Compliance")

st.markdown("""
| Regulation | Requirement | Implementation |
|-----------|-------------|---------------|
| **ECOA** (Equal Credit Opportunity Act) | No discriminatory lending | Disparate Impact Ratio ≥ 0.80 checked for all protected groups |
| **Fair Lending** | Explain rejection reasons | SHAP top-5 risk factors returned with every REJECT decision |
| **Model Risk** | Model validation & monitoring | KS, AUC, PSI tracked; drift triggers retraining alert |
| **GDPR** | Data minimisation | No PII stored; applicant_id is optional tracking field |
""")

st.markdown("---")
st.caption("Credit Risk Scoring Engine v1.0 · Built with Python, Streamlit, XGBoost, LightGBM · Deployed on Render")
