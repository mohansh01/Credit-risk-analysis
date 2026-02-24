"""
frontend/app.py — Main Dashboard (Home Page)
=============================================
Professional credit risk engine dashboard built with Streamlit.
Deployed on Render: single-service, no separate FastAPI server needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Page config (MUST be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="Credit Risk Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.utils import inject_css, page_header, model_status_sidebar

inject_css()

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.title("Credit Risk Engine")
    st.caption("v1.0  |  ML-Powered Lending")
    st.markdown("---")
    st.markdown("""
    **Navigation**
    - 🏠 Dashboard *(you are here)*
    - 🎯 Score Application
    - 📊 Batch Scoring
    - 📈 Reports
    - ℹ️ About
    """)

model_ready = model_status_sidebar()

# ── Header ───────────────────────────────────────────────────────────────
page_header(
    "🏦", "Credit Risk Scoring Engine",
    "ML-powered loan default prediction · SHAP explainability · ECOA compliant · Deployed on Render"
)

# ── Model status banner ───────────────────────────────────────────────────
if not model_ready:
    st.warning("""
    ⚠️ **Model not trained yet.**
    On Render the model trains automatically during the build phase.
    Locally, run: `python run.py --mode train`
    """)
else:
    st.success("✅ Model loaded and ready for predictions.")

st.markdown("<br>", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────
st.subheader("📊 Model Performance")
k1, k2, k3, k4, k5 = st.columns(5)

kpis = [
    (k1, "AUC-ROC",        "0.73",  "+0.21 vs baseline", "#1a56db"),
    (k2, "KS Statistic",   "0.38",  "Strong separation",  "#059669"),
    (k3, "Gini Coeff.",    "0.46",  "Good discrimination","#7c3aed"),
    (k4, "Training Rows",  "14,000","70% of 20k records", "#d97706"),
    (k5, "Features",       "76",    "Engineered from 19", "#db2777"),
]

for col, label, value, delta, color in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:{color}">
            <div class="label">{label}</div>
            <div class="value" style="color:{color}">{value}</div>
            <div class="delta">↑ {delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two-column section: Score gauge demo + pipeline flow ─────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("🔢 Score Range Interpretation")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=27,
        title={"text": "Sample: Grade A Borrower", "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#1a56db"},
            "steps": [
                {"range": [0,  15], "color": "#d1fae5"},
                {"range": [15, 25], "color": "#dbeafe"},
                {"range": [25, 40], "color": "#fef3c7"},
                {"range": [40, 55], "color": "#ffedd5"},
                {"range": [55,100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "#dc2626", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    | Grade | Probability | Decision |
    |-------|------------|---------|
    | **A** | 0–15%      | ✅ APPROVE |
    | **B** | 15–25%     | ✅ APPROVE |
    | **C** | 25–40%     | ✅ APPROVE |
    | **D** | 40–55%     | ❌ REJECT  |
    | **F** | 55%+       | ❌ REJECT  |
    """)

with right:
    st.subheader("⚙️ ML Pipeline")
    steps_df = pd.DataFrame({
        "Step": ["Raw Data", "Validation", "Feature Engineering",
                 "XGBoost", "LightGBM", "Meta-Learner", "Decision"],
        "Output": ["19 features", "Schema check", "76 features",
                   "Probability", "Probability", "Final probability", "APPROVE / REJECT"],
        "Type": ["Input","Process","Process","Model","Model","Model","Output"],
    })

    colors = {
        "Input":   "#dbeafe",
        "Process": "#fef3c7",
        "Model":   "#f0fdf4",
        "Output":  "#ede9fe",
    }

    rows = ""
    for _, row in steps_df.iterrows():
        bg = colors[row["Type"]]
        rows += f"""
        <tr>
            <td style="background:{bg};border-radius:6px;padding:6px 12px;font-weight:600">{row['Step']}</td>
            <td style="padding:6px 12px;color:#6b7280">{row['Output']}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
        <thead><tr><th>Stage</th><th>Output</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Score distribution chart ──────────────────────────────────────────────
st.subheader("📉 Simulated Score Distribution")

rng = np.random.RandomState(42)
approved_probs = rng.beta(2, 8, 800)
rejected_probs = rng.beta(6, 3, 200)

fig2 = go.Figure()
fig2.add_trace(go.Histogram(
    x=approved_probs, name="Approved",
    marker_color="#059669", opacity=0.75,
    nbinsx=30, histnorm="probability density",
))
fig2.add_trace(go.Histogram(
    x=rejected_probs, name="Rejected",
    marker_color="#dc2626", opacity=0.75,
    nbinsx=30, histnorm="probability density",
))
fig2.add_vline(x=0.50, line_dash="dash", line_color="#1a56db",
               annotation_text="Threshold (50%)", annotation_position="top right")
fig2.update_layout(
    barmode="overlay",
    xaxis_title="Default Probability",
    yaxis_title="Density",
    height=320,
    legend=dict(x=0.75, y=0.95),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=20, b=40),
)
fig2.update_xaxes(gridcolor="#f3f4f6")
fig2.update_yaxes(gridcolor="#f3f4f6")
st.plotly_chart(fig2, use_container_width=True)

# ── Tech stack footer ────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown("**🤖 Models**\nXGBoost · LightGBM · Logistic Regression (stacking)")
with c2: st.markdown("**📐 Explainability**\nSHAP · Feature importance · Rejection reasons")
with c3: st.markdown("**⚖️ Fairness**\nECOA · 4/5ths Rule · Disparate Impact analysis")
with c4: st.markdown("**🚀 Deployment**\nStreamlit · Render · Python 3.10")
