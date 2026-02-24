"""
pages/2_Batch.py — Batch Loan Application Scoring
==================================================
Score multiple applicants at once via JSON paste or CSV upload.
Shows a results table + approve/reject breakdown chart.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import io

st.set_page_config(
    page_title="Batch Scoring · Credit Risk Engine",
    page_icon="📊",
    layout="wide",
)

from frontend.utils import inject_css, page_header, model_status_sidebar, load_models

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Batch Scoring")
    st.caption("Score many applicants at once.")
    st.markdown("---")
    st.markdown("""
    **Required fields per applicant:**
    ```
    loan_amnt, annual_inc, dti
    int_rate, installment
    fico_range_low, fico_range_high
    ```
    All other fields are optional.
    """)

model_ready = model_status_sidebar()

# ── Header ────────────────────────────────────────────────────────────────
page_header("📊", "Batch Loan Scoring",
            "Score up to 10,000 loan applications at once. Paste JSON or upload a CSV file.")

if not model_ready:
    st.error("⚠️ Model not trained. Run `python run.py --mode train` first.")
    st.stop()

# ── Sample JSON helper ─────────────────────────────────────────────────────
SAMPLE_JSON = json.dumps({
    "applications": [
        {
            "applicant_id": "APP_001",
            "loan_amnt": 8000, "annual_inc": 90000, "dti": 8.0,
            "int_rate": 7.5, "installment": 250,
            "fico_range_low": 760, "fico_range_high": 764,
        },
        {
            "applicant_id": "APP_002",
            "loan_amnt": 30000, "annual_inc": 28000, "dti": 45.0,
            "int_rate": 28.0, "installment": 1100,
            "fico_range_low": 595, "fico_range_high": 599,
        },
        {
            "applicant_id": "APP_003",
            "loan_amnt": 15000, "annual_inc": 55000, "dti": 22.0,
            "int_rate": 14.49, "installment": 520,
            "fico_range_low": 670, "fico_range_high": 674,
        },
    ]
}, indent=2)

# ── Input method tabs ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 Paste JSON", "📂 Upload CSV"])

applications = None

with tab1:
    st.markdown("Paste a JSON array of applications. Click **Load Sample** to autofill.")

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("📥 Load Sample JSON"):
            st.session_state["batch_json"] = SAMPLE_JSON

    json_input = st.text_area(
        "JSON Input",
        value=st.session_state.get("batch_json", SAMPLE_JSON),
        height=280,
        key="json_area",
    )

    if st.button("▶ Score from JSON", type="primary", use_container_width=True):
        try:
            parsed = json.loads(json_input)
            if isinstance(parsed, dict) and "applications" in parsed:
                applications = parsed["applications"]
            elif isinstance(parsed, list):
                applications = parsed
            else:
                st.error("JSON must be a list of applications or `{\"applications\": [...]}`")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

with tab2:
    st.markdown("""
    Upload a CSV file with one row per applicant.
    **Required columns:** `loan_amnt`, `annual_inc`, `dti`, `int_rate`, `installment`,
    `fico_range_low`, `fico_range_high`
    """)

    template_df = pd.DataFrame([{
        "applicant_id": "APP_001",
        "loan_amnt": 8000, "annual_inc": 90000, "dti": 8.0,
        "int_rate": 7.5, "installment": 250,
        "fico_range_low": 760, "fico_range_high": 764,
    }])
    csv_bytes = template_df.to_csv(index=False).encode()
    st.download_button("📥 Download CSV Template", csv_bytes, "template.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.dataframe(df_up.head(5), use_container_width=True)
        st.caption(f"Loaded {len(df_up)} rows.")
        if st.button("▶ Score from CSV", type="primary", use_container_width=True):
            applications = df_up.to_dict("records")

# ── Default values for ALL optional fields ───────────────────────────────
# Without this, predict_single throws KeyError for missing fields like revol_util
FIELD_DEFAULTS = {
    "term":                    "36 months",
    "grade":                   "B",
    "home_ownership":          "RENT",
    "purpose":                 "debt_consolidation",
    "verification_status":     "Not Verified",
    "emp_length":              "5 years",
    "delinq_2yrs":             0.0,
    "open_acc":                10.0,
    "pub_rec":                 0.0,
    "revol_bal":               5000.0,
    "revol_util":              30.0,
    "total_acc":               20.0,
    "inq_last_6mths":          1.0,
    "mths_since_last_delinq":  None,
}

# ── Run predictions ────────────────────────────────────────────────────────
if applications:
    st.markdown("---")
    required = {"loan_amnt", "annual_inc", "dti", "int_rate", "installment",
                "fico_range_low", "fico_range_high"}

    missing_req = required - set(applications[0].keys())
    if missing_req:
        st.error(f"Missing required fields in first record: {missing_req}")
        st.stop()

    with st.spinner(f"Scoring {len(applications)} application(s)…"):
        try:
            load_models()
            from src.models.predict import predict_single

            results = []
            for app in applications:
                app_clean = {k: v for k, v in app.items() if v is not None}
                app_id = app_clean.pop("applicant_id", f"APP_{len(results)+1:03d}")
                # Fill in any missing optional fields with safe defaults
                for field, default in FIELD_DEFAULTS.items():
                    if field not in app_clean:
                        app_clean[field] = default
                r = predict_single(app_clean)
                r["applicant_id"] = app_id
                results.append(r)
        except Exception as e:
            st.error(f"Scoring error: {e}")
            st.stop()

    # Build results DataFrame
    rows = []
    for r in results:
        rows.append({
            "Applicant ID":  r.get("applicant_id", "—"),
            "Probability":   r["default_probability"],
            "Grade":         r["risk_grade"],
            "Decision":      r["decision"],
        })
    results_df = pd.DataFrame(rows)

    # ── Summary metrics ─────────────────────────────────────────────────
    st.markdown("## 📊 Results Summary")

    total    = len(results_df)
    approved = (results_df["Decision"] == "APPROVE").sum()
    rejected = total - approved
    avg_prob = results_df["Probability"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Applications", total)
    m2.metric("✅ Approved", approved, f"{approved/total:.0%}")
    m3.metric("❌ Rejected", rejected, f"-{rejected/total:.0%}")
    m4.metric("Avg. Default Prob.", f"{avg_prob:.1%}")

    # ── Charts ────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        pie = go.Figure(go.Pie(
            labels=["Approved", "Rejected"],
            values=[approved, rejected],
            marker_colors=["#059669", "#dc2626"],
            hole=0.55,
            textinfo="label+percent",
        ))
        pie.update_layout(
            title="Approval Breakdown",
            height=300, margin=dict(t=40, b=0, l=0, r=0),
            showlegend=False,
        )
        st.plotly_chart(pie, use_container_width=True)

    with ch2:
        hist = px.histogram(
            results_df, x="Probability", color="Decision",
            color_discrete_map={"APPROVE": "#059669", "REJECT": "#dc2626"},
            nbins=20, opacity=0.8, barmode="overlay",
            title="Probability Distribution",
            labels={"Probability": "Default Probability"},
        )
        hist.add_vline(x=0.50, line_dash="dash", line_color="#1a56db",
                       annotation_text="50% threshold")
        hist.update_layout(height=300, margin=dict(t=40, b=0),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(hist, use_container_width=True)

    # ── Results table ─────────────────────────────────────────────────────
    st.markdown("## 📋 Detailed Results")

    def style_decision(val):
        if val == "APPROVE":
            return "background-color:#d1fae5;color:#065f46;font-weight:600"
        return "background-color:#fee2e2;color:#991b1b;font-weight:600"

    def style_prob(val):
        if val < 0.25: return "color:#059669;font-weight:600"
        if val < 0.50: return "color:#d97706;font-weight:600"
        return "color:#dc2626;font-weight:600"

    display_df = results_df.copy()
    display_df["Probability"] = display_df["Probability"].map("{:.1%}".format)

    styled = display_df.style.applymap(style_decision, subset=["Decision"])
    st.dataframe(styled, use_container_width=True, height=400)

    # ── Download ──────────────────────────────────────────────────────────
    csv = results_df.to_csv(index=False).encode()
    st.download_button(
        "📥 Download Results CSV",
        csv, "credit_score_results.csv", "text/csv",
        use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Batch scoring supports up to 10,000 applications. Results are for demonstration only.")
