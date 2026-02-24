"""
pages/1_Score.py — Single Loan Application Scoring
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Score Application · Credit Risk Engine",
    page_icon="🎯",
    layout="wide",
)

from frontend.utils import inject_css, page_header, decision_card, model_status_sidebar, load_models

inject_css()

with st.sidebar:
    st.title("🎯 Score Application")
    st.caption("Fill the form and click **Score Application** to get an instant credit decision.")
    st.markdown("---")
    st.markdown("""
    **Required fields:**
    - Loan amount & interest rate
    - Annual income & DTI
    - FICO score range

    All other fields default to typical values if left blank.
    """)

model_ready = model_status_sidebar()

page_header("🎯", "Score Single Application",
            "Submit a loan application and receive an instant credit risk decision with explanation.")

if not model_ready:
    st.error("⚠️ Model not trained. Run `python run.py --mode train` first, then refresh.")
    st.stop()

# ── Preset data (defined before buttons) ─────────────────────────────────
PRESETS = {
    "safe": dict(
        loan_amnt=8000.0, annual_inc=90000.0, dti=8.0, int_rate=7.5,
        installment=250.0, fico_low=760, fico_high=764,
        term="36 months", grade="A", home="MORTGAGE",
        purpose="debt_consolidation", verification="Verified",
        delinq=0.0, open_acc=10.0, pub_rec=0.0,
        revol_bal=3000.0, revol_util=10.0, total_acc=20.0, inq=0.0,
        emp="10+ years",
    ),
    "medium": dict(
        loan_amnt=15000.0, annual_inc=55000.0, dti=22.0, int_rate=14.49,
        installment=520.0, fico_low=670, fico_high=674,
        term="36 months", grade="C", home="RENT",
        purpose="credit_card", verification="Not Verified",
        delinq=1.0, open_acc=8.0, pub_rec=0.0,
        revol_bal=12000.0, revol_util=55.0, total_acc=14.0, inq=2.0,
        emp="3 years",
    ),
    "risky": dict(
        loan_amnt=30000.0, annual_inc=28000.0, dti=45.0, int_rate=28.0,
        installment=1100.0, fico_low=595, fico_high=599,
        term="60 months", grade="G", home="RENT",
        purpose="small_business", verification="Not Verified",
        delinq=5.0, open_acc=3.0, pub_rec=2.0,
        revol_bal=25000.0, revol_util=95.0, total_acc=6.0, inq=6.0,
        emp="< 1 year",
    ),
}

# ── Session state — persists preset across form submissions ───────────────
# KEY FIX: Without session_state, preset resets to None on every rerun,
# causing the form to use its own default values, not the preset ones.
if "active_preset" not in st.session_state:
    st.session_state.active_preset = "safe"   # default on first load

# ── Quick-fill buttons ────────────────────────────────────────────────────
st.markdown("#### ⚡ Quick-fill Presets")
q1, q2, q3 = st.columns(3)

with q1:
    btn_safe = st.button("✅ Safe Borrower — Grade A", use_container_width=True,
                         type="primary" if st.session_state.active_preset == "safe" else "secondary")
with q2:
    btn_med  = st.button("⚠️ Medium Risk — Grade C",  use_container_width=True,
                         type="primary" if st.session_state.active_preset == "medium" else "secondary")
with q3:
    btn_risk = st.button("❌ Risky Borrower — Grade G", use_container_width=True,
                         type="primary" if st.session_state.active_preset == "risky" else "secondary")

# Update session state on button click and immediately rerun so form
# fields re-populate BEFORE the user hits "Score Application"
if btn_safe:
    st.session_state.active_preset = "safe"
    st.rerun()
if btn_med:
    st.session_state.active_preset = "medium"
    st.rerun()
if btn_risk:
    st.session_state.active_preset = "risky"
    st.rerun()

p = PRESETS[st.session_state.active_preset]

st.caption(f"Currently loaded: **{st.session_state.active_preset.upper()}** preset. "
           "Edit values freely, then click Score Application.")
st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────
with st.form("scoring_form"):
    st.markdown("### 📋 Loan Application Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**💰 Loan Details**")
        loan_amnt   = st.number_input("Loan Amount ($)",     500,   40000, int(p["loan_amnt"]),   step=500)
        int_rate    = st.number_input("Interest Rate (%)",   1.0,   35.0,  float(p["int_rate"]),  step=0.01, format="%.2f")
        installment = st.number_input("Monthly Install. ($)",10.0, 5000.0, float(p["installment"]),step=10.0)
        term        = st.selectbox("Loan Term",
                                   ["36 months", "60 months"],
                                   index=0 if p["term"] == "36 months" else 1)

    with c2:
        st.markdown("**🏠 Borrower Profile**")
        annual_inc = st.number_input("Annual Income ($)",  5000, 500000, int(p["annual_inc"]),  step=1000)
        dti        = st.number_input("Debt-to-Income (%)", 0.0,  99.0,  float(p["dti"]),        step=0.1, format="%.1f")
        home       = st.selectbox("Home Ownership",
                                  ["RENT","MORTGAGE","OWN","OTHER"],
                                  index=["RENT","MORTGAGE","OWN","OTHER"].index(p["home"]))
        emp_opts   = ["< 1 year","1 year","2 years","3 years","4 years",
                      "5 years","6 years","7 years","8 years","9 years","10+ years"]
        emp        = st.selectbox("Employment Length", emp_opts,
                                  index=emp_opts.index(p["emp"]) if p["emp"] in emp_opts else 10)

    with c3:
        st.markdown("**📊 Credit Profile**")
        fico_low  = st.number_input("FICO Score Low",  300, 850, int(p["fico_low"]))
        fico_high = st.number_input("FICO Score High", 300, 850, int(p["fico_high"]))
        grade_opts = ["A","B","C","D","E","F","G"]
        grade      = st.selectbox("Loan Grade", grade_opts,
                                  index=grade_opts.index(p["grade"]))
        purpose_opts = ["debt_consolidation","credit_card","home_improvement",
                        "major_purchase","small_business","medical","vacation","other"]
        purpose    = st.selectbox("Loan Purpose", purpose_opts,
                                  index=purpose_opts.index(p["purpose"]))

    st.markdown("**📜 Credit History**")
    h1, h2, h3, h4, h5, h6, h7 = st.columns(7)
    delinq     = h1.number_input("Delinquencies",  0,  30,   int(p["delinq"]))
    open_acc   = h2.number_input("Open Accounts",  0,  80,   int(p["open_acc"]))
    pub_rec    = h3.number_input("Public Records", 0,  20,   int(p["pub_rec"]))
    revol_bal  = h4.number_input("Revol. Bal ($)", 0, 500000,int(p["revol_bal"]), step=500)
    revol_util = h5.number_input("Revol. Util (%)",0.0,100.0,float(p["revol_util"]), step=1.0)
    total_acc  = h6.number_input("Total Accounts", 0,  200,  int(p["total_acc"]))
    inq        = h7.number_input("Inquiries (6m)", 0,  30,   int(p["inq"]))

    ver_opts  = ["Verified","Source Verified","Not Verified"]
    verification = st.selectbox("Verification Status", ver_opts,
                                index=ver_opts.index(p["verification"]))

    st.markdown("")
    submitted = st.form_submit_button("🚀 Score Application", use_container_width=True, type="primary")

# ── Prediction ─────────────────────────────────────────────────────────────
if submitted:
    if fico_high < fico_low:
        st.error("FICO High must be ≥ FICO Low.")
        st.stop()

    applicant = {
        "loan_amnt": float(loan_amnt), "annual_inc": float(annual_inc),
        "dti": float(dti), "int_rate": float(int_rate),
        "installment": float(installment),
        "fico_range_low": float(fico_low), "fico_range_high": float(fico_high),
        "term": term, "grade": grade, "home_ownership": home,
        "purpose": purpose, "verification_status": verification,
        "emp_length": emp,
        "delinq_2yrs": float(delinq), "open_acc": float(open_acc),
        "pub_rec": float(pub_rec), "revol_bal": float(revol_bal),
        "revol_util": float(revol_util), "total_acc": float(total_acc),
        "inq_last_6mths": float(inq),
    }

    with st.spinner("Scoring application…"):
        try:
            load_models()
            from src.models.predict import predict_single
            result = predict_single(applicant)
        except Exception as e:
            st.error(f"Scoring error: {e}")
            st.stop()

    prob      = result["default_probability"]
    grade_out = result["risk_grade"]
    dec       = result["decision"]
    factors   = result.get("top_risk_factors", [])

    st.markdown("---")
    st.markdown("## 📋 Decision")

    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        decision_card(dec, prob, grade_out)
        st.markdown("<br>", unsafe_allow_html=True)

        gauge_color = "#059669" if dec == "APPROVE" else "#dc2626"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            title={"text": "Default Probability", "font": {"size": 14}},
            number={"suffix": "%", "font": {"size": 30, "color": gauge_color}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": gauge_color},
                "steps": [
                    {"range": [0,  15], "color": "#d1fae5"},
                    {"range": [15, 25], "color": "#dbeafe"},
                    {"range": [25, 40], "color": "#fef3c7"},
                    {"range": [40, 55], "color": "#ffedd5"},
                    {"range": [55,100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#1a56db", "width": 3},
                    "thickness": 0.8, "value": 50,
                },
            },
        ))
        fig.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with res_right:
        st.markdown("#### 🔑 Key Risk Factors (Why this decision?)")

        if factors:
            rows = ""
            for i, f in enumerate(factors[:5], 1):
                direction = f.get("direction", "")
                impact    = f.get("shap_impact", 0)
                label     = f.get("label", f.get("feature", ""))
                val       = f.get("value", "")
                arrow     = "🔴 Increases Risk" if direction == "increases_risk" else "🟢 Reduces Risk"
                rows += f"""
                <tr>
                    <td><b>#{i}</b></td>
                    <td>{label}</td>
                    <td>{val}</td>
                    <td>{arrow}</td>
                    <td>{impact:+.3f}</td>
                </tr>"""
            st.markdown(f"""
            <table class="styled-table">
                <thead><tr>
                    <th>#</th><th>Feature</th><th>Your Value</th>
                    <th>Direction</th><th>SHAP Score</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.info("Risk factor details not available for this prediction.")

        st.markdown(f"""
        <div class="info-box">
            Decision threshold: <b>50%</b> — predicted default probability is <b>{prob:.1%}</b>.
            {"Above threshold → REJECT" if prob >= 0.50 else "Below threshold → APPROVE"}.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Predictions are for demonstration purposes only. Not financial advice.")
