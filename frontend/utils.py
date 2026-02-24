"""
frontend/utils.py — Shared CSS, styling helpers, and model loader.
"""

import sys
from pathlib import Path

# Make project root importable from any page
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────

SHARED_CSS = """
<style>
    /* ── Layout ── */
    .block-container { padding-top: 1.5rem; }

    /* ── Page header banner ── */
    .page-header {
        background: linear-gradient(135deg, #1a56db 0%, #1e429f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .page-header h1 { margin: 0; font-size: 1.9rem; }
    .page-header p  { margin: 0.4rem 0 0 0; opacity: 0.88; font-size: 1rem; }

    /* ── Metric cards ── */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #1a56db;
        margin-bottom: 0.8rem;
    }
    .metric-card .label { font-size: 0.82rem; color: #6b7280; font-weight: 500; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1e2432; }
    .metric-card .delta { font-size: 0.78rem; color: #059669; }

    /* ── Decision cards ── */
    .approve-card {
        background: linear-gradient(135deg, #059669, #047857);
        color: white;
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(5,150,105,0.35);
    }
    .reject-card {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        color: white;
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(220,38,38,0.35);
    }
    .decision-icon  { font-size: 3rem; margin-bottom: 0.5rem; }
    .decision-label { font-size: 1.8rem; font-weight: 800; letter-spacing: 2px; }
    .decision-sub   { font-size: 0.95rem; opacity: 0.9; margin-top: 0.4rem; }

    /* ── Grade pill ── */
    .grade-pill {
        display: inline-block;
        padding: 3px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .grade-A { background:#d1fae5; color:#065f46; }
    .grade-B { background:#dbeafe; color:#1e40af; }
    .grade-C { background:#fef3c7; color:#92400e; }
    .grade-D { background:#ffedd5; color:#9a3412; }
    .grade-F { background:#fee2e2; color:#991b1b; }

    /* ── Info box ── */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }

    /* ── Table ── */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .styled-table th {
        background: #f0f4f8;
        padding: 0.6rem 0.9rem;
        text-align: left;
        font-weight: 600;
        color: #374151;
        border-bottom: 2px solid #e5e7eb;
    }
    .styled-table td {
        padding: 0.6rem 0.9rem;
        border-bottom: 1px solid #f3f4f6;
        color: #4b5563;
    }
    .styled-table tr:hover td { background: #f9fafb; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #1e2432;
    }
    [data-testid="stSidebar"] * { color: #e5e7eb !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #9ca3af !important; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
</style>
"""


def inject_css():
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="page-header">
        <h1>{icon} {title}</h1>
        {"<p>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def decision_card(decision: str, probability: float, grade: str):
    pct = f"{probability:.1%}"
    if decision == "APPROVE":
        st.markdown(f"""
        <div class="approve-card">
            <div class="decision-icon">✅</div>
            <div class="decision-label">APPROVED</div>
            <div class="decision-sub">Default Probability: {pct} &nbsp;|&nbsp; Grade: {grade}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="reject-card">
            <div class="decision-icon">❌</div>
            <div class="decision-label">REJECTED</div>
            <div class="decision-sub">Default Probability: {pct} &nbsp;|&nbsp; Grade: {grade}</div>
        </div>
        """, unsafe_allow_html=True)


def grade_pill(grade: str) -> str:
    return f'<span class="grade-pill grade-{grade}">{grade}</span>'


def model_status_sidebar():
    models_path = ROOT / "models" / "artifacts" / "xgb_model.pkl"
    is_ready = models_path.exists()
    with st.sidebar:
        st.markdown("---")
        if is_ready:
            st.success("🟢 Model Ready")
        else:
            st.error("🔴 Model Not Trained")
            st.caption("Run: `python run.py --mode train`")
    return is_ready


@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    from src.models.predict import model_store
    model_store.load()
    return model_store
