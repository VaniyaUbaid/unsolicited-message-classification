from pathlib import Path

import streamlit as st

from src.prediction import MODEL_FILES, load_models
from src.ui import render_page


BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def get_models():
    return load_models(BASE_DIR / "models")


st.set_page_config(
    page_title="MessageGuard",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #29263a;
        --muted: #77738b;
        --purple: #7862c8;
        --purple-soft: #f0edfb;
        --line: #ebe8f2;
        --surface: #ffffff;
        --canvas: #faf9fc;
    }
    html, body, [class*="css"] { font-family: Inter, 'Segoe UI', sans-serif; }
    .stApp { background: var(--canvas); color: var(--ink); }
    h1, h2, h3 { font-family: Inter, 'Segoe UI', sans-serif; color: var(--ink); }
    h1 { letter-spacing: -0.035em; }
    [data-testid="stSidebar"] {
        background: #fff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] h1 { font-size: 1.65rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.3rem 0.15rem;
    }
    .brand-mark {
        color: var(--purple);
        font-family: Inter, 'Segoe UI', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.06em;
    }
    .brand-subtitle {
        color: var(--muted);
        font-size: 0.83rem;
        margin-top: -0.5rem;
        margin-bottom: 1.4rem;
    }
    .eyebrow {
        color: var(--purple);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }
    .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: 0 5px 22px rgba(49, 39, 82, 0.045);
        padding: 1.15rem 1.3rem;
    }
    .muted { color: var(--muted); }
    div.stButton > button[kind="primary"] {
        background: var(--purple);
        border-color: var(--purple);
    }
    div.stButton > button[kind="primary"]:hover {
        background: #6650b5;
        border-color: #6650b5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

models = get_models()
missing_files = [
    filename
    for experiment in MODEL_FILES.values()
    for filename in experiment.values()
    if not (BASE_DIR / "models" / filename).is_file()
]

with st.sidebar:
    st.markdown('<div class="brand-mark">MessageGuard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">Intelligent Message Classification &amp; Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Workspace",
        [
            "Message Classification",
            "Explainable Analysis",
            "Suspicious Patterns",
            "Performance Laboratory",
            "Prediction History",
            "Batch Analysis",
            "Robustness Testing",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Local inference · Saved models · No cloud APIs")
    if missing_files:
        st.warning("Some model files are missing. Affected selections are disabled.")

render_page(page, models, BASE_DIR)
