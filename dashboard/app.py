import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Customer Review Intelligence",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Sidebar Navigation ───────────────────────────────────
st.sidebar.title("Customer Review Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "EDA & Visuals",
        "Model Comparison",
        "Live Predictor",
        "Business Insights"
    ]
)

st.sidebar.markdown("---")

# ─── Route to Pages ───────────────────────────────────────
if page == "Overview":
    from pages.overview import show
    show()

elif page == "EDA & Visuals":
    from pages.eda import show
    show()

elif page == "Model Comparison":
    from pages.models import show
    show()

elif page == "Live Predictor":
    from pages.predictor import show
    show()

elif page == "Business Insights":
    from pages.insights import show
    show()