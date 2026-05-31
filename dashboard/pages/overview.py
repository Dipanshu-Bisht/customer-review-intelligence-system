import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocess import load_processed_data
from src.insights import get_sentiment_stats, get_health_label, get_recommendations


def show():
    st.title("Customer Review Intelligence System")
    st.markdown("End-to-end NLP pipeline analyzing 100,000 Amazon reviews.")
    st.markdown("---")

    # Load data
    with st.spinner("Loading data..."):
        df = load_processed_data()
        stats = get_sentiment_stats(df)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{stats['total']:,}")
    col2.metric("Positive Reviews", f"{stats['positive']:,}")
    col3.metric("Negative Reviews", f"{stats['negative']:,}")
    col4.metric("Health Score", f"{stats['health_score']}%")

    st.markdown("---")

    # Health label
    label = get_health_label(stats["health_score"])
    st.subheader(f"Overall Sentiment Health: {label}")

    # Recommendations
    st.subheader("Business Recommendations")
    recs = get_recommendations(stats)
    for r in recs:
        st.info(r)

    # Project info
    st.markdown("---")
    st.subheader("Project Pipeline")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.success("Data Preprocessing")
    col2.success("TF-IDF Features")
    col3.success("ML Models (3)")
    col4.success("Topic Modeling")
    col5.success("Sentiment Analysis")