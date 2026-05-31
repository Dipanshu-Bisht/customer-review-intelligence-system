import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocess import load_processed_data
from src.models import train_all
from config.config import LABEL_COLUMN, TEXT_COLUMN


# ─── Cache so it doesn't retrain on every page visit ──────
@st.cache_data
def get_results():
    df = load_processed_data()
    return train_all(df)


def show():
    st.title("Model Comparison")
    st.markdown("Training and comparing 3 ML classifiers on 100,000 Amazon reviews.")
    st.markdown("---")

    with st.spinner("Loading models... this may take a minute on first visit."):
        results = get_results()

    names      = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]
    f1s        = [results[n]["f1"] for n in names]
    precisions = [results[n]["precision"] for n in names]
    recalls    = [results[n]["recall"] for n in names]

    # ─── Metric Cards ─────────────────────────────────────
    st.subheader("Model Performance Summary")
    cols = st.columns(3)
    for i, name in enumerate(names):
        with cols[i]:
            st.metric("Model", name)
            st.metric("Accuracy", f"{results[name]['accuracy']*100:.2f}%")
            st.metric("F1 Score", f"{results[name]['f1']*100:.2f}%")
            st.metric("Precision", f"{results[name]['precision']*100:.2f}%")
            st.metric("Recall", f"{results[name]['recall']*100:.2f}%")

    st.markdown("---")

    # ─── Bar Chart Comparison ─────────────────────────────
    st.subheader("Metrics Comparison Chart")
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*width, accuracies, width, label="Accuracy",  color="#3498db")
    ax.bar(x - 0.5*width, f1s,        width, label="F1 Score",  color="#2ecc71")
    ax.bar(x + 0.5*width, precisions, width, label="Precision", color="#e67e22")
    ax.bar(x + 1.5*width, recalls,    width, label="Recall",    color="#9b59b6")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.8, 0.95)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ─── Best Model Highlight ─────────────────────────────
    best = max(results, key=lambda x: results[x]["f1"])
    st.subheader("Best Model")
    st.success(
        f"{best} — F1 Score: {results[best]['f1']*100:.2f}% | "
        f"Accuracy: {results[best]['accuracy']*100:.2f}%"
    )

    st.markdown("---")

    # ─── Classification Reports ───────────────────────────
    st.subheader("Detailed Classification Reports")
    for name in names:
        with st.expander(f"{name} — Full Report"):
            st.code(results[name]["report"])