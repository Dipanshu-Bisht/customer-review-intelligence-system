import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocess import load_processed_data
from src.insights import (
    get_sentiment_stats,
    get_health_label,
    get_recommendations,
    get_top_negative_reviews,
    get_top_positive_reviews,
    get_text_length_stats
)
from src.topic_modeling import load_lda, get_topics, assign_topics
from src.sentiment import summarize_reviews
from config.config import LABEL_COLUMN, TEXT_COLUMN, NUM_TOPICS


@st.cache_data
def load_data_with_topics():
    df = load_processed_data()
    lda, dictionary = load_lda()
    df["topic"] = assign_topics(df[TEXT_COLUMN], lda, dictionary)
    return df, get_topics(lda)


def show():
    st.title("Business Insights")
    st.markdown("Actionable intelligence derived from 100,000 Amazon reviews.")
    st.markdown("---")

    with st.spinner("Loading insights..."):
        df, topics = load_data_with_topics()
        stats = get_sentiment_stats(df)

    # ─── Health Score ─────────────────────────────────────
    st.subheader("Overall Business Health")
    col1, col2, col3 = st.columns(3)
    col1.metric("Health Score", f"{stats['health_score']}%")
    col2.metric("Positive Reviews", f"{stats['positive']:,}")
    col3.metric("Negative Reviews", f"{stats['negative']:,}")

    label = get_health_label(stats["health_score"])
    st.info(f"Sentiment Health: {label}")

    st.markdown("---")

    # ─── Topic wise Breakdown ─────────────────────────────
    st.subheader("Topic-wise Sentiment Breakdown")

    topic_data = []
    for t in topics:
        tid = t["topic_id"]
        topic_df = df[df["topic"] == tid]
        if len(topic_df) == 0:
            continue
        pos = int((topic_df[LABEL_COLUMN] == 1).sum())
        neg = int((topic_df[LABEL_COLUMN] == 0).sum())
        total = len(topic_df)
        topic_data.append({
            "Topic"        : f"Topic {tid+1}: {', '.join(t['words'][:3])}",
            "Total"        : total,
            "Positive"     : pos,
            "Negative"     : neg,
            "Positive %"   : round((pos/total)*100, 1),
            "Health"       : get_health_label(round((pos/total)*100, 1))
        })

    topic_df_display = pd.DataFrame(topic_data)
    st.dataframe(topic_df_display, use_container_width=True)

    # Topic health bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ecc71" if p >= 50 else "#e74c3c"
              for p in topic_df_display["Positive %"]]
    ax.barh(topic_df_display["Topic"],
            topic_df_display["Positive %"],
            color=colors)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Positive %")
    ax.set_title("Sentiment Health by Topic")
    ax.set_xlim(0, 100)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ─── AI Generated Summary ─────────────────────────────
    st.subheader("AI-Generated Review Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Negative Themes**")
        with st.spinner("Summarizing negative reviews..."):
            neg_summary = summarize_reviews(
                df[df[LABEL_COLUMN] == 0][TEXT_COLUMN], max_reviews=50
            )
        st.error(neg_summary)

    with col2:
        st.markdown("**Top Positive Themes**")
        with st.spinner("Summarizing positive reviews..."):
            pos_summary = summarize_reviews(
                df[df[LABEL_COLUMN] == 1][TEXT_COLUMN], max_reviews=50
            )
        st.success(pos_summary)

    st.markdown("---")

    # ─── Recommendations ──────────────────────────────────
    st.subheader("Business Recommendations")
    recs = get_recommendations(stats)
    for r in recs:
        st.info(r)

    st.markdown("---")

    # ─── Top Reviews ──────────────────────────────────────
    st.subheader("Most Detailed Reviews")
    tab1, tab2 = st.tabs(["Top Negative Reviews", "Top Positive Reviews"])

    with tab1:
        top_neg = get_top_negative_reviews(df, n=5)
        for i, row in top_neg.iterrows():
            with st.expander(f"Review — {row['text_length']} words"):
                st.write(row[TEXT_COLUMN])

    with tab2:
        top_pos = get_top_positive_reviews(df, n=5)
        for i, row in top_pos.iterrows():
            with st.expander(f"Review — {row['text_length']} words"):
                st.write(row[TEXT_COLUMN])