import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocess import load_processed_data
from src.insights import get_text_length_stats
from config.config import LABEL_COLUMN, TEXT_COLUMN


def show():
    st.title("EDA & Visuals")
    st.markdown("Exploratory analysis of 100,000 Amazon reviews.")
    st.markdown("---")

    with st.spinner("Loading data..."):
        df = load_processed_data()

    # ─── Sentiment Distribution ───────────────────────────
    st.subheader("Sentiment Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df[LABEL_COLUMN].value_counts()
        ax.bar(
            ["Negative", "Positive"],
            [counts[0], counts[1]],
            color=["#e74c3c", "#2ecc71"]
        )
        ax.set_ylabel("Count")
        ax.set_title("Review Count by Sentiment")
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            [counts[0], counts[1]],
            labels=["Negative", "Positive"],
            colors=["#e74c3c", "#2ecc71"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.set_title("Sentiment Split")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ─── Word Length Distribution ─────────────────────────
    st.subheader("Review Length Distribution")
    df["word_count"] = df[TEXT_COLUMN].apply(lambda x: len(str(x).split()))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(
            df[df[LABEL_COLUMN] == 1]["word_count"],
            bins=30, color="#2ecc71", alpha=0.7, label="Positive"
        )
        ax.hist(
            df[df[LABEL_COLUMN] == 0]["word_count"],
            bins=30, color="#e74c3c", alpha=0.7, label="Negative"
        )
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Frequency")
        ax.set_title("Word Count Distribution")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        stats = get_text_length_stats(df)
        st.markdown("### Length Stats")
        st.metric("Average Words", stats["avg_words"])
        st.metric("Positive Reviews Avg", stats["positive_avg"])
        st.metric("Negative Reviews Avg", stats["negative_avg"])
        st.metric("Max Words", stats["max_words"])

    st.markdown("---")

    # ─── WordClouds ───────────────────────────────────────
    st.subheader("Word Clouds")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Positive Reviews**")
        pos_text = " ".join(df[df[LABEL_COLUMN] == 1][TEXT_COLUMN].sample(2000).tolist())
        wc = WordCloud(
            width=600, height=300,
            background_color="white",
            colormap="Greens",
            max_words=100
        ).generate(pos_text)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Negative Reviews**")
        neg_text = " ".join(df[df[LABEL_COLUMN] == 0][TEXT_COLUMN].sample(2000).tolist())
        wc = WordCloud(
            width=600, height=300,
            background_color="white",
            colormap="Reds",
            max_words=100
        ).generate(neg_text)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()