import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import predict, load_model
from src.sentiment import predict_sentiment
from src.features import load_vectorizer
from config.config import TEXT_COLUMN


def show():
    st.title("Live Predictor")
    st.markdown("Type any review and get instant sentiment prediction from both ML model and DistilBERT transformer.")
    st.markdown("---")

    # ─── Input ────────────────────────────────────────────
    st.subheader("Enter a Review")
    user_input = st.text_area(
        "Review Text",
        placeholder="e.g. This product is absolutely amazing, highly recommend it!",
        height=150
    )

    model_choice = st.selectbox(
        "Choose ML Model",
        ["Logistic Regression", "Naive Bayes", "SVM"]
    )

    predict_btn = st.button("Analyze Sentiment")

    st.markdown("---")

    # ─── Prediction ───────────────────────────────────────
    if predict_btn:
        if not user_input.strip():
            st.warning("Please enter a review first.")
            return

        col1, col2 = st.columns(2)

        # ML Model Prediction
        with col1:
            st.subheader(f"ML Model — {model_choice}")
            with st.spinner("Predicting..."):
                try:
                    ml_result = predict(user_input, model_choice)
                    if ml_result["label"] == 1:
                        st.success(f"Positive Review")
                    else:
                        st.error(f"Negative Review")
                    st.markdown(f"**Prediction:** {ml_result['prediction']}")
                except Exception as e:
                    st.error(f"ML Model error: {e}")

        # DistilBERT Prediction
        with col2:
            st.subheader("DistilBERT Transformer")
            with st.spinner("Running transformer..."):
                try:
                    bert_result = predict_sentiment(user_input)
                    if bert_result["label"] == "POSITIVE":
                        st.success("Positive Review")
                    else:
                        st.error("Negative Review")
                    st.markdown(f"**Prediction:** {bert_result['sentiment']}")
                    st.markdown(f"**Confidence:** {bert_result['confidence']*100:.2f}%")
                except Exception as e:
                    st.error(f"Transformer error: {e}")

        st.markdown("---")

        # ─── Agreement Check ──────────────────────────────
        st.subheader("Model Agreement")
        try:
            ml_label = ml_result["label"]
            bert_label = 1 if bert_result["label"] == "POSITIVE" else 0

            if ml_label == bert_label:
                st.success("Both models agree on the prediction.")
            else:
                st.warning("Models disagree — the review may be ambiguous or mixed sentiment.")
        except:
            pass

        # ─── Sample Reviews ───────────────────────────────
    st.markdown("---")
    st.subheader("Try These Sample Reviews")

    samples = [
        "Absolutely fantastic product, exceeded all my expectations!",
        "Terrible quality, broke after one day. Complete waste of money.",
        "It is okay, nothing special but does the job.",
        "Best purchase I have made this year, highly recommend!",
        "Very disappointed, the description was completely misleading."
    ]

    for sample in samples:
        if st.button(sample[:60], key=sample):
            st.session_state["sample_text"] = sample
            st.info(f"Copied: {sample}")