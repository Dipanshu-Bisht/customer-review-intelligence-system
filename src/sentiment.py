import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from config.config import HF_MODEL

# ─── Load Pipelines (lazy loading) ────────────────────────
_sentiment_pipeline = None


def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("Loading DistilBERT sentiment model...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=HF_MODEL,
            truncation=True,
            max_length=512
        )
    return _sentiment_pipeline


# ─── Predict Single Text ──────────────────────────────────
def predict_sentiment(text: str) -> dict:
    pipe = get_sentiment_pipeline()
    result = pipe(text[:512])[0]
    return {
        "label"     : result["label"],
        "confidence": round(result["score"], 4),
        "sentiment" : "Positive ✅" if result["label"] == "POSITIVE" else "Negative ❌"
    }


# ─── Predict Batch ────────────────────────────────────────
def predict_batch(texts: pd.Series, batch_size: int = 32) -> pd.DataFrame:
    pipe = get_sentiment_pipeline()
    truncated = [str(t)[:512] for t in texts]
    results = pipe(truncated, batch_size=batch_size)
    return pd.DataFrame([{
        "text"      : truncated[i],
        "label"     : r["label"],
        "confidence": round(r["score"], 4)
    } for i, r in enumerate(results)])


# ─── Extractive Summarizer ────────────────────────────────
def summarize_reviews(texts: pd.Series, max_reviews: int = 50) -> str:
    sample = texts.dropna().sample(
        min(max_reviews, len(texts)), random_state=42
    ).tolist()

    if len(sample) < 3:
        return "Not enough reviews to summarize."

    vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sample)
    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    top_indices = scores.argsort()[-3:][::-1]
    top_reviews = [sample[i] for i in top_indices]

    summary = " | ".join([r[:100] for r in top_reviews])
    return f"Key insights: {summary}"


# ─── Run Directly ─────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocess import load_processed_data

    print("─── Single Prediction ───")
    test_texts = [
        "This product is absolutely amazing, I love it!",
        "Terrible quality, broke after one day. Total waste of money.",
        "It's okay, nothing special but gets the job done."
    ]
    for text in test_texts:
        result = predict_sentiment(text)
        print(f"Text      : {text[:60]}")
        print(f"Sentiment : {result['sentiment']}")
        print(f"Confidence: {result['confidence']}\n")

    print("─── Review Summarizer ───")
    df = load_processed_data()
    negative_reviews = df[df["label"] == 0]["text"].head(30)
    summary = summarize_reviews(negative_reviews)
    print(f"Summary:\n{summary}")