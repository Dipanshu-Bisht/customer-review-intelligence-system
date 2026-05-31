import os
import joblib
import pandas as pd    
from src.preprocess import load_processed_data

from sklearn.feature_extraction.text import TfidfVectorizer
from config.config import (
    MODELS_PATH,
    TEXT_COLUMN,
    MAX_FEATURES,
    NGRAM_RANGE
)

# ─── Path for saved vectorizer ────────────────────────────
VECTORIZER_PATH = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")


# ─── Build & Fit Vectorizer ───────────────────────────────
def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True       # Reduces impact of very frequent words
    )


def fit_vectorizer(texts: pd.Series) -> TfidfVectorizer:
    vectorizer = build_vectorizer()
    vectorizer.fit(texts)
    save_vectorizer(vectorizer)
    print(f"Vectorizer fitted on {len(texts)} samples ✅")
    return vectorizer


# ─── Transform Text ───────────────────────────────────────
def transform_text(vectorizer: TfidfVectorizer, texts: pd.Series):
    return vectorizer.transform(texts)


def fit_transform_text(texts: pd.Series):
    vectorizer = fit_vectorizer(texts)
    return vectorizer, transform_text(vectorizer, texts)


# ─── Save & Load Vectorizer ───────────────────────────────
def save_vectorizer(vectorizer: TfidfVectorizer):
    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")


def load_vectorizer() -> TfidfVectorizer:
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Vectorizer not found. Run models.py first.")
    return joblib.load(VECTORIZER_PATH)


# ─── Test Directly ────────────────────────────────────────
if __name__ == "__main__":
    df = load_processed_data()

    print("Fitting TF-IDF vectorizer...")
    vectorizer, X = fit_transform_text(df[TEXT_COLUMN])

    print(f"Matrix shape     : {X.shape}")
    print(f"Sample features  : {vectorizer.get_feature_names_out()[:10]}")