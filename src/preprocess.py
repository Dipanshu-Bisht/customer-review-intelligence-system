import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config.config import (
    DATA_RAW_PATH,
    DATA_PROCESSED_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
    STOPWORDS_LANG,
    MIN_WORD_LENGTH
)

# ─── Download NLTK Data (runs once) ───────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ─── Init ─────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words(STOPWORDS_LANG))


# ─── Individual Cleaning Steps ────────────────────────────
def remove_html(text: str) -> str:
    return re.sub(r"<.*?>", " ", text)

def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\S+", " ", text)

def remove_special_chars(text: str) -> str:
    return re.sub(r"[^a-zA-Z\s]", " ", text)

def to_lowercase(text: str) -> str:
    return text.lower()

def remove_stopwords(text: str) -> str:
    words = text.split()
    return " ".join(
        w for w in words
        if w not in stop_words and len(w) >= MIN_WORD_LENGTH
    )

def lemmatize(text: str) -> str:
    words = text.split()
    return " ".join(lemmatizer.lemmatize(w) for w in words)


# ─── Full Pipeline ────────────────────────────────────────
def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = to_lowercase(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text.strip()


# ─── Load & Process Dataset ───────────────────────────────
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW_PATH, header=None, names=[LABEL_COLUMN, "title", TEXT_COLUMN], nrows=100000)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    # Drop header row if it got read as data
    df = df[df[LABEL_COLUMN].astype(str).str.strip().isin(["1", "2"])]
    # Combine title + review text for richer NLP
    df[TEXT_COLUMN] = df["title"].astype(str) + " " + df[TEXT_COLUMN].astype(str)
    df = df[[LABEL_COLUMN, TEXT_COLUMN]]
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(lambda x: 1 if x == 2 else 0)
    return df

def preprocess_and_save() -> pd.DataFrame:
    print("Loading raw data...")
    df = load_raw_data()
    print(f"Total reviews loaded: {len(df)}")

    print("Cleaning text...")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).apply(clean_text)

    # Remove empty rows after cleaning
    df = df[df[TEXT_COLUMN].str.strip() != ""]

    print(f"Saving processed data to: {DATA_PROCESSED_PATH}")
    df.to_csv(DATA_PROCESSED_PATH, index=False)
    print("Done! ✅")
    return df


def load_processed_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED_PATH)


# ─── Run directly to test ─────────────────────────────────
if __name__ == "__main__":
    df = preprocess_and_save()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nLabel distribution:\n{df[LABEL_COLUMN].value_counts()}") 