import os
import joblib
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from config.config import (
    MODELS_PATH,
    TEXT_COLUMN,
    NUM_TOPICS,
    LDA_PASSES,
    RANDOM_STATE
)

# ─── Paths ────────────────────────────────────────────────
LDA_MODEL_PATH      = os.path.join(MODELS_PATH, "lda_model")
DICTIONARY_PATH     = os.path.join(MODELS_PATH, "lda_dictionary.pkl")


# ─── Prepare Corpus ───────────────────────────────────────
def prepare_corpus(texts: pd.Series):
    tokenized = [simple_preprocess(str(doc)) for doc in texts]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]
    return corpus, dictionary, tokenized


# ─── Train LDA ────────────────────────────────────────────
def train_lda(texts: pd.Series) -> LdaModel:
    print("Preparing corpus...")
    corpus, dictionary, _ = prepare_corpus(texts)

    print(f"Training LDA with {NUM_TOPICS} topics...")
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=LDA_PASSES,
        random_state=RANDOM_STATE
    )

    os.makedirs(MODELS_PATH, exist_ok=True)
    lda.save(LDA_MODEL_PATH)
    joblib.dump(dictionary, DICTIONARY_PATH)
    print(f"LDA model saved ✅")
    return lda, dictionary


# ─── Load LDA ─────────────────────────────────────────────
def load_lda():
    if not os.path.exists(LDA_MODEL_PATH):
        raise FileNotFoundError("LDA model not found. Train first.")
    lda = LdaModel.load(LDA_MODEL_PATH)
    dictionary = joblib.load(DICTIONARY_PATH)
    return lda, dictionary


# ─── Get Topics ───────────────────────────────────────────
def get_topics(lda: LdaModel, num_words: int = 8) -> list:
    topics = []
    for idx, topic in lda.show_topics(
        num_topics=NUM_TOPICS,
        num_words=num_words,
        formatted=False
    ):
        words = [word for word, _ in topic]
        topics.append({
            "topic_id": idx,
            "words"   : words,
            "label"   : f"Topic {idx + 1}"
        })
    return topics


# ─── Assign Topic to Each Review ──────────────────────────
def assign_topics(texts: pd.Series, lda: LdaModel, dictionary) -> pd.Series:
    def get_dominant_topic(text):
        bow = dictionary.doc2bow(simple_preprocess(str(text)))
        topics = lda.get_document_topics(bow)
        if not topics:
            return -1
        return max(topics, key=lambda x: x[1])[0]

    return texts.apply(get_dominant_topic)


# ─── Run Directly ─────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocess import load_processed_data

    df = load_processed_data()

    # Train on sample for speed
    sample = df.sample(20000, random_state=42)
    lda, dictionary = train_lda(sample[TEXT_COLUMN])

    print("\n─── Discovered Topics ───")
    topics = get_topics(lda)
    for t in topics:
        print(f"Topic {t['topic_id'] + 1}: {', '.join(t['words'])}")