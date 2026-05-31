import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    classification_report
)
from config.config import (
    MODELS_PATH,
    LABEL_COLUMN,
    TEXT_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)
from src.features import fit_transform_text, transform_text, load_vectorizer

# ─── Model Definitions ────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Naive Bayes"        : MultinomialNB(),
    "SVM"                : LinearSVC(random_state=RANDOM_STATE, max_iter=2000)
}


# ─── Train All Models ─────────────────────────────────────
def train_all(df: pd.DataFrame) -> dict:
    print("Vectorizing text...")
    vectorizer, X = fit_transform_text(df[TEXT_COLUMN])
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

    results = {}
    for name, model in MODELS.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "f1"       : round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall"   : round(recall_score(y_test, y_pred), 4),
            "report"   : classification_report(y_test, y_pred)
        }
        results[name] = metrics
        save_model(model, name)
        print(f"  Accuracy : {metrics['accuracy']}")
        print(f"  F1 Score : {metrics['f1']}\n")

    best = max(results, key=lambda x: results[x]["f1"])
    print(f"Best Model: {best} (F1: {results[best]['f1']}) 🏆")
    return results


# ─── Save & Load ──────────────────────────────────────────
def save_model(model, name: str):
    os.makedirs(MODELS_PATH, exist_ok=True)
    filename = name.lower().replace(" ", "_") + ".pkl"
    path = os.path.join(MODELS_PATH, filename)
    joblib.dump(model, path)
    print(f"  Saved → {path}")


def load_model(name: str):
    filename = name.lower().replace(" ", "_") + ".pkl"
    path = os.path.join(MODELS_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model '{name}' not found. Train first.")
    return joblib.load(path)


# ─── Predict Single Text ──────────────────────────────────
def predict(text: str, model_name: str = "SVM") -> dict:
    vectorizer = load_vectorizer()
    model = load_model(model_name)
    X = transform_text(vectorizer, pd.Series([text]))
    pred = model.predict(X)[0]
    return {
        "model"     : model_name,
        "prediction": "Positive ✅" if pred == 1 else "Negative ❌",
        "label"     : int(pred)
    }


# ─── Run Directly ─────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocess import load_processed_data
    df = load_processed_data()
    results = train_all(df)

    print("\n─── Final Results ───")
    for name, m in results.items():
        print(f"{name:25} | Acc: {m['accuracy']} | F1: {m['f1']}")