import pandas as pd
import numpy as np
from config.config import LABEL_COLUMN, TEXT_COLUMN, NUM_TOPICS


# ─── Overall Sentiment Stats ──────────────────────────────
def get_sentiment_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    positive = int((df[LABEL_COLUMN] == 1).sum())
    negative = int((df[LABEL_COLUMN] == 0).sum())
    score = round((positive / total) * 100, 2)

    return {
        "total"         : total,
        "positive"      : positive,
        "negative"      : negative,
        "positive_pct"  : round((positive / total) * 100, 2),
        "negative_pct"  : round((negative / total) * 100, 2),
        "health_score"  : score
    }


# ─── Health Score Label ───────────────────────────────────
def get_health_label(score: float) -> str:
    if score >= 75:
        return "🟢 Healthy"
    elif score >= 50:
        return "🟡 Moderate"
    else:
        return "🔴 Critical"


# ─── Top Negative Reviews ─────────────────────────────────
def get_top_negative_reviews(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    negative = df[df[LABEL_COLUMN] == 0].copy()
    negative["text_length"] = negative[TEXT_COLUMN].apply(len)
    # Longer negative reviews = more detailed complaints
    return negative.nlargest(n, "text_length")[[TEXT_COLUMN, "text_length"]]


# ─── Top Positive Reviews ─────────────────────────────────
def get_top_positive_reviews(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    positive = df[df[LABEL_COLUMN] == 1].copy()
    positive["text_length"] = positive[TEXT_COLUMN].apply(len)
    return positive.nlargest(n, "text_length")[[TEXT_COLUMN, "text_length"]]


# ─── Topic wise Sentiment Breakdown ───────────────────────
def get_topic_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if "topic" not in df.columns:
        return pd.DataFrame()

    topic_stats = []
    for topic_id in range(NUM_TOPICS):
        topic_df = df[df["topic"] == topic_id]
        if len(topic_df) == 0:
            continue
        pos = int((topic_df[LABEL_COLUMN] == 1).sum())
        neg = int((topic_df[LABEL_COLUMN] == 0).sum())
        total = len(topic_df)
        topic_stats.append({
            "topic"        : f"Topic {topic_id + 1}",
            "total"        : total,
            "positive"     : pos,
            "negative"     : neg,
            "positive_pct" : round((pos / total) * 100, 2),
            "health_score" : round((pos / total) * 100, 2)
        })

    return pd.DataFrame(topic_stats).sort_values("health_score")


# ─── Word Length Distribution ─────────────────────────────
def get_text_length_stats(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["word_count"] = df[TEXT_COLUMN].apply(lambda x: len(str(x).split()))
    return {
        "avg_words"   : round(df["word_count"].mean(), 1),
        "max_words"   : int(df["word_count"].max()),
        "min_words"   : int(df["word_count"].min()),
        "positive_avg": round(df[df[LABEL_COLUMN] == 1]["word_count"].mean(), 1),
        "negative_avg": round(df[df[LABEL_COLUMN] == 0]["word_count"].mean(), 1)
    }


# ─── Actionable Business Recommendations ─────────────────
def get_recommendations(stats: dict) -> list:
    recommendations = []
    score = stats["health_score"]

    if score < 50:
        recommendations.append("🚨 Critical: Over half of reviews are negative — immediate product/service review needed.")
    elif score < 75:
        recommendations.append("⚠️ Warning: Significant negative sentiment detected — investigate top complaint topics.")
    else:
        recommendations.append("✅ Positive: Customer sentiment is healthy — maintain current quality standards.")

    if stats["negative"] > stats["positive"] * 0.3:
        recommendations.append("📦 Action: High negative volume suggests systemic issues — prioritize root cause analysis.")

    recommendations.append(f"📊 Insight: {stats['positive_pct']}% positive vs {stats['negative_pct']}% negative reviews.")
    return recommendations


# ─── Run Directly ─────────────────────────────────────────
if __name__ == "__main__":
    from src.preprocess import load_processed_data

    df = load_processed_data()

    print("─── Sentiment Stats ───")
    stats = get_sentiment_stats(df)
    for k, v in stats.items():
        print(f"{k:20}: {v}")

    print(f"\nHealth Label: {get_health_label(stats['health_score'])}")

    print("\n─── Text Length Stats ───")
    length_stats = get_text_length_stats(df)
    for k, v in length_stats.items():
        print(f"{k:20}: {v}")

    print("\n─── Recommendations ───")
    recs = get_recommendations(stats)
    for r in recs:
        print(r)

    print("\n─── Top 3 Negative Reviews ───")
    top_neg = get_top_negative_reviews(df, n=3)
    for i, row in top_neg.iterrows():
        print(f"\n• {row[TEXT_COLUMN][:150]}...")