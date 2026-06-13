# Customer Review Intelligence System

An end-to-end NLP pipeline that analyzes 100,000 Amazon reviews using machine learning,
LDA topic modeling, and DistilBERT transformer — deployed as an interactive 5-page Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What This Project Does

Most sentiment tools just say "positive" or "negative." This system goes further — it tells you **why** customers feel the way they do, which product categories have the most complaints, and what actions a business should take.

Built as a learning project to understand NLP pipelines end-to-end — from raw text cleaning all the way to a live interactive dashboard.

---

## Results

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| **Logistic Regression** | 88.77% | **89.21%** | 88.79% | 89.64% |
| SVM | 88.52% | 88.93% | 88.83% | 89.03% |
| Naive Bayes | 85.94% | 86.53% | 85.94% | 87.12% |

**Best Model: Logistic Regression (F1: 89.21%)**

DistilBERT Transformer confidence on clear sentiment: **99.99%**

---

## Dashboard Pages

| Page | What It Shows |
|---|---|
| Overview | Health score, total reviews, sentiment stats, pipeline status |
| EDA & Visuals | Bar chart, pie chart, word count distribution, WordClouds |
| Model Comparison | Metrics comparison chart, classification reports for all 3 models |
| Live Predictor | Real-time prediction — ML model + DistilBERT side by side |
| Business Insights | Topic-wise breakdown, AI-generated summaries, top reviews |

---

## Tech Stack

| Area | Tools Used |
|---|---|
| Language | Python 3.12 |
| Data Handling | Pandas, NumPy |
| Text Processing | NLTK, Regex, WordCloud |
| ML Models | Scikit-learn — Logistic Regression, Naive Bayes, LinearSVC |
| Feature Engineering | TF-IDF Vectorizer with Bigrams |
| Topic Modeling | Gensim LDA |
| Deep Learning | HuggingFace Transformers, DistilBERT, PyTorch |
| Dashboard | Streamlit |
| Model Persistence | Joblib (.pkl files) |
| Version Control | Git, GitHub |

---

## Project Structure

```
customer-review-intelligence-system/
│
├── config/
│   └── config.py              # All settings, paths, and constants
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # Text cleaning pipeline
│   ├── features.py            # TF-IDF vectorization
│   ├── models.py              # Train, evaluate, save all 3 models
│   ├── topic_modeling.py      # LDA topic extraction
│   ├── sentiment.py           # HuggingFace DistilBERT wrapper
│   └── insights.py            # Business metrics and recommendations
│
├── dashboard/
│   ├── app.py                 # Main Streamlit entry point
│   └── pages/
│       ├── overview.py        # Overview page
│       ├── eda.py             # EDA & Visuals page
│       ├── models.py          # Model Comparison page
│       ├── predictor.py       # Live Predictor page
│       └── insights.py        # Business Insights page
│
├── data/
│   ├── raw/                   # Place train.csv here (see Dataset section)
│   ├── processed/             # Auto-generated after preprocessing
│   └── README.md              # Dataset download instructions
│
├── notebooks/
│   └── exploration.ipynb      # Data exploration before building pipeline
│
├── models/                    # Auto-generated after running run.py
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   └── lda_model
│
├── requirements.txt
├── requirements-torch.txt
├── run.py                     # One command to train all models
└── README.md
```

---

## Dataset

This project uses the **Amazon Reviews** dataset from Kaggle.

### Download Instructions
1. Go to: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
2. Login to Kaggle and click **Download**
3. Extract the zip file
4. Place `train.csv` inside `data/raw/` folder

### Dataset Info
- Total reviews: 3.6 million
- Columns: label (1=negative, 2=positive), title, review text
- This project uses **100,000 rows** for development

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Dipanshu-Bisht/customer-review-intelligence-system.git
cd customer-review-intelligence-system
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-torch.txt
```

### 4. Add dataset
Download `train.csv` from Kaggle link above and place in `data/raw/`

### 5. Train all models
```bash
python run.py
```
This will preprocess data, train all 3 ML models, and save them to `models/`

### 6. Launch dashboard
```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

---

## Key Learnings

- **TF-IDF with bigrams** captures phrases like "not good" better than simple Bag of Words
- **Logistic Regression** outperforms SVM and Naive Bayes on high-dimensional TF-IDF features
- **LDA** discovered Amazon's product categories (music, games, books, movies) without any predefined labels — purely from text patterns
- **Negative reviews average 40.7 words** vs 38.7 for positive — unhappy customers write more detailed complaints
- **DistilBERT** achieves 99.99% confidence on clear reviews — pre-trained models are far more powerful for individual predictions
- **Modular code** makes debugging and reuse much easier than single-file scripts

---

## Folder Design Philosophy

Each file in `src/` does exactly one thing:

| File | Single Responsibility |
|---|---|
| `preprocess.py` | Clean raw text only |
| `features.py` | TF-IDF vectorization only |
| `models.py` | Train, evaluate, save models only |
| `topic_modeling.py` | LDA topic extraction only |
| `sentiment.py` | HuggingFace inference only |
| `insights.py` | Business metrics only |

This makes each file easy to test, modify, and reuse independently.

---

## Author

**Dipanshu Bisht**
- GitHub: https://github.com/Dipanshu-Bisht
- Email: dipanshu.bisht01@gmail.com
- LinkedIn: https://www.linkedin.com/in/dipanshubisht23


---

## License

This project is open source and available under the [MIT License](LICENSE).
