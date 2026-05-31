import os

# ─── Base Paths ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_PATH        = os.path.join(BASE_DIR, "data", "raw", "train.csv")
DATA_PROCESSED_PATH  = os.path.join(BASE_DIR, "data", "processed", "cleaned.csv")
MODELS_PATH          = os.path.join(BASE_DIR, "models")

# ─── Column Names ─────────────────────────────────────────
TEXT_COLUMN      = "text"
LABEL_COLUMN     = "label"

# ─── Preprocessing ────────────────────────────────────────
STOPWORDS_LANG   = "english"
MIN_WORD_LENGTH  = 3

# ─── TF-IDF ───────────────────────────────────────────────
MAX_FEATURES     = 5000
NGRAM_RANGE      = (1, 2)

# ─── Model Training ───────────────────────────────────────
TEST_SIZE        = 0.2
RANDOM_STATE     = 42

# ─── Topic Modeling (LDA) ─────────────────────────────────
NUM_TOPICS       = 5
LDA_PASSES       = 10

# ─── HuggingFace ──────────────────────────────────────────
HF_MODEL         = "distilbert-base-uncased-finetuned-sst-2-english"
HF_SUMMARIZER    = "facebook/bart-large-cnn"

# ─── Dashboard ────────────────────────────────────────────
APP_TITLE        = "Customer Review Intelligence System"
APP_ICON         = "💡"