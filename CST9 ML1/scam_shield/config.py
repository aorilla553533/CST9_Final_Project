"""
Scam Shield configuration constants.

Centralizes paths, feature lists, encoding maps, and hyperparameter grids so the
training, evaluation, and figure-generation modules stay consistent with the
paper "Scam Shield: Predicting User Defense Outcomes in Scam Conversations
Using Random Forest Classification".
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figures"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

DATASET_FILE = DATA_DIR / "unified_scam_dataset.csv"
CLEANED_FILE = DATA_DIR / "cleaned_dataset.csv"
ENGINEERED_FILE = DATA_DIR / "engineered_features.csv"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Target and leakage columns (per paper, Section "Feature Selection")
# ---------------------------------------------------------------------------
TARGET_COLUMN = "user_success"
LEAKAGE_COLUMNS = ["is_scam", "scam_category", "effectiveness_score"]

# Conversation identifier and timestamp - tracking IDs, not generalizable features
DROP_COLUMNS = ["conversation_id", "timestamp"]

# ---------------------------------------------------------------------------
# Categorical encodings (paper: "ordinal encoding on a ranked scale")
# ---------------------------------------------------------------------------
# response_type ranks countermeasure strength. Higher rank = stronger defense.
RESPONSE_TYPE_ORDER = [
    "ignored",                # user ignores the message
    "engaged",        # user replies but complies
    "delayed",       # user asks for time
    "rejected",                # user explicitly rejects
    "questioned",              # user asks clarifying questions
    "expert_mention",          # mentions awareness of fraud
    "verification",            # user verifies with a third party (bank, etc.)
    "refused",                 # outright refuses / reports
]

RISK_LEVEL_ORDER = ["low", "medium", "high", "critical"]

# ---------------------------------------------------------------------------
# Feature engineering keyword sets
# ---------------------------------------------------------------------------
VERIFICATION_KEYWORDS = [
    "call the bank", "call my bank", "call my bank", "verify", "confirm with",
    "official branch", "official site", "official website", "official number",
    "in person", "branch", "customer service", "official app", "log in to",
    "log into my", "logged into my", "check my account", "check directly",
    "official hotline", "official channel",
]

REFUSAL_KEYWORDS = [
    "no thank you", "i refuse", "i will not", "i won't", "i wont",
    "do not send", "stop contacting", "report", "spam", "scam",
    "fraud", "phishing", "block", "i am not interested",
    "i'm not interested", "no thanks", "this is a scam",
    "you are reporting fraud",
]

URGENCY_KEYWORDS = [
    "urgent", "immediately", "now", "asap", "right away", "today",
    "expires", "expire", "deadline", "limited time", "act now",
    "suspend", "suspended", "locked", "lock", "final notice",
]

AUTHORITY_KEYWORDS = [
    "bank", "irs", "government", "police", "fbi", "tax", "court",
    "lawyer", "official", "agent", "officer", "authority",
    "federal", "treasury", "support team",
]

THREAT_KEYWORDS = [
    "arrest", "lawsuit", "legal action", "fine", "penalty",
    "consequences", "shut down", "close your account", "freeze",
    "warrant", "prosecute", "jail", "prison",
]

FEE_KEYWORDS = [
    "fee", "payment", "pay", "$", "usd", "wire", "transfer",
    "gift card", "bitcoin", "crypto", "deposit", "tax", "fine",
]

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
TEST_SIZE = 0.30

# Hyperparameter grid for the Balanced Random Forest
PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [10, 15, 20, None],
}

CV_FOLDS = 5

# ---------------------------------------------------------------------------
# TF-IDF settings
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_SUBLINEAR_TF = True