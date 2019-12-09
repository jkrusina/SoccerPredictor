# Length of random name prefix for folders
FOLDER_PREFIX_LEN = 4
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = f"{DATE_FORMAT}T%H-%M-%S"
# Folders pattern: (prefix)_(TIMESTAMP_FORMAT)_(N_EPOCHS)
FOLDER_NAME_PATTERN = r"([a-zA-Z]+)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_(\d+)$"
# Columns names for bet on matches dataframe
BET_ON_MATCHES_COLS = ["id", "date", "home", "away", "bet_on_team", "pred_perc", "odds_wd", "bet_won"]
# Features which should be scaled
FEATURES_TO_SCALE = ["rating", "goals", "shots", "errors", "red_cards", "season", "odds_wd"]
# Features which should be encoded
FEATURES_TO_LENC = ["team", "future_opponent"]
TRAINING_METRICS = ["loss", "acc"]
VERBOSITY_LEVELS = range(0, 2)
# Whether to apply best threshold selection on predicted matches
APPLY_THRESHOLD_SELECTION = True

# Additional model settings:
# Force ConfigProto single threads settings
FORCE_SINGLE_THREADS = True
FORCE_SINGLE_CPU = True
# Whether should models use different initial weights
RANDOM_WEIGHTS = False
STATEFUL = True
BATCH_SIZE = 1
# How long to wait before tracking performance of models
TRACK_PERF_FROM_EPOCH = 10
# Number of samples to predict ahead
NPREDICT = 1
# Lr and dropout for team in test dataset
LR = 3e-4
DROPOUT = 0.0
# Lr and dropout for teams not in test dataset
NONTEST_LR = 0.01
NONTEST_DROPOUT = 0.0
# Min and max season used
MIN_SEASON = 13
MAX_SEASON = 17

# File names and dirs
DATA_DIR = "./data/"
ASSETS_DIR = "assets/"
MODEL_DIR = "models/"
DB_DIR = "db/"
IMG_DIR = "images/"
TRAIN_STATS_FILE = "train_stats.pickle"
TEST_STATS_FILE = "test_stats.pickle"
BEST_TRAIN_STATS_FILE = "best_train_stats.pickle"
BEST_TEST_STATS_FILE = "best_test_stats.pickle"
PREDICT_STATS_FILE = "prediction.pickle"
MODEL_SETTINGS_FILE = "model_settings.json"
DATA_FILE = "data.npy"
DB_FILE = "soccer.db"

# Tensorboard settings
TB_ROOT_DIR = "logs/"
TB_TRAIN_DIR = "train/"
TB_TEST_DIR = "test/"
TB_FLUSH_SECS = 60
TB_MAX_QUEUE = 100

# Basic features used
FEATURES_COMMON = [
    "team",
    "league",
    "season",
    "ashome",
    "rating",
    "goals",
    "shots",
    "errors",
    "red_cards",
    "future_ashome",
    "future_opponent"
]

# Features used specifically for current target variable
FEATURES_WD = [
    "wd",
    "odds_wd",
    "future_odds_wd",
]

# Columns which must be present in the dataset
REQUIRED_COLUMNS = [
    "id",
    "date",
    "season",
    "league",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "home_rating",
    "home_errors",
    "home_red_cards",
    "away_rating",
    "away_errors",
    "away_red_cards",
    "home_odds_wd",
    "away_odds_wd",
    "home_shots",
    "away_shots"
]

# Target columns which last values must be present in the dataset
REQUIRED_TARGET_COLUMNS = [
    "id",
    "date",
    "season",
    "league",
    "home",
    "away",
    "home_odds_wd",
    "away_odds_wd"
]
