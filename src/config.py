from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

OUTPUT_DIR = ROOT_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
MISSING_VALUES_DIR = OUTPUT_DIR / "missing_values"
FEATURE_SELECTION_DIR = OUTPUT_DIR / "feature_selection"
MODEL_TRAINING_DIR = OUTPUT_DIR / "model_training"
MODEL_COMPARISON_DIR = OUTPUT_DIR / "model_comparison"

ALL_DIRS = [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, EDA_DIR,
            MISSING_VALUES_DIR, FEATURE_SELECTION_DIR, MODEL_TRAINING_DIR,
            MODEL_COMPARISON_DIR]

RAW_FILE = DATA_RAW_DIR / "global_power_plant_database.csv"

TARGET_COL = "estimated_generation_gwh"

NUMERIC_FEATURE_COLS = ["capacity_mw", "latitude", "longitude", "commissioning_year"]
CATEGORICAL_FEATURE_COLS = ["fuel1", "country_long"]

DROP_COLS = [
    "name", "gppd_idnr", "owner", "source", "url", "geolocation_source",
    "fuel2", "fuel3", "fuel4", "country",
    "generation_gwh_2013", "generation_gwh_2014", "generation_gwh_2015",
    "generation_gwh_2016", "year_of_capacity_data",
]

RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.02
TEST_SIZE = 0.2

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
GB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "min_samples_split": [2, 5],
}
RIDGE_PARAM_GRID = {"alpha": [0.01, 0.1, 1, 10, 100]}