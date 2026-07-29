from sklearn.model_selection import train_test_split
from src import config
from src.utils import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """Standard train/test split (no grouping needed for this dataset)."""

    def __init__(self, test_size: float = config.TEST_SIZE, random_state: int = config.RANDOM_STATE):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df, feature_cols: list):
        X = df[feature_cols]
        y = df[config.TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        logger.info("Split -> train=%d rows, test=%d rows", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test
