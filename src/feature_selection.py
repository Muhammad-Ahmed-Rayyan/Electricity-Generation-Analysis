import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Selects features based on correlation with the target."""

    def __init__(self, threshold: float = config.CORRELATION_THRESHOLD):
        self.threshold = threshold
        self.selected_features_ = []

    @staticmethod
    def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        return df

    def compute_correlation_with_target(self, df_encoded: pd.DataFrame) -> pd.Series:
        corr_matrix = df_encoded.corr(numeric_only=True)
        corr_with_target = corr_matrix[config.TARGET_COL].abs().sort_values(ascending=False)
        corr_with_target.to_csv(config.FEATURE_SELECTION_DIR / "correlation_with_target.csv")
        return corr_with_target

    def plot_correlation_bar(self, corr_series: pd.Series):
        plt.figure(figsize=(12, 7))
        corr_series.drop(labels=[config.TARGET_COL], errors="ignore").plot.bar()
        plt.title(f"Feature Correlation with {config.TARGET_COL}")
        plt.tight_layout()
        plt.savefig(config.FEATURE_SELECTION_DIR / "correlation_bar.png")
        plt.close()

    def select(self, df_encoded: pd.DataFrame) -> list:
        corr_series = self.compute_correlation_with_target(df_encoded)
        self.plot_correlation_bar(corr_series)

        candidates = corr_series.drop(labels=[config.TARGET_COL], errors="ignore")
        selected = candidates[candidates > self.threshold].index.tolist()

        self.selected_features_ = selected
        logger.info("Selected %d features (|corr| > %.3f): %s", len(selected), self.threshold, selected)
        return selected

    def run(self, df: pd.DataFrame):
        df_encoded = self.encode_categoricals(df)
        selected_features = self.select(df_encoded)
        return df_encoded, selected_features