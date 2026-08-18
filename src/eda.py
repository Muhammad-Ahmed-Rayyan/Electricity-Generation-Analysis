import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import norm  # noqa: E402

from src import config  # noqa: E402
from src.utils import get_logger  # noqa: E402

import missingno as msno

logger = get_logger(__name__)


class EDAAnalyzer:
    """Generates and saves exploratory plots and summary statistics."""

    def __init__(self, output_dir=config.EDA_DIR):
        self.output_dir = output_dir

    def _save(self, filename: str):
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = df.describe()
        summary.to_csv(self.output_dir / "summary_statistics.csv")
        return summary

    def plot_target_distribution(self, df: pd.DataFrame):
        plt.figure(figsize=(8, 5))
        plt.hist(df[config.TARGET_COL], bins=50, density=True, color="skyblue", edgecolor="black")
        mu, std = df[config.TARGET_COL].mean(), df[config.TARGET_COL].std()
        x = np.linspace(df[config.TARGET_COL].min(), df[config.TARGET_COL].max(), 100)
        plt.plot(x, norm.pdf(x, mu, std), color="red", lw=2, label="Normal curve")
        plt.xlabel(config.TARGET_COL)
        plt.ylabel("Density")
        plt.title("Distribution of Estimated Generation (GWh)")
        plt.legend()
        self._save("target_distribution.png")

    def plot_categorical_counts(self, df: pd.DataFrame, column: str, filename: str, top_n: int = 15):
        if column not in df.columns:
            logger.warning("Column '%s' not found, skipping", column)
            return
        plt.figure(figsize=(12, 6))
        top_categories = df[column].value_counts().nlargest(top_n).index
        sns.countplot(x=column, data=df[df[column].isin(top_categories)], order=top_categories)
        plt.title(f"Count of {column} (top {top_n})")
        plt.xticks(rotation=45, ha="right")
        self._save(filename)

    def plot_correlation_heatmap(self, df: pd.DataFrame, filename: str):
        numeric_df = df[config.NUMERIC_FEATURE_COLS + [config.TARGET_COL]].select_dtypes(include=np.number)
        plt.figure(figsize=(9, 7))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap — Numeric Features")
        self._save(filename)

    def plot_capacity_vs_generation(self, df: pd.DataFrame, filename: str):
        plt.figure(figsize=(9, 6))
        sns.scatterplot(x="capacity_mw", y=config.TARGET_COL, data=df, alpha=0.5)
        plt.title("Capacity (MW) vs Estimated Generation (GWh)")
        self._save(filename)

    def plot_boxplot(self, df: pd.DataFrame, columns: list, filename: str, title: str):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[columns])
        plt.title(title)
        self._save(filename)

    def run(self, df: pd.DataFrame):
        logger.info("Running EDA suite -> %s", self.output_dir)
        self.summary_statistics(df)
        self.plot_target_distribution(df)
        self.plot_categorical_counts(df, "fuel1", "count_fuel1.png")
        self.plot_categorical_counts(df, "country_long", "count_country.png")
        self.plot_correlation_heatmap(df, "numeric_feature_correlation.png")
        self.plot_capacity_vs_generation(df, "capacity_vs_generation.png")
        self.plot_boxplot(df, config.NUMERIC_FEATURE_COLS, "numeric_feature_boxplot.png",
                          "Boxplot of Numeric Features")
        logger.info("EDA complete")

    def plot_missing_value_matrix(self, df: pd.DataFrame, filename: str):
        """Visualizes WHERE missing values occur, run on raw data pre-cleaning."""
        msno.matrix(df, figsize=(12, 6))
        plt.savefig(self.output_dir / filename, bbox_inches="tight")
        plt.close()

    def plot_missing_value_bar(self, df: pd.DataFrame, filename: str):
        plt.figure(figsize=(12, 6))
        msno.bar(df)
        self._save(filename)

    def run_raw_data_eda(self, raw_df: pd.DataFrame):
        """Run BEFORE cleaning — shows the data quality problem visually."""
        logger.info("Running EDA on RAW (uncleaned) data")
        self.plot_missing_value_matrix(raw_df, "raw_missing_value_matrix.png")
        self.plot_missing_value_bar(raw_df, "raw_missing_value_bar.png")