import pandas as pd
from sklearn.impute import SimpleImputer

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Cleans the raw power plant dataset: drops noise columns, dedupes, imputes."""

    @staticmethod
    def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in config.DROP_COLS if c in df.columns]
        return df.drop(columns=cols_to_drop)

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        logger.info("Removed %d duplicate rows", before - len(df))
        return df

    @staticmethod
    def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=[config.TARGET_COL])
        logger.info("Dropped %d rows with missing target", before - len(df))
        return df

    @staticmethod
    def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = [c for c in config.NUMERIC_FEATURE_COLS if c in df.columns]
        categorical_cols = [c for c in config.CATEGORICAL_FEATURE_COLS if c in df.columns]

        if numeric_cols:
            imputer = SimpleImputer(strategy="median")
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        for col in categorical_cols:
            df[col] = df[col].fillna("Unknown")

        logger.info("Imputed %d numeric / %d categorical columns", len(numeric_cols), len(categorical_cols))
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.drop_irrelevant_columns(df)
        df = self.remove_duplicates(df)
        df = self.drop_missing_target(df)
        df = self.impute_missing_values(df)
        df.to_csv(config.DATA_PROCESSED_DIR / "cleaned_power_plants.csv", index=False)
        return df
