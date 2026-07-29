import datetime
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Derives plant_age and fuel-mix features."""

    @staticmethod
    def add_plant_age(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        current_year = datetime.datetime.now().year
        df["plant_age"] = current_year - df["commissioning_year"]
        df.loc[df["plant_age"] < 0, "plant_age"] = 0
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_plant_age(df)
        logger.info("Added engineered feature: plant_age")
        return df