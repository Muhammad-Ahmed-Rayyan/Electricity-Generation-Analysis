from pathlib import Path
import pandas as pd
from src import config
from src.utils import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Loads the raw Global Power Plant Database CSV."""

    def __init__(self, path: Path = config.RAW_FILE):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Expected data file not found: {self.path}\n"
                f"Download it from https://www.kaggle.com/datasets/eshaan90/global-power-plant-database "
                f"and place it at '{config.RAW_FILE}'."
            )
        df = pd.read_csv(self.path, low_memory=False)
        logger.info("Loaded %s -> shape=%s", self.path.name, df.shape)
        return df