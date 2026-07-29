import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from src.utils import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Computes standard regression metrics on the test set."""

    @staticmethod
    def regression_metrics(y_true, y_pred) -> dict:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return {"RMSE": rmse, "MAE": mae, "R2_Score": r2}

    def evaluate(self, model_name: str, y_test, y_pred) -> dict:
        metrics = self.regression_metrics(y_test, y_pred)
        logger.info("%s -> RMSE=%.4f MAE=%.4f R2=%.4f", model_name, metrics["RMSE"], metrics["MAE"], metrics["R2_Score"])
        return {"Model": model_name, **metrics}