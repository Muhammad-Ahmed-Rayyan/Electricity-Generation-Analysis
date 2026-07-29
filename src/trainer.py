import joblib
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.models import ModelSpec
from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


RMSE_SCORER = make_scorer(_rmse, greater_is_better=False)


class ModelTrainer:
    """Fits, optionally tunes, and saves regression models."""

    def __init__(self, models_dir=config.MODELS_DIR):
        self.models_dir = models_dir

    def tune(self, spec: ModelSpec, X, y, cv: int = 5):
        if not spec.param_grid:
            logger.info("No hyperparameter grid for %s, skipping tuning", spec.name)
            return spec.estimator

        logger.info("Tuning %s with %d-fold CV", spec.name, cv)
        search = GridSearchCV(
            estimator=spec.estimator,
            param_grid=spec.param_grid,
            scoring=RMSE_SCORER,
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X, y)
        logger.info("Best params for %s: %s (CV RMSE=%.4f)",
                    spec.name, search.best_params_, -search.best_score_)
        return search.best_estimator_

    def train(self, spec: ModelSpec, X_train, y_train, tune: bool = False):
        estimator = spec.estimator
        if tune:
            estimator = self.tune(spec, X_train, y_train)
        estimator.fit(X_train, y_train)
        logger.info("Trained %s on %d rows", spec.name, len(X_train))
        return estimator

    def save_model(self, model, model_key: str):
        path = self.models_dir / f"{model_key}.pkl"
        joblib.dump(model, path)
        logger.info("Saved model -> %s", path)
        return path