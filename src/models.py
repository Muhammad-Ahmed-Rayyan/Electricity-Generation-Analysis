from dataclasses import dataclass, field
from typing import Any, Dict

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from src import config


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    param_grid: Dict = field(default_factory=dict)


def get_model_registry() -> Dict[str, ModelSpec]:
    """Returns the set of models to train and compare."""
    return {
        "linear_regression": ModelSpec(
            name="Linear Regression",
            estimator=LinearRegression(),
            param_grid={},
        ),
        "ridge_regression": ModelSpec(
            name="Ridge Regression",
            estimator=Ridge(random_state=config.RANDOM_STATE),
            param_grid=config.RIDGE_PARAM_GRID,
        ),
        "random_forest": ModelSpec(
            name="Random Forest",
            estimator=RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1),
            param_grid=config.RF_PARAM_GRID,
        ),
        "gradient_boosting": ModelSpec(
            name="Gradient Boosting",
            estimator=GradientBoostingRegressor(random_state=config.RANDOM_STATE),
            param_grid=config.GB_PARAM_GRID,
        ),
    }
