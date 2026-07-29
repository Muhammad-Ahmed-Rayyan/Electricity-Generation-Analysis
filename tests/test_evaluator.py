import numpy as np

from src.evaluator import ModelEvaluator


def test_regression_metrics_are_zero_for_perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = ModelEvaluator.regression_metrics(y_true, y_pred)
    assert metrics["RMSE"] == 0
    assert metrics["MAE"] == 0
    assert metrics["R2_Score"] == 1.0


def test_regression_metrics_detect_error():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    metrics = ModelEvaluator.regression_metrics(y_true, y_pred)
    assert metrics["RMSE"] > 0
    assert metrics["MAE"] > 0
