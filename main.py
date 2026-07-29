import argparse
import json

import pandas as pd

from src import config
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.data_splitter import DataSplitter
from src.eda import EDAAnalyzer
from src.evaluator import ModelEvaluator
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.models import get_model_registry
from src.trainer import ModelTrainer
from src.utils import ensure_project_directories, get_logger
from src.visualizer import ResultVisualizer

logger = get_logger("pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Power Plant Generation Predictor pipeline")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (slower)")
    parser.add_argument("--sample", type=int, default=None,
                         help="Randomly subsample this many rows before splitting (for quick runs)")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_directories()

    logger.info("STEP 1/8: Loading raw data")
    raw_df = DataLoader().load()

    logger.info("STEP 2/8: Cleaning data")
    clean_df = DataCleaner().run(raw_df)

    if args.sample and args.sample < len(clean_df):
        clean_df = clean_df.sample(n=args.sample, random_state=config.RANDOM_STATE)
        logger.info("Subsampled to %d rows", len(clean_df))

    logger.info("STEP 3/8: Exploratory Data Analysis")
    EDAAnalyzer().run(clean_df)

    logger.info("STEP 4/8: Feature engineering")
    engineered_df = FeatureEngineer().run(clean_df)

    logger.info("STEP 5/8: Feature selection")
    selector = FeatureSelector()
    encoded_df, selected_features = selector.run(engineered_df)

    if not selected_features:
        logger.warning("No features passed the correlation threshold; falling back to defaults")
        selected_features = config.NUMERIC_FEATURE_COLS + config.CATEGORICAL_FEATURE_COLS + ["plant_age"]

    logger.info("STEP 6/8: Splitting data")
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.split(encoded_df, selected_features)

    logger.info("STEP 7/8: Training, evaluating, and comparing models")
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    visualizer = ResultVisualizer()

    registry = get_model_registry()
    all_metrics = []

    for key, spec in registry.items():
        model = trainer.train(spec, X_train, y_train, tune=args.tune)
        trainer.save_model(model, key)

        y_pred = model.predict(X_test)
        result = evaluator.evaluate(spec.name, y_test, y_pred)
        visualizer.plot_residuals(key, y_test, y_pred)
        all_metrics.append(result)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(config.MODEL_COMPARISON_DIR / "all_model_metrics.csv", index=False)
    visualizer.plot_model_comparison(metrics_df)

    logger.info("STEP 8/8: Writing results summary")
    best_model = metrics_df.loc[metrics_df["RMSE"].idxmin()]
    summary = {
        "n_rows_used": len(encoded_df),
        "selected_features": selected_features,
        "best_model": best_model["Model"],
        "best_model_rmse": float(best_model["RMSE"]),
        "all_models": metrics_df.to_dict(orient="records"),
    }
    with open(config.MODEL_COMPARISON_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Pipeline complete. Best model: %s (RMSE=%.4f)", best_model["Model"], best_model["RMSE"])
    print("\n=== Model Comparison ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()