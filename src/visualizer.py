import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)


class ResultVisualizer:
    def __init__(self, training_dir=config.MODEL_TRAINING_DIR, comparison_dir=config.MODEL_COMPARISON_DIR):
        self.training_dir = training_dir
        self.comparison_dir = comparison_dir

    def plot_residuals(self, model_key: str, y_test, y_pred):
        residuals = y_test.values - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, bins=40, kde=True, color="steelblue")
        plt.title(f"{model_key} — Residual Distribution")
        plt.xlabel("Residual = Actual − Predicted (GWh)")
        plt.tight_layout()
        plt.savefig(self.training_dir / f"{model_key}_residuals.png")
        plt.close()

    def plot_model_comparison(self, metrics_df):
        for metric in ["RMSE", "MAE", "R2_Score"]:
            plt.figure(figsize=(8, 5))
            sns.barplot(x="Model", y=metric, data=metrics_df)
            plt.title(f"Comparison of {metric} Across Models")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(self.comparison_dir / f"comparison_{metric}.png")
            plt.close()
        logger.info("Saved comparison plots -> %s", self.comparison_dir)
