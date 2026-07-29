# Power Plant Generation Predictor

An end-to-end, production-oriented Data Science pipeline that predicts a power
plant's **estimated annual electricity generation (GWh)** from its capacity,
fuel type, and location — built for Task 1 of the Alphatron Technologies
Research Internship Program 2026 (Batch 2).

## Project Overview

Using the **Global Power Plant Database** (World Resources Institute), the
pipeline:

1. Loads the raw dataset (28,664 plants, 22 columns).
2. Cleans it — removes duplicate rows, drops rows with a missing target,
   imputes missing numeric/categorical values.
3. Runs exploratory data analysis (distributions, correlations, fuel-type
   and country breakdowns).
4. Engineers a `plant_age` feature from `commissioning_year`.
5. Selects features by correlation with the target.
6. Splits data into train/test sets.
7. Trains four regression models: Linear Regression, Ridge, Random Forest,
   Gradient Boosting.
8. Tunes hyperparameters with `GridSearchCV` (5-fold CV).
9. Evaluates each model on RMSE, MAE, and R².
10. Compares all models and saves plots, metrics, and a results summary.

## Dataset

Source: **Global Power Plant Database** (WRI), CC Attribution 4.0 License.
Download: https://www.kaggle.com/datasets/eshaan90/global-power-plant-database

Place the CSV at `data/raw/global_power_plant_database.csv`. Columns include:

| Column | Description |
|---|---|
| `country`, `country_long` | Plant location |
| `capacity_mw` | Nameplate generating capacity |
| `latitude`, `longitude` | Geolocation |
| `fuel1`–`fuel4` | Primary and secondary fuel types |
| `commissioning_year` | Year the plant became operational |
| `estimated_generation_gwh` | **Target** — estimated annual generation |

The dataset is genuinely unclean: duplicate entries, missing
`commissioning_year` for many older plants, and missing generation estimates
for a meaningful share of plants — all handled explicitly in
`src/data_cleaner.py`.

## Installation

```bash
git clone <your-repo-url>
cd Electricity-Generation-Analysis
```

### Virtual environment setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure
```
Electricity-Generation-Analysis/
├── data/
│   ├── raw/                  # global_power_plant_database.csv
│   └── processed/            # cleaned_power_plants.csv (generated + tracked)
├── models/                   # trained .pkl models
├── outputs/
│   ├── eda/
│   ├── feature_selection/
│   ├── model_training/
│   └── model_comparison/
├── src/                      # OOP pipeline modules
├── tests/                    # pytest unit tests
├── main.py                   # pipeline entry point
└── requirements.txt
```

| Module | Responsibility |
|---|---|
| `data_loader.py` | Reads the raw CSV |
| `data_cleaner.py` | Dedupes, drops missing-target rows, imputes |
| `eda.py` | Generates EDA plots and summary statistics |
| `feature_engineering.py` | Adds `plant_age` |
| `feature_selection.py` | Correlation-based feature selection |
| `data_splitter.py` | Train/test split |
| `models.py` | Model + hyperparameter grid registry |
| `trainer.py` | Fits/tunes models, saves to `models/` |
| `evaluator.py` | RMSE / MAE / R² |
| `visualizer.py` | Residual plots, model comparison charts |

## How to Run

```bash
# Full pipeline with hyperparameter tuning (recommended for final results)
python main.py --tune

# Faster run without tuning
python main.py

# Quick smoke test on a subsample
python main.py --sample 5000
```

## Running Tests

```bash
pytest tests/ -v
```

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`: installs
dependencies, lints with `flake8`, runs unit tests, and smoke-tests the full
pipeline on a small sample.

## Results

Full pipeline run on the complete cleaned dataset (27,438 rows: 21,950 train
/ 5,488 test), with `--tune` enabled (`GridSearchCV`, 5-fold CV):

Selected features: `capacity_mw`, `fuel1`, `country_long`

| Model | RMSE (GWh) | MAE (GWh) | R² |
|---|---|---|---|
| Linear Regression | 1242.90 | 326.27 | 0.816 |
| Ridge Regression | 1242.90 | 326.26 | 0.816 |
| Random Forest | 876.20 | 140.86 | 0.908 |
| **Gradient Boosting** | **805.28** | 164.82 | **0.923** |

**Best model: Gradient Boosting** (RMSE = 805.28, R² = 0.923), tuned to
`{n_estimators: 200, max_depth: 5, learning_rate: 0.1, min_samples_split: 2}`.

### Observations

- Both linear models plateau around R²≈0.82 — capacity, fuel type, and
  country alone have a roughly linear relationship with generation, but
  can't capture the non-linear interactions (e.g. how fuel type changes the
  capacity→generation ratio) that tree-based models pick up.
- **Random Forest has the lowest MAE (140.86)** despite a higher RMSE than
  Gradient Boosting — it's more consistently close on typical plants, while
  Gradient Boosting handles large/outlier plants better, which is why it
  wins on RMSE (which penalizes big errors more heavily).
- Ridge tuned to `alpha=100`, a strong regularization strength, which barely
  moved its score versus plain Linear Regression — suggesting multicollinearity
  isn't a major issue with only 3 selected features.

All plots and the full metrics/summary are available in `outputs/`:
- `outputs/eda/` — distribution, correlation, and fuel/country breakdown plots
- `outputs/feature_selection/` — correlation rankings and selected features
- `outputs/model_training/` — per-model residual distribution plots
- `outputs/model_comparison/` — RMSE/MAE/R² comparison charts,
  `all_model_metrics.csv`, `results_summary.json`
- `models/` — every trained model, saved as `.pkl`

## Design Notes

- **Data leakage avoided**: raw yearly generation columns
  (`generation_gwh_2013`–`2016`) are excluded from features since they would
  leak the target (`estimated_generation_gwh` is derived from them).
- **Target-missing rows dropped** rather than imputed — imputing the label
  itself would fabricate ground truth, which is inappropriate for a
  supervised regression target.
- **Hyperparameter tuning** uses standard 5-fold `GridSearchCV`, scored on
  RMSE, for all models except Linear Regression (no hyperparameters to tune).