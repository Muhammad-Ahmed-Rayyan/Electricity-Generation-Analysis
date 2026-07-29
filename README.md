<div align="center">

# ⚡ Electricity Generation Analysis

*An End-to-End, Production-Oriented Data Science Pipeline for Power Plant Generation Forecasting*

![Last Commit](https://img.shields.io/github/last-commit/Muhammad-Ahmed-Rayyan/Electricity-Generation-Analysis)
![Python](https://img.shields.io/badge/Python-100%25-blue?logo=python)
![languages](https://img.shields.io/github/languages/count/Muhammad-Ahmed-Rayyan/Electricity-Generation-Analysis)

<br>

Built with the tools and technologies:

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![GitHubActions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

</div>

---

## 🧠 Project Summary

**Electricity Generation Analysis** is an end-to-end, production-oriented Data Science pipeline that predicts a power plant's **estimated annual electricity generation (GWh)** from its capacity, fuel type, and location.


Using the **Global Power Plant Database** (World Resources Institute), the pipeline:

1. Loads the raw dataset (28,664 plants, 22 columns).
2. Cleans it — removes duplicate rows, drops rows with a missing target, imputes missing numeric/categorical values.
3. Runs exploratory data analysis (distributions, correlations, fuel-type and country breakdowns).
4. Engineers a `plant_age` feature from `commissioning_year`.
5. Selects features by correlation with the target.
6. Splits data into train/test sets.
7. Trains four regression models: Linear Regression, Ridge, Random Forest, Gradient Boosting.
8. Tunes hyperparameters with `GridSearchCV` (5-fold CV).
9. Evaluates each model on RMSE, MAE, and R².
10. Compares all models and saves plots, metrics, and a results summary.

---

## 🚀 Features

- 🧹 **Automated Data Cleaning** — dedupes rows, drops missing-target rows, imputes missing numeric/categorical values
- 📊 **Exploratory Data Analysis** — distributions, correlations, fuel-type and country breakdowns
- 🏗️ **Feature Engineering** — derives `plant_age` from `commissioning_year`
- 🎯 **Correlation-Based Feature Selection**
- 🤖 **Four Regression Models** — Linear Regression, Ridge, Random Forest, Gradient Boosting
- 🔍 **Hyperparameter Tuning** — `GridSearchCV` with 5-fold cross-validation
- 📈 **Full Model Evaluation** — RMSE, MAE, and R² for every model
- 🖼️ **Auto-Generated Visualizations** — EDA plots, residual plots, model comparison charts
- ✅ **Unit Tested** — `pytest` coverage for cleaning, feature selection, and evaluation
- 🔄 **CI/CD Pipeline** — automated linting, testing, and pipeline smoke tests via GitHub Actions

---

## 📊 Dataset

Source: **Global Power Plant Database** (WRI), CC Attribution 4.0 License.  
Download: [global-power-plant-database](https://www.kaggle.com/datasets/eshaan90/global-power-plant-database)

Place the CSV at `data/raw/global_power_plant_database.csv`. Columns include:

| Column | Description |
|---|---|
| `country`, `country_long` | Plant location |
| `capacity_mw` | Nameplate generating capacity |
| `latitude`, `longitude` | Geolocation |
| `fuel1`–`fuel4` | Primary and secondary fuel types |
| `commissioning_year` | Year the plant became operational |
| `estimated_generation_gwh` | **Target** — estimated annual generation |

The dataset is genuinely unclean: duplicate entries, missing `commissioning_year` for many older plants, and missing generation estimates for a meaningful share of plants — all handled explicitly in `src/data_cleaner.py`.

---

## 🔧 Setup & Installation

> Make sure Python 3.8+ is installed.

```bash
# Clone the repo
git clone https://github.com/Muhammad-Ahmed-Rayyan/Electricity-Generation-Analysis.git
cd Electricity-Generation-Analysis
```

### 🐍 Virtual Environment Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 📦 Dependencies

```bash
pip install -r requirements.txt
```

---

## 🗃️ Project Structure

```bash
electric-generation-cost-predictor/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/  # global_power_plant_database.csv
│   └── processed/  # cleaned_power_plants.csv (generated + tracked)
├── models/  # trained .pkl models
├── outputs/
│   ├── eda/
│   ├── missing_values/
│   ├── feature_selection/
│   ├── model_training/
│   └── model_comparison/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── data_splitter.py
│   ├── models.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── visualizer.py
├── tests/
│   ├── __init__.py
│   ├── test_data_cleaner.py
│   ├── test_feature_selection.py
│   └── test_evaluator.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

### 🧩 Module Overview

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

---

## ▶️ How to Run

```bash
# Full pipeline with hyperparameter tuning (recommended for final results)
python main.py --tune

# Faster run without tuning
python main.py

# Quick smoke test on a subsample
python main.py --sample 5000
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 🧹 Linting

Flake8 configuration lives in `setup.cfg`:
```ini
[flake8]
max-line-length = 110
ignore = E203, W503
```

Run locally before pushing:
```bash
flake8 src tests main.py
```

---

## 🔄 CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`: installs dependencies, lints with `flake8`, runs unit tests, and smoke-tests the full pipeline on a small sample.

---

## 📈 Results

Full pipeline run on the complete cleaned dataset (27,438 rows: 21,950 train / 5,488 test), with `--tune` enabled (`GridSearchCV`, 5-fold CV):

Selected features: `capacity_mw`, `fuel1`, `country_long`

| Model | RMSE (GWh) | MAE (GWh) | R² |
|---|---|---|---|
| Linear Regression | 1242.90 | 326.27 | 0.816 |
| Ridge Regression | 1242.90 | 326.26 | 0.816 |
| Random Forest | 876.20 | 140.86 | 0.908 |
| **Gradient Boosting** | **805.28** | 164.82 | **0.923** |

**Best model: Gradient Boosting** (RMSE = 805.28, R² = 0.923), tuned to `{n_estimators: 200, max_depth: 5, learning_rate: 0.1, min_samples_split: 2}`.

---

### 🔍 Observations

- Both linear models plateau around R²≈0.82 — capacity, fuel type, and country alone have a roughly linear relationship with generation, but can't capture the non-linear interactions (e.g. how fuel type changes the capacity→generation ratio) that tree-based models pick up.
- **Random Forest has the lowest MAE (140.86)** despite a higher RMSE than Gradient Boosting — it's more consistently close on typical plants, while Gradient Boosting handles large/outlier plants better, which is why it wins on RMSE (which penalizes big errors more heavily).
- Ridge tuned to `alpha=100`, a strong regularization strength, which barely moved its score versus plain Linear Regression — suggesting multicollinearity isn't a major issue with only 3 selected features.

All plots and the full metrics/summary are available in `outputs/`:
- `outputs/eda/` — distribution, correlation, and fuel/country breakdown plots
- `outputs/feature_selection/` — correlation rankings and selected features
- `outputs/model_training/` — per-model residual distribution plots
- `outputs/model_comparison/` — RMSE/MAE/R² comparison charts, `all_model_metrics.csv`, `results_summary.json`

---

## 💾 Note on Trained Models

`models/linear_regression.pkl`, `models/ridge_regression.pkl`, and `models/gradient_boosting.pkl` are included directly in this repository.

`models/random_forest.pkl` is **excluded** — its trained size exceeds GitHub's 100MB per-file limit even after compression, due to the large ensemble size (`n_estimators=300`, `max_depth=20`) selected by hyperparameter tuning. It is fully reproducible; regenerate it with:

```bash
python main.py --tune
```

---

## 📝 Design Notes

- **Data leakage avoided**: raw yearly generation columns (`generation_gwh_2013`–`2016`) are excluded from features since they would leak the target (`estimated_generation_gwh` is derived from them).
- **Target-missing rows dropped** rather than imputed — imputing the label itself would fabricate ground truth, which is inappropriate for a supervised regression target.
- **Hyperparameter tuning** uses standard 5-fold `GridSearchCV`, scored on RMSE, for all models except Linear Regression (no hyperparameters to tune).

---

<div align="center">

⭐ Found this project useful? Drop a star on GitHub!

</div>