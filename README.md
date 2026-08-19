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
2. **Validates** the raw data against expected schema, dtypes, and value ranges, and produces a standalone validation report *before* any cleaning happens.
3. Cleans it — removes duplicate rows, drops rows with a missing target, imputes missing numeric/categorical values.
4. Runs exploratory data analysis on **both the raw and cleaned data** — missing-value matrices/bar charts on the raw data, plus distributions, correlations, fuel-type and country breakdowns on the cleaned data.
5. Engineers a `plant_age` feature from `commissioning_year`.
6. Selects features by correlation with the target.
7. Splits data into train/test sets.
8. Trains four regression models: Linear Regression, Ridge, Random Forest, Gradient Boosting.
9. Tunes hyperparameters with `GridSearchCV` (5-fold CV) — **enabled by default**, with every search logged to a standing JSON artifact.
10. Evaluates each model on RMSE, MAE, and R².
11. Compares all models and saves plots, metrics, and a results summary.

---

## 🚀 Features

- ✅ **Data Validation** — schema, dtype, missingness, and value-range checks on raw data, saved as a standalone report
- 🧹 **Automated Data Cleaning** — dedupes rows, drops missing-target rows, imputes missing numeric/categorical values
- 📊 **Exploratory Data Analysis** — raw-data missingness visualizations plus distributions, correlations, fuel-type and country breakdowns on cleaned data
- 🏗️ **Feature Engineering** — derives `plant_age` from `commissioning_year`
- 🎯 **Correlation-Based Feature Selection**
- 🤖 **Four Regression Models** — Linear Regression, Ridge, Random Forest, Gradient Boosting
- 🔍 **Hyperparameter Tuning** — `GridSearchCV` with 5-fold cross-validation, **on by default**, with results logged to `hyperparameter_tuning_log.json`
- 📈 **Full Model Evaluation** — RMSE, MAE, and R² for every model
- 🖼️ **Auto-Generated Visualizations** — raw-data missingness plots, EDA plots, residual plots, model comparison charts
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

The dataset is genuinely unclean: duplicate entries, missing `commissioning_year` for many older plants, and missing generation estimates for a meaningful share of plants — all handled explicitly in `src/data_cleaner.py`, and quantified explicitly in `src/data_validator.py` (see [Data Validation](#-data-validation) below).

---

## 🔎 Data Validation

Before any cleaning happens, `src/data_validator.py` runs a standalone validation pass on the raw dataset and saves the full report to `outputs/data_validation_report.txt`. This is a distinct step from cleaning — it exists to *quantify* how messy the raw data actually is, rather than silently fixing it.

**Schema check:** 0 missing expected columns, 0 unexpected columns — the raw file matches the expected 22-column structure.

**Missingness by column (raw data, pre-cleaning):**

| Column | Missing % |
|---|---|
| `fuel4` | 99.62% |
| `fuel3` | 98.97% |
| `generation_gwh_2013` | 98.47% |
| `generation_gwh_2014` | 98.41% |
| `generation_gwh_2015` | 96.66% |
| `fuel2` | 94.05% |
| `generation_gwh_2016` | 70.95% |
| `commissioning_year` | 47.84% |
| `year_of_capacity_data` | 43.67% |
| `owner` | 36.88% |
| `estimated_generation_gwh` (target) | 3.94% |
| `geolocation_source` | 1.47% |
| `name` | 0.07% |
| `fuel1` | 0.01% |

**Duplicates:** 0 duplicate rows found by the validator's raw-data check (duplicates found later during cleaning were exact-match duplicates surfaced via `drop_duplicates()`, which uses a stricter full-row comparison — see [Design Notes](#-design-notes)).

**Range violations:** 17 rows have a `latitude` outside the valid [-90, 90] range, and 4 rows have a `longitude` outside the valid [-180, 180] range — confirming the raw data contains genuine coordinate entry errors, not just missing values.

This report is what justified the cleaning decisions in `data_cleaner.py`: columns with >90% missingness (`fuel2/3/4`, `generation_gwh_2013/2014/2015`) were dropped rather than imputed, since imputing that much of a column would mean fabricating almost all of its values rather than genuinely filling gaps.

**Raw-data missingness visualizations** (`outputs/eda/raw_missing_value_bar.png`, `outputs/eda/raw_missing_value_matrix.png`):

The bar chart confirms the pattern above visually — `fuel2/3/4` and the yearly `generation_gwh_*` columns are almost entirely empty, while `capacity_mw`, `latitude`, `longitude`, and `fuel1` are essentially complete. The matrix view shows the missingness in `commissioning_year` and `owner` is **not randomly scattered** — it clusters in contiguous blocks of rows, suggesting these gaps come from specific data sources/batches within the database rather than random data-entry omissions.

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
│   ├── data_validation_report.txt
│   ├── eda/
│   ├── missing_values/
│   ├── feature_selection/
│   ├── model_training/
│   └── model_comparison/
│       └── hyperparameter_tuning_log.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── data_validator.py
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
├── setup.cfg
├── .gitignore
└── README.md
```

### 🧩 Module Overview

| Module | Responsibility |
|---|---|
| `data_loader.py` | Reads the raw CSV |
| `data_validator.py` | Validates schema, dtypes, missingness, and value ranges on raw data; saves `data_validation_report.txt` |
| `data_cleaner.py` | Dedupes, drops missing-target rows, imputes |
| `eda.py` | Generates raw-data missingness plots and cleaned-data EDA plots/statistics |
| `feature_engineering.py` | Adds `plant_age` |
| `feature_selection.py` | Correlation-based feature selection |
| `data_splitter.py` | Train/test split |
| `models.py` | Model + hyperparameter grid registry |
| `trainer.py` | Fits/tunes models, saves to `models/`, logs tuning results |
| `evaluator.py` | RMSE / MAE / R² |
| `visualizer.py` | Residual plots, model comparison charts |

---

## ▶️ How to Run

```bash
# Full pipeline — hyperparameter tuning runs by default
python main.py

# Skip tuning for a faster run
python main.py --no-tune

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

Full pipeline run on the complete cleaned dataset (27,438 rows: 21,950 train / 5,488 test), with hyperparameter tuning enabled (`GridSearchCV`, 5-fold CV):

Selected features: `capacity_mw`, `fuel1`, `country_long`

| Model | RMSE (GWh) | MAE (GWh) | R² |
|---|---|---|---|
| Linear Regression | 1242.90 | 326.27 | 0.816 |
| Ridge Regression | 1242.90 | 326.26 | 0.816 |
| Random Forest | 876.20 | 140.86 | 0.908 |
| **Gradient Boosting** | **805.28** | 164.82 | **0.923** |

**Best model: Gradient Boosting** (RMSE = 805.28, R² = 0.923), tuned to `{n_estimators: 200, max_depth: 5, learning_rate: 0.1, min_samples_split: 2}`.

---

## 🎛️ Hyperparameter Tuning Log

Every model with tunable parameters was searched with `GridSearchCV` (5-fold cross-validation, scored on RMSE). Full machine-readable results are saved to `outputs/model_comparison/hyperparameter_tuning_log.json`; summarized here:

| Model | Grid Searched | Best Parameters | Best CV RMSE |
|---|---|---|---|
| Ridge Regression | `alpha`: [0.01, 0.1, 1, 10, 100] | `alpha=100` | 1091.29 |
| Random Forest | `n_estimators`: [100,200,300], `max_depth`: [None,10,20], `min_samples_split`: [2,5], `min_samples_leaf`: [1,2] | `n_estimators=300, max_depth=20, min_samples_split=2, min_samples_leaf=1` | 926.67 |
| Gradient Boosting | `n_estimators`: [100,200], `max_depth`: [3,5], `learning_rate`: [0.05,0.1], `min_samples_split`: [2,5] | `n_estimators=200, max_depth=5, learning_rate=0.1, min_samples_split=2` | 864.09 |

Linear Regression has no tunable hyperparameters and was trained directly. Note that each model's **CV RMSE during tuning** is higher than its **final test-set RMSE** in the Results table above — this is expected, since CV RMSE is averaged across 5 folds of the training set (a more conservative estimate), while the Results table reports performance on the single held-out test set after fitting on the full training data.

---

### 🔍 Observations

- Both linear models plateau around R²≈0.82 — capacity, fuel type, and country alone have a roughly linear relationship with generation, but can't capture the non-linear interactions (e.g. how fuel type changes the capacity→generation ratio) that tree-based models pick up.
- **Random Forest has the lowest MAE (140.86)** despite a higher RMSE than Gradient Boosting — it's more consistently close on typical plants, while Gradient Boosting handles large/outlier plants better, which is why it wins on RMSE (which penalizes big errors more heavily).
- Ridge tuned to `alpha=100`, a strong regularization strength, which barely moved its score versus plain Linear Regression — suggesting multicollinearity isn't a major issue with only 3 selected features.
- Tuning delivered the largest gains for the tree-based models (Random Forest, Gradient Boosting), and negligible gains for the linear models — consistent with linear models having far fewer degrees of freedom to tune in the first place.

All plots and the full metrics/summary are available in `outputs/`:
- `outputs/data_validation_report.txt` — raw-data validation findings
- `outputs/eda/` — raw-data missingness plots, plus distribution, correlation, and fuel/country breakdown plots on cleaned data
- `outputs/feature_selection/` — correlation rankings and selected features
- `outputs/model_training/` — per-model residual distribution plots
- `outputs/model_comparison/` — RMSE/MAE/R² comparison charts, `all_model_metrics.csv`, `results_summary.json`, `hyperparameter_tuning_log.json`

---

## 💾 Note on Trained Models

`models/linear_regression.pkl`, `models/ridge_regression.pkl`, and `models/gradient_boosting.pkl` are included directly in this repository.

`models/random_forest.pkl` is **excluded** — its trained size exceeds GitHub's 100MB per-file limit even after compression, due to the large ensemble size (`n_estimators=300`, `max_depth=20`) selected by hyperparameter tuning. It is fully reproducible; regenerate it with:

```bash
python main.py
```

---

## 📝 Design Notes

- **Data validation runs before cleaning**, as a distinct step, to give visibility into raw data quality independent of any fixes applied later.
- **Data leakage avoided**: raw yearly generation columns (`generation_gwh_2013`–`2016`) are excluded from features since they would leak the target (`estimated_generation_gwh` is derived from them).
- **Target-missing rows dropped** rather than imputed — imputing the label itself would fabricate ground truth, which is inappropriate for a supervised regression target.
- **Hyperparameter tuning runs by default** using standard 5-fold `GridSearchCV`, scored on RMSE, for all models except Linear Regression (no hyperparameters to tune) — every search result is logged to `hyperparameter_tuning_log.json` regardless of how the pipeline is run.

---

<div align="center">

⭐ Found this project useful? Drop a star on GitHub!

</div>