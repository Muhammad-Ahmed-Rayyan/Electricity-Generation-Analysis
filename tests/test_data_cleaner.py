import numpy as np
import pandas as pd

from src.data_cleaner import DataCleaner


def test_remove_duplicates_drops_exact_dupes():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    result = DataCleaner.remove_duplicates(df)
    assert len(result) == 2


def test_drop_missing_target_removes_null_rows():
    df = pd.DataFrame({"estimated_generation_gwh": [1.0, np.nan, 3.0], "x": [1, 2, 3]})
    result = DataCleaner.drop_missing_target(df)
    assert result["estimated_generation_gwh"].isna().sum() == 0
    assert len(result) == 2


def test_impute_missing_values_fills_numeric_and_categorical():
    df = pd.DataFrame({
        "capacity_mw": [10.0, np.nan, 30.0],
        "latitude": [1.0, 2.0, 3.0],
        "longitude": [1.0, 2.0, 3.0],
        "commissioning_year": [2000, 2005, np.nan],
        "fuel1": ["Coal", None, "Gas"],
        "country_long": ["A", "B", None],
    })
    result = DataCleaner.impute_missing_values(df)
    assert result["capacity_mw"].isna().sum() == 0
    assert result["fuel1"].isna().sum() == 0
