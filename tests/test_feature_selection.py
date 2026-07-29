import pandas as pd

from src.feature_selection import FeatureSelector


def test_encode_categoricals_converts_object_columns_to_numeric():
    df = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1, 2, 3]})
    encoded = FeatureSelector.encode_categoricals(df)
    assert pd.api.types.is_numeric_dtype(encoded["cat"])


def test_select_keeps_only_features_above_threshold():
    df = pd.DataFrame({
        "strong_feature": [1, 2, 3, 4, 5],
        "noise_feature": [5, 1, 4, 2, 3],
        "Cost_USD_per_MWh": [1.1, 2.0, 3.1, 4.2, 5.0],
    })
    selector = FeatureSelector(threshold=0.9)
    selected = selector.select(df)
    assert "strong_feature" in selected
    assert "noise_feature" not in selected