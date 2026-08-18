import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "country", "country_long", "name", "gppd_idnr", "capacity_mw",
    "latitude", "longitude", "fuel1", "fuel2", "fuel3", "fuel4",
    "commissioning_year", "owner", "source", "url", "geolocation_source",
    "year_of_capacity_data", "generation_gwh_2013", "generation_gwh_2014",
    "generation_gwh_2015", "generation_gwh_2016", "estimated_generation_gwh",
]

VALID_RANGES = {
    "capacity_mw": (0, 25000),          # largest real plants are ~22,500 MW
    "latitude": (-90, 90),
    "longitude": (-180, 180),
    "commissioning_year": (1880, 2026),  # first power plants ~1880s
}


class DataValidator:
    """Validates schema, dtypes, and value ranges of the raw dataset."""

    def __init__(self):
        self.report_lines = []

    def _log(self, line: str):
        self.report_lines.append(line)
        logger.info(line)

    def validate_schema(self, df: pd.DataFrame) -> bool:
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
        self._log(f"Schema check: {len(missing_cols)} missing expected columns, "
                   f"{len(extra_cols)} unexpected columns")
        if missing_cols:
            self._log(f"  Missing: {missing_cols}")
        if extra_cols:
            self._log(f"  Unexpected: {extra_cols}")
        return len(missing_cols) == 0

    def validate_missingness(self, df: pd.DataFrame) -> pd.Series:
        missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
        self._log("Missingness by column (%):")
        for col, pct in missing_pct[missing_pct > 0].items():
            self._log(f"  {col}: {pct:.2f}%")
        return missing_pct

    def validate_duplicates(self, df: pd.DataFrame) -> int:
        n_dupes = df.duplicated().sum()
        self._log(f"Duplicate rows found: {n_dupes} ({n_dupes / len(df) * 100:.2f}%)")
        return n_dupes

    def validate_value_ranges(self, df: pd.DataFrame) -> dict:
        violations = {}
        for col, (low, high) in VALID_RANGES.items():
            if col not in df.columns:
                continue
            out_of_range = df[(df[col] < low) | (df[col] > high)]
            violations[col] = len(out_of_range)
            if len(out_of_range) > 0:
                self._log(f"Range violation: {col} has {len(out_of_range)} values "
                           f"outside expected [{low}, {high}]")
        return violations

    def validate_dtypes(self, df: pd.DataFrame) -> dict:
        dtype_report = df.dtypes.astype(str).to_dict()
        self._log("Column dtypes: " + ", ".join(f"{k}={v}" for k, v in dtype_report.items()))
        return dtype_report

    def run(self, df: pd.DataFrame) -> dict:
        self._log("=== DATA VALIDATION REPORT (raw data, pre-cleaning) ===")
        self._log(f"Shape: {df.shape}")

        schema_ok = self.validate_schema(df)
        missingness = self.validate_missingness(df)
        n_duplicates = self.validate_duplicates(df)
        range_violations = self.validate_value_ranges(df)
        dtypes = self.validate_dtypes(df)

        report_path = config.OUTPUT_DIR / "data_validation_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(self.report_lines))
        logger.info("Saved validation report -> %s", report_path)

        return {
            "schema_ok": schema_ok,
            "missingness_pct": missingness.to_dict(),
            "n_duplicates": int(n_duplicates),
            "range_violations": range_violations,
            "dtypes": dtypes,
        }