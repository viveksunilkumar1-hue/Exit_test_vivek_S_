import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


OUTPUT_DIR = Path(__file__).parent / "artifacts"
DEFAULT_DATA = Path("G3.csv")


def parse_amount(value):
    """Parse numbers, currency strings, and K/M/B/T suffixes."""
    if pd.isna(value):
        return np.nan

    text = re.sub(r"[$€£,\s]", "", str(value).strip().upper())
    match = re.fullmatch(r"(-?\d*\.?\d+)([KMBT]?)", text)

    if not match:
        return pd.to_numeric(value, errors="coerce")

    multiplier = {
        "": 1,
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
        "T": 1_000_000_000_000,
    }[match.group(2)]

    return float(match.group(1)) * multiplier


def find_revenue_column(columns):
    normalized = {col.lower().replace(" ", "_"): col for col in columns}
    for candidate in ("annual_revenue", "revenue"):
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        f"Revenue column not found. Available columns: {list(columns)}"
    )


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    target = find_revenue_column(df.columns)
    df[target] = df[target].map(parse_amount)
    df = df[df[target].notna() & (df[target] > 0)].copy()

    # Remove IDs, names, and revenue-derived columns to reduce leakage.
    excluded = {
        target,
        "enterprise_group_id",
        "group_root_lei",
        "display_name",
        "enterprise_group_name",
        "website",
    }
    excluded.update(
        col for col in df.columns
        if col != target and (
            "revenue" in col.lower()
            or col.lower().endswith("_source")
        )
    )
    X = df.drop(columns=[c for c in excluded if c in df.columns]).copy()

    # Parse common numeric fields that may be stored as text.
    for col in ("employees", "employees_year", "market_cap",
                "max_hierarchy_depth"):
        if col in X.columns:
            X[col] = X[col].map(parse_amount)

    # Convert the market-cap date into numeric components.
    if "market_cap_date" in X.columns:
        dates = pd.to_datetime(X.pop("market_cap_date"), errors="coerce")
        X["market_cap_year"] = dates.dt.year
        X["market_cap_month"] = dates.dt.month

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns remain after cleaning.")

    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(
                strategy="constant", fill_value="Unknown"
            )),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Train/test evaluation uses log revenue, as revenue is often skewed.
    y = np.log1p(df[target])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predicted = np.maximum(np.expm1(model.predict(X_test)), 0)
    actual = np.expm1(y_test)

    print(f"Test MAE:  {mean_absolute_error(actual, predicted):,.2f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(actual, predicted)):,.2f}")
    print(f"Test R²:   {r2_score(actual, predicted):.3f}")

    # Refit on all available labeled data for the deployed model.
    model.fit(X, y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "revenue_model.joblib")

    schema = {
        "target": target,
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }
    (OUTPUT_DIR / "feature_schema.json").write_text(
        json.dumps(schema, indent=2)
    )

    print(f"Saved model and schema in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()