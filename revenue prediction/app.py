import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

model = joblib.load(ARTIFACT_DIR / "revenue_model.joblib")
schema = json.loads((ARTIFACT_DIR / "feature_schema.json").read_text())

FEATURES = schema["feature_columns"]
NUMERIC_FEATURES = set(schema["numeric_columns"])
TARGET_NAME = schema["target"]

app = Flask(__name__)


def parse_amount(value):
    if value is None or str(value).strip() == "":
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


def predict_revenue(payload):
    unknown = set(payload) - set(FEATURES)
    if unknown:
        raise ValueError(f"Unknown feature(s): {', '.join(sorted(unknown))}")

    row = {}
    for feature in FEATURES:
        value = payload.get(feature)

        if feature in NUMERIC_FEATURES:
            parsed = parse_amount(value)
            if value not in (None, "") and pd.isna(parsed):
                raise ValueError(f"'{feature}' must be a valid number.")
            row[feature] = parsed
        else:
            row[feature] = (
                str(value).strip() if value not in (None, "") else np.nan
            )

    input_df = pd.DataFrame([row], columns=FEATURES)
    predicted_log_revenue = float(model.predict(input_df)[0])
    revenue = max(0.0, float(np.expm1(predicted_log_revenue)))
    return revenue


@app.get("/")
def home():
    fields = [
        {
            "name": feature,
            "label": feature.replace("_", " ").title(),
            "kind": "number" if feature in NUMERIC_FEATURES else "text",
        }
        for feature in FEATURES
    ]
    return render_template(
        "index.html",
        fields=fields,
        target_name=TARGET_NAME.replace("_", " ").title(),
    )


@app.post("/predict")
def predict_form():
    try:
        revenue = predict_revenue(request.form)
        return render_template(
            "index.html",
            fields=[
                {
                    "name": feature,
                    "label": feature.replace("_", " ").title(),
                    "kind": "number" if feature in NUMERIC_FEATURES else "text",
                }
                for feature in FEATURES
            ],
            target_name=TARGET_NAME.replace("_", " ").title(),
            prediction=f"{revenue:,.2f}",
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object of feature values."}), 400

    try:
        revenue = predict_revenue(payload)
        return jsonify({
            "target": TARGET_NAME,
            "predicted_revenue": revenue,
        })
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)