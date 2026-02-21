import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)


# Path Setup

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# Load Models

regressor = joblib.load(os.path.join(MODEL_DIR, "regressor.pkl"))
classifier = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))

# Feature List

FEATURES = [
    "electricity_kwh",
    "lpg_kg",
    "fuel_litres",
    "flights_year",
    "public_transport",
    "meat_days_week",
    "waste_kg_week",
    "brand_co2"
]


# Home Route

@app.route("/")
def home():
    return jsonify({"message": "Carbon Intelligence Backend Running"})


# Predict Route

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate input
        for feature in FEATURES:
            if feature not in data:
                return jsonify({"error": f"Missing field: {feature}"}), 400

        input_df = pd.DataFrame(
            [[data[f] for f in FEATURES]],
            columns=FEATURES
        )

        co2_prediction = regressor.predict(input_df)[0]
        category_prediction = classifier.predict(input_df)[0]

        return jsonify({
            "predicted_total_CO2": round(float(co2_prediction), 2),
            "predicted_category": str(category_prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# Run App

if __name__ == "__main__":
    app.run(debug=True)