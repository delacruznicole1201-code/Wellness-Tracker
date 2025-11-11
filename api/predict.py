from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("mental_health_model.pkl")
le = joblib.load("label_encoder.pkl")

# Feature names expected by the model
feature_names = ["Academic_Avg", "Personal_Avg", "Social_Avg", "Career_Avg"]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        inputs = data.get("inputs")

        # Validate input format: must be 4 values
        if not isinstance(inputs, list) or len(inputs) != 4:
            return jsonify({"error": "Provide exactly 4 numeric inputs: Academic, Personal, Social, Career"}), 400

        # Convert and validate numeric
        try:
            numeric_inputs = [float(i) for i in inputs]
        except:
            return jsonify({"error": "All inputs must be numeric"}), 400

        # Convert input into DataFrame to match model format
        features_df = pd.DataFrame([numeric_inputs], columns=feature_names)

        # Predict
        prediction_num = model.predict(features_df)[0]
        prediction_label = le.inverse_transform([prediction_num])[0]

        return jsonify({
            "prediction": prediction_label
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Mental Health Prediction API is running!"})


if __name__ == "__main__":
    # For cloud hosting
    app.run(host="0.0.0.0", port=5000)
