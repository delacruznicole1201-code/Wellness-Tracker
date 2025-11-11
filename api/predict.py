from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("mental_health_model.pkl")
le = joblib.load("label_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        inputs = data.get("inputs")

        if not isinstance(inputs, list) or len(inputs) != 4:
            return jsonify({"error": "Provide 4 numeric inputs: Academic, Personal, Social, Career"}), 400

        # Convert inputs to float
        try:
            numeric_inputs = [float(i) for i in inputs]
        except ValueError:
            return jsonify({"error": "All inputs must be numeric"}), 400

        # Convert to DataFrame
        feature_names = ["Academic_Avg", "Personal_Avg", "Social_Avg", "Career_Avg"]
        features_df = pd.DataFrame([numeric_inputs], columns=feature_names)

        # Predict
        prediction = model.predict(features_df)[0]
        result_label = le.inverse_transform([prediction])[0]

        return jsonify({"prediction": result_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Mental Health Prediction API is running!"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
