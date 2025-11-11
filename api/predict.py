from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("mental_health_model.pkl")

# ✅ Label Mapping (model output → text label)
label_map = {
    0: "Normal",
    1: "Stress",
    2: "Burnout",
    3: "Anxiety",
    4: "Depression"
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        inputs = data.get("inputs")

        # Validate input format
        if not isinstance(inputs, list) or len(inputs) != 4:
            return jsonify({"error": "Provide 4 numeric inputs: Academic, Personal, Social, Career"}), 400

        # Validate numeric
        try:
            numeric_inputs = [float(i) for i in inputs]
        except:
            return jsonify({"error": "All inputs must be numeric"}), 400

        features = np.array(numeric_inputs).reshape(1, -1)
        prediction = int(model.predict(features)[0])

        # ✅ Convert numeric prediction to label text
        result_label = label_map.get(prediction, "Unknown Result")

        return jsonify({"prediction": result_label}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Mental Health Prediction API is running!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
