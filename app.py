import os
import sys
from flask import Flask, render_template, request, jsonify
from healthfusion_engine import HealthFusionEngine, PATIENT_SCENARIOS, DISEASE_METADATA

app = Flask(__name__, template_folder="templates")

# Initialize HealthFusion AI Engine
print("=" * 70)
print("  HEALTHFUSIONNET — CLINICAL DECISION SUPPORT DASHBOARD SERVER")
print("=" * 70)
engine = HealthFusionEngine()
print("Engine successfully bound to Flask server.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/scenarios", methods=["GET"])
def get_scenarios():
    return jsonify(PATIENT_SCENARIOS)

@app.route("/api/diseases", methods=["GET"])
def get_diseases():
    return jsonify(DISEASE_METADATA)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    biomarkers = data.get("biomarkers", {})
    retinal_severity = float(data.get("retinal_severity", 0.0))
    sensor_wearable = bool(data.get("sensor_wearable_active", True))

    prediction = engine.predict(
        biomarkers_dict=biomarkers,
        retinal_severity=retinal_severity,
        sensor_wearable_active=sensor_wearable
    )
    return jsonify(prediction)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", "")
    current_prediction = data.get("prediction", {})
    biomarkers = data.get("biomarkers", {})

    reply = engine.chat_response(user_msg, current_prediction, biomarkers)
    return jsonify({"reply": reply})

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "framework": "HealthFusionNet v4.1",
        "author": "Mayank Kumar Jaiswal (Q-ID: 23030096)",
        "macro_test_roc_auc": 0.8564
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n* HealthFusionNet Dashboard running at: http://127.0.0.1:{port}")
    print("* Press Ctrl+C to terminate the server.\n")
    app.run(host="127.0.0.1", port=port, debug=False)
