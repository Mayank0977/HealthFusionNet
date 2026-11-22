import json
from app import app, engine, PATIENT_SCENARIOS

print("=" * 60)
print("Testing HealthFusionNet Clinical Dashboard In-Process via Flask test_client()")
print("=" * 60)

client = app.test_client()

# Test 1: Health endpoint
res = client.get("/api/health")
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
print("[PASS] /api/health:", res.get_json())

# Test 2: Scenarios endpoint
res = client.get("/api/scenarios")
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
scenarios = res.get_json()
assert len(scenarios) == 5, f"Expected 5 scenarios, got {len(scenarios)}"
print(f"[PASS] /api/scenarios: {len(scenarios)} patient archetypes verified.")

# Test 3: Prediction endpoint
diab_data = scenarios["diabetic_retinopathy"]["biomarkers"]
payload = {
    "biomarkers": diab_data,
    "retinal_severity": 0.6,
    "sensor_wearable_active": True
}
res = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
pred = res.get_json()
assert "health_score" in pred, "Missing health_score"
assert "risks" in pred, "Missing risks"
assert len(pred["risks"]) == 5, f"Expected 5 disease risks, got {len(pred['risks'])}"
print("[PASS] /api/predict:")
print(f"       Health Score: {pred['health_score']}/100 ({pred['tier']})")
print(f"       CVD Risk: {pred['risks']['CVD']['probability']}% ({pred['risks']['CVD']['level']})")
print(f"       Diabetes Risk: {pred['risks']['Diabetes']['probability']}% ({pred['risks']['Diabetes']['level']})")
print(f"       Hypertension: {pred['risks']['Hypertension']['probability']}% ({pred['risks']['Hypertension']['level']})")
print(f"       Sleep Risk: {pred['risks']['Sleep_Risk']['probability']}% ({pred['risks']['Sleep_Risk']['level']})")
print(f"       Depression Risk: {pred['risks']['Depression_Risk']['probability']}% ({pred['risks']['Depression_Risk']['level']})")
print(f"       Top SHAP: {[s['feature'] + ': ' + str(s['impact']) + '%' for s in pred['shap_top_drivers'][:3]]}")
print(f"       Actionable Recommendations: {len(pred['recommendations'])} items")

# Test 4: Chatbot endpoint
chat_payload = {
    "message": "Why is the diabetes risk so high for this patient?",
    "prediction": pred,
    "biomarkers": diab_data
}
res = client.post("/api/chat", data=json.dumps(chat_payload), content_type="application/json")
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
chat_resp = res.get_json()
assert "reply" in chat_resp and len(chat_resp["reply"]) > 20, "Invalid chat response"
print("[PASS] /api/chat response snippet:")
print("       " + chat_resp["reply"][:180].replace("\n", " "))

# Test 5: Root index.html template rendering
res = client.get("/")
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
html = res.get_data(as_text=True)
assert "HealthFusionNet" in html, "Missing HealthFusionNet brand"
assert "id=\"scenarios\"" in html, "Missing scenarios section"
assert "id=\"radarChart\"" in html, "Missing radar chart canvas"
assert "id=\"printable-report\"" in html, "Missing printable report"
print(f"[PASS] / (index.html): Successfully rendered ({len(html)} bytes).")

print("=" * 60)
print("ALL CLINICAL DASHBOARD TESTS PASSED WITH 100% SUCCESS!")
print("=" * 60)
