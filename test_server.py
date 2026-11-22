import os
import subprocess
import time
import urllib.request
import json
import sys

app_dir = r"C:\Users\bsfar\.gemini\antigravity\scratch\HealthFusionNet"
app_path = os.path.join(app_dir, "app.py")

print("Starting test server process at:", app_path)
proc = subprocess.Popen([sys.executable, app_path], cwd=app_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

url = "http://127.0.0.1:5000"
ready = False

# Wait for server to be up
for attempt in range(80):
    time.sleep(1)
    try:
        req = urllib.request.urlopen(f"{url}/api/health", timeout=2)
        if req.status == 200:
            print(f"Server is up and responsive on attempt {attempt}!")
            ready = True
            break
    except Exception as e:
        if attempt % 10 == 0:
            print(f"Waiting for server... (attempt {attempt})")

if not ready:
    print("Server failed to start in time. Output:")
    proc.terminate()
    out, _ = proc.communicate(timeout=5)
    print(out)
    sys.exit(1)

# Test 1: Health check
with urllib.request.urlopen(f"{url}/api/health") as resp:
    data = json.loads(resp.read().decode())
    print("Health check:", data)

# Test 2: Scenarios endpoint
with urllib.request.urlopen(f"{url}/api/scenarios") as resp:
    scenarios = json.loads(resp.read().decode())
    print(f"Scenarios loaded: {len(scenarios)} ({list(scenarios.keys())})")

# Test 3: Prediction endpoint
req_data = json.dumps({
    "biomarkers": scenarios["diabetic_retinopathy"]["biomarkers"],
    "retinal_severity": 0.6,
    "sensor_wearable_active": True
}).encode("utf-8")

req = urllib.request.Request(f"{url}/api/predict", data=req_data, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    pred = json.loads(resp.read().decode())
    print("Prediction test:")
    print(f"  Health Score: {pred['health_score']} ({pred['tier']})")
    print("  Risks:", {k: f"{v['probability']}%" for k, v in pred['risks'].items()})
    print("  Top SHAP:", [d['feature'] for d in pred['shap_top_drivers'][:3]])

# Test 4: Chat endpoint
chat_data = json.dumps({
    "message": "Why is diabetes risk elevated for this patient?",
    "prediction": pred,
    "biomarkers": scenarios["diabetic_retinopathy"]["biomarkers"]
}).encode("utf-8")

chat_req = urllib.request.Request(f"{url}/api/chat", data=chat_data, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(chat_req) as resp:
    chat_resp = json.loads(resp.read().decode())
    print("Chat response snippet:\n ", chat_resp["reply"][:150], "...")

# Test 5: Root HTML check
with urllib.request.urlopen(f"{url}/") as resp:
    html = resp.read().decode("utf-8")
    print(f"Root HTML fetched successfully! Length: {len(html)} bytes | Contains 'HealthFusionNet': {'HealthFusionNet' in html}")

# Terminate cleanly
proc.terminate()
proc.wait()
print("\nAll endpoints verified successfully!")
