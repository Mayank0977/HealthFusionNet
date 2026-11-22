from healthfusion_engine import HealthFusionEngine, PATIENT_SCENARIOS

engine = HealthFusionEngine()
scenario = PATIENT_SCENARIOS['diabetic_retinopathy']
pred = engine.predict(scenario['biomarkers'], retinal_severity=0.6, sensor_wearable_active=True)

print("Health Score:", pred['health_score'], pred['tier'])
for k, v in pred['risks'].items():
    print(f"  {k}: {v['probability']}% ({v['level']})")
print("Top SHAP:", pred['shap_top_drivers'][:2])
print("Chat test:\n", engine.chat_response('Why is diabetes risk high?', pred, scenario['biomarkers'])[:200])
