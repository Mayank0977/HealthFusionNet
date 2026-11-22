# HealthFusionNet: Multi-Hybrid Explainable AI Clinical Dashboard

**Lead Researcher:** Mayank Kumar Jaiswal (Q-ID: 23030096)  
**Framework:** HealthFusionNet v4.1  
**Cohort:** $N = 3,067$ Common NHANES Participants  
**Macro Test ROC-AUC:** **0.8564** ($+11.43\%$ relative improvement over clinical baseline)

---

## Overview

**HealthFusionNet** is a state-of-the-art multimodal clinical decision support framework that predicts 5 chronic lifestyle disorders simultaneously:
1. **Cardiovascular Disease (CVD)**
2. **Type 2 Diabetes Mellitus**
3. **Essential Hypertension**
4. **Sleep-Related Disorders**
5. **Major Depression Risk**

This repository contains the interactive, animated, web-based clinical dashboard (`app.py`), enabling clinicians and researchers to:
- Test preloaded **patient archetype scenarios** (Healthy Adult, Diabetic Retinopathy, Hypertensive Cardiac Strain, Insomnia & Depressive Strain, Geriatric Multi-Morbidity).
- Tweak **15 clinical biomarkers** in real-time with physiological sliders and immediate risk updates.
- Upload and inspect **digital retinal fundus scans** (Vision Transformer microvascular feature extraction).
- Toggle between **Continuous Smartwatch Telemetry** and **Phone-Only Mode** to evaluate the mobile ecosystem advantage.
- View real-time **0–100 Composite Health Scores** and calibrated 5-disease risk meters.
- Explore **Explainable AI (SHAP) factor attributions** isolating individual risk escalators vs. protective factors.
- Consult an embedded **AI Clinical Copilot** (Chatbot) for personalized pathophysiological interpretations and lifestyle interventions.
- Generate and print an official, high-resolution **Clinical Health Risk Assessment Report** (`window.print()`).

---

## Architecture Summary

```
                       [ 15 Clinical Biomarkers ]   --> Dense Projection (128-dim)
                       [ Continuous Accelerometry ] --> BiLSTM Motion Encoder (128-dim)
                       [ Retinal Fundus Imaging ]  --> Vision Transformer ViT (128-dim)
                       [ Patient Relational Graph ] --> Multi-Task GCN (128-dim)
                                       │
                                       ▼
                       [ Multimodal Cross-Attention & GMU ]
                                       │
                                       ▼
                    [ Shared Comorbidity Bottleneck (128-dim) ]
                     ├── Task Head 1: Cardiovascular Disease (Sigmoid)
                     ├── Task Head 2: Type 2 Diabetes (Sigmoid)
                     ├── Task Head 3: Essential Hypertension (Sigmoid)
                     ├── Task Head 4: Sleep-Related Disorders (Sigmoid)
                     └── Task Head 5: Major Depression Risk (Sigmoid)
```

---

## Quickstart: Launching the Dashboard

### 1. Requirements
Ensure Python 3.10+ and the required packages are installed:
```bash
pip install flask tensorflow numpy pandas matplotlib scikit-learn
```

### 2. Start the Web Server
Launch the Flask dashboard application:
```bash
python app.py
```

### 3. Open in Browser
Navigate to:
```
http://127.0.0.1:5000
```

---

## Key Features

### 1. Interactive Patient Archetypes
Choose from 5 preloaded patient scenarios to instantly populate all physiological and imaging parameters:
- **Healthy Young Adult**: 32yo, optimal HbA1c (5.1%), normal blood pressure (116/74 mmHg), 8h sleep.
- **Type 2 Diabetes & Microvascular Retinopathy**: 58yo, HbA1c 9.2%, fasting glucose 178 mg/dL, retinal microaneurysms.
- **Hypertensive Cardiac Strain**: 64yo, Stage 2 hypertension (164/98 mmHg), high sedentary time.
- **Severe Insomnia & Depressive Strain**: 41yo, sleep curtailment (4.2h), PHQ-9 index 17/27.
- **Geriatric Multi-Morbidity**: 74yo, compound metabolic, vascular, and circadian burden.

### 2. Real-Time Biomarker Control Room
Interactive sliders for all 15 clinical biomarkers:
- **Metabolic**: Fasting Glucose, HbA1c, BMI, Weight, Waist Circumference.
- **Cardiovascular**: Systolic BP, Diastolic BP, Resting Heart Rate.
- **Circadian & Lifestyle**: Sleep Duration, Sedentary Minutes, Vigorous/Moderate Physical Activity.
- **Neuro-Affective**: PHQ-9 Score, Age.

### 3. Multimodal Imaging & Sensor Studio
- **Retinal Fundus Scanner**: Simulated microvascular lesion severity slider (Grade 0: Normal to Grade 4: Proliferative Retinopathy) with laser scan animation.
- **Smartwatch Actigraphy Switch**: Toggle continuous wearable telemetry on or off to evaluate the mobile ecosystem advantage ($+2.07\%$ sleep risk gain).

### 4. 0–100 Composite Health Score Engine
Calculated via calibrated morbidity loss weighting:
$$\text{Health Score} = 100 \times \left(1 - \sum_{d=1}^5 \beta_d \cdot \hat{y}_d\right)$$
- **Optimal Health**: $>75$
- **Mild Lifestyle Risk**: $55 - 75$
- **Moderate Clinical Risk**: $35 - 55$
- **High / Critical Risk**: $<35$

### 5. Explainable AI (SHAP) Factor Attribution
Visualizes top positive risk contributors (red) and protective factors (green) derived from Game-Theoretic Shapley values.

### 6. Interactive AI Clinical Copilot (Chatbot)
Click the floating robot icon to open the **HealthFusion Assistant**. Ask natural language questions like:
- *"Why is this patient's diabetes risk elevated?"*
- *"What happens if HbA1c drops to 5.8%?"*
- *"Suggest a personalized 7-day lifestyle intervention."*

### 7. Official Printable Clinical Report
Click **Generate Clinical Report** to open a comprehensive medical diagnostic report with:
- Patient demographics & file ID.
- Complete vital signs & laboratory biomarkers table.
- Multi-task risk matrix & calibrated health score.
- Top SHAP attribution breakdown.
- Tailored clinical recommendations.
- Attending physician review and signature block.
- One-click print/save to PDF (`window.print()`).

---

## REST API Endpoints

- `GET /api/scenarios`: Returns preloaded patient archetype configurations.
- `GET /api/diseases`: Returns clinical definitions and diagnostic thresholds for all 5 diseases.
- `POST /api/predict`: Runs multimodal inference given JSON biomarkers and imaging parameters.
- `POST /api/chat`: Generates context-aware clinical copilot consultation responses.
- `GET /api/health`: Health-check and model metadata endpoint.
