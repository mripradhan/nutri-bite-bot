import os
import zipfile
import numpy as np
import pandas as pd
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "models")

# Load TabNet model from standalone model_params.json + network.pt
def load_tabnet(model_dir):
    params_path = os.path.join(model_dir, "model_params.json")
    weights_path = os.path.join(model_dir, "network.pt")
    tmp_zip = os.path.join(model_dir, "_tabnet_test_tmp.zip")
    try:
        with zipfile.ZipFile(tmp_zip, "w") as zf:
            zf.write(params_path, "model_params.json")
            zf.write(weights_path, "network.pt")
        model = TabNetClassifier()
        model.load_model(tmp_zip)
        return model
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)

tabnet_model = load_tabnet(MODEL_DIR)

# Load preprocessing parameters (extract raw numpy arrays to avoid sklearn version issues)
_raw_imputer = joblib.load(os.path.join(MODEL_DIR, 'imputer.joblib'))
_raw_scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
features = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))

IMPUTER_FILL_VALUES = _raw_imputer.statistics_
SCALER_MEAN = _raw_scaler.mean_
SCALER_SCALE = _raw_scaler.scale_

def preprocess(X_df):
    """Manual imputation + scaling to avoid sklearn version issues."""
    X = X_df.values.astype(np.float64)
    for col_idx in range(X.shape[1]):
        mask = np.isnan(X[:, col_idx])
        if mask.any():
            X[mask, col_idx] = IMPUTER_FILL_VALUES[col_idx]
    X = (X - SCALER_MEAN) / SCALER_SCALE
    return X

# Test patient with diabetes risk factors
patient = {
    'age': 55,
    'sex_male': 1,
    'has_htn': 1,
    'has_dm': 1,        # Has diabetes
    'has_ckd': 1,
    'serum_sodium': 144,
    'serum_potassium': 5.6,
    'creatinine': 2.3,
    'egfr': 28,
    'hba1c': 8.4,       # High HbA1c (diabetic)
    'fbs': 170,         # High fasting blood sugar
    'sbp': 152,
    'dbp': 94,
    'bmi': 32           # Obese
}

# Prepare features
X = pd.DataFrame([patient])[features]
X = preprocess(X).astype(np.float32)  # TabNet requires float32

# Get predictions
label_map = {0: 'low', 1: 'moderate', 2: 'high'}

targets = [
    'sodium_sensitivity',
    'potassium_sensitivity',
    'protein_restriction',
    'carb_sensitivity',
]

print("=" * 50)
print("CLINICAL RISK STRATIFICATION RESULTS (TabNet)")
print("=" * 50)
print(f"\nPatient: Age {patient['age']}, Diabetes: Yes, CKD: Yes")
print(f"HbA1c: {patient['hba1c']}%, FBS: {patient['fbs']} mg/dL, BMI: {patient['bmi']}")
print("\nRisk Levels:")
print("-" * 50)

for target in targets:
    pred = tabnet_model.predict(X)[0]
    proba = tabnet_model.predict_proba(X)[0]
    print(f"  {target:25s}: {label_map[pred]:10s} ({proba[pred]*100:.1f}% confidence)")

print("=" * 50)