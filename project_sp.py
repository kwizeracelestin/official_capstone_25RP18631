# =========================================================
# INTELLIGENT MOTOR FAULT DETECTION & HEALTH MONITORING
# FINAL Spyder-Safe Version (CSV-based, Robust)
# =========================================================

import os
import pandas as pd
import numpy as np

# ---- Safe joblib import ----
try:
    import joblib
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "❌ joblib is not installed. Install it using: pip install joblib"
    )

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# BASE DIRECTORY
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# FILE PATHS
# =========================================================
DATASET_FILE = os.path.join(BASE_DIR, "dataset.csv")
MODEL_FILE = os.path.join(BASE_DIR, "project_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "project_scaler.pkl")
ENCODER_FILE = os.path.join(BASE_DIR, "label_encoder.pkl")

# =========================================================
# LOAD DATASET
# =========================================================
def load_dataset():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"❌ Dataset not found: {DATASET_FILE}")

    df = pd.read_csv(DATASET_FILE)
    print("✅ Dataset loaded successfully")
    print(f"📊 Dataset shape: {df.shape}")
    return df

# =========================================================
# TRAIN MODEL
# =========================================================
def train_and_save_model(df):

    # --- Detect label column ---
    possible_labels = ["Fault_Label", "fault", "label", "Condition", "condition"]
    label_col = None

    for col in possible_labels:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError("❌ No fault/condition label column found")

    print(f"🔍 Label column used: {label_col}")

    # --- Split features & target ---
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # --- Keep numeric features only ---
    X = X.select_dtypes(include=["number"])

    if X.shape[1] == 0:
        raise ValueError("❌ No numeric feature columns found")

    print(f"📈 Numeric features used: {X.shape[1]}")

    # --- Encode labels ---
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Model ---
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test_scaled)

    print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))

    # ✅ FIX: convert class names to strings
    target_names = encoder.classes_.astype(str)

    print(
        "\n📊 Classification Report:\n",
        classification_report(y_test, y_pred, target_names=target_names)
    )

    # --- Save artifacts ---
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    print("💾 Model, scaler & encoder saved successfully")

# =========================================================
# TEST PREDICTION
# =========================================================
def test_prediction(df):

    if not all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, ENCODER_FILE]):
        raise FileNotFoundError("❌ Model files missing")

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)

    # Match training features
    feature_cols = df.drop(columns=[encoder.classes_.name], errors="ignore")
    feature_cols = feature_cols.select_dtypes(include=["number"]).columns.tolist()

    sample = pd.DataFrame(
        [df[feature_cols].mean().values],
        columns=feature_cols
    )

    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)
    label = encoder.inverse_transform(pred)[0]

    print("\n🔍 SAMPLE PREDICTION RESULT")
    print("⚙️ Motor Condition:", label)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("🚀 Starting Intelligent Motor Fault Detection System\n")

    df = load_dataset()

    if not all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, ENCODER_FILE]):
        print("⚠️ Training model (artifacts missing)...")
        train_and_save_model(df)
    else:
        print("ℹ️ Existing model found — skipping training")

    test_prediction(df)

    print("\n✅ Program executed successfully (Spyder)")
