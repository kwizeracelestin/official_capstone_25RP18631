# =========================================================
# INTELLIGENT MOTOR FAULT DETECTION & HEALTH MONITORING
# FINAL STABLE VERSION (Spyder Compatible, CSV-based)
# =========================================================

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# FILE NAMES (same directory as this script)
# =========================================================
DATASET_FILE = "dataset.csv"
MODEL_FILE = "project_model.pkl"
SCALER_FILE = "project_scaler.pkl"
ENCODER_FILE = "label_encoder.pkl"

# =========================================================
# LOAD DATASET
# =========================================================
def load_dataset():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    df = pd.read_csv(DATASET_FILE)
    print("✅ Dataset loaded successfully")
    print("📊 Dataset shape:", df.shape)
    return df

# =========================================================
# TRAIN MODEL
# =========================================================
def train_and_save_model(df):

    # ---- Detect label column automatically ----
    possible_labels = ["Fault_Label", "fault", "label", "Condition", "condition"]
    label_col = None

    for col in possible_labels:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError("❌ No fault/condition label column found in dataset")

    print(f"🔍 Label column used: {label_col}")

    # ---- Separate label and features ----
    y = df[label_col]

    # ✅ Keep ONLY numeric feature columns
    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=["number"])

    if X.empty:
        raise ValueError("❌ No numeric feature columns found in dataset")

    print(f"📈 Number of numeric features: {X.shape[1]}")

    # ---- Encode labels ----
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # ---- Train-test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # ---- Feature scaling ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Train model ----
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # ---- Evaluate model ----
    y_pred = model.predict(X_test_scaled)

    print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # ---- Save model files ----
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    print("\n💾 Model, scaler, and encoder saved successfully")

# =========================================================
# TEST PREDICTION (DEMO)
# =========================================================
def test_prediction(df):

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)

    # Use the same numeric features
    feature_cols = (
        df.drop(df.columns[-1], axis=1)
          .select_dtypes(include=["number"])
          .columns
          .tolist()
    )

    # Example input: mean of each feature
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

    # Train model only if files are missing
    if not all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, ENCODER_FILE]):
        print("⚠️ No trained model found — training now...\n")
        train_and_save_model(df)
    else:
        print("ℹ️ Existing model found — skipping training")

    test_prediction(df)

    print("\n✅ Program executed successfully (Spyder)")
