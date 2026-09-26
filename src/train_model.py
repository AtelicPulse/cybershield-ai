"""
Model training for CyberShield AI: baseline + main model + anomaly detector.
Run: python src/train_model.py
"""
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)

def load_train_test():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    with open(PROCESSED_DIR / "feature_columns.txt") as f:
        feature_cols = f.read().splitlines()
    X_train, y_train = train[feature_cols], train["is_malicious"]
    X_test, y_test = test[feature_cols], test["is_malicious"]
    return X_train, X_test, y_train, y_test, feature_cols

def balance_classes(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

def train_baseline(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, max_depth=8):
    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=max_depth, learning_rate=0.1,
        eval_metric="logloss", random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_isolation_forest(X_train):
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(X_train)
    return model

def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    print(f"\n=== {name} ===\n{classification_report(y_test, preds)}\nConfusion matrix:\n{cm}")
    return {"name": name, "report": report, "confusion_matrix": cm.tolist()}

def main():
    X_train, X_test, y_train, y_test, feature_cols = load_train_test()
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)
    results = []

    baseline = train_baseline(X_train_bal, y_train_bal)
    results.append(evaluate(baseline, X_test, y_test, "Logistic Regression (baseline)"))
    joblib.dump(baseline, MODELS_DIR / "baseline_logreg.joblib")

    rf = train_random_forest(X_train_bal, y_train_bal)
    results.append(evaluate(rf, X_test, y_test, "Random Forest"))
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")

    xgb_model = train_xgboost(X_train_bal, y_train_bal, max_depth=8)  # confirmed best depth
    results.append(evaluate(xgb_model, X_test, y_test, "XGBoost"))
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_main.joblib")

    iso = train_isolation_forest(X_train_bal)
    joblib.dump(iso, MODELS_DIR / "isolation_forest.joblib")

    with open(MODELS_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll models trained and saved to models/")

if __name__ == "__main__":
    main()
