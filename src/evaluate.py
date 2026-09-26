"""
Reproducible evaluation script + simple rule-based baseline.
Run: python src/evaluate.py
"""
import pandas as pd
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def rule_based_predict(df, packet_rate_threshold=2.0):
    """Features are StandardScaler-scaled -- this threshold is in z-score units."""
    col = "Flow Packets/s" if "Flow Packets/s" in df.columns else None
    if col is None:
        raise KeyError("Flow Packets/s column not found -- check feature_columns.txt")
    return (df[col] > packet_rate_threshold).astype(int)

def plot_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    with open(PROCESSED_DIR / "feature_columns.txt") as f:
        feature_cols = f.read().splitlines()
    X_test, y_test = test[feature_cols], test["is_malicious"]

    model = joblib.load(MODELS_DIR / "xgboost_main.joblib")
    preds = model.predict(X_test)
    print("=== Main model (XGBoost) ===")
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    plot_confusion_matrix(cm, ["Benign", "Malicious"], "XGBoost - Confusion Matrix",
                           FIG_DIR / "xgboost_confusion_matrix.png")

    rule_preds = rule_based_predict(test)
    print("\n=== Rule-based baseline ===")
    print(classification_report(y_test, rule_preds))

    with open(MODELS_DIR / "training_results.json") as f:
        training_results = json.load(f)
    logreg_report = training_results["Logistic Regression"]["malicious_metrics"]

    comparison = {
        "Rule-based": classification_report(y_test, rule_preds, output_dict=True)["1"],
        "Logistic Regression": {
            "precision": logreg_report["precision"],
            "recall": logreg_report["recall"],
            "f1-score": logreg_report["f1_score"],
        },
        "XGBoost": classification_report(y_test, preds, output_dict=True)["1"],
    }
    with open(FIG_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved confusion matrix + comparison to {FIG_DIR}")

if __name__ == "__main__":
    main()
