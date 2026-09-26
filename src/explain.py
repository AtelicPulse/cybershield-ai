"""
SHAP explainability wrapper.
Run: python src/explain.py
"""
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

_model = None
_explainer = None
_feature_cols = None

def load_explainer():
    global _model, _explainer, _feature_cols
    if _model is None:
        _model = joblib.load(MODELS_DIR / "xgboost_main.joblib")
        _explainer = shap.TreeExplainer(_model)
        with open(PROCESSED_DIR / "feature_columns.txt") as f:
            _feature_cols = f.read().splitlines()
    return _model, _explainer, _feature_cols

def global_summary_plot(sample_size=2000):
    model, explainer, feature_cols = load_explainer()
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    sample = test[feature_cols].sample(min(sample_size, len(test)), random_state=42)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_global_summary.png", dpi=150)
    plt.close()
    print(f"Saved global SHAP summary to {FIG_DIR / 'shap_global_summary.png'}")

def explain_prediction(row, top_n=4):
    model, explainer, feature_cols = load_explainer()
    row_df = row.to_frame().T if isinstance(row, pd.Series) else row
    row_df = row_df[feature_cols]
    shap_values = explainer.shap_values(row_df)
    values = shap_values[0] if hasattr(shap_values, "__len__") else shap_values
    pairs = list(zip(feature_cols, values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]

if __name__ == "__main__":
    global_summary_plot()
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    example = test.iloc[0]
    top_features = explain_prediction(example)
    print("\nExample local explanation (row 0):")
    for feat, val in top_features:
        print(f"  {feat}: {val:+.4f}")
