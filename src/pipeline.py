"""
Integration function: ties model + SHAP + risk engine together.
"""
import pandas as pd
from explain import load_explainer, explain_prediction
from risk_engine import get_risk
from db import log_prediction

def analyze_flow(row, flow_id="unknown"):
    model, explainer, feature_cols = load_explainer()
    row_df = row.to_frame().T if isinstance(row, pd.Series) else row
    row_df = row_df[feature_cols]

    prediction = int(model.predict(row_df)[0])
    proba = float(model.predict_proba(row_df)[0][1])

    top_features = explain_prediction(row_df.iloc[0])
    risk = get_risk(prediction, proba)

    log_prediction(flow_id, "Malicious" if prediction else "Benign", proba, risk["tier"], risk["action"])

    return {
        "flow_id": flow_id,
        "prediction": "Malicious" if prediction else "Benign",
        "confidence": proba,
        "top_features": top_features,
        "risk_tier": risk["tier"],
        "action": risk["action"],
    }
