import pandas as pd
import numpy as np
from src.explain import load_model_and_feature_columns, explain_prediction
from src.risk_engine import get_risk
from src.db import log_prediction

# Global variables to store model and feature columns once loaded
_model = None
_feature_columns = None

def _load_resources():
    global _model, _feature_columns
    if _model is None or _feature_columns is None:
        _model, _feature_columns = load_model_and_feature_columns()
        if _model is None or _feature_columns is None:
            raise RuntimeError("Failed to load model or feature columns.")

def analyze_flow(row, flow_id="unknown"):
    """
    Analyzes a single network flow, predicts its maliciousness, explains the prediction,
    assesses risk, and logs the outcome.

    Args:
        row (pd.Series or list): The input features for a single network flow.
        flow_id (str): A unique identifier for the flow.

    Returns:
        dict: A dictionary containing the analysis results.
    """
    _load_resources() # Ensure resources are loaded

    # Convert row to DataFrame for prediction if it's a Series or list
    if isinstance(row, pd.Series):
        X_input = pd.DataFrame([row.reindex(_feature_columns)], columns=_feature_columns)
    elif isinstance(row, list):
        X_input = pd.DataFrame([row], columns=_feature_columns)
    else:
        raise ValueError("Input row must be a pandas Series or list.")

    # 1. Predict and get confidence
    prediction_raw = _model.predict(X_input)[0]
    # Get probability for the malicious class (index 1)
    confidence = _model.predict_proba(X_input)[0][1]
    
    # Convert raw prediction (0 or 1) to string for output
    prediction_label = "Malicious" if prediction_raw == 1 else "Benign"

    # 2. Get SHAP explanation for top 4 features
    top_features = explain_prediction(_model, X_input.iloc[0], _feature_columns, top_n=4)

    # 3. Get risk tier and action
    risk_result = get_risk(prediction_raw, confidence)
    risk_tier = risk_result['tier']
    action = risk_result['action']

    # 4. Log the prediction
    log_prediction(flow_id, prediction_raw, confidence, risk_tier, action)

    return {
        "flow_id": flow_id,
        "prediction": prediction_label,
        "confidence": float(confidence), # Ensure JSON serializable
        "top_features": top_features,
        "risk_tier": risk_tier,
        "action": action
    }

