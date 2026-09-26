"""
Risk-tier and recommended-action logic. No ML here -- pure rules.
"""

def get_risk(prediction: int, confidence: float):
    if prediction == 0 or confidence < 0.5:
        return {"tier": "Low", "action": "Monitor"}
    if confidence < 0.7:
        return {"tier": "Medium", "action": "Administrator verification recommended"}
    if confidence < 0.9:
        return {"tier": "High", "action": "Investigate soon"}
    return {"tier": "Critical", "action": "Immediate investigation"}
