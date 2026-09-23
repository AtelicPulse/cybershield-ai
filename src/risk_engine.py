def get_risk(prediction, confidence):
    """
    Determines the risk tier and action based on model prediction and confidence.

    Args:
        prediction (int): 0 for benign, 1 for malicious.
        confidence (float): Model's predicted probability of the malicious class (0-1).

    Returns:
        dict: A dictionary with 'tier' and 'action'.
    """
    if prediction == 0 or confidence < 0.5:
        return {"tier": "Low", "action": "Monitor"}
    elif 0.5 <= confidence < 0.7:
        return {"tier": "Medium", "action": "Administrator verification recommended"}
    elif 0.7 <= confidence < 0.9:
        return {"tier": "High", "action": "Investigate soon"}
    else:  # confidence >= 0.9
        return {"tier": "Critical", "action": "Immediate investigation"}
