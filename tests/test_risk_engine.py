import pytest
from src.risk_engine import get_risk

def test_risk_low_benign_prediction():
    # Prediction is benign (0), should always be Low
    result = get_risk(0, 0.95)
    assert result == {"tier": "Low", "action": "Monitor"}

def test_risk_low_malicious_low_confidence():
    # Malicious prediction, but confidence < 0.5
    result = get_risk(1, 0.49)
    assert result == {"tier": "Low", "action": "Monitor"}

def test_risk_medium():
    # Malicious prediction, confidence 0.5-0.7
    result = get_risk(1, 0.65)
    assert result == {"tier": "Medium", "action": "Administrator verification recommended"}
    result = get_risk(1, 0.50)
    assert result == {"tier": "Medium", "action": "Administrator verification recommended"}
    result = get_risk(1, 0.69999)
    assert result == {"tier": "Medium", "action": "Administrator verification recommended"}

def test_risk_high():
    # Malicious prediction, confidence 0.7-0.9
    result = get_risk(1, 0.85)
    assert result == {"tier": "High", "action": "Investigate soon"}
    result = get_risk(1, 0.70)
    assert result == {"tier": "High", "action": "Investigate soon"}
    result = get_risk(1, 0.89999)
    assert result == {"tier": "High", "action": "Investigate soon"}

def test_risk_critical():
    # Malicious prediction, confidence >= 0.9
    result = get_risk(1, 0.90)
    assert result == {"tier": "Critical", "action": "Immediate investigation"}
    result = get_risk(1, 0.99)
    assert result == {"tier": "Critical", "action": "Immediate investigation"}
