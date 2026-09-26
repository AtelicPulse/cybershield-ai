import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from risk_engine import get_risk

def test_benign_is_low():
    assert get_risk(prediction=0, confidence=0.95)["tier"] == "Low"

def test_critical_threshold():
    result = get_risk(prediction=1, confidence=0.95)
    assert result["tier"] == "Critical"
    assert "Immediate" in result["action"]

def test_medium_threshold():
    assert get_risk(prediction=1, confidence=0.6)["tier"] == "Medium"

def test_low_confidence_malicious_is_low():
    assert get_risk(prediction=1, confidence=0.3)["tier"] == "Low"
