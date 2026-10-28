import json
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

ROOT = Path(__file__).resolve().parents[1]
FEATURES = json.loads((ROOT / "feature_names.json").read_text(encoding="utf-8"))
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["feature_count"] == 30


def test_prediction_endpoint():
    # Values are synthetic smoke-test inputs, not a real patient record.
    payload = {"features": {name: 1.0 for name in FEATURES}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in ["B", "M"]
    assert body["prediction_label"] in ["benign", "malignant"]
    assert "probabilities" in body
    assert "disclaimer" in body


def test_missing_feature_is_rejected():
    payload = {"features": {name: 1.0 for name in FEATURES[:-1]}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
