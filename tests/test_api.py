from fastapi.testclient import TestClient
from src.serving.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_high_risk():
    response = client.post("/predict", json={
        "subject": "Server outage",
        "body": "Our production server is down",
        "type": "Incident",
        "queue": "Technical Support",
        "tags": ["Outage", "Crash"]
    })
    assert response.status_code == 200
    assert "escalation_risk" in response.json()

def test_predict_invalid_input():
    response = client.post("/predict", json={"subject": "Missing fields"})
    assert response.status_code == 422