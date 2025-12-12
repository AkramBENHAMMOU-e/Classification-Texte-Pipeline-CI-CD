from fastapi.testclient import TestClient
import sys
import os
import pytest

# Ajout du dossier root au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.app import app

@pytest.fixture(scope="module")
def client():
    """Fixture qui crée un TestClient avec lifespan events activés."""
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """Vérifie que l'endpoint /health répond 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True

def test_predict_endpoint(client):
    """Vérifie que l'endpoint /predict retourne une prédiction."""
    payload = {"text": "Computer graphics involve GPU and CPU."}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction_class_id" in data
    assert isinstance(data["prediction_class_id"], int)
    assert data["status"] == "success"

def test_predict_empty_text(client):
    """Vérifie le comportement avec une string vide."""
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

