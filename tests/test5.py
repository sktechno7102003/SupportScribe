import os
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_triage_valid_request():
    payload = {
        "ticket_id": "TEST-1001",
        "raw_text": "I was double charged on my invoice INV-4029! Please issue a refund."
    }
    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200, f"Triage endpoint failed: {response.text}"
    
    data = response.json()
    assert data["ticket_id"] == "TEST-1001"
    assert data["urgency_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert data["primary_category"] in ["BILLING", "TECHNICAL_BUG", "ACCOUNT_ACCESS", "FEATURE_REQUEST"]
    assert data["sentiment"] in ["POSITIVE", "NEUTRAL", "NEGATIVE", "FRUSTRATED"]
    assert isinstance(data["extracted_entities"], list)
    assert len(data["recommended_action"]) > 0
    assert data["processing_time_ms"] > 0.0

def test_triage_invalid_schema_rejection():
    # Sending missing raw_text should trigger 422 Unprocessable Entity
    invalid_payload = {"ticket_id": "TEST-BAD"}
    response = client.post("/api/v1/triage", json=invalid_payload)
    assert response.status_code == 422

def test_load_test_report_exists():
    report_path = "reports/api_load_test_report.json"
    assert os.path.exists(report_path), "reports/api_load_test_report.json missing! Run scripts/load_test.py."