"""Tests for the web application and API endpoints."""

import pytest
from fastapi.testclient import TestClient

from eagleeye.api.app import app
from eagleeye.core.database import init_db


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    """Use a temporary database for all web tests."""
    db_url = f"sqlite:///{tmp_path / 'test_web.db'}"
    monkeypatch.setenv("EAGLEEYE_DB_URL", db_url)
    # Re-init with temp db
    from eagleeye.core import config
    monkeypatch.setattr(config, "DATABASE_URL", db_url)
    init_db(db_url)


@pytest.fixture
def client():
    return TestClient(app)


class TestWebPages:
    """Test that all HTML pages load successfully."""

    def test_dashboard(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "EagleEye" in resp.text
        assert "Dashboard" in resp.text

    def test_incidents_page(self, client):
        resp = client.get("/incidents")
        assert resp.status_code == 200
        assert "Incident" in resp.text

    def test_analysis_page(self, client):
        resp = client.get("/analysis")
        assert resp.status_code == 200
        assert "Analysis" in resp.text

    def test_predictions_page(self, client):
        resp = client.get("/predictions")
        assert resp.status_code == 200
        assert "Predictions" in resp.text

    def test_alerts_page(self, client):
        resp = client.get("/alerts")
        assert resp.status_code == 200
        assert "Alerts" in resp.text

    def test_intel_page(self, client):
        resp = client.get("/intel")
        assert resp.status_code == 200
        assert "Intelligence" in resp.text

    def test_reports_page(self, client):
        resp = client.get("/reports")
        assert resp.status_code == 200
        assert "Reports" in resp.text


class TestAPIEndpoints:
    """Test core API endpoints work through the web app."""

    def test_system_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "operational"
        assert "target_states" in data

    def test_list_incidents_empty(self, client):
        resp = client.get("/api/incidents?days=30")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_create_incident(self, client):
        payload = {
            "date": "2024-06-15T03:00:00",
            "state": "Zamfara",
            "lga": "Anka",
            "incident_type": "village_raid",
            "casualties_killed": 5,
            "casualties_injured": 3,
            "kidnapped_count": 10,
            "confidence_score": 0.8,
            "threat_level": 4,
        }
        resp = client.post("/api/incidents", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert data["state"] == "Zamfara"

    def test_create_and_list_incident(self, client):
        from datetime import datetime, timedelta
        recent_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%dT03:00:00")
        payload = {
            "date": recent_date,
            "state": "Kaduna",
            "incident_type": "kidnapping",
            "casualties_killed": 1,
            "confidence_score": 0.7,
            "threat_level": 3,
        }
        resp_create = client.post("/api/incidents", json=payload)
        assert resp_create.status_code == 200
        resp = client.get("/api/incidents?days=30")
        data = resp.json()
        assert data["total"] >= 1

    def test_submit_intel_report(self, client):
        payload = {
            "content": "Armed bandits attacked a village in Anka LGA of Zamfara State, killing 10 people and kidnapping 20 residents.",
            "source": "Test Report",
        }
        resp = client.post("/api/intel/report", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processed"

    def test_get_alerts_empty(self, client):
        resp = client.get("/api/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_threat_scores_empty(self, client):
        resp = client.get("/api/threat/scores")
        assert resp.status_code == 200

    def test_static_files(self, client):
        resp = client.get("/static/css/style.css")
        assert resp.status_code == 200
        assert "EagleEye" in resp.text

    def test_static_js(self, client):
        resp = client.get("/static/js/app.js")
        assert resp.status_code == 200
        assert "function api" in resp.text
