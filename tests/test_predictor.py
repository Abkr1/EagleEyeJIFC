"""Tests for ML prediction models."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from eagleeye.intel.report_parser import ReportParser
from eagleeye.models.anomaly_detector import AnomalyDetector
from eagleeye.models.predictor import BanditActivityPredictor
from eagleeye.models.threat_scorer import ThreatScorer


@pytest.fixture
def training_df():
    """Create a training dataset with enough samples."""
    base_date = datetime(2024, 1, 1)
    records = []
    for i in range(60):
        records.append({
            "id": i + 1,
            "date": base_date + timedelta(days=i * 2),
            "state": ["Zamfara", "Kaduna", "Katsina", "Sokoto", "Niger", "Kebbi"][i % 6],
            "lga": ["Anka", "Birnin Gwari", "Batsari", "Rabah", "Shiroro", "Zuru"][i % 6],
            "location_name": f"Location_{i}",
            "latitude": 10.0 + (i % 10) * 0.3,
            "longitude": 5.0 + (i % 8) * 0.4,
            "incident_type": [
                "village_raid", "kidnapping", "highway_ambush",
                "cattle_rustling", "attack_on_security_forces",
                "market_attack", "mining_site_attack",
            ][i % 7],
            "description": f"Test incident {i}",
            "casualties_killed": i % 10,
            "casualties_injured": i % 6,
            "kidnapped_count": (i % 5) * 4,
            "cattle_stolen": (i % 4) * 30,
            "estimated_bandits": 15 + i % 40,
            "threat_level": (i % 5) + 1,
            "confidence_score": 0.75,
            "source": "Test",
        })
    return pd.DataFrame(records)


class TestBanditActivityPredictor:
    def test_train(self, training_df):
        predictor = BanditActivityPredictor()
        result = predictor.train(training_df)
        assert result["status"] == "success"
        assert result["type_model_accuracy"] >= 0
        assert result["location_model_accuracy"] >= 0
        assert predictor.is_trained

    def test_predict(self, training_df):
        predictor = BanditActivityPredictor()
        predictor.train(training_df)

        context = {
            "date": datetime.now(),
            "state": "Zamfara",
            "recent_incidents_7d": 3,
            "recent_incidents_14d": 6,
            "recent_incidents_30d": 12,
            "recent_killed_7d": 5,
            "recent_killed_30d": 15,
            "last_incident_days_ago": 2,
            "current_threat_level": 4,
        }
        prediction = predictor.predict(context)
        assert "predicted_incident_types" in prediction
        assert "predicted_locations" in prediction
        assert "threat_level" in prediction
        assert "recommendations" in prediction
        assert len(prediction["predicted_incident_types"]) > 0

    def test_predict_multi_day(self, training_df):
        predictor = BanditActivityPredictor()
        predictor.train(training_df)

        context = {
            "date": datetime.now(),
            "state": "Kaduna",
            "recent_incidents_7d": 2,
            "recent_incidents_30d": 8,
        }
        predictions = predictor.predict_multi_day(context, horizon_days=7)
        assert len(predictions) == 7
        for p in predictions:
            assert "day_offset" in p

    def test_insufficient_data(self):
        predictor = BanditActivityPredictor()
        small_df = pd.DataFrame([
            {"id": 1, "date": "2024-01-01", "state": "Zamfara", "lga": "Anka",
             "incident_type": "village_raid", "casualties_killed": 1,
             "casualties_injured": 0, "kidnapped_count": 0, "cattle_stolen": 0,
             "estimated_bandits": 10, "threat_level": 3, "confidence_score": 0.7}
        ])
        result = predictor.train(small_df)
        assert result["status"] == "error"

    def test_predict_without_training(self):
        predictor = BanditActivityPredictor()
        result = predictor.predict({"date": datetime.now()})
        assert "error" in result


class TestThreatScorer:
    def test_score_state(self, training_df):
        scorer = ThreatScorer(training_df)
        score = scorer.score_state("Zamfara")
        assert "overall_score" in score
        assert "threat_level" in score
        assert 1 <= score["threat_level"] <= 5

    def test_score_all_states(self, training_df):
        scorer = ThreatScorer(training_df)
        scores = scorer.score_all_states()
        assert len(scores) > 0
        # Should be sorted by score descending
        for i in range(len(scores) - 1):
            assert scores[i]["overall_score"] >= scores[i + 1]["overall_score"]

    def test_score_unknown_state(self, training_df):
        scorer = ThreatScorer(training_df)
        score = scorer.score_state("NonexistentState")
        assert score["overall_score"] == 0
        assert score["threat_level"] == 1

    def test_score_lga(self, training_df):
        scorer = ThreatScorer(training_df)
        score = scorer.score_lga("Zamfara", "Anka")
        assert "overall_score" in score


class TestAnomalyDetector:
    def test_fit(self, training_df):
        detector = AnomalyDetector()
        result = detector.fit(training_df)
        assert result["status"] == "success"
        assert detector.is_fitted

    def test_detect(self, training_df):
        detector = AnomalyDetector()
        detector.fit(training_df)
        anomalies = detector.detect(training_df)
        assert isinstance(anomalies, list)
        for a in anomalies:
            assert "anomaly_score" in a
            assert "anomaly_reasons" in a

    def test_detect_without_fit(self):
        detector = AnomalyDetector()
        result = detector.detect(pd.DataFrame())
        assert result[0].get("error") is not None

    def test_pattern_breaks(self, training_df):
        detector = AnomalyDetector()
        breaks = detector.detect_pattern_breaks(training_df)
        assert isinstance(breaks, list)


class TestReportParser:
    def test_parse_village_raid(self):
        parser = ReportParser()
        text = (
            "Armed bandits attacked Badarawa village in Anka Local Government Area "
            "of Zamfara State on January 15, 2024. The gunmen, numbering about 50, "
            "arrived on motorcycles and killed 12 people, injured 8 others, and "
            "kidnapped 25 residents. They also rustled over 200 cattle. The attackers "
            "were armed with AK-47 rifles and machine guns."
        )
        incidents = parser.parse_report(text, source="Test")
        assert len(incidents) >= 1
        inc = incidents[0]
        assert inc["state"] == "Zamfara"
        assert inc["lga"] == "Anka"
        assert inc["incident_type"] == "village_raid"
        assert inc["casualties_killed"] >= 12
        assert inc["kidnapped_count"] >= 25

    def test_parse_kidnapping(self):
        parser = ReportParser()
        text = (
            "Suspected bandits kidnapped 15 travellers along the Birnin Gwari-Kaduna "
            "highway in Kaduna State. The attackers, who were on motorcycles and armed "
            "with AK-47 rifles, set up a roadblock before abducting the victims."
        )
        incidents = parser.parse_report(text, source="Test")
        assert len(incidents) >= 1
        inc = incidents[0]
        assert inc["state"] == "Kaduna"
        assert inc["incident_type"] in ("kidnapping", "highway_ambush")

    def test_parse_no_incident(self):
        parser = ReportParser()
        text = "The weather in Lagos was sunny today. Markets were busy."
        incidents = parser.parse_report(text)
        assert len(incidents) == 0

    def test_extract_weapons(self):
        parser = ReportParser()
        text = (
            "Bandits armed with AK-47 rifles, machine guns and RPGs attacked "
            "a village in Zamfara State."
        )
        incidents = parser.parse_report(text)
        if incidents:
            weapons = incidents[0].get("weapons_observed", "")
            assert "AK-47" in weapons

    def test_extract_vehicles(self):
        parser = ReportParser()
        text = (
            "The gunmen arrived on motorcycles and horses to attack the "
            "community in Katsina State. Over 30 bandits killed 5 people."
        )
        incidents = parser.parse_report(text)
        if incidents:
            vehicles = incidents[0].get("vehicles_used", "")
            assert "motorcycle" in vehicles
