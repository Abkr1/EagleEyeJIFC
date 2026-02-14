"""Tests for analysis modules."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from eagleeye.analysis.correlation import CorrelationAnalyzer
from eagleeye.analysis.spatial import SpatialAnalyzer, haversine_distance
from eagleeye.analysis.temporal import TemporalAnalyzer


@pytest.fixture
def sample_df():
    """Create a sample incidents DataFrame for testing."""
    base_date = datetime(2024, 1, 1)
    records = []
    for i in range(40):
        records.append({
            "id": i + 1,
            "date": base_date + timedelta(days=i * 3),
            "state": ["Zamfara", "Kaduna", "Katsina", "Sokoto"][i % 4],
            "lga": ["Anka", "Birnin Gwari", "Batsari", "Rabah"][i % 4],
            "location_name": f"Village_{i}",
            "latitude": 10.0 + (i % 10) * 0.3,
            "longitude": 5.0 + (i % 8) * 0.4,
            "incident_type": [
                "village_raid", "kidnapping", "highway_ambush",
                "cattle_rustling", "attack_on_security_forces",
            ][i % 5],
            "description": f"Test incident {i}",
            "casualties_killed": i % 8,
            "casualties_injured": i % 5,
            "kidnapped_count": (i % 6) * 3,
            "cattle_stolen": (i % 3) * 50,
            "estimated_bandits": 20 + i % 30,
            "threat_level": (i % 5) + 1,
            "confidence_score": 0.7,
            "source": "Test",
        })
    return pd.DataFrame(records)


class TestTemporalAnalyzer:
    def test_monthly_trend(self, sample_df):
        analyzer = TemporalAnalyzer(sample_df)
        trend = analyzer.monthly_trend()
        assert not trend.empty
        assert "incident_count" in trend.columns

    def test_seasonal_pattern(self, sample_df):
        analyzer = TemporalAnalyzer(sample_df)
        pattern = analyzer.seasonal_pattern()
        assert "dry_early" in pattern
        assert "rainy" in pattern
        assert "peak_season" in pattern

    def test_day_of_week_pattern(self, sample_df):
        analyzer = TemporalAnalyzer(sample_df)
        pattern = analyzer.day_of_week_pattern()
        assert "peak_day" in pattern
        assert "Monday" in pattern

    def test_frequency_analysis(self, sample_df):
        analyzer = TemporalAnalyzer(sample_df)
        freq = analyzer.frequency_analysis()
        assert "trend" in freq
        assert "incidents_per_week" in freq
        assert freq["trend"] in ("escalating", "de-escalating", "stable")

    def test_frequency_analysis_by_state(self, sample_df):
        analyzer = TemporalAnalyzer(sample_df)
        freq = analyzer.frequency_analysis(state="Zamfara")
        assert "recent_count" in freq

    def test_detect_surge_periods(self, sample_df):
        analyzer = TemporalAnalyzer(sample_df)
        surges = analyzer.detect_surge_periods(threshold_multiplier=1.5)
        assert isinstance(surges, list)

    def test_predict_next_window(self, sample_df):
        analyzer = TemporalAnalyzer(sample_df)
        prediction = analyzer.predict_next_window()
        assert "expected_incidents" in prediction
        assert "current_trend" in prediction

    def test_empty_dataframe(self):
        analyzer = TemporalAnalyzer(pd.DataFrame())
        assert analyzer.monthly_trend().empty
        assert analyzer.seasonal_pattern() == {}


class TestSpatialAnalyzer:
    def test_hotspot_analysis(self, sample_df):
        analyzer = SpatialAnalyzer(sample_df)
        hotspots = analyzer.hotspot_analysis(top_n=5)
        assert len(hotspots) <= 5
        if hotspots:
            assert "risk_score" in hotspots[0]
            assert "state" in hotspots[0]

    def test_hotspot_analysis_by_state(self, sample_df):
        analyzer = SpatialAnalyzer(sample_df)
        hotspots = analyzer.hotspot_analysis(state="Zamfara")
        for spot in hotspots:
            assert spot["state"] == "Zamfara"

    def test_state_comparison(self, sample_df):
        analyzer = SpatialAnalyzer(sample_df)
        comparison = analyzer.state_comparison()
        assert len(comparison) > 0
        assert "total_incidents" in comparison[0]

    def test_geographic_clustering(self, sample_df):
        analyzer = SpatialAnalyzer(sample_df)
        clusters = analyzer.geographic_clustering(radius_km=50)
        assert isinstance(clusters, list)

    def test_haversine_distance(self):
        # Approximate distance between Gusau and Kaduna
        dist = haversine_distance(12.17, 6.66, 10.52, 7.43)
        assert 150 < dist < 250

    def test_movement_corridors(self, sample_df):
        analyzer = SpatialAnalyzer(sample_df)
        corridors = analyzer.movement_corridors(days_window=10)
        assert isinstance(corridors, list)


class TestCorrelationAnalyzer:
    def test_attack_sequence_patterns(self, sample_df):
        analyzer = CorrelationAnalyzer(sample_df)
        patterns = analyzer.attack_sequence_patterns()
        assert isinstance(patterns, list)

    def test_escalation_indicators(self, sample_df):
        analyzer = CorrelationAnalyzer(sample_df)
        indicators = analyzer.escalation_indicators()
        if "status" not in indicators:
            assert "escalation_score" in indicators
            assert "escalation_level" in indicators

    def test_vulnerability_windows(self, sample_df):
        analyzer = CorrelationAnalyzer(sample_df)
        windows = analyzer.vulnerability_windows()
        if windows:
            assert "monthly_risk" in windows
            assert "peak_vulnerability_months" in windows
