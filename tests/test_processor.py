"""Tests for the data processing module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from eagleeye.core.database import Base, get_engine
from eagleeye.data.processor import DataProcessor
from eagleeye.data.schemas import IncidentCreate


@pytest.fixture
def processor(tmp_path):
    """Use a file-based temporary SQLite database so init_db and DataProcessor share state."""
    db_file = tmp_path / "test.db"
    db_url = f"sqlite:///{db_file}"
    proc = DataProcessor(db_url)
    yield proc
    proc.close()


@pytest.fixture
def sample_incident():
    return IncidentCreate(
        date=datetime(2024, 3, 15),
        state="Zamfara",
        lga="Anka",
        location_name="Test Village",
        incident_type="village_raid",
        description="Test incident",
        casualties_killed=5,
        casualties_injured=3,
        kidnapped_count=10,
        cattle_stolen=50,
        estimated_bandits=30,
        confidence_score=0.8,
        threat_level=4,
    )


def test_ingest_incident(processor, sample_incident):
    incident = processor.ingest_incident(sample_incident)
    assert incident.id is not None
    assert incident.state == "Zamfara"
    assert incident.incident_type == "village_raid"
    assert incident.casualties_killed == 5


def test_ingest_bulk_incidents(processor):
    data = [
        {
            "date": "2024-01-15",
            "state": "Zamfara",
            "lga": "Anka",
            "incident_type": "village_raid",
            "casualties_killed": 3,
        },
        {
            "date": "2024-02-20",
            "state": "Kaduna",
            "lga": "Birnin Gwari",
            "incident_type": "highway_ambush",
            "casualties_killed": 2,
        },
    ]
    incidents = processor.ingest_bulk_incidents(data)
    assert len(incidents) == 2
    assert incidents[0].state == "Zamfara"
    assert incidents[1].state == "Kaduna"


def test_normalize_state(processor):
    assert processor._normalize_state("zamfara") == "Zamfara"
    assert processor._normalize_state("KADUNA") == "Kaduna"
    assert processor._normalize_state("  katsina  ") == "Katsina"


def test_normalize_incident_type(processor):
    assert processor._normalize_incident_type("village_raid") == "village_raid"
    assert processor._normalize_incident_type("kidnap") == "kidnapping"
    assert processor._normalize_incident_type("ambush") == "highway_ambush"
    assert processor._normalize_incident_type("rustling") == "cattle_rustling"


def test_get_incidents_dataframe(processor, sample_incident):
    processor.ingest_incident(sample_incident)
    df = processor.get_incidents_dataframe()
    assert len(df) == 1
    assert df.iloc[0]["state"] == "Zamfara"


def test_get_incidents_filtered(processor):
    data = [
        {"date": "2024-01-15", "state": "Zamfara", "incident_type": "village_raid", "casualties_killed": 3},
        {"date": "2024-02-20", "state": "Kaduna", "incident_type": "kidnapping", "casualties_killed": 1},
        {"date": "2024-03-10", "state": "Zamfara", "incident_type": "kidnapping", "casualties_killed": 0},
    ]
    processor.ingest_bulk_incidents(data)

    # Filter by state
    df = processor.get_incidents_dataframe(state="Zamfara")
    assert len(df) == 2

    # Filter by type
    df = processor.get_incidents_dataframe(incident_type="kidnapping")
    assert len(df) == 2


def test_ingest_from_json_file(processor):
    data = {
        "incidents": [
            {"date": "2024-01-15", "state": "Zamfara", "incident_type": "village_raid", "casualties_killed": 3},
            {"date": "2024-02-20", "state": "Kaduna", "incident_type": "kidnapping", "casualties_killed": 1},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = f.name

    incidents = processor.ingest_from_json_file(temp_path)
    assert len(incidents) == 2
    Path(temp_path).unlink()


def test_tag_incident(processor, sample_incident):
    incident = processor.ingest_incident(sample_incident)
    processor.tag_incident(incident.id, ["dry_season", "large_group", "motorcycle"])

    # Verify tags are stored
    from eagleeye.core.database import IncidentTag
    tags = processor.session.query(IncidentTag).filter_by(incident_id=incident.id).all()
    assert len(tags) == 3
    tag_names = [t.tag for t in tags]
    assert "dry_season" in tag_names
