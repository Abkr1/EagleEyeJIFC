"""
Data processing module for cleaning, normalizing, and structuring
raw incident data and intelligence reports.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dateutil import parser as dateparser

from eagleeye.core.config import INCIDENT_TYPES, TARGET_STATES
from eagleeye.core.database import (
    Incident,
    IncidentTag,
    IntelReport,
    get_session,
    init_db,
)
from eagleeye.data.schemas import IncidentCreate

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes raw data into structured incident records."""

    def __init__(self, db_url: Optional[str] = None):
        if db_url:
            init_db(db_url)
            self.session = get_session(db_url)
        else:
            init_db()
            self.session = get_session()

    def ingest_incident(self, data: IncidentCreate) -> Incident:
        """Validate and store a single incident record."""
        incident = Incident(
            date=data.date,
            state=self._normalize_state(data.state),
            lga=data.lga,
            location_name=data.location_name,
            latitude=data.latitude,
            longitude=data.longitude,
            incident_type=self._normalize_incident_type(data.incident_type),
            description=data.description,
            casualties_killed=data.casualties_killed,
            casualties_injured=data.casualties_injured,
            kidnapped_count=data.kidnapped_count,
            cattle_stolen=data.cattle_stolen,
            estimated_bandits=data.estimated_bandits,
            weapons_observed=data.weapons_observed,
            vehicles_used=data.vehicles_used,
            duration_hours=data.duration_hours,
            security_response=data.security_response,
            source=data.source,
            source_url=data.source_url,
            confidence_score=data.confidence_score,
            threat_level=data.threat_level,
        )
        self.session.add(incident)
        self.session.commit()
        self.session.refresh(incident)
        logger.info(f"Ingested incident #{incident.id}: {data.incident_type} in {data.state}")
        return incident

    def ingest_bulk_incidents(self, incidents_data: list[dict]) -> list[Incident]:
        """Ingest multiple incidents from a list of dictionaries."""
        ingested = []
        for data in incidents_data:
            try:
                # Parse date if it's a string
                if isinstance(data.get("date"), str):
                    data["date"] = dateparser.parse(data["date"])
                validated = IncidentCreate(**data)
                incident = self.ingest_incident(validated)
                ingested.append(incident)
            except Exception as e:
                logger.error(f"Failed to ingest incident: {e} | Data: {data}")
        return ingested

    def ingest_from_json_file(self, file_path: str | Path) -> list[Incident]:
        """Load and ingest incidents from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return []

        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("incidents", [data])

        return self.ingest_bulk_incidents(data)

    def ingest_intel_report(self, title: str, content: str,
                            source: str = "", source_url: str = "",
                            published_date: Optional[datetime] = None,
                            region: Optional[str] = None) -> IntelReport:
        """Store a raw intelligence report for later processing."""
        report = IntelReport(
            title=title,
            content=content,
            source=source,
            source_url=source_url,
            published_date=published_date,
            region=region,
        )
        self.session.add(report)
        self.session.commit()
        self.session.refresh(report)
        logger.info(f"Ingested intel report #{report.id}: {title}")
        return report

    def tag_incident(self, incident_id: int, tags: list[str]) -> None:
        """Add tags to an incident."""
        for tag_name in tags:
            tag = IncidentTag(incident_id=incident_id, tag=tag_name.lower().strip())
            self.session.add(tag)
        self.session.commit()

    def get_incidents_dataframe(
        self,
        state: Optional[str] = None,
        incident_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Query incidents and return as a pandas DataFrame for analysis."""
        query = self.session.query(Incident)

        if state:
            query = query.filter(Incident.state == self._normalize_state(state))
        if incident_type:
            query = query.filter(Incident.incident_type == incident_type)
        if start_date:
            query = query.filter(Incident.date >= start_date)
        if end_date:
            query = query.filter(Incident.date <= end_date)

        incidents = query.order_by(Incident.date).all()

        if not incidents:
            return pd.DataFrame()

        records = []
        for inc in incidents:
            records.append({
                "id": inc.id,
                "date": inc.date,
                "state": inc.state,
                "lga": inc.lga,
                "location_name": inc.location_name,
                "latitude": inc.latitude,
                "longitude": inc.longitude,
                "incident_type": inc.incident_type,
                "description": inc.description,
                "casualties_killed": inc.casualties_killed,
                "casualties_injured": inc.casualties_injured,
                "kidnapped_count": inc.kidnapped_count,
                "cattle_stolen": inc.cattle_stolen,
                "estimated_bandits": inc.estimated_bandits,
                "threat_level": inc.threat_level,
                "confidence_score": inc.confidence_score,
                "source": inc.source,
            })

        return pd.DataFrame(records)

    @staticmethod
    def _normalize_state(state: str) -> str:
        """Normalize state name to standard format."""
        state_clean = state.strip().title()
        # Handle common variations
        variations = {
            "Fct": "FCT",
            "Fct Abuja": "FCT",
        }
        return variations.get(state_clean, state_clean)

    @staticmethod
    def _normalize_incident_type(incident_type: str) -> str:
        """Normalize incident type to standard categories."""
        cleaned = incident_type.lower().strip().replace(" ", "_").replace("-", "_")

        # Map common variations to standard types
        mapping = {
            "raid": "village_raid",
            "village_attack": "village_raid",
            "community_attack": "village_raid",
            "kidnap": "kidnapping",
            "abduction": "kidnapping",
            "rustling": "cattle_rustling",
            "ambush": "highway_ambush",
            "road_attack": "highway_ambush",
            "highway_attack": "highway_ambush",
            "attack_security": "attack_on_security_forces",
            "military_attack": "attack_on_security_forces",
            "mining_attack": "mining_site_attack",
            "gang_clash": "inter_gang_clash",
            "infighting": "inter_gang_clash",
        }

        if cleaned in INCIDENT_TYPES:
            return cleaned
        return mapping.get(cleaned, cleaned)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the database session."""
        self.session.close()
