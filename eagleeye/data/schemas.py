"""
Pydantic schemas for data validation and serialization.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class IncidentCreate(BaseModel):
    """Schema for creating a new incident record."""

    date: datetime
    state: str
    lga: Optional[str] = None
    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    incident_type: str
    description: Optional[str] = None
    casualties_killed: int = 0
    casualties_injured: int = 0
    kidnapped_count: int = 0
    cattle_stolen: int = 0
    estimated_bandits: Optional[int] = None
    weapons_observed: Optional[str] = None
    vehicles_used: Optional[str] = None
    duration_hours: Optional[float] = None
    security_response: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    threat_level: int = Field(default=2, ge=1, le=5)


class IncidentResponse(IncidentCreate):
    """Schema for returning incident data."""

    id: int
    created_at: datetime
    updated_at: datetime
    tags: list[str] = []

    class Config:
        from_attributes = True


class BanditGroupCreate(BaseModel):
    """Schema for creating a bandit group record."""

    name: str
    aliases: Optional[str] = None
    known_leader: Optional[str] = None
    estimated_size: Optional[int] = None
    primary_state: Optional[str] = None
    primary_territory: Optional[str] = None
    weapons_capability: Optional[str] = None
    primary_activity: Optional[str] = None
    alliances: Optional[str] = None
    rivals: Optional[str] = None
    status: str = "active"
    notes: Optional[str] = None


class IntelReportCreate(BaseModel):
    """Schema for ingesting a raw intelligence report."""

    title: Optional[str] = None
    content: str
    source: Optional[str] = None
    source_url: Optional[str] = None
    published_date: Optional[datetime] = None
    region: Optional[str] = None


class PredictionResponse(BaseModel):
    """Schema for a prediction output."""

    id: int
    generated_at: datetime
    prediction_date: datetime
    state: str
    lga: Optional[str] = None
    predicted_incident_type: Optional[str] = None
    probability: float
    threat_level: int
    reasoning: Optional[str] = None
    contributing_factors: Optional[str] = None
    recommended_actions: Optional[str] = None
    model_version: Optional[str] = None

    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    """Schema for an alert output."""

    id: int
    created_at: datetime
    alert_type: str
    severity: int
    state: Optional[str] = None
    lga: Optional[str] = None
    title: str
    description: str
    acknowledged: bool = False

    class Config:
        from_attributes = True


class ThreatAssessment(BaseModel):
    """Schema for a regional threat assessment."""

    state: str
    assessment_date: datetime
    overall_threat_level: int = Field(ge=1, le=5)
    threat_label: str
    incident_count_last_30_days: int = 0
    incident_trend: str = "stable"
    primary_threat_type: Optional[str] = None
    hotspot_lgas: list[str] = []
    active_groups: list[str] = []
    prediction_summary: Optional[str] = None
    recommended_posture: Optional[str] = None


class AnalysisSummary(BaseModel):
    """Schema for an analysis summary report."""

    analysis_date: datetime
    period_start: datetime
    period_end: datetime
    total_incidents: int
    incidents_by_state: dict[str, int] = {}
    incidents_by_type: dict[str, int] = {}
    casualty_total: int = 0
    kidnapped_total: int = 0
    most_active_state: Optional[str] = None
    escalation_detected: bool = False
    pattern_shifts: list[str] = []
    high_risk_areas: list[str] = []
