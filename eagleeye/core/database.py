"""
Database models for EagleEye incident tracking and intelligence storage.
Uses SQLAlchemy ORM with SQLite (upgradeable to PostgreSQL).
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

class Base(DeclarativeBase):
    pass


class Incident(Base):
    """A recorded security incident involving bandit activity."""

    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    state = Column(String(50), nullable=False, index=True)
    lga = Column(String(100), nullable=True)
    location_name = Column(String(200), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    incident_type = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=True)
    casualties_killed = Column(Integer, default=0)
    casualties_injured = Column(Integer, default=0)
    kidnapped_count = Column(Integer, default=0)
    cattle_stolen = Column(Integer, default=0)
    estimated_bandits = Column(Integer, nullable=True)
    weapons_observed = Column(String(200), nullable=True)
    vehicles_used = Column(String(200), nullable=True)
    duration_hours = Column(Float, nullable=True)
    security_response = Column(Text, nullable=True)
    source = Column(String(200), nullable=True)
    source_url = Column(String(500), nullable=True)
    confidence_score = Column(Float, default=0.5)
    threat_level = Column(Integer, default=2)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    tags = relationship("IncidentTag", back_populates="incident", cascade="all, delete-orphan")
    linked_groups = relationship("GroupActivity", back_populates="incident", cascade="all, delete-orphan")


class BanditGroup(Base):
    """Known or suspected bandit groups and their identifiers."""

    __tablename__ = "bandit_groups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    aliases = Column(Text, nullable=True)
    known_leader = Column(String(200), nullable=True)
    estimated_size = Column(Integer, nullable=True)
    primary_state = Column(String(50), nullable=True)
    primary_territory = Column(Text, nullable=True)
    weapons_capability = Column(String(200), nullable=True)
    primary_activity = Column(String(100), nullable=True)
    alliances = Column(Text, nullable=True)
    rivals = Column(Text, nullable=True)
    status = Column(String(50), default="active")
    first_seen = Column(DateTime, nullable=True)
    last_seen = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    activities = relationship("GroupActivity", back_populates="group", cascade="all, delete-orphan")


class GroupActivity(Base):
    """Links bandit groups to specific incidents."""

    __tablename__ = "group_activities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    group_id = Column(Integer, ForeignKey("bandit_groups.id"), nullable=False)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=False)
    role = Column(String(50), default="perpetrator")
    confidence = Column(Float, default=0.5)

    group = relationship("BanditGroup", back_populates="activities")
    incident = relationship("Incident", back_populates="linked_groups")


class IncidentTag(Base):
    """Tags and labels attached to incidents for categorization."""

    __tablename__ = "incident_tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=False)
    tag = Column(String(100), nullable=False, index=True)

    incident = relationship("Incident", back_populates="tags")


class IntelReport(Base):
    """Raw intelligence reports ingested into the system."""

    __tablename__ = "intel_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=True)
    content = Column(Text, nullable=False)
    source = Column(String(200), nullable=True)
    source_url = Column(String(500), nullable=True)
    published_date = Column(DateTime, nullable=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Integer, default=0)
    extracted_incidents_count = Column(Integer, default=0)
    region = Column(String(100), nullable=True)


class Prediction(Base):
    """Model predictions for future bandit activity."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generated_at = Column(DateTime, default=datetime.utcnow)
    prediction_date = Column(DateTime, nullable=False)
    state = Column(String(50), nullable=False)
    lga = Column(String(100), nullable=True)
    predicted_incident_type = Column(String(50), nullable=True)
    probability = Column(Float, nullable=False)
    threat_level = Column(Integer, nullable=False)
    reasoning = Column(Text, nullable=True)
    contributing_factors = Column(Text, nullable=True)
    recommended_actions = Column(Text, nullable=True)
    model_version = Column(String(50), nullable=True)
    validated = Column(Integer, default=0)
    actual_outcome = Column(Text, nullable=True)


class AppSetting(Base):
    """Key-value settings for persistent app configuration."""

    __tablename__ = "app_settings"

    key = Column(String(200), primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Alert(Base):
    """Generated alerts based on analysis and predictions."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    alert_type = Column(String(50), nullable=False)
    severity = Column(Integer, nullable=False)
    state = Column(String(50), nullable=True)
    lga = Column(String(100), nullable=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)
    acknowledged = Column(Integer, default=0)
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)


_engine_cache = {}


def get_engine(url: str = None):
    if url is None:
        from eagleeye.core.config import DATABASE_URL
        url = DATABASE_URL
    if url in _engine_cache:
        return _engine_cache[url]
    kwargs = {"echo": False}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"timeout": 30}
    eng = create_engine(url, **kwargs)
    # Enable WAL mode for SQLite to allow concurrent reads + writes
    if url.startswith("sqlite"):
        from sqlalchemy import event, text
        @event.listens_for(eng, "connect")
        def _set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    _engine_cache[url] = eng
    return eng


def get_session(url: str = None) -> Session:
    engine = get_engine(url)
    session_factory = sessionmaker(bind=engine)
    return session_factory()


def init_db(url: str = None):
    """Create all tables in the database."""
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    return engine
