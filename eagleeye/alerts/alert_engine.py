"""
Alert generation engine that produces security alerts based on
analysis results, predictions, and detected anomalies.
"""

import logging
from datetime import datetime
from typing import Optional

from eagleeye.core.database import Alert, Prediction, get_session, init_db

logger = logging.getLogger(__name__)


class AlertEngine:
    """
    Generates and manages security alerts based on:
    - High threat level predictions
    - Detected anomalies
    - Escalation indicators
    - Pattern breaks
    - Surge detection
    """

    ALERT_TYPES = {
        "prediction_high_threat": "Prediction indicates high threat level",
        "anomaly_detected": "Anomalous activity pattern detected",
        "escalation_warning": "Escalation indicators detected",
        "pattern_break": "Break in established pattern detected",
        "surge_detected": "Surge in activity detected",
        "new_target_area": "Activity in previously untargeted area",
        "lull_warning": "Extended lull may indicate planning phase",
        "cross_border": "Cross-border activity pattern detected",
    }

    def __init__(self, db_url: Optional[str] = None):
        if db_url:
            init_db(db_url)
            self.session = get_session(db_url)
        else:
            init_db()
            self.session = get_session()

    def generate_prediction_alert(self, prediction: dict) -> Optional[dict]:
        """Generate alert from a high-threat prediction."""
        threat_level = prediction.get("threat_level", 1)
        if threat_level < 3:
            return None

        severity = threat_level
        top_type = prediction.get("predicted_incident_types", [{}])[0]
        top_location = prediction.get("predicted_locations", [{}])[0]

        title = (
            f"{self._severity_label(severity)} THREAT: "
            f"Predicted {top_type.get('type', 'activity')} activity "
            f"in {top_location.get('state', 'region')}"
        )

        description = self._build_prediction_description(prediction)
        recommendations = prediction.get("recommendations", [])

        alert = self._store_alert(
            alert_type="prediction_high_threat",
            severity=severity,
            state=top_location.get("state"),
            title=title,
            description=description + "\n\nRecommendations:\n" + "\n".join(f"- {r}" for r in recommendations),
        )

        return alert

    def generate_anomaly_alert(self, anomaly: dict) -> dict:
        """Generate alert from a detected anomaly."""
        severity = min(5, max(3, int(anomaly.get("anomaly_score", 0.5) * 5)))

        title = (
            f"ANOMALY: Unusual {anomaly.get('incident_type', 'activity')} "
            f"in {anomaly.get('state', 'Unknown')}"
        )

        reasons = anomaly.get("anomaly_reasons", [])
        description = (
            f"Anomalous activity detected on {anomaly.get('date', 'unknown date')}.\n"
            f"Location: {anomaly.get('state', 'Unknown')}, {anomaly.get('lga', 'Unknown')}\n"
            f"Type: {anomaly.get('incident_type', 'Unknown')}\n"
            f"Anomaly Score: {anomaly.get('anomaly_score', 0):.2f}\n\n"
            f"Reasons:\n" + "\n".join(f"- {r}" for r in reasons)
        )

        return self._store_alert(
            alert_type="anomaly_detected",
            severity=severity,
            state=anomaly.get("state"),
            lga=anomaly.get("lga"),
            title=title,
            description=description,
        )

    def generate_escalation_alert(self, indicators: dict, state: str) -> Optional[dict]:
        """Generate alert from escalation indicators."""
        score = indicators.get("escalation_score", 0)
        if score < 4:
            return None

        severity = 5 if score >= 8 else 4

        title = f"ESCALATION WARNING: {state} — {indicators.get('escalation_level', 'moderate').upper()} level"

        factors = []
        if indicators.get("frequency_escalating"):
            factors.append(f"Frequency up {indicators.get('frequency_change_pct', 0):.0f}%")
        if indicators.get("lethality_escalating"):
            factors.append(f"Casualties up {indicators.get('casualty_change_pct', 0):.0f}%")
        if indicators.get("geographic_expansion"):
            factors.append(f"Geographic expansion: {indicators.get('recent_lgas_affected', 0)} LGAs (was {indicators.get('previous_lgas_affected', 0)})")
        if indicators.get("tactics_escalating"):
            factors.append(f"Tactics shifting to more severe types (shift: {indicators.get('severity_shift', 0):.1f})")

        description = (
            f"Escalation detected in {state}.\n"
            f"Escalation Score: {score}/10\n\n"
            f"Contributing factors:\n" + "\n".join(f"- {f}" for f in factors)
        )

        return self._store_alert(
            alert_type="escalation_warning",
            severity=severity,
            state=state,
            title=title,
            description=description,
        )

    def generate_pattern_break_alert(self, pattern_break: dict) -> dict:
        """Generate alert from a detected pattern break."""
        severity_map = {"critical": 5, "high": 4, "medium": 3, "low": 2}
        severity = severity_map.get(pattern_break.get("severity", "medium"), 3)

        title = f"PATTERN BREAK: {pattern_break.get('type', 'Unknown').replace('_', ' ').title()}"

        description = (
            f"{pattern_break.get('description', 'Pattern break detected')}\n\n"
            f"Details: {pattern_break.get('details', 'N/A')}"
        )

        return self._store_alert(
            alert_type="pattern_break",
            severity=severity,
            title=title,
            description=description,
        )

    def generate_surge_alert(self, surge: dict, state: Optional[str] = None) -> dict:
        """Generate alert from a detected activity surge."""
        surge_factor = surge.get("surge_factor", 1)
        severity = 5 if surge_factor >= 3 else 4 if surge_factor >= 2 else 3

        title = (
            f"SURGE DETECTED: {surge.get('incident_count', 0)} incidents "
            f"in week of {surge.get('week_start', 'unknown')}"
        )

        description = (
            f"Activity surge detected: {surge.get('incident_count', 0)} incidents "
            f"vs weekly average of {surge.get('weekly_average', 0):.1f}\n"
            f"Surge factor: {surge_factor:.1f}x normal\n"
            f"Week: {surge.get('week_start', 'unknown')}"
        )

        return self._store_alert(
            alert_type="surge_detected",
            severity=severity,
            state=state,
            title=title,
            description=description,
        )

    def get_active_alerts(self, state: Optional[str] = None,
                           min_severity: int = 1) -> list[dict]:
        """Retrieve active (unacknowledged) alerts."""
        query = self.session.query(Alert).filter(
            Alert.acknowledged == 0,
            Alert.severity >= min_severity,
        )
        if state:
            query = query.filter(Alert.state == state)

        alerts = query.order_by(Alert.severity.desc(), Alert.created_at.desc()).all()

        return [
            {
                "id": a.id,
                "created_at": a.created_at.isoformat(),
                "alert_type": a.alert_type,
                "severity": a.severity,
                "severity_label": self._severity_label(a.severity),
                "state": a.state,
                "lga": a.lga,
                "title": a.title,
                "description": a.description,
            }
            for a in alerts
        ]

    def acknowledge_alert(self, alert_id: int, acknowledged_by: str) -> bool:
        """Mark an alert as acknowledged."""
        alert = self.session.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.acknowledged = 1
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow()
            self.session.commit()
            return True
        return False

    def _store_alert(self, alert_type: str, severity: int, title: str,
                     description: str, state: Optional[str] = None,
                     lga: Optional[str] = None) -> dict:
        """Store an alert in the database."""
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            state=state,
            lga=lga,
            title=title,
            description=description,
        )
        self.session.add(alert)
        self.session.commit()
        self.session.refresh(alert)

        logger.info(f"Alert #{alert.id}: [{self._severity_label(severity)}] {title}")

        return {
            "id": alert.id,
            "alert_type": alert_type,
            "severity": severity,
            "severity_label": self._severity_label(severity),
            "title": title,
            "state": state,
            "created_at": alert.created_at.isoformat(),
        }

    @staticmethod
    def _severity_label(level: int) -> str:
        labels = {1: "LOW", 2: "MODERATE", 3: "ELEVATED", 4: "HIGH", 5: "CRITICAL"}
        return labels.get(level, "UNKNOWN")

    @staticmethod
    def _build_prediction_description(prediction: dict) -> str:
        types = prediction.get("predicted_incident_types", [])
        locations = prediction.get("predicted_locations", [])

        desc = f"Prediction for {prediction.get('prediction_date', 'unknown date')}\n\n"
        desc += "Predicted activity types:\n"
        for t in types[:3]:
            desc += f"  - {t.get('type', '?')}: {t.get('probability', 0):.1%} probability\n"

        desc += "\nMost likely locations:\n"
        for loc in locations[:3]:
            desc += f"  - {loc.get('state', '?')}: {loc.get('probability', 0):.1%} probability\n"

        desc += f"\nOverall threat level: {prediction.get('threat_label', 'UNKNOWN')}"
        desc += f"\nConfidence: {prediction.get('confidence', 0):.1%}"

        return desc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.session.close()
