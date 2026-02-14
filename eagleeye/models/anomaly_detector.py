"""
Anomaly detection module for identifying unusual patterns that
may indicate new tactics, shifting operations, or imminent large-scale attacks.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalous patterns in bandit activity that deviate from
    established baselines. Anomalies may indicate:
    - New tactics or weapons being used
    - Unusual geographic targeting
    - Abnormal timing patterns
    - Coordinated multi-location attacks
    - Sudden escalation or lull before major operation
    """

    def __init__(self, contamination: float = 0.1):
        """
        Args:
            contamination: Expected proportion of anomalies in the data.
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.baseline_stats: dict = {}

    def fit(self, incidents_df: pd.DataFrame) -> dict:
        """
        Fit the anomaly detector on historical incident data
        to establish baseline patterns.
        """
        features = self._extract_features(incidents_df)
        if features.empty or len(features) < 10:
            return {"status": "insufficient_data", "min_required": 10}

        X = self.scaler.fit_transform(features)
        self.model.fit(X)
        self.is_fitted = True

        # Store baseline statistics
        self.baseline_stats = {
            "mean_features": dict(zip(features.columns, features.mean().round(2))),
            "std_features": dict(zip(features.columns, features.std().round(2))),
            "training_samples": len(features),
            "fitted_at": datetime.utcnow().isoformat(),
        }

        return {"status": "success", "baseline": self.baseline_stats}

    def detect(self, incidents_df: pd.DataFrame) -> list[dict]:
        """
        Run anomaly detection on incident data and return flagged anomalies.
        """
        if not self.is_fitted:
            return [{"error": "Model not fitted. Call fit() first."}]

        features = self._extract_features(incidents_df)
        if features.empty:
            return []

        X = self.scaler.transform(features)
        predictions = self.model.predict(X)
        anomaly_scores = self.model.decision_function(X)

        anomalies = []
        for idx, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
            if pred == -1:  # Anomaly
                row = incidents_df.iloc[idx]
                anomalies.append({
                    "incident_id": int(row.get("id", idx)),
                    "date": row["date"].isoformat() if isinstance(row["date"], datetime) else str(row["date"]),
                    "state": row.get("state", "Unknown"),
                    "lga": row.get("lga", "Unknown"),
                    "incident_type": row.get("incident_type", "Unknown"),
                    "anomaly_score": round(float(-score), 3),  # Higher = more anomalous
                    "description": row.get("description", ""),
                    "anomaly_reasons": self._explain_anomaly(features.iloc[idx]),
                })

        anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)
        return anomalies

    def detect_pattern_breaks(self, incidents_df: pd.DataFrame,
                                window_days: int = 30) -> list[dict]:
        """
        Detect breaks in established patterns that may indicate
        tactical shifts or new operational phases.
        """
        df = incidents_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        if len(df) < 20:
            return [{"status": "insufficient_data"}]

        breaks = []

        # Check for sudden location shifts
        recent = df.tail(window_days) if len(df) > window_days else df
        historical = df.head(len(df) - len(recent)) if len(df) > window_days else pd.DataFrame()

        if not historical.empty:
            # New LGAs being targeted
            hist_lgas = set(historical["lga"].dropna())
            recent_lgas = set(recent["lga"].dropna())
            new_targets = recent_lgas - hist_lgas
            if new_targets:
                breaks.append({
                    "type": "new_geographic_targets",
                    "description": f"Attacks detected in previously untargeted LGAs",
                    "details": list(new_targets),
                    "severity": "high",
                })

            # New incident types appearing
            hist_types = set(historical["incident_type"])
            recent_types = set(recent["incident_type"])
            new_types = recent_types - hist_types
            if new_types:
                breaks.append({
                    "type": "new_tactic",
                    "description": f"New attack types observed",
                    "details": list(new_types),
                    "severity": "high",
                })

            # Significant frequency change
            hist_weekly = len(historical) / max((historical["date"].max() - historical["date"].min()).days / 7, 1)
            recent_weekly = len(recent) / max((recent["date"].max() - recent["date"].min()).days / 7, 1)
            if hist_weekly > 0:
                freq_change = (recent_weekly - hist_weekly) / hist_weekly * 100
                if abs(freq_change) > 50:
                    breaks.append({
                        "type": "frequency_shift",
                        "description": f"Attack frequency {'increased' if freq_change > 0 else 'decreased'} by {abs(freq_change):.0f}%",
                        "change_percent": round(freq_change, 1),
                        "severity": "high" if freq_change > 0 else "medium",
                    })

            # Casualty severity change
            hist_avg_killed = historical["casualties_killed"].mean()
            recent_avg_killed = recent["casualties_killed"].mean()
            if hist_avg_killed > 0:
                lethality_change = (recent_avg_killed - hist_avg_killed) / hist_avg_killed * 100
                if lethality_change > 50:
                    breaks.append({
                        "type": "lethality_increase",
                        "description": f"Average casualties per incident increased by {lethality_change:.0f}%",
                        "severity": "critical",
                    })

        return breaks

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract numerical features for anomaly detection."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        features = pd.DataFrame()
        features["month"] = df["date"].dt.month
        features["day_of_week"] = df["date"].dt.dayofweek
        features["casualties_killed"] = df["casualties_killed"].fillna(0)
        features["kidnapped_count"] = df["kidnapped_count"].fillna(0)
        features["threat_level"] = df["threat_level"].fillna(2)

        # Encode incident type as numerical
        type_severity = {
            "camp_sighting": 1, "displacement": 1,
            "cattle_rustling": 2, "negotiation_ransom": 2,
            "kidnapping": 3, "mining_site_attack": 3, "arms_smuggling": 3,
            "highway_ambush": 4, "village_raid": 4, "market_attack": 4,
            "reprisal_attack": 4, "infrastructure_attack": 4,
            "attack_on_security_forces": 5, "inter_gang_clash": 3,
        }
        features["type_severity"] = df["incident_type"].map(type_severity).fillna(3)

        # Estimated bandit count
        features["estimated_bandits"] = df.get("estimated_bandits", pd.Series([0] * len(df))).fillna(0)

        return features

    def _explain_anomaly(self, feature_row: pd.Series) -> list[str]:
        """Explain why a data point was flagged as anomalous."""
        reasons = []

        for feature_name, value in feature_row.items():
            if feature_name in self.baseline_stats.get("mean_features", {}):
                mean = self.baseline_stats["mean_features"][feature_name]
                std = self.baseline_stats["std_features"].get(feature_name, 1)
                if std > 0:
                    z_score = abs(value - mean) / std
                    if z_score > 2:
                        direction = "above" if value > mean else "below"
                        reasons.append(
                            f"{feature_name} is {z_score:.1f} std devs {direction} baseline "
                            f"(value: {value:.1f}, baseline: {mean:.1f})"
                        )

        if not reasons:
            reasons.append("Combination of features is unusual compared to baseline")

        return reasons
