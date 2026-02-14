"""
Threat scoring module for assessing regional threat levels
based on multiple weighted factors.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from eagleeye.core.config import TARGET_STATES, THREAT_LEVELS

logger = logging.getLogger(__name__)


class ThreatScorer:
    """
    Calculates composite threat scores for regions based on:
    - Recent incident frequency and trend
    - Casualty severity
    - Geographic spread of attacks
    - Attack type escalation
    - Historical baseline comparison
    """

    # Weight configuration for threat factors
    WEIGHTS = {
        "frequency": 0.25,
        "severity": 0.20,
        "trend": 0.20,
        "geographic_spread": 0.15,
        "tactic_escalation": 0.10,
        "recency": 0.10,
    }

    def __init__(self, incidents_df: pd.DataFrame):
        self.df = incidents_df.copy()
        if not self.df.empty:
            self.df["date"] = pd.to_datetime(self.df["date"])

    def score_state(self, state: str, assessment_date: Optional[datetime] = None) -> dict:
        """
        Calculate a comprehensive threat score for a state.

        Returns:
            Dict with overall score (0-100), threat level (1-5),
            component scores, and assessment details.
        """
        if assessment_date is None:
            assessment_date = datetime.now()

        state_df = self.df[self.df["state"].str.lower() == state.lower()]
        if state_df.empty:
            return {
                "state": state,
                "overall_score": 0,
                "threat_level": 1,
                "threat_label": "LOW",
                "status": "no_data",
            }

        scores = {}

        # 1. Frequency score (0-100)
        scores["frequency"] = self._frequency_score(state_df, assessment_date)

        # 2. Severity score (0-100)
        scores["severity"] = self._severity_score(state_df, assessment_date)

        # 3. Trend score (0-100)
        scores["trend"] = self._trend_score(state_df, assessment_date)

        # 4. Geographic spread score (0-100)
        scores["geographic_spread"] = self._geographic_spread_score(state_df, assessment_date)

        # 5. Tactic escalation score (0-100)
        scores["tactic_escalation"] = self._tactic_escalation_score(state_df, assessment_date)

        # 6. Recency score (0-100)
        scores["recency"] = self._recency_score(state_df, assessment_date)

        # Weighted composite score
        overall_score = sum(
            scores[factor] * weight
            for factor, weight in self.WEIGHTS.items()
        )
        overall_score = min(100, max(0, overall_score))

        # Map to threat level
        threat_level = self._score_to_level(overall_score)

        return {
            "state": state,
            "assessment_date": assessment_date.isoformat(),
            "overall_score": round(overall_score, 1),
            "threat_level": threat_level,
            "threat_label": self._level_label(threat_level),
            "component_scores": {k: round(v, 1) for k, v in scores.items()},
            "weights": self.WEIGHTS,
            "incidents_last_30d": len(state_df[state_df["date"] >= assessment_date - timedelta(days=30)]),
            "incidents_last_90d": len(state_df[state_df["date"] >= assessment_date - timedelta(days=90)]),
        }

    def score_all_states(self, assessment_date: Optional[datetime] = None) -> list[dict]:
        """Score all target states and rank them by threat level."""
        scores = []
        for state in TARGET_STATES:
            score = self.score_state(state, assessment_date)
            scores.append(score)

        scores.sort(key=lambda x: x["overall_score"], reverse=True)
        return scores

    def score_lga(self, state: str, lga: str,
                  assessment_date: Optional[datetime] = None) -> dict:
        """Calculate threat score for a specific LGA."""
        if assessment_date is None:
            assessment_date = datetime.now()

        lga_df = self.df[
            (self.df["state"].str.lower() == state.lower())
            & (self.df["lga"].str.lower() == lga.lower())
        ]
        if lga_df.empty:
            return {
                "state": state,
                "lga": lga,
                "overall_score": 0,
                "threat_level": 1,
                "threat_label": "LOW",
                "status": "no_data",
            }

        freq = self._frequency_score(lga_df, assessment_date)
        sev = self._severity_score(lga_df, assessment_date)
        rec = self._recency_score(lga_df, assessment_date)

        overall = freq * 0.4 + sev * 0.3 + rec * 0.3
        overall = min(100, max(0, overall))

        return {
            "state": state,
            "lga": lga,
            "overall_score": round(overall, 1),
            "threat_level": self._score_to_level(overall),
            "threat_label": self._level_label(self._score_to_level(overall)),
            "total_incidents": len(lga_df),
            "last_incident": lga_df["date"].max().isoformat(),
        }

    def _frequency_score(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Score based on incident frequency in last 30 days."""
        recent = df[df["date"] >= ref_date - timedelta(days=30)]
        count = len(recent)
        # Normalize: 0 incidents = 0, 10+ incidents = 100
        return min(100, count * 10)

    def _severity_score(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Score based on casualty severity in last 30 days."""
        recent = df[df["date"] >= ref_date - timedelta(days=30)]
        if recent.empty:
            return 0
        killed = recent["casualties_killed"].sum()
        kidnapped = recent["kidnapped_count"].sum()
        severity = killed * 3 + kidnapped * 2
        return min(100, severity * 2)

    def _trend_score(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Score based on whether activity is increasing."""
        recent_30d = len(df[df["date"] >= ref_date - timedelta(days=30)])
        prev_30d = len(df[
            (df["date"] >= ref_date - timedelta(days=60))
            & (df["date"] < ref_date - timedelta(days=30))
        ])

        if prev_30d == 0:
            return 50 if recent_30d > 0 else 0

        change = (recent_30d - prev_30d) / prev_30d
        # Map: -100% change = 0, 0% = 50, +100% = 100
        return min(100, max(0, 50 + change * 50))

    def _geographic_spread_score(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Score based on how many LGAs are affected."""
        recent = df[df["date"] >= ref_date - timedelta(days=30)]
        if recent.empty:
            return 0
        lga_count = recent["lga"].nunique()
        # Normalize: 1 LGA = 10, 10+ LGAs = 100
        return min(100, lga_count * 10)

    def _tactic_escalation_score(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Score based on whether attack types are becoming more severe."""
        severity_map = {
            "camp_sighting": 1, "displacement": 1,
            "cattle_rustling": 2, "negotiation_ransom": 2,
            "kidnapping": 3, "mining_site_attack": 3, "arms_smuggling": 3,
            "highway_ambush": 4, "village_raid": 4, "market_attack": 4,
            "reprisal_attack": 4, "infrastructure_attack": 4,
            "attack_on_security_forces": 5, "inter_gang_clash": 3,
        }

        recent = df[df["date"] >= ref_date - timedelta(days=30)]
        previous = df[
            (df["date"] >= ref_date - timedelta(days=60))
            & (df["date"] < ref_date - timedelta(days=30))
        ]

        if recent.empty or previous.empty:
            return 30

        recent_severity = recent["incident_type"].map(severity_map).mean()
        prev_severity = previous["incident_type"].map(severity_map).mean()

        if pd.isna(recent_severity) or pd.isna(prev_severity):
            return 30

        diff = recent_severity - prev_severity
        # Map: -2 diff = 0, 0 = 50, +2 = 100
        return min(100, max(0, 50 + diff * 25))

    def _recency_score(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Score based on how recently the last incident occurred."""
        if df.empty:
            return 0
        last = df["date"].max()
        days_ago = (ref_date - last).days
        # Very recent = high score
        if days_ago <= 1:
            return 100
        elif days_ago <= 7:
            return 80
        elif days_ago <= 14:
            return 60
        elif days_ago <= 30:
            return 40
        elif days_ago <= 60:
            return 20
        return 5

    @staticmethod
    def _score_to_level(score: float) -> int:
        """Convert 0-100 score to 1-5 threat level."""
        if score >= 80:
            return 5
        elif score >= 60:
            return 4
        elif score >= 40:
            return 3
        elif score >= 20:
            return 2
        return 1

    @staticmethod
    def _level_label(level: int) -> str:
        labels = {1: "LOW", 2: "MODERATE", 3: "ELEVATED", 4: "HIGH", 5: "CRITICAL"}
        return labels.get(level, "UNKNOWN")
