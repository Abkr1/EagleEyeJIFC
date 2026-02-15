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
    # Frequency and severity (absolute measures) dominate
    WEIGHTS = {
        "frequency": 0.30,
        "severity": 0.30,
        "trend": 0.10,
        "geographic_spread": 0.15,
        "tactic_escalation": 0.05,
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
        """Score all target states and rank them by threat level.

        Uses cross-state normalization so component scores are relative
        to the most-affected state, preventing low-activity states from
        scoring disproportionately high on relative metrics.
        """
        if assessment_date is None:
            assessment_date = datetime.now()

        # Collect raw component values for all states first
        raw_data = {}
        for state in TARGET_STATES:
            state_df = self.df[self.df["state"].str.lower() == state.lower()]
            if state_df.empty:
                raw_data[state] = None
                continue
            raw_data[state] = {
                "df": state_df,
                "frequency": self._raw_frequency(state_df, assessment_date),
                "severity": self._raw_severity(state_df, assessment_date),
                "trend": self._raw_trend(state_df, assessment_date),
                "geographic_spread": self._raw_geographic_spread(state_df, assessment_date),
                "tactic_escalation": self._tactic_escalation_score(state_df, assessment_date),
                "recency": self._recency_score(state_df, assessment_date),
            }

        # Find max values across states for normalization
        active = {s: d for s, d in raw_data.items() if d is not None}
        if not active:
            return [self._empty_score(s) for s in TARGET_STATES]

        max_freq = max((d["frequency"] for d in active.values()), default=1) or 1
        max_sev = max((d["severity"] for d in active.values()), default=1) or 1
        max_spread = max((d["geographic_spread"] for d in active.values()), default=1) or 1

        scores = []
        for state in TARGET_STATES:
            if raw_data[state] is None:
                scores.append(self._empty_score(state))
                continue

            d = raw_data[state]
            state_df = d["df"]

            # Normalize frequency, severity, spread relative to cross-state max
            components = {
                "frequency": min(100, (d["frequency"] / max_freq) * 100),
                "severity": min(100, (d["severity"] / max_sev) * 100),
                "trend": d["trend"],
                "geographic_spread": min(100, (d["geographic_spread"] / max_spread) * 100),
                "tactic_escalation": d["tactic_escalation"],
                "recency": d["recency"],
            }

            overall = sum(
                components[factor] * weight
                for factor, weight in self.WEIGHTS.items()
            )
            overall = min(100, max(0, overall))
            threat_level = self._score_to_level(overall)

            scores.append({
                "state": state,
                "assessment_date": assessment_date.isoformat(),
                "overall_score": round(overall, 1),
                "threat_level": threat_level,
                "threat_label": self._level_label(threat_level),
                "component_scores": {k: round(v, 1) for k, v in components.items()},
                "weights": self.WEIGHTS,
                "incidents_last_30d": len(state_df[state_df["date"] >= assessment_date - timedelta(days=30)]),
                "incidents_last_90d": len(state_df[state_df["date"] >= assessment_date - timedelta(days=90)]),
            })

        scores.sort(key=lambda x: x["overall_score"], reverse=True)
        return scores

    def _empty_score(self, state: str) -> dict:
        return {
            "state": state,
            "overall_score": 0,
            "threat_level": 1,
            "threat_label": "LOW",
            "status": "no_data",
        }

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

    def _raw_frequency(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Raw incident count in last 30 days (for cross-state normalization)."""
        recent = df[df["date"] >= ref_date - timedelta(days=30)]
        return float(len(recent))

    def _frequency_score(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Score based on incident frequency in last 30 days."""
        return min(100, self._raw_frequency(df, ref_date) * 10)

    def _raw_severity(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Raw casualty severity in last 30 days (for cross-state normalization)."""
        recent = df[df["date"] >= ref_date - timedelta(days=30)]
        if recent.empty:
            return 0.0
        killed = recent["casualties_killed"].sum()
        kidnapped = recent["kidnapped_count"].sum()
        injured = recent["casualties_injured"].sum() if "casualties_injured" in recent.columns else 0
        return float(killed * 3 + kidnapped * 2 + injured)

    def _severity_score(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Score based on casualty severity in last 30 days."""
        return min(100, self._raw_severity(df, ref_date) * 2)

    def _trend_score(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Trend score for single-state use (delegates to _raw_trend)."""
        return self._raw_trend(df, ref_date)

    def _raw_trend(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Trend score weighted by absolute volume so small fluctuations
        in low-activity states don't dominate."""
        recent_30d = len(df[df["date"] >= ref_date - timedelta(days=30)])
        prev_30d = len(df[
            (df["date"] >= ref_date - timedelta(days=60))
            & (df["date"] < ref_date - timedelta(days=30))
        ])

        if prev_30d == 0:
            # No baseline — score proportional to current count, capped
            return min(100, recent_30d * 5) if recent_30d > 0 else 0.0

        pct_change = (recent_30d - prev_30d) / prev_30d
        # Base trend from percentage change: -100%→0, 0%→50, +100%→100
        base = min(100, max(0, 50 + pct_change * 50))
        # Dampen for low-volume states: scale by volume factor
        # (states with few incidents shouldn't score 100 on trend)
        volume_factor = min(1.0, max(recent_30d, prev_30d) / 10.0)
        return base * volume_factor

    def _raw_geographic_spread(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Raw LGA count affected in last 30 days (for cross-state normalization)."""
        recent = df[df["date"] >= ref_date - timedelta(days=30)]
        if recent.empty:
            return 0.0
        return float(recent["lga"].nunique())

    def _geographic_spread_score(self, df: pd.DataFrame, ref_date: datetime) -> float:
        """Score based on how many LGAs are affected."""
        return min(100, self._raw_geographic_spread(df, ref_date) * 10)

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
