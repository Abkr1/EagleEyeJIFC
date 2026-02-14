"""
Correlation analysis module for identifying relationships between
bandit activities and external factors.

Analyzes: attack pattern correlations, escalation chains, tactic evolution,
and contextual factors (seasonal, economic, security operations).
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Identifies correlations and relationships between incidents and factors."""

    def __init__(self, incidents_df: pd.DataFrame):
        self.df = incidents_df.copy()
        if not self.df.empty:
            self.df["date"] = pd.to_datetime(self.df["date"])
            self.df = self.df.sort_values("date")

    def attack_sequence_patterns(self, max_gap_days: int = 14,
                                  state: Optional[str] = None) -> list[dict]:
        """
        Identify common sequences of attack types.

        Example: kidnapping often followed by ransom negotiation,
        or cattle rustling followed by reprisal attack.
        """
        df = self._filter_state(state)
        if len(df) < 3:
            return []

        sequences = Counter()
        records = df.to_dict("records")

        for i in range(len(records) - 1):
            gap = (records[i + 1]["date"] - records[i]["date"]).total_seconds() / 86400
            if gap <= max_gap_days:
                pair = (records[i]["incident_type"], records[i + 1]["incident_type"])
                sequences[pair] += 1

        # Also check 3-event sequences
        triple_sequences = Counter()
        for i in range(len(records) - 2):
            gap1 = (records[i + 1]["date"] - records[i]["date"]).total_seconds() / 86400
            gap2 = (records[i + 2]["date"] - records[i + 1]["date"]).total_seconds() / 86400
            if gap1 <= max_gap_days and gap2 <= max_gap_days:
                triple = (records[i]["incident_type"],
                          records[i + 1]["incident_type"],
                          records[i + 2]["incident_type"])
                triple_sequences[triple] += 1

        results = []
        for (t1, t2), count in sequences.most_common(15):
            if count >= 2:
                results.append({
                    "sequence": [t1, t2],
                    "occurrences": count,
                    "pattern_type": "pair",
                    "interpretation": self._interpret_sequence([t1, t2]),
                })

        for (t1, t2, t3), count in triple_sequences.most_common(10):
            if count >= 2:
                results.append({
                    "sequence": [t1, t2, t3],
                    "occurrences": count,
                    "pattern_type": "triple",
                    "interpretation": self._interpret_sequence([t1, t2, t3]),
                })

        return results

    def escalation_indicators(self, state: Optional[str] = None,
                               window_days: int = 30) -> dict:
        """
        Detect indicators that activity is escalating.

        Escalation indicators:
        - Increasing incident frequency
        - Rising casualty counts
        - Expanding geographic spread
        - Shift to more severe attack types
        - Larger estimated group sizes
        """
        df = self._filter_state(state)
        if len(df) < 5:
            return {"status": "insufficient_data"}

        now = df["date"].max()
        recent = df[df["date"] >= now - timedelta(days=window_days)]
        previous = df[
            (df["date"] >= now - timedelta(days=window_days * 2))
            & (df["date"] < now - timedelta(days=window_days))
        ]

        if previous.empty:
            return {"status": "no_comparison_period"}

        indicators = {}

        # Frequency change
        freq_change = (len(recent) - len(previous)) / max(len(previous), 1) * 100
        indicators["frequency_change_pct"] = round(freq_change, 1)
        indicators["frequency_escalating"] = freq_change > 20

        # Casualty change
        recent_casualties = recent["casualties_killed"].sum()
        prev_casualties = previous["casualties_killed"].sum()
        cas_change = (recent_casualties - prev_casualties) / max(prev_casualties, 1) * 100
        indicators["casualty_change_pct"] = round(cas_change, 1)
        indicators["lethality_escalating"] = cas_change > 20

        # Geographic spread
        recent_lgas = recent["lga"].nunique()
        prev_lgas = previous["lga"].nunique()
        indicators["recent_lgas_affected"] = recent_lgas
        indicators["previous_lgas_affected"] = prev_lgas
        indicators["geographic_expansion"] = recent_lgas > prev_lgas

        # Attack type severity shift
        severity_map = {
            "village_raid": 4, "attack_on_security_forces": 5,
            "kidnapping": 3, "highway_ambush": 4,
            "cattle_rustling": 2, "market_attack": 4,
            "mining_site_attack": 3, "reprisal_attack": 4,
            "arms_smuggling": 3, "camp_sighting": 1,
            "displacement": 2, "negotiation_ransom": 2,
            "inter_gang_clash": 3, "infrastructure_attack": 4,
        }
        recent_severity = recent["incident_type"].map(severity_map).mean()
        prev_severity = previous["incident_type"].map(severity_map).mean()
        indicators["severity_shift"] = round(recent_severity - prev_severity, 2) if pd.notna(recent_severity) and pd.notna(prev_severity) else 0
        indicators["tactics_escalating"] = indicators["severity_shift"] > 0.5

        # Composite escalation score (0-10)
        score = 0
        if indicators["frequency_escalating"]:
            score += 3
        if indicators["lethality_escalating"]:
            score += 3
        if indicators["geographic_expansion"]:
            score += 2
        if indicators["tactics_escalating"]:
            score += 2

        indicators["escalation_score"] = score
        indicators["escalation_level"] = (
            "critical" if score >= 8 else
            "high" if score >= 6 else
            "moderate" if score >= 4 else
            "low"
        )

        return indicators

    def tactic_correlation_matrix(self, state: Optional[str] = None) -> dict:
        """
        Build a correlation matrix showing which attack types tend to
        co-occur in the same region and time period.
        """
        df = self._filter_state(state)
        if df.empty:
            return {}

        # Group incidents by state-month
        df["period"] = df["date"].dt.to_period("M").astype(str)
        grouped = df.groupby(["state", "period"])["incident_type"].apply(list)

        # Count co-occurrences
        co_occurrence = Counter()
        for types_in_period in grouped:
            unique_types = set(types_in_period)
            for t1, t2 in combinations(sorted(unique_types), 2):
                co_occurrence[(t1, t2)] += 1

        results = []
        for (t1, t2), count in co_occurrence.most_common(20):
            results.append({
                "type_1": t1,
                "type_2": t2,
                "co_occurrence_count": count,
                "interpretation": self._interpret_correlation(t1, t2),
            })

        return {"correlations": results}

    def retaliatory_pattern_detection(self, state: Optional[str] = None,
                                       max_gap_days: int = 10) -> list[dict]:
        """
        Detect retaliatory attack patterns: security operation followed
        by bandit retaliation, or inter-group attacks.
        """
        df = self._filter_state(state)
        if len(df) < 2:
            return []

        retaliatory_pairs = [
            ("attack_on_security_forces", "village_raid"),
            ("attack_on_security_forces", "highway_ambush"),
            ("inter_gang_clash", "village_raid"),
            ("cattle_rustling", "reprisal_attack"),
            ("kidnapping", "village_raid"),
        ]

        patterns = []
        records = df.to_dict("records")

        for i in range(len(records) - 1):
            for j in range(i + 1, len(records)):
                gap = (records[j]["date"] - records[i]["date"]).total_seconds() / 86400
                if gap > max_gap_days:
                    break

                pair = (records[i]["incident_type"], records[j]["incident_type"])
                if pair in retaliatory_pairs:
                    patterns.append({
                        "trigger_event": records[i]["incident_type"],
                        "trigger_date": records[i]["date"].isoformat(),
                        "trigger_location": records[i].get("lga", "Unknown"),
                        "response_event": records[j]["incident_type"],
                        "response_date": records[j]["date"].isoformat(),
                        "response_location": records[j].get("lga", "Unknown"),
                        "gap_days": round(gap, 1),
                        "same_lga": records[i].get("lga") == records[j].get("lga"),
                    })

        return patterns

    def vulnerability_windows(self, state: Optional[str] = None) -> dict:
        """
        Identify time periods when an area is most vulnerable.
        Combines temporal patterns with incident severity data.
        """
        df = self._filter_state(state)
        if df.empty:
            return {}

        # Hour-of-day vulnerability (if data is available)
        df["hour"] = df["date"].dt.hour
        hour_severity = df.groupby("hour").agg(
            count=("id", "count"),
            avg_killed=("casualties_killed", "mean"),
        )

        # Month vulnerability
        df["month"] = df["date"].dt.month
        month_severity = df.groupby("month").agg(
            count=("id", "count"),
            avg_killed=("casualties_killed", "mean"),
            total_killed=("casualties_killed", "sum"),
        )

        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December",
        }

        monthly_risk = {}
        for month, row in month_severity.iterrows():
            risk_score = row["count"] * (1 + row["avg_killed"])
            monthly_risk[month_names.get(month, str(month))] = {
                "incidents": int(row["count"]),
                "avg_casualties": round(row["avg_killed"], 1),
                "total_casualties": int(row["total_killed"]),
                "risk_score": round(risk_score, 1),
            }

        # Find peak vulnerability months
        sorted_months = sorted(monthly_risk.items(),
                               key=lambda x: x[1]["risk_score"], reverse=True)
        peak_months = [m[0] for m in sorted_months[:3]]

        return {
            "monthly_risk": monthly_risk,
            "peak_vulnerability_months": peak_months,
            "safest_months": [m[0] for m in sorted_months[-3:]],
        }

    @staticmethod
    def _interpret_sequence(sequence: list[str]) -> str:
        """Provide human-readable interpretation of attack sequences."""
        interpretations = {
            ("kidnapping", "negotiation_ransom"): "Standard kidnap-for-ransom operation cycle",
            ("cattle_rustling", "reprisal_attack"): "Rustling triggering community/vigilante reprisal",
            ("village_raid", "displacement"): "Raid causing population displacement",
            ("highway_ambush", "kidnapping"): "Roadblock/ambush leading to abductions",
            ("attack_on_security_forces", "village_raid"): "Possible retaliation after engaging security forces",
            ("camp_sighting", "village_raid"): "Staging from camps before raids",
            ("arms_smuggling", "village_raid"): "Weapons acquisition preceding major attack",
            ("inter_gang_clash", "village_raid"): "Displaced gang attacking civilians after turf war",
        }
        key = tuple(sequence[:2])
        return interpretations.get(key, "Pattern under analysis")

    @staticmethod
    def _interpret_correlation(type_1: str, type_2: str) -> str:
        """Interpret why two incident types co-occur."""
        if "kidnapping" in (type_1, type_2) and "negotiation_ransom" in (type_1, type_2):
            return "Kidnapping operations naturally lead to ransom negotiations"
        if "village_raid" in (type_1, type_2) and "displacement" in (type_1, type_2):
            return "Village raids cause civilian displacement"
        if "cattle_rustling" in (type_1, type_2) and "reprisal_attack" in (type_1, type_2):
            return "Rustling often provokes retaliatory strikes"
        return "Co-occurring activities in the same operational area"

    def _filter_state(self, state: Optional[str]) -> pd.DataFrame:
        if state and not self.df.empty:
            return self.df[self.df["state"].str.lower() == state.lower()]
        return self.df
