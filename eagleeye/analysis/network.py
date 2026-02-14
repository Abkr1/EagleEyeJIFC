"""
Network analysis module for understanding bandit group structures,
alliances, territorial control, and inter-group dynamics.
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """Analyzes bandit group networks, territories, and relationships."""

    def __init__(self, incidents_df: pd.DataFrame, groups_data: Optional[list[dict]] = None):
        """
        Args:
            incidents_df: DataFrame of incidents.
            groups_data: Optional list of known bandit group records.
        """
        self.df = incidents_df.copy()
        if not self.df.empty:
            self.df["date"] = pd.to_datetime(self.df["date"])
        self.groups = groups_data or []

    def territory_mapping(self) -> dict:
        """
        Map bandit territorial control based on incident patterns.

        Areas with sustained, high-frequency attacks from the same type
        of operations suggest territorial control or operating bases.
        """
        if self.df.empty:
            return {}

        territories = {}
        for state in self.df["state"].unique():
            state_df = self.df[self.df["state"] == state]
            lga_analysis = {}

            for lga in state_df["lga"].dropna().unique():
                lga_df = state_df[state_df["lga"] == lga]

                # Calculate control indicators
                incident_count = len(lga_df)
                date_range = (lga_df["date"].max() - lga_df["date"].min()).days
                frequency = incident_count / max(date_range / 30, 1)  # incidents per month

                # High frequency + sustained activity = likely territory
                sustained = date_range > 90
                high_freq = frequency > 2

                lga_analysis[lga] = {
                    "incident_count": incident_count,
                    "date_range_days": date_range,
                    "incidents_per_month": round(frequency, 1),
                    "sustained_activity": sustained,
                    "dominant_types": dict(lga_df["incident_type"].value_counts().head(3)),
                    "total_killed": int(lga_df["casualties_killed"].sum()),
                    "control_indicator": "high" if (sustained and high_freq) else "medium" if sustained else "low",
                    "last_activity": lga_df["date"].max().isoformat(),
                }

            territories[state] = {
                "lgas": lga_analysis,
                "total_lgas_affected": len(lga_analysis),
                "high_control_lgas": [
                    lga for lga, data in lga_analysis.items()
                    if data["control_indicator"] == "high"
                ],
            }

        return territories

    def operational_pattern_profiling(self) -> list[dict]:
        """
        Profile distinct operational patterns that may correspond to
        different bandit groups or factions.

        Different groups often have signature tactics:
        - Some specialize in kidnapping (Zamfara corridor groups)
        - Some focus on cattle rustling (Katsina/Zamfara border)
        - Some conduct large-scale village raids (Kaduna southern groups)
        """
        if self.df.empty:
            return []

        # Cluster by LGA + incident type combination
        profiles = defaultdict(lambda: {
            "locations": set(),
            "states": set(),
            "incident_types": Counter(),
            "dates": [],
            "casualties": [],
            "estimated_sizes": [],
        })

        for _, row in self.df.iterrows():
            # Create a regional key
            key = (row["state"], row.get("lga", "Unknown"))
            profile = profiles[key]
            profile["locations"].add(row.get("location_name", "Unknown"))
            profile["states"].add(row["state"])
            profile["incident_types"][row["incident_type"]] += 1
            profile["dates"].append(row["date"])
            profile["casualties"].append(row["casualties_killed"])
            if pd.notna(row.get("estimated_bandits")):
                profile["estimated_sizes"].append(row["estimated_bandits"])

        results = []
        for (state, lga), profile in profiles.items():
            if len(profile["dates"]) < 3:
                continue

            dominant_tactic = profile["incident_types"].most_common(1)[0][0]
            dates = sorted(profile["dates"])
            avg_interval = np.mean([(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)])

            results.append({
                "state": state,
                "lga": lga,
                "incident_count": sum(profile["incident_types"].values()),
                "dominant_tactic": dominant_tactic,
                "tactic_distribution": dict(profile["incident_types"]),
                "avg_interval_days": round(avg_interval, 1),
                "total_casualties": sum(profile["casualties"]),
                "estimated_group_size": round(np.mean(profile["estimated_sizes"])) if profile["estimated_sizes"] else None,
                "active_period": {
                    "start": min(dates).isoformat(),
                    "end": max(dates).isoformat(),
                    "duration_days": (max(dates) - min(dates)).days,
                },
                "locations": list(profile["locations"])[:10],
            })

        results.sort(key=lambda x: x["incident_count"], reverse=True)
        return results

    def supply_route_analysis(self) -> dict:
        """
        Identify likely supply and smuggling routes based on
        arms_smuggling incidents and geographic patterns.
        """
        smuggling = self.df[self.df["incident_type"].isin(["arms_smuggling", "mining_site_attack"])]
        if smuggling.empty:
            return {"status": "no_smuggling_data"}

        routes = []
        for state in smuggling["state"].unique():
            state_data = smuggling[smuggling["state"] == state]
            lgas = state_data["lga"].value_counts()
            routes.append({
                "state": state,
                "affected_lgas": lgas.to_dict(),
                "incident_count": len(state_data),
                "date_range": {
                    "first": state_data["date"].min().isoformat(),
                    "last": state_data["date"].max().isoformat(),
                },
            })

        return {"identified_routes": routes}

    def group_activity_timeline(self, state: Optional[str] = None) -> dict:
        """
        Create a timeline of activity showing operational tempo
        and identifying periods of increased coordination.
        """
        df = self.df if state is None else self.df[self.df["state"].str.lower() == state.lower()]
        if df.empty:
            return {}

        # Weekly activity timeline
        weekly = df.groupby(df["date"].dt.to_period("W")).agg(
            incident_count=("id", "count"),
            killed=("casualties_killed", "sum"),
            kidnapped=("kidnapped_count", "sum"),
            types=("incident_type", lambda x: list(x)),
        ).reset_index()

        timeline = []
        for _, row in weekly.iterrows():
            week_start = row["date"].start_time.strftime("%Y-%m-%d")
            types_counter = Counter(row["types"])

            # Detect coordinated operations (multiple incidents of same type in one week)
            coordinated = any(count >= 3 for count in types_counter.values())

            timeline.append({
                "week_start": week_start,
                "incident_count": int(row["incident_count"]),
                "killed": int(row["killed"]),
                "kidnapped": int(row["kidnapped"]),
                "incident_types": dict(types_counter),
                "coordinated_activity": coordinated,
                "intensity": (
                    "high" if row["incident_count"] >= 5 else
                    "medium" if row["incident_count"] >= 2 else
                    "low"
                ),
            })

        return {
            "state": state or "all",
            "timeline": timeline,
            "total_weeks": len(timeline),
            "high_intensity_weeks": sum(1 for t in timeline if t["intensity"] == "high"),
            "coordinated_weeks": sum(1 for t in timeline if t["coordinated_activity"]),
        }

    def identify_operational_lulls(self, state: Optional[str] = None,
                                    min_gap_days: int = 14) -> list[dict]:
        """
        Identify periods of low/no activity (lulls) which may indicate:
        - Regrouping/resupply
        - Internal disputes
        - Security pressure working
        - Planning for larger operations
        """
        df = self.df if state is None else self.df[self.df["state"].str.lower() == state.lower()]
        if len(df) < 2:
            return []

        df_sorted = df.sort_values("date")
        dates = df_sorted["date"].values

        lulls = []
        for i in range(len(dates) - 1):
            gap = (pd.Timestamp(dates[i + 1]) - pd.Timestamp(dates[i])).days
            if gap >= min_gap_days:
                # Check what happened after the lull
                after_lull = df_sorted[df_sorted["date"] > pd.Timestamp(dates[i + 1])]
                post_lull_intensity = len(after_lull[
                    after_lull["date"] <= pd.Timestamp(dates[i + 1]) + timedelta(days=14)
                ])

                lulls.append({
                    "start_date": pd.Timestamp(dates[i]).isoformat(),
                    "end_date": pd.Timestamp(dates[i + 1]).isoformat(),
                    "gap_days": gap,
                    "post_lull_incidents_14d": post_lull_intensity,
                    "followed_by_surge": post_lull_intensity >= 3,
                    "interpretation": (
                        "Lull followed by surge - possible regrouping/planning"
                        if post_lull_intensity >= 3
                        else "Extended quiet period"
                    ),
                })

        lulls.sort(key=lambda x: x["gap_days"], reverse=True)
        return lulls
