"""
Spatial analysis module for identifying geographic patterns in bandit activity.

Analyzes: hotspot identification, movement corridors, territory mapping,
geographic clustering, and spatial prediction.
"""

import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in km."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


class SpatialAnalyzer:
    """Analyzes geographic patterns in incident data."""

    def __init__(self, incidents_df: pd.DataFrame):
        self.df = incidents_df.copy()
        if not self.df.empty:
            self.df["date"] = pd.to_datetime(self.df["date"])

    def hotspot_analysis(self, state: Optional[str] = None,
                         top_n: int = 10) -> list[dict]:
        """
        Identify geographic hotspots — LGAs with the highest incident concentration.
        """
        df = self._filter_state(state)
        if df.empty:
            return []

        lga_stats = df.groupby(["state", "lga"]).agg(
            incident_count=("id", "count"),
            total_killed=("casualties_killed", "sum"),
            total_kidnapped=("kidnapped_count", "sum"),
            avg_threat_level=("threat_level", "mean"),
            last_incident=("date", "max"),
        ).reset_index()

        lga_stats = lga_stats.sort_values("incident_count", ascending=False)

        hotspots = []
        for _, row in lga_stats.head(top_n).iterrows():
            # Get dominant incident type for this LGA
            lga_df = df[(df["state"] == row["state"]) & (df["lga"] == row["lga"])]
            dominant_type = lga_df["incident_type"].mode().iloc[0] if not lga_df["incident_type"].mode().empty else "unknown"

            # Calculate recency score (higher = more recent activity)
            days_since_last = (datetime.now() - row["last_incident"]).days if pd.notna(row["last_incident"]) else 999
            recency_score = max(0, 1 - (days_since_last / 365))

            hotspots.append({
                "state": row["state"],
                "lga": row["lga"],
                "incident_count": int(row["incident_count"]),
                "total_killed": int(row["total_killed"]),
                "total_kidnapped": int(row["total_kidnapped"]),
                "avg_threat_level": round(row["avg_threat_level"], 1),
                "dominant_incident_type": dominant_type,
                "last_incident": row["last_incident"].isoformat() if pd.notna(row["last_incident"]) else None,
                "days_since_last_incident": days_since_last,
                "recency_score": round(recency_score, 2),
                "risk_score": round(row["incident_count"] * recency_score * row["avg_threat_level"], 1),
            })

        # Sort by composite risk score
        hotspots.sort(key=lambda x: x["risk_score"], reverse=True)
        return hotspots

    def movement_corridors(self, days_window: int = 7,
                            max_distance_km: float = 150) -> list[dict]:
        """
        Detect likely movement corridors by analyzing sequential incidents
        that are geographically close within a time window.

        The assumption: if incidents happen in Location A then Location B
        within a few days and within a reasonable travel distance, it may
        indicate the same group moving along a corridor.
        """
        df = self.df.dropna(subset=["latitude", "longitude"]).sort_values("date")
        if len(df) < 2:
            return []

        corridors = []
        records = df.to_dict("records")

        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                time_diff = (records[j]["date"] - records[i]["date"]).total_seconds() / 86400
                if time_diff > days_window:
                    break
                if time_diff <= 0:
                    continue

                dist = haversine_distance(
                    records[i]["latitude"], records[i]["longitude"],
                    records[j]["latitude"], records[j]["longitude"],
                )

                if 5 < dist < max_distance_km:  # Skip if same location or too far
                    corridors.append({
                        "from_location": records[i].get("location_name", records[i].get("lga", "Unknown")),
                        "from_state": records[i]["state"],
                        "from_coords": [records[i]["latitude"], records[i]["longitude"]],
                        "to_location": records[j].get("location_name", records[j].get("lga", "Unknown")),
                        "to_state": records[j]["state"],
                        "to_coords": [records[j]["latitude"], records[j]["longitude"]],
                        "distance_km": round(dist, 1),
                        "time_gap_days": round(time_diff, 1),
                        "from_date": records[i]["date"].isoformat(),
                        "to_date": records[j]["date"].isoformat(),
                        "from_type": records[i]["incident_type"],
                        "to_type": records[j]["incident_type"],
                    })

        # Score corridors by frequency (same from/to pair appearing multiple times)
        corridor_freq = Counter()
        for c in corridors:
            key = (c["from_state"], c["from_location"], c["to_state"], c["to_location"])
            corridor_freq[key] += 1

        for c in corridors:
            key = (c["from_state"], c["from_location"], c["to_state"], c["to_location"])
            c["frequency"] = corridor_freq[key]

        corridors.sort(key=lambda x: x["frequency"], reverse=True)
        return corridors

    def state_comparison(self) -> list[dict]:
        """Compare threat levels and activity across all target states."""
        if self.df.empty:
            return []

        results = []
        for state in self.df["state"].unique():
            state_df = self.df[self.df["state"] == state]
            recent_30d = state_df[state_df["date"] >= datetime.now() - timedelta(days=30)]
            recent_90d = state_df[state_df["date"] >= datetime.now() - timedelta(days=90)]

            results.append({
                "state": state,
                "total_incidents": len(state_df),
                "last_30_days": len(recent_30d),
                "last_90_days": len(recent_90d),
                "total_killed": int(state_df["casualties_killed"].sum()),
                "total_kidnapped": int(state_df["kidnapped_count"].sum()),
                "avg_threat_level": round(state_df["threat_level"].mean(), 1),
                "most_common_type": state_df["incident_type"].mode().iloc[0] if not state_df["incident_type"].mode().empty else "unknown",
                "unique_lgas_affected": state_df["lga"].nunique(),
                "last_incident": state_df["date"].max().isoformat(),
            })

        results.sort(key=lambda x: x["last_30_days"], reverse=True)
        return results

    def geographic_clustering(self, radius_km: float = 30) -> list[dict]:
        """
        Identify geographic clusters of incidents using a simple
        density-based approach. Incidents within radius_km of each other
        are grouped into clusters.
        """
        df = self.df.dropna(subset=["latitude", "longitude"])
        if len(df) < 2:
            return []

        points = df[["latitude", "longitude", "id", "state", "lga", "incident_type"]].to_dict("records")
        visited = set()
        clusters = []

        for i, point in enumerate(points):
            if i in visited:
                continue

            cluster = [point]
            visited.add(i)

            for j, other in enumerate(points):
                if j in visited:
                    continue
                dist = haversine_distance(
                    point["latitude"], point["longitude"],
                    other["latitude"], other["longitude"],
                )
                if dist <= radius_km:
                    cluster.append(other)
                    visited.add(j)

            if len(cluster) >= 3:  # Minimum 3 incidents to form a cluster
                lats = [p["latitude"] for p in cluster]
                lons = [p["longitude"] for p in cluster]
                clusters.append({
                    "center_lat": round(np.mean(lats), 4),
                    "center_lon": round(np.mean(lons), 4),
                    "incident_count": len(cluster),
                    "radius_km": radius_km,
                    "states": list(set(p["state"] for p in cluster)),
                    "lgas": list(set(p["lga"] for p in cluster if p.get("lga"))),
                    "incident_types": dict(Counter(p["incident_type"] for p in cluster)),
                })

        clusters.sort(key=lambda x: x["incident_count"], reverse=True)
        return clusters

    def cross_border_analysis(self) -> list[dict]:
        """
        Analyze incidents that occur near state borders, which may indicate
        bandits exploiting jurisdictional gaps between security forces.
        """
        if self.df.empty:
            return []

        # Analyze pairs of states with incidents close in time
        state_pairs = defaultdict(list)
        df_sorted = self.df.sort_values("date")

        for i in range(len(df_sorted)):
            for j in range(i + 1, min(i + 50, len(df_sorted))):
                row_i = df_sorted.iloc[i]
                row_j = df_sorted.iloc[j]

                if row_i["state"] == row_j["state"]:
                    continue

                time_diff = (row_j["date"] - row_i["date"]).total_seconds() / 86400
                if time_diff > 5:
                    break

                pair = tuple(sorted([row_i["state"], row_j["state"]]))
                state_pairs[pair].append({
                    "time_gap_days": round(time_diff, 1),
                    "states": list(pair),
                    "from_lga": row_i.get("lga"),
                    "to_lga": row_j.get("lga"),
                })

        results = []
        for pair, events in state_pairs.items():
            results.append({
                "state_pair": list(pair),
                "cross_border_events": len(events),
                "avg_time_gap_days": round(np.mean([e["time_gap_days"] for e in events]), 1),
                "common_lgas": list(set(
                    e["from_lga"] for e in events if e["from_lga"]
                ) | set(
                    e["to_lga"] for e in events if e["to_lga"]
                )),
            })

        results.sort(key=lambda x: x["cross_border_events"], reverse=True)
        return results

    def _filter_state(self, state: Optional[str]) -> pd.DataFrame:
        if state and not self.df.empty:
            return self.df[self.df["state"].str.lower() == state.lower()]
        return self.df
