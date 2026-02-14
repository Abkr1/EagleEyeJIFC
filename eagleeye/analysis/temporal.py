"""
Temporal analysis module for identifying time-based patterns in bandit activity.

Analyzes: seasonal trends, day-of-week patterns, attack frequency changes,
escalation/de-escalation cycles, and predictive time windows.
"""

import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """Analyzes time-based patterns in incident data."""

    def __init__(self, incidents_df: pd.DataFrame):
        """
        Args:
            incidents_df: DataFrame with at least 'date', 'state', 'incident_type' columns.
        """
        self.df = incidents_df.copy()
        if not self.df.empty:
            self.df["date"] = pd.to_datetime(self.df["date"])
            self.df = self.df.sort_values("date")
            self.df["month"] = self.df["date"].dt.month
            self.df["day_of_week"] = self.df["date"].dt.dayofweek
            self.df["week_of_year"] = self.df["date"].dt.isocalendar().week.astype(int)
            self.df["hour"] = self.df["date"].dt.hour
            self.df["year"] = self.df["date"].dt.year

    def monthly_trend(self, state: Optional[str] = None) -> pd.DataFrame:
        """Calculate monthly incident counts and trends."""
        df = self._filter_state(state)
        if df.empty:
            return pd.DataFrame()

        monthly = df.groupby(df["date"].dt.to_period("M")).agg(
            incident_count=("id", "count"),
            killed=("casualties_killed", "sum"),
            kidnapped=("kidnapped_count", "sum"),
        ).reset_index()
        monthly["date"] = monthly["date"].dt.to_timestamp()

        # Calculate rolling average
        if len(monthly) >= 3:
            monthly["rolling_avg_3m"] = monthly["incident_count"].rolling(3).mean()
            monthly["trend"] = monthly["incident_count"].diff()

        return monthly

    def seasonal_pattern(self, state: Optional[str] = None) -> dict:
        """
        Identify seasonal patterns in bandit activity.

        Key insight for Northern Nigeria: dry season (Oct-May) typically sees
        more bandit movement due to better road conditions and harvest periods.
        Rainy season (Jun-Sep) can reduce mobility but push bandits to highways.
        """
        df = self._filter_state(state)
        if df.empty:
            return {}

        monthly_counts = df.groupby("month")["id"].count()

        # Define seasons for the Sahel region
        seasons = {
            "dry_early": [10, 11, 12],    # Oct-Dec: early dry season, post-harvest raids
            "dry_peak": [1, 2, 3],          # Jan-Mar: peak dry season, high mobility
            "dry_late": [4, 5],             # Apr-May: late dry, pre-planting
            "rainy": [6, 7, 8, 9],          # Jun-Sep: rainy season
        }

        season_data = {}
        for season_name, months in seasons.items():
            counts = [monthly_counts.get(m, 0) for m in months]
            season_data[season_name] = {
                "months": months,
                "total_incidents": sum(counts),
                "avg_per_month": np.mean(counts) if counts else 0,
                "peak_month": months[np.argmax(counts)] if counts else None,
            }

        # Identify the most dangerous season
        peak_season = max(season_data, key=lambda s: season_data[s]["avg_per_month"])
        season_data["peak_season"] = peak_season

        return season_data

    def day_of_week_pattern(self, state: Optional[str] = None) -> dict:
        """Analyze which days of the week see the most activity."""
        df = self._filter_state(state)
        if df.empty:
            return {}

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
        day_counts = df.groupby("day_of_week")["id"].count()

        pattern = {}
        for day_num, day_name in enumerate(day_names):
            pattern[day_name] = int(day_counts.get(day_num, 0))

        peak_day = max(pattern, key=pattern.get) if pattern else None
        pattern["peak_day"] = peak_day

        # Market day analysis - many attacks correlate with market days
        pattern["market_day_correlation"] = self._check_market_day_correlation(df)

        return pattern

    def frequency_analysis(self, state: Optional[str] = None,
                           window_days: int = 30) -> dict:
        """
        Analyze attack frequency and detect escalation/de-escalation.

        Returns:
            Dict with frequency metrics and escalation indicators.
        """
        df = self._filter_state(state)
        if df.empty:
            return {"status": "no_data"}

        now = df["date"].max()
        recent = df[df["date"] >= now - timedelta(days=window_days)]
        previous = df[
            (df["date"] >= now - timedelta(days=window_days * 2))
            & (df["date"] < now - timedelta(days=window_days))
        ]

        recent_count = len(recent)
        previous_count = len(previous)

        # Calculate change
        if previous_count > 0:
            change_pct = ((recent_count - previous_count) / previous_count) * 100
        else:
            change_pct = 100.0 if recent_count > 0 else 0.0

        # Determine trend
        if change_pct > 25:
            trend = "escalating"
        elif change_pct < -25:
            trend = "de-escalating"
        else:
            trend = "stable"

        # Calculate average days between incidents
        if len(df) >= 2:
            date_diffs = df["date"].diff().dropna().dt.total_seconds() / 86400
            avg_interval = date_diffs.mean()
            min_interval = date_diffs.min()
        else:
            avg_interval = None
            min_interval = None

        return {
            "window_days": window_days,
            "recent_count": recent_count,
            "previous_count": previous_count,
            "change_percent": round(change_pct, 1),
            "trend": trend,
            "avg_days_between_incidents": round(avg_interval, 1) if avg_interval else None,
            "min_days_between_incidents": round(min_interval, 1) if min_interval else None,
            "incidents_per_week": round(recent_count / (window_days / 7), 1),
        }

    def detect_surge_periods(self, threshold_multiplier: float = 2.0,
                              state: Optional[str] = None) -> list[dict]:
        """
        Detect periods of abnormally high activity (surges).

        A surge is defined as a week with incident count exceeding
        threshold_multiplier * the overall weekly average.
        """
        df = self._filter_state(state)
        if df.empty:
            return []

        weekly = df.groupby([df["date"].dt.isocalendar().year,
                             df["date"].dt.isocalendar().week])["id"].count()
        avg_weekly = weekly.mean()
        threshold = avg_weekly * threshold_multiplier

        surges = []
        for (year, week), count in weekly.items():
            if count >= threshold:
                # Find the date range for this week
                week_start = datetime.strptime(f"{year}-W{week:02d}-1", "%G-W%V-%u")
                surges.append({
                    "year": int(year),
                    "week": int(week),
                    "week_start": week_start.strftime("%Y-%m-%d"),
                    "incident_count": int(count),
                    "weekly_average": round(avg_weekly, 1),
                    "surge_factor": round(count / avg_weekly, 1),
                })

        return sorted(surges, key=lambda x: x["surge_factor"], reverse=True)

    def incident_type_evolution(self, state: Optional[str] = None) -> dict:
        """Track how the mix of incident types changes over time."""
        df = self._filter_state(state)
        if df.empty:
            return {}

        # Group by quarter
        df["quarter"] = df["date"].dt.to_period("Q")
        quarterly = df.groupby(["quarter", "incident_type"])["id"].count().unstack(fill_value=0)

        evolution = {}
        for quarter in quarterly.index:
            q_str = str(quarter)
            counts = quarterly.loc[quarter].to_dict()
            total = sum(counts.values())
            evolution[q_str] = {
                "counts": {k: int(v) for k, v in counts.items()},
                "total": total,
                "dominant_type": max(counts, key=counts.get) if counts else None,
                "proportions": {k: round(v / total, 2) for k, v in counts.items()} if total > 0 else {},
            }

        return evolution

    def predict_next_window(self, state: Optional[str] = None,
                             horizon_days: int = 14) -> dict:
        """
        Basic temporal prediction: estimate likely activity level
        in the next N days based on historical patterns.
        """
        df = self._filter_state(state)
        if len(df) < 10:
            return {"status": "insufficient_data", "min_required": 10}

        freq = self.frequency_analysis(state)
        seasonal = self.seasonal_pattern(state)

        current_month = datetime.now().month

        # Find which season we're in
        current_season = "rainy"
        for season_name, data in seasonal.items():
            if season_name in ("peak_season",):
                continue
            if isinstance(data, dict) and current_month in data.get("months", []):
                current_season = season_name
                break

        # Estimate expected incidents
        weekly_rate = freq.get("incidents_per_week", 0)
        expected_incidents = round(weekly_rate * (horizon_days / 7), 1)

        # Adjust by seasonal factor
        if seasonal and current_season != "peak_season":
            season_avg = seasonal.get(current_season, {}).get("avg_per_month", 1)
            overall_avg = np.mean([
                s.get("avg_per_month", 0) for k, s in seasonal.items()
                if isinstance(s, dict) and "avg_per_month" in s
            ])
            if overall_avg > 0:
                seasonal_factor = season_avg / overall_avg
                expected_incidents = round(expected_incidents * seasonal_factor, 1)

        return {
            "horizon_days": horizon_days,
            "expected_incidents": expected_incidents,
            "current_trend": freq.get("trend", "unknown"),
            "current_season": current_season,
            "weekly_rate": weekly_rate,
            "warning": "escalating" if freq.get("trend") == "escalating" else "normal",
        }

    def _filter_state(self, state: Optional[str]) -> pd.DataFrame:
        if state and not self.df.empty:
            return self.df[self.df["state"].str.lower() == state.lower()]
        return self.df

    @staticmethod
    def _check_market_day_correlation(df: pd.DataFrame) -> str:
        """
        Check if attacks correlate with typical market days.
        In Northern Nigeria, markets often run on specific weekdays
        that vary by locality.
        """
        if df.empty:
            return "insufficient_data"

        day_counts = df["day_of_week"].value_counts()
        if len(day_counts) < 3:
            return "insufficient_data"

        # Check if distribution is significantly non-uniform
        expected = len(df) / 7
        max_count = day_counts.max()
        if max_count > expected * 1.5:
            peak_day = day_counts.idxmax()
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
            return f"possible_correlation_with_{day_names[peak_day]}"

        return "no_significant_correlation"
