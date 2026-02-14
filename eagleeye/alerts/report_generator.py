"""
Intelligence report generator that produces structured security
assessment reports from analysis results.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from eagleeye.analysis.correlation import CorrelationAnalyzer
from eagleeye.analysis.spatial import SpatialAnalyzer
from eagleeye.analysis.temporal import TemporalAnalyzer
from eagleeye.core.config import TARGET_STATES
from eagleeye.models.threat_scorer import ThreatScorer

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive intelligence assessment reports."""

    def __init__(self, incidents_df: pd.DataFrame):
        self.df = incidents_df.copy()
        if not self.df.empty:
            self.df["date"] = pd.to_datetime(self.df["date"])

    def generate_situation_report(self, period_days: int = 30,
                                   state: Optional[str] = None) -> dict:
        """
        Generate a Situation Report (SITREP) covering recent activity.

        This is the primary output format for security decision-makers.
        """
        now = datetime.now()
        period_start = now - timedelta(days=period_days)

        df = self.df[self.df["date"] >= period_start]
        if state:
            df = df[df["state"].str.lower() == state.lower()]

        if df.empty:
            return {
                "report_type": "SITREP",
                "period": f"Last {period_days} days",
                "status": "No incidents recorded in this period",
                "generated_at": now.isoformat(),
            }

        # Run analysis
        temporal = TemporalAnalyzer(df)
        spatial = SpatialAnalyzer(df)
        correlation = CorrelationAnalyzer(df)
        scorer = ThreatScorer(df)

        freq = temporal.frequency_analysis(state)
        hotspots = spatial.hotspot_analysis(state, top_n=5)
        escalation = correlation.escalation_indicators(state)
        state_scores = scorer.score_all_states()

        report = {
            "report_type": "SITREP",
            "classification": "RESTRICTED",
            "generated_at": now.isoformat(),
            "period": {
                "start": period_start.isoformat(),
                "end": now.isoformat(),
                "days": period_days,
            },
            "scope": state or "All target states",

            "executive_summary": self._generate_executive_summary(
                df, freq, escalation, state
            ),

            "incident_statistics": {
                "total_incidents": len(df),
                "total_killed": int(df["casualties_killed"].sum()),
                "total_injured": int(df["casualties_injured"].sum()),
                "total_kidnapped": int(df["kidnapped_count"].sum()),
                "incidents_by_state": df.groupby("state")["id"].count().to_dict(),
                "incidents_by_type": df.groupby("incident_type")["id"].count().to_dict(),
                "avg_incidents_per_week": freq.get("incidents_per_week", 0),
                "trend": freq.get("trend", "unknown"),
            },

            "threat_assessment": {
                "state_scores": state_scores,
                "highest_threat_state": state_scores[0]["state"] if state_scores else None,
                "escalation_indicators": escalation,
            },

            "geographic_analysis": {
                "hotspots": hotspots,
                "states_affected": df["state"].nunique(),
                "lgas_affected": df["lga"].nunique(),
            },

            "pattern_analysis": {
                "attack_sequences": correlation.attack_sequence_patterns(state=state),
                "vulnerability_windows": correlation.vulnerability_windows(state=state),
            },

            "recommendations": self._generate_recommendations(
                freq, escalation, hotspots, state_scores
            ),
        }

        self._enhance_with_claude(report, "SITREP")
        return report

    def generate_threat_briefing(self, state: str) -> dict:
        """
        Generate a focused threat briefing for a specific state.
        Designed for operational commanders and security coordinators.
        """
        now = datetime.now()
        state_df = self.df[self.df["state"].str.lower() == state.lower()]

        if state_df.empty:
            return {"state": state, "status": "no_data"}

        temporal = TemporalAnalyzer(state_df)
        spatial = SpatialAnalyzer(state_df)
        scorer = ThreatScorer(state_df)
        correlation = CorrelationAnalyzer(state_df)

        score = scorer.score_state(state)
        freq = temporal.frequency_analysis()
        seasonal = temporal.seasonal_pattern()
        hotspots = spatial.hotspot_analysis(top_n=5)
        escalation = correlation.escalation_indicators()

        result = {
            "report_type": "THREAT_BRIEFING",
            "state": state,
            "generated_at": now.isoformat(),
            "threat_level": score.get("threat_level", 1),
            "threat_label": score.get("threat_label", "LOW"),
            "overall_score": score.get("overall_score", 0),

            "current_situation": {
                "trend": freq.get("trend", "unknown"),
                "incidents_per_week": freq.get("incidents_per_week", 0),
                "change_from_previous": f"{freq.get('change_percent', 0):.0f}%",
                "escalation_level": escalation.get("escalation_level", "low"),
            },

            "key_hotspots": hotspots[:5],

            "seasonal_context": {
                "current_season": self._get_current_season(),
                "seasonal_risk": seasonal.get(self._get_current_season(), {}),
                "peak_season": seasonal.get("peak_season", "unknown"),
            },

            "operational_recommendations": self._state_specific_recommendations(
                state, score, freq, escalation, hotspots
            ),
        }
        self._enhance_with_claude(result, "THREAT_BRIEFING")
        return result

    def generate_predictive_outlook(self, predictions: list[dict],
                                     state: Optional[str] = None) -> dict:
        """
        Generate a forward-looking predictive outlook report
        based on model predictions.
        """
        now = datetime.now()

        if not predictions:
            return {"status": "no_predictions_available"}

        # Aggregate predictions
        high_threat_days = [p for p in predictions if p.get("threat_level", 0) >= 4]
        avg_threat = sum(p.get("threat_level", 0) for p in predictions) / len(predictions)

        # Collect all predicted types
        type_freq = {}
        for pred in predictions:
            for t in pred.get("predicted_incident_types", []):
                name = t.get("type", "unknown")
                type_freq[name] = type_freq.get(name, 0) + t.get("probability", 0)

        top_predicted_types = sorted(type_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        result = {
            "report_type": "PREDICTIVE_OUTLOOK",
            "generated_at": now.isoformat(),
            "scope": state or "All target states",
            "horizon_days": len(predictions),

            "summary": {
                "average_threat_level": round(avg_threat, 1),
                "high_threat_days": len(high_threat_days),
                "high_threat_percentage": f"{len(high_threat_days) / len(predictions) * 100:.0f}%",
            },

            "predicted_activity_types": [
                {"type": t, "aggregate_probability": round(p / len(predictions), 3)}
                for t, p in top_predicted_types
            ],

            "high_threat_periods": [
                {
                    "date": p.get("prediction_date"),
                    "threat_level": p.get("threat_level"),
                    "threat_label": p.get("threat_label"),
                    "primary_type": p.get("predicted_incident_types", [{}])[0].get("type", "unknown"),
                }
                for p in high_threat_days
            ],

            "recommendations": list(set(
                rec
                for p in high_threat_days
                for rec in p.get("recommendations", [])
            )),
        }
        self._enhance_with_claude(result, "PREDICTIVE_OUTLOOK")
        return result

    def _enhance_with_claude(self, report: dict, report_type: str) -> None:
        """Add Claude AI narrative to a report. Modifies report in-place."""
        try:
            from eagleeye.llm.claude_engine import engine
            narrative = engine.generate_report_narrative(report_type, report)
            if narrative:
                report["narrative"] = narrative
        except Exception as e:
            logger.debug(f"Claude report enhancement failed: {e}")

    def _generate_executive_summary(self, df: pd.DataFrame, freq: dict,
                                     escalation: dict,
                                     state: Optional[str]) -> str:
        """Generate a text executive summary."""
        scope = state or "the Northwest region"
        trend = freq.get("trend", "stable")
        total = len(df)
        killed = int(df["casualties_killed"].sum())
        kidnapped = int(df["kidnapped_count"].sum())

        summary = (
            f"During the reporting period, {total} security incidents were recorded "
            f"across {scope}. "
        )

        if killed > 0:
            summary += f"A total of {killed} fatalities and {kidnapped} abductions were reported. "

        if trend == "escalating":
            summary += (
                f"Activity is ESCALATING with a {freq.get('change_percent', 0):.0f}% increase "
                f"compared to the previous period. "
            )
        elif trend == "de-escalating":
            summary += (
                f"Activity has decreased by {abs(freq.get('change_percent', 0)):.0f}% "
                f"compared to the previous period. "
            )
        else:
            summary += "Activity levels remain relatively stable compared to the previous period. "

        esc_level = escalation.get("escalation_level", "low")
        if esc_level in ("critical", "high"):
            summary += (
                f"WARNING: Escalation indicators are at {esc_level.upper()} level. "
                f"Multiple factors suggest increased threat posture."
            )

        return summary

    def _generate_recommendations(self, freq: dict, escalation: dict,
                                   hotspots: list, state_scores: list) -> list[str]:
        """Generate strategic recommendations."""
        recs = []

        if freq.get("trend") == "escalating":
            recs.append("URGENT: Increase operational tempo and patrol frequency across affected areas")
            recs.append("Deploy additional rapid response units to escalation zones")

        if escalation.get("geographic_expansion"):
            recs.append("Geographic expansion detected — extend security coverage to newly affected LGAs")

        if escalation.get("tactics_escalating"):
            recs.append("Tactics are becoming more severe — review and upgrade response protocols")

        if hotspots:
            top_spot = hotspots[0]
            recs.append(
                f"Priority deployment to {top_spot['lga']}, {top_spot['state']} "
                f"(highest risk score: {top_spot['risk_score']})"
            )

        if state_scores:
            critical_states = [s for s in state_scores if s.get("threat_level", 0) >= 4]
            if critical_states:
                names = ", ".join(s["state"] for s in critical_states)
                recs.append(f"States at HIGH/CRITICAL threat: {names} — prioritize resource allocation")

        recs.append("Maintain and strengthen community early warning networks")
        recs.append("Continue intelligence gathering on bandit logistics and supply routes")

        return recs

    def _state_specific_recommendations(self, state: str, score: dict,
                                         freq: dict, escalation: dict,
                                         hotspots: list) -> list[str]:
        """Generate state-specific operational recommendations."""
        recs = []
        threat = score.get("threat_level", 1)

        if threat >= 4:
            recs.append(f"Maintain HIGH alert posture across {state}")

        if freq.get("trend") == "escalating":
            recs.append("Increase frequency of joint security patrols")

        if hotspots:
            for spot in hotspots[:3]:
                recs.append(f"Reinforce security presence in {spot['lga']} LGA")

        # State-specific context
        state_notes = {
            "Zamfara": "Monitor Zamfara-Sokoto-Katsina tri-border corridor for cross-state movement",
            "Kaduna": "Focus on Southern Kaduna and Birnin Gwari axis — known high-risk corridor",
            "Katsina": "Strengthen border surveillance along Katsina-Niger Republic frontier",
            "Sokoto": "Monitor eastern Sokoto LGAs bordering Zamfara",
            "Niger": "Focus on Shiroro-Munya-Rafi triangle and Kontagora axis",
            "Kebbi": "Monitor Danko-Wasagu and Zuru areas bordering Zamfara",
        }
        if state in state_notes:
            recs.append(state_notes[state])

        return recs

    def generate_templated_sitrep(self, template: str, period_days: int = 30,
                                    state: Optional[str] = None,
                                    context_texts: list[str] = None) -> dict:
        """
        Generate a SITREP using a user-provided template format.
        Gathers the same data as generate_situation_report() but passes it
        to Claude with the user's template for formatting.
        Falls back to standard SITREP if Claude is unavailable.
        """
        # Gather data the same way as standard SITREP
        now = datetime.now()
        period_start = now - timedelta(days=period_days)

        df = self.df[self.df["date"] >= period_start]
        if state:
            df = df[df["state"].str.lower() == state.lower()]

        if df.empty:
            return {
                "report_type": "TEMPLATED_SITREP",
                "period": f"Last {period_days} days",
                "status": "No incidents recorded in this period",
                "generated_at": now.isoformat(),
            }

        temporal = TemporalAnalyzer(df)
        spatial = SpatialAnalyzer(df)
        correlation = CorrelationAnalyzer(df)
        scorer = ThreatScorer(df)

        freq = temporal.frequency_analysis(state)
        hotspots = spatial.hotspot_analysis(state, top_n=5)
        escalation = correlation.escalation_indicators(state)
        state_scores = scorer.score_all_states()

        data = {
            "period": {"start": period_start.isoformat(), "end": now.isoformat(), "days": period_days},
            "scope": state or "All target states",
            "incident_statistics": {
                "total_incidents": len(df),
                "total_killed": int(df["casualties_killed"].sum()),
                "total_injured": int(df["casualties_injured"].sum()),
                "total_kidnapped": int(df["kidnapped_count"].sum()),
                "incidents_by_state": df.groupby("state")["id"].count().to_dict(),
                "incidents_by_type": df.groupby("incident_type")["id"].count().to_dict(),
                "avg_incidents_per_week": freq.get("incidents_per_week", 0),
                "trend": freq.get("trend", "unknown"),
            },
            "threat_assessment": {
                "state_scores": state_scores,
                "escalation_indicators": escalation,
            },
            "hotspots": hotspots,
            "recommendations": self._generate_recommendations(freq, escalation, hotspots, state_scores),
        }

        result = {
            "report_type": "TEMPLATED_SITREP",
            "classification": "RESTRICTED",
            "generated_at": now.isoformat(),
            "data": data,
        }

        # Include uploaded context in data if provided
        if context_texts:
            data["additional_context"] = context_texts

        # Try Claude with template
        try:
            from eagleeye.llm.claude_engine import engine
            narrative = engine.generate_templated_sitrep(template, data, context_texts)
            if narrative:
                result["narrative"] = narrative
                return result
        except Exception as e:
            logger.debug(f"Claude templated SITREP failed: {e}")

        # Fallback: generate standard SITREP
        logger.info("Falling back to standard SITREP (Claude unavailable)")
        result["narrative"] = self._generate_executive_summary(df, freq, escalation, state)
        result["fallback"] = True
        return result

    def generate_area_threat_briefing(self, area_name: str = "", state: str = "",
                                       lat: float = None, lon: float = None) -> dict:
        """
        Generate a terrain-integrated area threat briefing.
        Uses TerrainAnalyzer for OSM data and AI for JIFC narrative.
        """
        from eagleeye.analysis.terrain import TerrainAnalyzer

        now = datetime.now()
        terrain_analyzer = TerrainAnalyzer()

        # Get terrain data
        terrain = terrain_analyzer.analyze_terrain(
            area_name=area_name, state=state, lat=lat, lon=lon,
        )

        if "error" in terrain and "area_info" not in terrain:
            return {
                "report_type": "AREA_THREAT_BRIEFING",
                "error": terrain["error"],
                "generated_at": now.isoformat(),
            }

        area_info = terrain.get("area_info", {})
        resolved_name = area_info.get("area_name", area_name)

        # Gather incident data for the state
        state_df = self.df[self.df["state"].str.lower() == state.lower()] if state else self.df
        incident_data = {}
        if not state_df.empty:
            scorer = ThreatScorer(state_df)
            state_score = scorer.score_state(state) if state else {}
            recent_30d = state_df[state_df["date"] >= now - timedelta(days=30)]
            incident_data = {
                "incidents": {
                    "total_all_time": len(state_df),
                    "total_last_30d": len(recent_30d),
                    "killed_last_30d": int(recent_30d["casualties_killed"].sum()) if not recent_30d.empty else 0,
                    "kidnapped_last_30d": int(recent_30d["kidnapped_count"].sum()) if not recent_30d.empty else 0,
                    "types_last_30d": recent_30d.groupby("incident_type")["id"].count().to_dict() if not recent_30d.empty else {},
                    "lgas_affected": recent_30d["lga"].value_counts().head(5).to_dict() if not recent_30d.empty else {},
                },
                "threat_scores": state_score,
            }

        result = {
            "report_type": "AREA_THREAT_BRIEFING",
            "generated_at": now.isoformat(),
            "area": resolved_name,
            "state": state,
            "area_info": area_info,
            "terrain": {
                "tactical_summary": terrain.get("tactical_summary", {}),
                "settlements": terrain.get("settlements", [])[:10],
                "road_summary": terrain.get("road_summary", {}),
                "waterways": terrain.get("waterways", [])[:10],
                "infrastructure": terrain.get("infrastructure", [])[:10],
                "natural_features": terrain.get("natural_features", [])[:10],
            },
            "incident_summary": incident_data,
        }

        # Try AI for JIFC narrative
        try:
            from eagleeye.llm.claude_engine import engine
            narrative = engine.generate_terrain_threat_briefing(
                resolved_name, state, terrain, incident_data,
            )
            if narrative:
                result["narrative"] = narrative
        except Exception as e:
            logger.debug(f"Claude terrain briefing failed: {e}")

        return result

    @staticmethod
    def _get_current_season() -> str:
        month = datetime.now().month
        if month in [10, 11, 12]:
            return "dry_early"
        elif month in [1, 2, 3]:
            return "dry_peak"
        elif month in [4, 5]:
            return "dry_late"
        return "rainy"
