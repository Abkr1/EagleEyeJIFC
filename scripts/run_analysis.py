#!/usr/bin/env python3
"""
Script to run a full analysis cycle and generate reports.

Usage:
    python scripts/run_analysis.py [--state STATE] [--days DAYS]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eagleeye.alerts.report_generator import ReportGenerator
from eagleeye.analysis.correlation import CorrelationAnalyzer
from eagleeye.analysis.network import NetworkAnalyzer
from eagleeye.analysis.spatial import SpatialAnalyzer
from eagleeye.analysis.temporal import TemporalAnalyzer
from eagleeye.core.config import TARGET_STATES
from eagleeye.data.processor import DataProcessor
from eagleeye.models.threat_scorer import ThreatScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run EagleEye analysis")
    parser.add_argument("--state", type=str, default=None, help="Focus on a specific state")
    parser.add_argument("--days", type=int, default=365, help="Analysis period in days")
    parser.add_argument("--output", type=str, default=None, help="Output file for JSON report")
    args = parser.parse_args()

    processor = DataProcessor()
    df = processor.get_incidents_dataframe(state=args.state)

    if df.empty:
        logger.error("No data available. Run scripts/train_model.py first to ingest data.")
        sys.exit(1)

    logger.info(f"Analyzing {len(df)} incidents...")

    # Run all analyses
    temporal = TemporalAnalyzer(df)
    spatial = SpatialAnalyzer(df)
    correlation = CorrelationAnalyzer(df)
    network = NetworkAnalyzer(df)
    scorer = ThreatScorer(df)

    report = {
        "threat_scores": scorer.score_all_states(),
        "temporal": {
            "frequency": temporal.frequency_analysis(args.state),
            "seasonal": temporal.seasonal_pattern(args.state),
            "surges": temporal.detect_surge_periods(state=args.state),
        },
        "spatial": {
            "hotspots": spatial.hotspot_analysis(args.state),
            "state_comparison": spatial.state_comparison(),
        },
        "correlation": {
            "attack_sequences": correlation.attack_sequence_patterns(state=args.state),
            "escalation": correlation.escalation_indicators(state=args.state),
        },
        "network": {
            "territories": network.territory_mapping(),
            "profiles": network.operational_pattern_profiling(),
            "lulls": network.identify_operational_lulls(args.state),
        },
    }

    # Generate SITREP
    generator = ReportGenerator(df)
    report["sitrep"] = generator.generate_situation_report(
        period_days=args.days, state=args.state
    )

    # Output
    output_json = json.dumps(report, indent=2, default=str)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        logger.info(f"Report saved to {args.output}")
    else:
        print(output_json)

    processor.close()


if __name__ == "__main__":
    main()
