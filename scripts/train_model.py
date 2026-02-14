#!/usr/bin/env python3
"""
Script to train the EagleEye prediction model on historical data.

Usage:
    python scripts/train_model.py [--data-file PATH]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eagleeye.core.config import HISTORICAL_DIR, MODELS_DIR
from eagleeye.core.database import init_db
from eagleeye.data.processor import DataProcessor
from eagleeye.models.anomaly_detector import AnomalyDetector
from eagleeye.models.predictor import BanditActivityPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train EagleEye prediction models")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to JSON incident data file to ingest before training")
    args = parser.parse_args()

    # Initialize database
    logger.info("Initializing database...")
    init_db()

    processor = DataProcessor()

    # Ingest data file if provided
    if args.data_file:
        data_path = Path(args.data_file)
    else:
        data_path = HISTORICAL_DIR / "sample_incidents.json"

    if data_path.exists():
        logger.info(f"Ingesting data from {data_path}...")
        incidents = processor.ingest_from_json_file(data_path)
        logger.info(f"Ingested {len(incidents)} incidents")
    else:
        logger.warning(f"Data file not found: {data_path}")

    # Get all incidents for training
    df = processor.get_incidents_dataframe()
    if df.empty:
        logger.error("No incident data available for training.")
        sys.exit(1)

    logger.info(f"Training on {len(df)} total incidents...")

    # Train prediction model
    logger.info("Training BanditActivityPredictor...")
    predictor = BanditActivityPredictor()
    result = predictor.train(df)

    if result.get("status") == "success":
        logger.info(f"Prediction model trained successfully!")
        logger.info(f"  Type accuracy: {result['type_model_accuracy']:.3f}")
        logger.info(f"  Location accuracy: {result['location_model_accuracy']:.3f}")
        logger.info(f"  Top features (type): {list(result['feature_importance_type'].keys())[:5]}")

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "predictor_latest.pkl"
        predictor.save_model(model_path)
        logger.info(f"  Model saved: {model_path}")
    else:
        logger.error(f"Training failed: {result}")

    # Train anomaly detector
    logger.info("Training AnomalyDetector...")
    detector = AnomalyDetector()
    ad_result = detector.fit(df)
    logger.info(f"Anomaly detector: {ad_result.get('status', 'unknown')}")

    processor.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
