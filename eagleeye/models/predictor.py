"""
Main prediction model for anticipating bandit activity.

Uses ensemble ML methods trained on historical incident data to predict:
- Where the next attacks are likely to occur
- What type of attacks to expect
- When activity is likely to spike
- Which areas face the highest threat
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from eagleeye.core.config import INCIDENT_TYPES, ML_CONFIG, TARGET_STATES

logger = logging.getLogger(__name__)


class BanditActivityPredictor:
    """
    Predicts future bandit activity based on historical patterns.

    Features used for prediction:
    - Temporal: month, day_of_week, season, days_since_last_incident
    - Spatial: state, LGA, regional incident density
    - Contextual: recent incident counts, recent casualty trends,
      active incident types, escalation indicators
    """

    # Top incident types for state-level prior features
    TOP_TYPES_FOR_FEATURES = [
        "village_raid", "kidnapping", "attack_on_security_forces",
        "cattle_rustling", "highway_ambush", "camp_sighting",
    ]

    def __init__(self):
        self.location_model: Optional[RandomForestClassifier] = None
        self.type_model: Optional[RandomForestClassifier] = None
        self.state_encoder = LabelEncoder()
        self.lga_encoder = LabelEncoder()
        self.type_encoder = LabelEncoder()
        self.state_type_priors: dict = {}  # {state: {type: proportion}}
        self.lga_type_priors: dict = {}   # {lga: {type: proportion}}
        self.is_trained = False
        self.model_version = "0.2.0"
        self.training_stats: dict = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw incident data for ML training.
        """
        features = pd.DataFrame()

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Temporal features
        features["month"] = df["date"].dt.month
        features["day_of_week"] = df["date"].dt.dayofweek
        features["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        features["is_dry_season"] = features["month"].apply(
            lambda m: 1 if m in [10, 11, 12, 1, 2, 3, 4, 5] else 0
        )
        features["is_harvest_period"] = features["month"].apply(
            lambda m: 1 if m in [10, 11, 12] else 0
        )
        features["is_ramadan_approx"] = features["month"].apply(
            lambda m: 1 if m in [3, 4] else 0  # Approximate, varies yearly
        )

        # Location features
        features["state_encoded"] = self.state_encoder.fit_transform(
            df["state"].fillna("Unknown")
        )
        features["lga_encoded"] = self.lga_encoder.fit_transform(
            df["lga"].fillna("Unknown")
        )

        # Rolling window features (lookback)
        for window in [7, 14, 30]:
            col_name = f"incidents_last_{window}d"
            counts = []
            for idx, row in df.iterrows():
                cutoff = row["date"] - timedelta(days=window)
                count = len(df[(df["date"] >= cutoff) & (df["date"] < row["date"])])
                counts.append(count)
            features[col_name] = counts

        # Casualties in recent window
        for window in [7, 30]:
            col_name = f"killed_last_{window}d"
            killed = []
            for idx, row in df.iterrows():
                cutoff = row["date"] - timedelta(days=window)
                total = df[
                    (df["date"] >= cutoff) & (df["date"] < row["date"])
                ]["casualties_killed"].sum()
                killed.append(total)
            features[col_name] = killed

        # Days since last incident in same state
        days_since_last = []
        for idx, row in df.iterrows():
            prev_in_state = df[
                (df["state"] == row["state"]) & (df["date"] < row["date"])
            ]
            if len(prev_in_state) > 0:
                gap = (row["date"] - prev_in_state["date"].max()).days
            else:
                gap = 365  # Default large value
            days_since_last.append(gap)
        features["days_since_last_in_state"] = days_since_last

        # Incident severity features
        features["casualties_killed"] = df["casualties_killed"].fillna(0)
        features["kidnapped_count"] = df["kidnapped_count"].fillna(0)
        features["threat_level"] = df["threat_level"].fillna(2)

        # State-type prior features: what fraction of incidents in each state
        # are of each major type. Captures structural tendencies per state.
        state_type_counts = df.groupby(["state", "incident_type"]).size().unstack(fill_value=0)
        state_totals = state_type_counts.sum(axis=1)
        state_type_pcts = state_type_counts.div(state_totals, axis=0)

        # Store priors for use at prediction time
        self.state_type_priors = {
            state: {
                t: round(float(state_type_pcts.at[state, t]), 4)
                if t in state_type_pcts.columns and state in state_type_pcts.index
                else 0.0
                for t in self.TOP_TYPES_FOR_FEATURES
            }
            for state in df["state"].unique()
        }

        for t in self.TOP_TYPES_FOR_FEATURES:
            col = f"state_{t}_pct"
            if t in state_type_pcts.columns:
                features[col] = df["state"].map(state_type_pcts[t]).fillna(0)
            else:
                features[col] = 0.0

        # State-specific incident counts in last 30 days
        state_30d_counts = []
        for idx, row in df.iterrows():
            cutoff = row["date"] - timedelta(days=30)
            count = len(df[
                (df["state"] == row["state"]) &
                (df["date"] >= cutoff) &
                (df["date"] < row["date"])
            ])
            state_30d_counts.append(count)
        features["incidents_in_state_last_30d"] = state_30d_counts

        # LGA-type prior features: what fraction of incidents in each LGA
        # are of each major type. More granular than state-level priors.
        lga_type_counts = df.groupby(["lga", "incident_type"]).size().unstack(fill_value=0)
        lga_totals = lga_type_counts.sum(axis=1)
        lga_type_pcts = lga_type_counts.div(lga_totals, axis=0)

        self.lga_type_priors = {
            lga: {
                t: round(float(lga_type_pcts.at[lga, t]), 4)
                if t in lga_type_pcts.columns and lga in lga_type_pcts.index
                else 0.0
                for t in self.TOP_TYPES_FOR_FEATURES
            }
            for lga in df["lga"].fillna("Unknown").unique()
        }

        lga_col = df["lga"].fillna("Unknown")
        for t in self.TOP_TYPES_FOR_FEATURES:
            col = f"lga_{t}_pct"
            if t in lga_type_pcts.columns:
                features[col] = lga_col.map(lga_type_pcts[t]).fillna(0)
            else:
                features[col] = 0.0

        # Target variables
        features["incident_type"] = df["incident_type"]
        features["state"] = df["state"]

        return features

    def train(self, incidents_df: pd.DataFrame) -> dict:
        """
        Train the prediction models on historical incident data.

        Returns:
            Dict with training metrics and model performance stats.
        """
        if len(incidents_df) < ML_CONFIG["min_samples_per_region"]:
            return {
                "status": "error",
                "message": f"Need at least {ML_CONFIG['min_samples_per_region']} incidents to train",
                "available": len(incidents_df),
            }

        logger.info(f"Training on {len(incidents_df)} incidents...")

        features = self.prepare_features(incidents_df)
        feature_cols = [
            "month", "day_of_week", "week_of_year", "is_dry_season",
            "is_harvest_period", "is_ramadan_approx", "state_encoded",
            "lga_encoded", "incidents_last_7d", "incidents_last_14d",
            "incidents_last_30d", "killed_last_7d", "killed_last_30d",
            "days_since_last_in_state", "casualties_killed",
            "kidnapped_count", "threat_level",
            "incidents_in_state_last_30d",
        ] + [f"state_{t}_pct" for t in self.TOP_TYPES_FOR_FEATURES] \
          + [f"lga_{t}_pct" for t in self.TOP_TYPES_FOR_FEATURES]

        X = features[feature_cols]
        y_type = self.type_encoder.fit_transform(features["incident_type"])
        y_state = features["state_encoded"].values

        # Stratified train/test split — ensures minority classes in both sets
        # Split indices once so both models use the same train/test partition
        indices = np.arange(len(X))
        try:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=ML_CONFIG["test_size"],
                random_state=ML_CONFIG["random_state"],
                stratify=y_type,
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                indices,
                test_size=ML_CONFIG["test_size"],
                random_state=ML_CONFIG["random_state"],
            )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_type_train, y_type_test = y_type[train_idx], y_type[test_idx]
        y_state_train, y_state_test = y_state[train_idx], y_state[test_idx]

        # Model 1: Predict incident TYPE
        self.type_model = RandomForestClassifier(
            n_estimators=ML_CONFIG["n_estimators"],
            max_depth=None,
            min_samples_leaf=2,
            random_state=ML_CONFIG["random_state"],
            n_jobs=-1,
        )
        self.type_model.fit(X_train, y_type_train)
        type_score = self.type_model.score(X_test, y_type_test)

        # Per-class metrics for type model
        y_type_pred = self.type_model.predict(X_test)
        type_class_report = classification_report(
            y_type_test, y_type_pred,
            labels=range(len(self.type_encoder.classes_)),
            target_names=self.type_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        per_class_f1 = {
            cls: round(type_class_report[cls]["f1-score"], 3)
            for cls in self.type_encoder.classes_
            if cls in type_class_report
        }

        # Model 2: Predict incident LOCATION (state level)
        # No class weighting needed — location accuracy is already high
        self.location_model = RandomForestClassifier(
            n_estimators=ML_CONFIG["n_estimators"],
            max_depth=None,
            min_samples_leaf=2,
            random_state=ML_CONFIG["random_state"],
            n_jobs=-1,
        )
        self.location_model.fit(X_train, y_state_train)
        location_score = self.location_model.score(X_test, y_state_test)

        # Cross-validation scores
        cv_folds = min(5, len(X) // 5 + 1)
        type_cv = cross_val_score(self.type_model, X, y_type, cv=cv_folds)
        location_cv = cross_val_score(self.location_model, X, y_state, cv=cv_folds)

        # Feature importance
        type_importance = dict(zip(
            feature_cols,
            [round(float(x), 4) for x in self.type_model.feature_importances_],
        ))
        location_importance = dict(zip(
            feature_cols,
            [round(float(x), 4) for x in self.location_model.feature_importances_],
        ))

        # Class distribution info
        unique_types, type_counts = np.unique(y_type, return_counts=True)
        class_distribution = {
            self.type_encoder.classes_[t]: int(c)
            for t, c in zip(unique_types, type_counts)
        }

        self.is_trained = True
        self.training_stats = {
            "status": "success",
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "type_model_accuracy": round(type_score, 3),
            "type_model_cv_mean": round(float(type_cv.mean()), 3),
            "type_model_cv_std": round(float(type_cv.std()), 3),
            "type_model_per_class_f1": per_class_f1,
            "type_model_macro_f1": round(type_class_report["macro avg"]["f1-score"], 3),
            "type_model_weighted_f1": round(type_class_report["weighted avg"]["f1-score"], 3),
            "location_model_accuracy": round(location_score, 3),
            "location_model_cv_mean": round(float(location_cv.mean()), 3),
            "class_distribution": class_distribution,
            "feature_importance_type": dict(sorted(
                type_importance.items(), key=lambda x: x[1], reverse=True
            )),
            "feature_importance_location": dict(sorted(
                location_importance.items(), key=lambda x: x[1], reverse=True
            )),
            "trained_at": datetime.utcnow().isoformat(),
            "model_version": self.model_version,
        }

        logger.info(
            f"Training complete. Type accuracy: {type_score:.3f} "
            f"(macro F1: {type_class_report['macro avg']['f1-score']:.3f}), "
            f"Location accuracy: {location_score:.3f}"
        )
        return self.training_stats

    def predict(self, current_context: dict) -> dict:
        """
        Generate predictions based on current context.

        Args:
            current_context: Dict with keys like:
                - date: prediction target date
                - state: target state (optional)
                - recent_incidents_7d: count of recent incidents
                - recent_incidents_30d: count of recent incidents
                - recent_killed_7d: recent casualties
                - recent_killed_30d: recent casualties
                - last_incident_days_ago: days since last incident
        """
        if not self.is_trained:
            return {"error": "Model not trained. Call train() first."}

        date = pd.Timestamp(current_context.get("date", datetime.now()))

        state = current_context.get("state", "Zamfara")
        state_priors = self.state_type_priors.get(state, {})

        fv = {
            "month": date.month,
            "day_of_week": date.dayofweek,
            "week_of_year": date.isocalendar().week,
            "is_dry_season": 1 if date.month in [10, 11, 12, 1, 2, 3, 4, 5] else 0,
            "is_harvest_period": 1 if date.month in [10, 11, 12] else 0,
            "is_ramadan_approx": 1 if date.month in [3, 4] else 0,
            "state_encoded": self._safe_encode_state(state),
            "lga_encoded": 0,
            "incidents_last_7d": current_context.get("recent_incidents_7d", 0),
            "incidents_last_14d": current_context.get("recent_incidents_14d", 0),
            "incidents_last_30d": current_context.get("recent_incidents_30d", 0),
            "killed_last_7d": current_context.get("recent_killed_7d", 0),
            "killed_last_30d": current_context.get("recent_killed_30d", 0),
            "days_since_last_in_state": current_context.get("last_incident_days_ago", 7),
            "casualties_killed": 0,
            "kidnapped_count": 0,
            "threat_level": current_context.get("current_threat_level", 3),
            "incidents_in_state_last_30d": current_context.get("recent_incidents_in_state_30d", 0),
        }
        lga = current_context.get("lga", "Unknown")
        lga_priors = self.lga_type_priors.get(lga, {})

        for t in self.TOP_TYPES_FOR_FEATURES:
            fv[f"state_{t}_pct"] = state_priors.get(t, 0.0)
            fv[f"lga_{t}_pct"] = lga_priors.get(t, 0.0)

        feature_vector = pd.DataFrame([fv])

        # Predict incident type
        type_probs = self.type_model.predict_proba(feature_vector)[0]
        type_classes = self.type_encoder.classes_
        type_predictions = sorted(
            zip(type_classes, type_probs),
            key=lambda x: x[1],
            reverse=True,
        )

        # Predict location (state)
        location_probs = self.location_model.predict_proba(feature_vector)[0]
        location_classes = self.state_encoder.classes_
        location_predictions = sorted(
            zip(location_classes, location_probs),
            key=lambda x: x[1],
            reverse=True,
        )

        # Generate threat level based on predictions
        top_type_prob = type_predictions[0][1]
        threat_level = self._calculate_threat_level(
            top_type_prob,
            current_context.get("recent_incidents_7d", 0),
            current_context.get("recent_killed_7d", 0),
        )

        return {
            "prediction_date": date.isoformat(),
            "predicted_incident_types": [
                {"type": t, "probability": round(float(p), 3)}
                for t, p in type_predictions[:5]
            ],
            "predicted_locations": [
                {"state": s, "probability": round(float(p), 3)}
                for s, p in location_predictions
            ],
            "threat_level": threat_level,
            "threat_label": self._threat_label(threat_level),
            "confidence": round(float(top_type_prob), 3),
            "model_version": self.model_version,
            "recommendations": self._generate_recommendations(
                type_predictions[0][0], threat_level, current_context
            ),
        }

    def predict_multi_day(self, current_context: dict,
                           horizon_days: int = 14) -> list[dict]:
        """Generate predictions for multiple days ahead."""
        predictions = []
        base_date = pd.Timestamp(current_context.get("date", datetime.now()))

        for day_offset in range(1, horizon_days + 1):
            ctx = current_context.copy()
            ctx["date"] = base_date + timedelta(days=day_offset)
            pred = self.predict(ctx)
            pred["day_offset"] = day_offset
            predictions.append(pred)

        return predictions

    def save_model(self, path: str | Path) -> None:
        """Save trained model to disk."""
        path = Path(path)
        model_data = {
            "type_model": self.type_model,
            "location_model": self.location_model,
            "state_encoder": self.state_encoder,
            "lga_encoder": self.lga_encoder,
            "type_encoder": self.type_encoder,
            "state_type_priors": self.state_type_priors,
            "lga_type_priors": self.lga_type_priors,
            "training_stats": self.training_stats,
            "model_version": self.model_version,
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str | Path) -> None:
        """Load a previously trained model from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        self.type_model = model_data["type_model"]
        self.location_model = model_data["location_model"]
        self.state_encoder = model_data["state_encoder"]
        self.lga_encoder = model_data["lga_encoder"]
        self.type_encoder = model_data["type_encoder"]
        self.state_type_priors = model_data.get("state_type_priors", {})
        self.lga_type_priors = model_data.get("lga_type_priors", {})
        self.training_stats = model_data["training_stats"]
        self.model_version = model_data["model_version"]
        self.is_trained = True
        logger.info(f"Model loaded from {path} (version {self.model_version})")

    def _safe_encode_state(self, state: str) -> int:
        """Encode a state name, handling unseen values."""
        try:
            return int(self.state_encoder.transform([state])[0])
        except ValueError:
            return 0

    @staticmethod
    def _calculate_threat_level(top_probability: float,
                                 recent_incidents: int,
                                 recent_killed: int) -> int:
        """Calculate composite threat level (1-5)."""
        score = top_probability * 2

        if recent_incidents > 5:
            score += 1.5
        elif recent_incidents > 2:
            score += 0.8

        if recent_killed > 10:
            score += 1.5
        elif recent_killed > 3:
            score += 0.8

        # Map to 1-5 scale
        if score >= 4:
            return 5
        elif score >= 3:
            return 4
        elif score >= 2:
            return 3
        elif score >= 1:
            return 2
        return 1

    @staticmethod
    def _threat_label(level: int) -> str:
        labels = {1: "LOW", 2: "MODERATE", 3: "ELEVATED", 4: "HIGH", 5: "CRITICAL"}
        return labels.get(level, "UNKNOWN")

    @staticmethod
    def _generate_recommendations(predicted_type: str, threat_level: int,
                                   context: dict) -> list[str]:
        """Generate actionable security recommendations."""
        recommendations = []

        if threat_level >= 4:
            recommendations.append("Increase patrol frequency in identified hotspot areas")
            recommendations.append("Alert rapid response units and place on standby")

        if predicted_type == "village_raid":
            recommendations.append("Reinforce community early warning systems in vulnerable villages")
            recommendations.append("Pre-position security assets along known approach routes")
        elif predicted_type == "kidnapping":
            recommendations.append("Increase checkpoints on highways and known kidnapping corridors")
            recommendations.append("Issue travel advisories for high-risk routes")
        elif predicted_type == "highway_ambush":
            recommendations.append("Deploy mobile patrols on major highways")
            recommendations.append("Coordinate with transport unions for convoy arrangements")
        elif predicted_type == "cattle_rustling":
            recommendations.append("Increase surveillance of grazing corridors and cattle routes")
            recommendations.append("Coordinate with herder communities for early warning")
        elif predicted_type == "attack_on_security_forces":
            recommendations.append("Review base security and patrol protocols")
            recommendations.append("Increase intelligence gathering on potential targets")
        elif predicted_type == "market_attack":
            recommendations.append("Increase security presence at major markets on market days")
            recommendations.append("Deploy plainclothes surveillance at market access points")

        if context.get("recent_incidents_7d", 0) > 3:
            recommendations.append("Current surge pattern detected — consider area-wide security operation")

        return recommendations
