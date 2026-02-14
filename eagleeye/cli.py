"""
Command-line interface for EagleEye platform.
"""

import json
import logging
import sys

import click
from rich.console import Console
from rich.table import Table

from eagleeye.core.config import MODELS_DIR, TARGET_STATES
from eagleeye.core.database import init_db
from eagleeye.data.processor import DataProcessor

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@click.group()
def main():
    """EagleEye — AI Security Intelligence Platform for Northern Nigeria."""
    pass


@main.command()
def init():
    """Initialize the database and create tables."""
    init_db()
    console.print("[green]Database initialized successfully.[/green]")


@main.command()
@click.argument("file_path")
def ingest(file_path):
    """Ingest incident data from a JSON file."""
    processor = DataProcessor()
    try:
        incidents = processor.ingest_from_json_file(file_path)
        console.print(f"[green]Ingested {len(incidents)} incidents from {file_path}[/green]")
    finally:
        processor.close()


@main.command()
@click.option("--state", default=None, help="Filter by state")
@click.option("--days", default=30, help="Number of days to look back")
def threats(state, days):
    """Display current threat levels for all states."""
    from datetime import datetime, timedelta

    from eagleeye.models.threat_scorer import ThreatScorer

    processor = DataProcessor()
    try:
        df = processor.get_incidents_dataframe(state=state)
        if df.empty:
            console.print("[yellow]No incident data available.[/yellow]")
            return

        scorer = ThreatScorer(df)
        scores = scorer.score_all_states()

        table = Table(title="EagleEye — Threat Assessment")
        table.add_column("State", style="bold")
        table.add_column("Threat Level")
        table.add_column("Score")
        table.add_column("Last 30d")
        table.add_column("Last 90d")

        for s in scores:
            level = s["threat_level"]
            label = s["threat_label"]
            style = {
                "CRITICAL": "bold red",
                "HIGH": "red",
                "ELEVATED": "yellow",
                "MODERATE": "cyan",
                "LOW": "green",
            }.get(label, "white")

            table.add_row(
                s["state"],
                f"[{style}]{label}[/{style}]",
                str(s["overall_score"]),
                str(s.get("incidents_last_30d", "N/A")),
                str(s.get("incidents_last_90d", "N/A")),
            )

        console.print(table)
    finally:
        processor.close()


@main.command()
def train():
    """Train the prediction model on historical data."""
    from eagleeye.models.predictor import BanditActivityPredictor

    processor = DataProcessor()
    try:
        df = processor.get_incidents_dataframe()
        if df.empty:
            console.print("[yellow]No data available for training.[/yellow]")
            return

        pred = BanditActivityPredictor()
        result = pred.train(df)

        if result.get("status") == "success":
            console.print(f"[green]Model trained successfully![/green]")
            console.print(f"  Type model accuracy: {result['type_model_accuracy']:.1%}")
            console.print(f"  Location model accuracy: {result['location_model_accuracy']:.1%}")

            model_path = MODELS_DIR / "predictor_latest.pkl"
            pred.save_model(model_path)
            console.print(f"  Model saved to: {model_path}")
        else:
            console.print(f"[red]Training failed: {result.get('message', 'Unknown error')}[/red]")
    finally:
        processor.close()


@main.command()
@click.option("--state", default=None, help="Target state")
@click.option("--days", default=14, help="Prediction horizon in days")
def predict(state, days):
    """Generate predictions for future bandit activity."""
    from datetime import datetime, timedelta

    from eagleeye.models.predictor import BanditActivityPredictor

    model_path = MODELS_DIR / "predictor_latest.pkl"
    pred = BanditActivityPredictor()

    try:
        pred.load_model(model_path)
    except FileNotFoundError:
        console.print("[red]No trained model found. Run 'eagleeye train' first.[/red]")
        return

    processor = DataProcessor()
    try:
        df = processor.get_incidents_dataframe(state=state)
        now = datetime.now()
        recent_7d = df[df["date"] >= now - timedelta(days=7)] if not df.empty else df

        context = {
            "date": now,
            "state": state or "Zamfara",
            "recent_incidents_7d": len(recent_7d),
            "recent_incidents_30d": len(df),
            "recent_killed_7d": int(recent_7d["casualties_killed"].sum()) if not recent_7d.empty else 0,
        }

        predictions = pred.predict_multi_day(context, days)

        table = Table(title=f"EagleEye — {days}-Day Forecast ({state or 'All States'})")
        table.add_column("Date")
        table.add_column("Threat Level")
        table.add_column("Top Predicted Type")
        table.add_column("Probability")

        for p in predictions:
            level = p["threat_label"]
            style = {
                "CRITICAL": "bold red", "HIGH": "red",
                "ELEVATED": "yellow", "MODERATE": "cyan", "LOW": "green",
            }.get(level, "white")

            top_type = p["predicted_incident_types"][0] if p["predicted_incident_types"] else {}
            table.add_row(
                p["prediction_date"][:10],
                f"[{style}]{level}[/{style}]",
                top_type.get("type", "N/A"),
                f"{top_type.get('probability', 0):.1%}",
            )

        console.print(table)
    finally:
        processor.close()


@main.command()
@click.option("--state", default=None, help="Filter by state")
@click.option("--days", default=30, help="Reporting period")
def sitrep(state, days):
    """Generate a situation report."""
    from eagleeye.alerts.report_generator import ReportGenerator

    processor = DataProcessor()
    try:
        df = processor.get_incidents_dataframe()
        if df.empty:
            console.print("[yellow]No data available.[/yellow]")
            return

        generator = ReportGenerator(df)
        report = generator.generate_situation_report(period_days=days, state=state)
        console.print_json(json.dumps(report, indent=2, default=str))
    finally:
        processor.close()


@main.command()
@click.option("--port", default=8000, help="Port to listen on")
def serve(port):
    """Start the EagleEye web application and API server."""
    import uvicorn
    console.print(f"[green]Starting EagleEye server on http://0.0.0.0:{port}[/green]")
    console.print(f"  Dashboard:   http://localhost:{port}/")
    console.print(f"  API Docs:    http://localhost:{port}/docs")
    uvicorn.run("eagleeye.api.app:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()
