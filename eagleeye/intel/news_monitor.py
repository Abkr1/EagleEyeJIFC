"""
News monitoring module that continuously collects, parses, and processes
security news for the target region.
"""

import logging
from datetime import datetime
from typing import Optional

from eagleeye.data.collector import NewsCollector
from eagleeye.data.processor import DataProcessor
from eagleeye.data.schemas import IncidentCreate
from eagleeye.intel.report_parser import ReportParser

logger = logging.getLogger(__name__)


class NewsMonitor:
    """
    Monitors Nigerian news sources for security-related reporting,
    extracts structured incident data, and stores it in the database.
    """

    def __init__(self, db_url: Optional[str] = None):
        self.collector = NewsCollector()
        self.parser = ReportParser()
        self.processor = DataProcessor(db_url)
        self.stats = {
            "articles_collected": 0,
            "articles_parsed": 0,
            "incidents_extracted": 0,
            "incidents_stored": 0,
        }

    def run_collection_cycle(self, queries: Optional[list[str]] = None) -> dict:
        """
        Run a full collection cycle:
        1. Collect articles from news sources
        2. Fetch full article content
        3. Parse for incident data
        4. Store in database
        """
        if queries is None:
            queries = [
                "bandits attack",
                "kidnapping northern nigeria",
                "zamfara security",
                "kaduna bandits",
                "katsina attack",
                "niger state bandits",
            ]

        all_articles = []
        for query in queries:
            articles = self.collector.collect_from_all_sources(query)
            all_articles.extend(articles)

        all_articles = self.collector._deduplicate(all_articles)
        self.stats["articles_collected"] = len(all_articles)
        logger.info(f"Collected {len(all_articles)} unique articles")

        incidents_extracted = []
        for article in all_articles:
            content = self.collector.fetch_article_content(article["url"])
            if not content or not content.get("content"):
                continue

            self.stats["articles_parsed"] += 1

            # Parse the article content
            parsed = self.parser.parse_report(
                text=content["content"],
                source=article.get("source", ""),
                published_date=self._parse_date(content.get("published_date")),
            )

            for incident_data in parsed:
                incident_data["source_url"] = article["url"]
                incidents_extracted.append(incident_data)

        self.stats["incidents_extracted"] = len(incidents_extracted)
        logger.info(f"Extracted {len(incidents_extracted)} incidents from articles")

        # Store in database
        stored = self.processor.ingest_bulk_incidents(incidents_extracted)
        self.stats["incidents_stored"] = len(stored)

        return {
            "status": "completed",
            "stats": self.stats.copy(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def process_single_report(self, text: str, source: str = "",
                                source_url: str = "",
                                published_date: Optional[datetime] = None) -> list[dict]:
        """
        Process a single report submitted manually.

        Returns:
            List of extracted and stored incidents.
        """
        parsed = self.parser.parse_report(text=text, source=source,
                                          published_date=published_date)
        stored_incidents = []

        for incident_data in parsed:
            incident_data["source"] = source
            incident_data["source_url"] = source_url
            parser_used = incident_data.pop("_parser", "regex")
            try:
                if isinstance(incident_data.get("date"), str):
                    incident_data["date"] = datetime.fromisoformat(
                        incident_data["date"].replace("Z", "+00:00")
                    )
                validated = IncidentCreate(**{
                    k: v for k, v in incident_data.items()
                    if k in IncidentCreate.model_fields
                })
                stored = self.processor.ingest_incident(validated)
                stored_incidents.append({
                    "id": stored.id,
                    "state": stored.state,
                    "lga": stored.lga,
                    "location_name": stored.location_name,
                    "incident_type": stored.incident_type,
                    "date": stored.date.isoformat(),
                    "casualties_killed": stored.casualties_killed,
                    "casualties_injured": stored.casualties_injured,
                    "kidnapped_count": stored.kidnapped_count,
                    "confidence_score": stored.confidence_score,
                    "parser": parser_used,
                })
            except Exception as e:
                logger.error(f"Failed to store incident: {e}")
                try:
                    self.processor.session.rollback()
                except Exception:
                    pass

        return stored_incidents

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            from dateutil import parser as dateparser
            return dateparser.parse(date_str)
        except (ValueError, OverflowError):
            return None

    def close(self):
        self.processor.close()
