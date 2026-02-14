"""
Data collection module for gathering security reports and incident data
from online sources (Nigerian news outlets, conflict databases, OSINT).
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from eagleeye.core.config import (
    BANDIT_KEYWORDS,
    NEWS_SOURCES,
    TARGET_STATES,
)

logger = logging.getLogger(__name__)


class NewsCollector:
    """Collects security-related news articles from Nigerian news sources."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EagleEye Security Research Bot/0.1",
        })
        self.collected_articles: list[dict] = []

    def search_source(self, source_name: str, base_url: str, query: str) -> list[dict]:
        """Search a news source for relevant articles."""
        articles = []
        search_url = f"{base_url}/search?q={query}"

        try:
            response = self.session.get(search_url, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract article links - generic approach for most Nigerian news sites
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                text = link.get_text(strip=True)

                if self._is_relevant_article(text, href):
                    full_url = urljoin(base_url, href)
                    articles.append({
                        "title": text,
                        "url": full_url,
                        "source": source_name,
                        "collected_at": datetime.utcnow().isoformat(),
                    })

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch from {source_name}: {e}")

        return articles

    def collect_from_all_sources(self, query: str = "bandits") -> list[dict]:
        """Collect articles from all configured news sources."""
        all_articles = []

        for source_name, base_url in NEWS_SOURCES.items():
            logger.info(f"Collecting from {source_name}...")
            articles = self.search_source(source_name, base_url, query)
            all_articles.extend(articles)
            logger.info(f"  Found {len(articles)} articles from {source_name}")

        self.collected_articles = all_articles
        return all_articles

    def collect_by_state(self, state: str) -> list[dict]:
        """Collect articles specific to a state."""
        queries = [
            f"bandits {state}",
            f"kidnapping {state}",
            f"security {state} attack",
        ]
        articles = []
        for query in queries:
            articles.extend(self.collect_from_all_sources(query))
        return self._deduplicate(articles)

    def fetch_article_content(self, url: str) -> Optional[dict]:
        """Fetch full content of a single article."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Extract main content
            title = ""
            title_tag = soup.find("h1")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Try common article body selectors
            content = ""
            for selector in ["article", ".post-content", ".entry-content",
                             ".article-content", ".story-body", "main"]:
                body = soup.select_one(selector)
                if body:
                    content = body.get_text(separator="\n", strip=True)
                    break

            if not content:
                # Fallback: get all paragraph text
                paragraphs = soup.find_all("p")
                content = "\n".join(p.get_text(strip=True) for p in paragraphs)

            # Extract date if available
            date_str = None
            time_tag = soup.find("time")
            if time_tag:
                date_str = time_tag.get("datetime", time_tag.get_text(strip=True))

            return {
                "title": title,
                "content": content,
                "url": url,
                "published_date": date_str,
                "fetched_at": datetime.utcnow().isoformat(),
            }

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch article {url}: {e}")
            return None

    def _is_relevant_article(self, text: str, href: str) -> bool:
        """Check if an article link is relevant to bandit activity."""
        combined = (text + " " + href).lower()

        has_bandit_keyword = any(kw in combined for kw in BANDIT_KEYWORDS)
        has_state_keyword = any(s.lower() in combined for s in TARGET_STATES)

        # Must have at least one bandit keyword to be relevant
        return has_bandit_keyword and (has_state_keyword or "north" in combined)

    @staticmethod
    def _deduplicate(articles: list[dict]) -> list[dict]:
        """Remove duplicate articles based on URL."""
        seen_urls = set()
        unique = []
        for article in articles:
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                unique.append(article)
        return unique


class ACLEDCollector:
    """
    Collector for ACLED (Armed Conflict Location & Event Data) format data.
    ACLED provides structured conflict event data for Nigeria.
    """

    API_BASE = "https://api.acleddata.com/acled/read"

    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        self.api_key = api_key
        self.email = email
        self.session = requests.Session()

    def fetch_events(
        self,
        country: str = "Nigeria",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        admin1: Optional[str] = None,
    ) -> list[dict]:
        """
        Fetch conflict events from ACLED API.

        Args:
            country: Country name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            admin1: State/province filter
        """
        if not self.api_key or not self.email:
            logger.warning("ACLED API key/email not configured. Skipping ACLED collection.")
            return []

        params = {
            "key": self.api_key,
            "email": self.email,
            "country": country,
            "limit": 0,  # No limit
        }

        if start_date:
            params["event_date"] = start_date
            params["event_date_where"] = ">="
        if end_date:
            params["event_date"] = end_date
            params["event_date_where"] = "<="
        if admin1:
            params["admin1"] = admin1

        try:
            response = self.session.get(self.API_BASE, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.RequestException as e:
            logger.error(f"Failed to fetch ACLED data: {e}")
            return []

    def fetch_northwest_events(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """Fetch all events for the Northwest Nigeria target states."""
        all_events = []
        for state in TARGET_STATES:
            events = self.fetch_events(
                country="Nigeria",
                start_date=start_date,
                end_date=end_date,
                admin1=state,
            )
            all_events.extend(events)
            logger.info(f"Fetched {len(events)} events for {state}")
        return all_events

    def convert_to_incident(self, acled_event: dict) -> dict:
        """Convert an ACLED event record to EagleEye incident format."""
        return {
            "date": acled_event.get("event_date"),
            "state": acled_event.get("admin1"),
            "lga": acled_event.get("admin2"),
            "location_name": acled_event.get("location"),
            "latitude": float(acled_event["latitude"]) if acled_event.get("latitude") else None,
            "longitude": float(acled_event["longitude"]) if acled_event.get("longitude") else None,
            "incident_type": self._map_event_type(acled_event.get("event_type", "")),
            "description": acled_event.get("notes", ""),
            "casualties_killed": int(acled_event.get("fatalities", 0)),
            "source": "ACLED",
            "source_url": acled_event.get("source_url", ""),
            "confidence_score": 0.85,
        }

    @staticmethod
    def _map_event_type(acled_type: str) -> str:
        """Map ACLED event types to EagleEye incident types."""
        mapping = {
            "Battles": "attack_on_security_forces",
            "Violence against civilians": "village_raid",
            "Explosions/Remote violence": "infrastructure_attack",
            "Strategic developments": "camp_sighting",
            "Riots": "reprisal_attack",
            "Protests": "displacement",
        }
        return mapping.get(acled_type, "village_raid")
