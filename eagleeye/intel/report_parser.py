"""
Intelligence report parser for extracting structured incident data
from unstructured security reports, news articles, and field reports.

Uses NLP techniques to identify key entities:
- Locations (states, LGAs, villages)
- Dates and times
- Incident types (raid, kidnapping, etc.)
- Casualty figures
- Group identifiers
- Weapons and vehicles
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from dateutil import parser as dateparser

from eagleeye.core.config import (
    ACTIVITY_KEYWORDS,
    BANDIT_KEYWORDS,
    INCIDENT_TYPES,
    LOCATION_KEYWORDS,
    TARGET_STATES,
)

logger = logging.getLogger(__name__)


class ReportParser:
    """
    Parses unstructured security reports into structured incident data.
    Designed for Nigerian security reporting conventions.
    """

    # Common LGAs in the target states
    KNOWN_LGAS = {
        "Zamfara": [
            "Anka", "Bakura", "Birnin Magaji", "Bukkuyum", "Bungudu",
            "Gummi", "Gusau", "Kaura Namoda", "Maradun", "Maru",
            "Shinkafi", "Talata Mafara", "Tsafe", "Zurmi",
        ],
        "Kaduna": [
            "Birnin Gwari", "Chikun", "Giwa", "Igabi", "Jaba",
            "Jema'a", "Kachia", "Kaduna North", "Kaduna South",
            "Kagarko", "Kajuru", "Kaura", "Kauru", "Kubau",
            "Kudan", "Lere", "Sanga", "Soba", "Zangon Kataf", "Zaria",
        ],
        "Katsina": [
            "Batsari", "Danmusa", "Dutsin-Ma", "Faskari", "Jibia",
            "Kankara", "Katsina", "Kurfi", "Malumfashi", "Mani",
            "Safana", "Sabuwa", "Dandume", "Funtua",
        ],
        "Sokoto": [
            "Bodinga", "Dange Shuni", "Gada", "Goronyo", "Gudu",
            "Gwadabawa", "Illela", "Isa", "Kebbe", "Rabah",
            "Sabon Birni", "Silame", "Sokoto North", "Sokoto South",
            "Tambuwal", "Tangaza", "Tureta", "Wamako", "Wurno",
        ],
        "Niger": [
            "Agaie", "Agwara", "Bida", "Borgu", "Bosso", "Chanchaga",
            "Edati", "Gbako", "Gurara", "Kontagora", "Lapai",
            "Lavun", "Magama", "Mariga", "Mashegu", "Mokwa",
            "Munya", "Paikoro", "Rafi", "Rijau", "Shiroro",
            "Suleja", "Tafa", "Wushishi",
        ],
        "Kebbi": [
            "Aleiro", "Arewa Dandi", "Argungu", "Augie", "Bagudo",
            "Birnin Kebbi", "Bunza", "Dandi", "Fakai", "Gwandu",
            "Jega", "Kalgo", "Koko/Besse", "Maiyama", "Ngaski",
            "Sakaba", "Shanga", "Suru", "Wasagu/Danko", "Yauri", "Zuru",
        ],
    }

    # Number word mappings
    NUMBER_WORDS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
        "ninety": 90, "hundred": 100, "hundreds": 200, "dozens": 24,
        "scores": 40, "several": 5, "many": 10, "few": 3,
        "numerous": 15, "over a hundred": 100, "about": 0,
    }

    # Build a regex alternation for number words (longest first to match "over a hundred" before "over")
    _NUM_WORD_PATTERN = "|".join(
        re.escape(w) for w in sorted(NUMBER_WORDS.keys(), key=len, reverse=True)
    )

    def parse_report(self, text: str, source: str = "",
                     published_date: Optional[datetime] = None) -> list[dict]:
        """
        Parse a security report and extract structured incident(s).

        Tries Claude AI first for higher-quality extraction, then falls back
        to regex-based parsing if Claude is unavailable or fails.
        """
        text = text.strip()
        if not text:
            return []

        # Try Claude AI first
        claude_result = self._try_claude_parse(text, source, published_date)
        if claude_result is not None:
            # Tag incidents as Claude-extracted
            for inc in claude_result:
                inc["_parser"] = "claude"
            return claude_result

        # Fall back to regex-based parsing
        incidents = self._regex_parse(text, source, published_date)
        for inc in incidents:
            inc["_parser"] = "regex"
        return incidents

    def _try_claude_parse(self, text: str, source: str,
                          published_date: Optional[datetime]) -> Optional[list[dict]]:
        """Try to parse using Claude AI. Returns None to fall back to regex."""
        try:
            from eagleeye.llm.claude_engine import engine
            result = engine.parse_report(text, source, published_date)
            if result is not None and len(result) > 0:
                return result
            return None
        except Exception as e:
            logger.debug(f"Claude parse unavailable: {e}")
            return None

    def _regex_parse(self, text: str, source: str = "",
                     published_date: Optional[datetime] = None) -> list[dict]:
        """Regex-based parsing fallback. Original parse logic."""
        segments = self._split_into_segments(text)

        incidents = []

        # First try the full text as one incident
        full_incident = self._extract_incident(text, source, published_date)
        if full_incident:
            incidents.append(full_incident)

        # Then try each segment independently
        for segment in segments:
            if len(segment) < 50:
                continue
            if not self._has_incident_indicators(segment):
                continue
            inc = self._extract_incident(segment, source, published_date)
            if inc and not self._is_duplicate(inc, incidents):
                incidents.append(inc)

        return incidents

    def _split_into_segments(self, text: str) -> list[str]:
        """
        Split text into candidate segments that might each contain an incident.
        Splits on paragraph breaks, single newlines, and sentence boundaries
        where a new location/incident context begins.
        """
        # First split on any newline boundaries (single or double)
        raw_blocks = re.split(r'\n\s*\n|\n', text)
        raw_blocks = [b.strip() for b in raw_blocks if b.strip()]

        segments = []
        for block in raw_blocks:
            # If block is short enough, keep as-is
            if len(block) < 300:
                segments.append(block)
                continue

            # Split long blocks on sentence boundaries that start a new incident
            # Look for sentences that begin with location/actor cues
            sentences = re.split(
                r'(?<=[.!?])\s+(?=[A-Z])',
                block
            )

            # Group consecutive sentences that belong to the same incident
            current_group = []
            for sent in sentences:
                if current_group and self._starts_new_incident(sent):
                    segments.append(" ".join(current_group))
                    current_group = [sent]
                else:
                    current_group.append(sent)
            if current_group:
                segments.append(" ".join(current_group))

        return segments

    def _starts_new_incident(self, sentence: str) -> bool:
        """Check if a sentence likely starts a new/separate incident."""
        lower = sentence.lower().strip()

        # Starts with a location cue
        location_starters = [
            r'^in\s+\w+',
            r'^(?:also\s+)?in\s+(?:a\s+)?(?:separate|related|another|similar)',
            r'^meanwhile',
            r'^separately',
            r'^in\s+(?:the\s+)?same\s+vein',
            r'^(?:armed\s+)?(?:bandits|gunmen|attackers|kidnappers|terrorists)',
            r'^(?:suspected|unknown)\s+(?:armed\s+)?(?:bandits|gunmen)',
            r'^(?:a\s+)?(?:group|gang)\s+of',
            r'^no fewer than',
            r'^at least',
            r'^about\s+\d+',
            r'^(?:on|last)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
        ]

        for pattern in location_starters:
            if re.search(pattern, lower):
                return True

        # Contains a new state mention that differs from surrounding context
        for state in TARGET_STATES:
            if self._state_in_text(state, lower):
                return True

        return False

    @staticmethod
    def _state_in_text(state: str, text_lower: str) -> bool:
        """Check if a state name appears in text with word-boundary awareness."""
        if state == "Niger":
            # Must match "Niger State" or standalone "Niger" not followed by
            # "ia" (Nigeria), "ian" (Nigerian), " River", " Delta", " Republic"
            if re.search(r'\bniger\s+state\b', text_lower):
                return True
            # Reject Niger + common non-state contexts
            if re.search(r'\bniger(?:\s*ia[n]?|\s+river|\s+delta|\s+republic)\b', text_lower) \
                    and not re.search(r'\bniger\s+state\b', text_lower):
                # Only return True if there's ALSO a standalone Niger State reference
                return False
            # Standalone "Niger" not in the above exclusions
            return bool(re.search(r'\bniger\b(?!\s*ia|\s+river|\s+delta|\s+republic)', text_lower))
        return state.lower() in text_lower

    def _extract_incident(self, text: str, source: str,
                           published_date: Optional[datetime]) -> Optional[dict]:
        """Extract a single incident from text."""
        if not self._has_incident_indicators(text):
            return None

        incident = {
            "source": source,
            "confidence_score": 0.5,
        }

        # Extract date
        date = self._extract_date(text, published_date)
        if date:
            incident["date"] = date.isoformat()
        elif published_date:
            incident["date"] = published_date.isoformat()
        else:
            incident["date"] = datetime.now().isoformat()

        # Extract location
        state, lga, location = self._extract_location(text)
        incident["state"] = state or "Unknown"
        incident["lga"] = lga
        incident["location_name"] = location

        # Extract incident type
        incident["incident_type"] = self._extract_incident_type(text)

        # Extract casualties
        killed, injured, kidnapped = self._extract_casualties(text)
        incident["casualties_killed"] = killed
        incident["casualties_injured"] = injured
        incident["kidnapped_count"] = kidnapped

        # Extract cattle/livestock stolen
        incident["cattle_stolen"] = self._extract_cattle_count(text)

        # Extract bandit count
        incident["estimated_bandits"] = self._extract_bandit_count(text)

        # Extract weapons
        incident["weapons_observed"] = self._extract_weapons(text)

        # Extract vehicles
        incident["vehicles_used"] = self._extract_vehicles(text)

        # Description is the cleaned text
        incident["description"] = text[:1000]

        # Calculate confidence based on how many fields were extracted
        filled = sum(1 for v in incident.values() if v and v != "Unknown")
        incident["confidence_score"] = min(0.95, filled / 15)

        # Calculate threat level
        incident["threat_level"] = self._calculate_threat_level(incident)

        # Only return if we have a minimum viable incident
        if incident["state"] == "Unknown" and not lga:
            return None

        return incident

    # Day name to weekday number (Monday=0, Sunday=6)
    _DAY_NAMES = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }

    def _extract_date(self, text: str, fallback: Optional[datetime]) -> Optional[datetime]:
        """Extract date from text. Uses dayfirst=True for Nigerian date format (DD/MM/YYYY)."""
        reference_date = fallback or datetime.now()

        # Common date patterns in Nigerian news
        patterns = [
            r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*\d{4})',
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4})',
            r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s*,?\s*\d{4})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return dateparser.parse(match.group(1), fuzzy=True, dayfirst=True)
                except (ValueError, OverflowError):
                    continue

        # Relative day references — resolve relative to report date, not now()
        relative_patterns = [
            (r'(?:on\s+)?last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', True),
            (r'(?:on\s+|early\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|night))?', False),
            (r'(yesterday)', None),
            (r'(today)', None),
        ]

        text_lower = text.lower()
        for pattern, is_last in relative_patterns:
            match = re.search(pattern, text_lower)
            if match:
                word = match.group(1).lower().strip()
                if word == "yesterday":
                    return reference_date - timedelta(days=1)
                elif word == "today":
                    return reference_date
                elif word in self._DAY_NAMES:
                    target_day = self._DAY_NAMES[word]
                    ref_day = reference_date.weekday()
                    days_back = (ref_day - target_day) % 7
                    if days_back == 0 and is_last:
                        days_back = 7
                    elif days_back == 0:
                        days_back = 0  # same day
                    return reference_date - timedelta(days=days_back)

        # Last week / last month
        if "last week" in text_lower:
            return reference_date - timedelta(days=7)
        if "last month" in text_lower:
            return reference_date - timedelta(days=30)

        return fallback

    def _extract_location(self, text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract state, LGA, and specific location from text."""
        text_lower = text.lower()
        state = None
        lga = None
        location = None

        # Find state — use word-boundary-aware matching
        for s in TARGET_STATES:
            if self._state_in_text(s, text_lower):
                state = s
                break

        # Also check for "X State" pattern
        if not state:
            for s in TARGET_STATES:
                if re.search(rf'\b{s.lower()}\s+state\b', text_lower):
                    state = s
                    break

        # Find LGA — search in matched state first, then all states
        if state and state in self.KNOWN_LGAS:
            for lga_name in self.KNOWN_LGAS[state]:
                if lga_name.lower() in text_lower:
                    lga = lga_name
                    break

        # If no state found, try to find LGA across all states to determine state
        if not state:
            for s, lgas in self.KNOWN_LGAS.items():
                for lga_name in lgas:
                    if lga_name.lower() in text_lower:
                        state = s
                        lga = lga_name
                        break
                if state:
                    break

        # If state found but no LGA yet, search all states' LGAs
        if state and not lga:
            for s, lgas in self.KNOWN_LGAS.items():
                for lga_name in lgas:
                    if lga_name.lower() in text_lower:
                        # Prefer LGA in the matched state, but accept others
                        if s == state:
                            lga = lga_name
                            break
                if lga:
                    break

        # Extract specific location/village name
        village_patterns = [
            r'(?:village|community|town)\s+(?:of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:village|community|town)',
            r'in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:area|district|ward)',
            r'attacked?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:invaded?|stormed?|raided?|sacked?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]

        for pattern in village_patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1)
                # Filter out common false positives
                skip_words = {
                    "The", "They", "This", "That", "These", "Those", "Monday",
                    "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                    "Sunday", "January", "February", "March", "April", "May",
                    "June", "July", "August", "September", "October",
                    "November", "December", "Armed", "Local", "State",
                    "Security", "According", "However", "Meanwhile",
                }
                if candidate not in skip_words:
                    location = candidate
                    break

        return state, lga, location

    def _extract_incident_type(self, text: str) -> str:
        """Determine the incident type from text content."""
        text_lower = text.lower()

        type_indicators = {
            "kidnapping": [
                "kidnap", "abduct", "taken hostage", "seized", "captive",
                "held for ransom", "whisked away", "taken away", "took away",
                "carried away",
            ],
            "village_raid": [
                "raid", "raided", "village", "community attacked",
                "invaded", "overran", "stormed", "sacked", "burnt",
                "burning", "set ablaze", "razed", "houses destroyed",
            ],
            "cattle_rustling": [
                "rustl", "cattle", "livestock stolen", "herds", "cows stolen",
                "animals stolen", "livestock raided",
            ],
            "highway_ambush": [
                "ambush", "highway", "motorist", "traveller", "commuter",
                "roadblock", "waylaid", "along the road", "road users",
                "intercepted", "on the road",
            ],
            "market_attack": ["market", "traders", "market day", "shopping"],
            "attack_on_security_forces": [
                "soldier", "military", "police", "security operatives",
                "troops", "army", "barracks", "military base", "police station",
                "security personnel", "officers", "checkpoint",
            ],
            "mining_site_attack": [
                "mining", "mine site", "gold mine", "miners", "mining site",
            ],
            "reprisal_attack": [
                "reprisal", "retaliat", "revenge", "aveng", "in response to",
            ],
            "arms_smuggling": [
                "arms", "ammunition", "weapons cache", "smuggl", "arms deal",
            ],
            "inter_gang_clash": [
                "rival", "gang clash", "infighting", "inter-group",
                "rival gang", "warring factions",
            ],
            "displacement": [
                "displac", "fled", "refuge", "IDPs", "internally displaced",
                "fled their homes", "took refuge", "seek shelter",
            ],
            "negotiation_ransom": [
                "ransom", "negotiat", "payment demanded", "release",
                "paid ransom", "demanded ransom",
            ],
            "infrastructure_attack": [
                "telecom", "bridge", "road block", "infrastructure",
                "power line", "mast", "tower destroyed",
            ],
            "camp_sighting": [
                "camp", "hideout", "base camp", "encampment", "den",
            ],
        }

        scores = {}
        for inc_type, keywords in type_indicators.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[inc_type] = score

        if scores:
            return max(scores, key=scores.get)

        # Fallback: check for general attack indicators
        if any(kw in text_lower for kw in ["attack", "kill", "shot", "dead"]):
            return "village_raid"
        return "village_raid"  # Default

    def _extract_casualties(self, text: str) -> tuple[int, int, int]:
        """Extract killed, injured, and kidnapped counts."""
        killed = 0
        injured = 0
        kidnapped = 0

        text_lower = text.lower()

        # Build number pattern: digits or number words
        num = rf'(\d+|{self._NUM_WORD_PATTERN})'

        # Killed patterns
        kill_patterns = [
            rf'{num}\s*(?:people|persons|villagers|residents|civilians|soldiers|policemen|farmers|victims|others)?\s*(?:were\s+)?(?:killed|murdered|slain|dead|shot dead|lost their lives|massacred|butchered)',
            rf'kill(?:ed|ing)\s*(?:about|over|at least|no fewer than|not less than|approximately|around)?\s*{num}',
            rf'(?:about|over|at least|no fewer than|not less than|approximately|around)\s*{num}\s*(?:people|persons|villagers|residents|civilians|victims)?\s*(?:were\s+)?(?:killed|dead|slain|murdered|massacred|shot dead|lost their lives)',
            rf'{num}\s*(?:people|persons)?\s*(?:were\s+)?massacred',
            rf'death toll[^.]*?{num}',
            rf'claimed\s+(?:the\s+)?(?:lives?\s+of\s+)?{num}',
            rf'left\s+{num}\s*(?:people|persons|villagers)?\s*dead',
        ]

        for pattern in kill_patterns:
            match = re.search(pattern, text_lower)
            if match:
                killed = max(killed, self._parse_number(match.group(1)))

        # Injured patterns
        injury_patterns = [
            rf'{num}\s*(?:people|persons|others|villagers|victims)?\s*(?:were\s+)?(?:injured|wounded|hurt|hospitalized|maimed)',
            rf'injur(?:ed|ing)\s*(?:about|over|at least|no fewer than)?\s*{num}',
            rf'(?:about|over|at least|no fewer than)\s*{num}\s*(?:people|persons|others)?\s*(?:were\s+)?(?:injured|wounded|hurt)',
            rf'left\s+{num}\s*(?:people|persons|others)?\s*(?:injured|wounded)',
        ]

        for pattern in injury_patterns:
            match = re.search(pattern, text_lower)
            if match:
                injured = max(injured, self._parse_number(match.group(1)))

        # Kidnapped patterns
        kidnap_patterns = [
            rf'{num}\s*(?:people|persons|women|children|students|villagers|travellers|passengers|victims|residents|others)?\s*(?:were\s+)?(?:kidnapped|abducted|taken|seized|captured|whisked away|carried away)',
            rf'kidnap(?:ped|ping)\s*(?:about|over|at least|no fewer than|not less than)?\s*{num}',
            rf'abduct(?:ed|ing)\s*(?:about|over|at least|no fewer than)?\s*{num}',
            rf'(?:about|over|at least|no fewer than|not less than)\s*{num}\s*(?:people|persons|women|children|students|villagers|travellers|residents)?\s*(?:were\s+)?(?:kidnapped|abducted|taken)',
            rf'took\s+(?:away\s+)?{num}\s*(?:people|persons|women|children|villagers|hostages)',
        ]

        for pattern in kidnap_patterns:
            match = re.search(pattern, text_lower)
            if match:
                kidnapped = max(kidnapped, self._parse_number(match.group(1)))

        return killed, injured, kidnapped

    def _extract_cattle_count(self, text: str) -> int:
        """Extract number of cattle/livestock stolen."""
        num = rf'(\d+|{self._NUM_WORD_PATTERN})'
        patterns = [
            rf'{num}\s*(?:cattle|cows|livestock|animals|herds|goats|sheep)\s*(?:were\s+)?(?:stolen|rustled|taken|carted|raided|driven away)',
            rf'rustl(?:ed|ing)\s*(?:about|over)?\s*{num}\s*(?:cattle|cows|livestock)',
            rf'(?:stole|carted away|made away with)\s*{num}\s*(?:cattle|cows|livestock|animals)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return self._parse_number(match.group(1))
        return 0

    def _extract_bandit_count(self, text: str) -> Optional[int]:
        """Extract estimated number of bandits involved."""
        num = rf'(\d+|{self._NUM_WORD_PATTERN})'
        patterns = [
            rf'(?:about|over|at least|no fewer than|approximately|around)?\s*{num}\s*(?:armed\s+)?(?:bandits|gunmen|armed men|attackers|terrorists|herdsmen|assailants|criminals|kidnappers)',
            rf'(?:bandits|gunmen|armed men|attackers)\s*(?:numbering|estimated at|numbering about|numbering over|suspected to be about)\s*(?:about|over)?\s*{num}',
            rf'(?:group|gang)\s+of\s+(?:about|over|approximately)?\s*{num}',
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                val = self._parse_number(match.group(1))
                if val > 0:
                    return val

        # Check for word-based counts
        for word, num_val in self.NUMBER_WORDS.items():
            pattern = rf'\b{re.escape(word)}\b\s+(?:armed\s+)?(?:bandits|gunmen|armed men|attackers)'
            if re.search(pattern, text.lower()) and num_val > 0:
                return num_val

        return None

    def _extract_weapons(self, text: str) -> Optional[str]:
        """Extract weapons mentioned in the report."""
        weapons = []
        weapon_terms = [
            "AK-47", "AK47", "rifles", "rifle", "machine gun",
            "RPG", "rocket", "IED", "explosive", "machete", "cutlass",
            "dane gun", "locally made gun", "automatic weapon",
            "sophisticated weapon", "heavy weapon", "grenade",
            "pump action", "double barrel", "pistol", "gun",
        ]
        text_lower = text.lower()
        for weapon in weapon_terms:
            if weapon.lower() in text_lower:
                weapons.append(weapon)
        return ", ".join(weapons) if weapons else None

    def _extract_vehicles(self, text: str) -> Optional[str]:
        """Extract vehicles used by bandits."""
        vehicles = []
        vehicle_terms = [
            "motorcycle", "motorbike", "okada", "vehicle",
            "truck", "hilux", "pickup", "horse", "on foot",
            "motorcycles", "horses",
        ]
        text_lower = text.lower()
        for vehicle in vehicle_terms:
            if vehicle.lower() in text_lower:
                vehicles.append(vehicle)
        return ", ".join(vehicles) if vehicles else None

    def _has_incident_indicators(self, text: str) -> bool:
        """
        Check if text contains indicators of a security incident.
        Relaxed: allows action + location even without explicit actor keywords.
        Also checks for casualty mentions as strong incident signals.
        """
        text_lower = text.lower()
        has_actor = any(kw in text_lower for kw in BANDIT_KEYWORDS)
        has_action = any(kw in text_lower for kw in ACTIVITY_KEYWORDS)
        has_location = any(self._state_in_text(s, text_lower) for s in TARGET_STATES)

        # Also check if any LGA is mentioned (implies location even without state name)
        has_lga = False
        if not has_location:
            for lgas in self.KNOWN_LGAS.values():
                if any(lga.lower() in text_lower for lga in lgas):
                    has_lga = True
                    break

        # Check for casualty numbers as a strong incident signal
        has_casualties = bool(re.search(
            r'\d+\s*(?:killed|dead|murdered|kidnapped|abducted|injured|wounded)',
            text_lower
        ))

        # Original: (actor OR action) AND location
        # Improved: multiple paths to qualify
        if (has_actor or has_action) and (has_location or has_lga):
            return True
        if has_casualties and (has_location or has_lga):
            return True
        if has_actor and has_action:
            # Actor + action is strong enough even without explicit state mention
            return True
        return False

    def _parse_number(self, text: str) -> int:
        """Parse a number from text, handling digits and number words."""
        if not text:
            return 0
        text = text.strip().lower()

        # Try digit parse first
        try:
            return int(text.replace(",", ""))
        except (ValueError, AttributeError):
            pass

        # Try number word lookup
        if text in self.NUMBER_WORDS:
            return self.NUMBER_WORDS[text]

        return 0

    @staticmethod
    def _is_duplicate(incident: dict, existing: list[dict]) -> bool:
        """
        Check if an incident is a duplicate of an existing one.
        Requires same state + type + LGA AND similar description content.
        """
        for existing_inc in existing:
            if (incident.get("state") == existing_inc.get("state")
                    and incident.get("incident_type") == existing_inc.get("incident_type")
                    and incident.get("lga") == existing_inc.get("lga")):
                # Check if descriptions share significant overlap
                desc1 = set(incident.get("description", "").lower().split())
                desc2 = set(existing_inc.get("description", "").lower().split())
                if not desc1 or not desc2:
                    return True
                overlap = len(desc1 & desc2) / max(len(desc1), len(desc2))
                if overlap > 0.6:
                    return True
        return False

    @staticmethod
    def _calculate_threat_level(incident: dict) -> int:
        """Calculate threat level for an extracted incident."""
        score = 0
        killed = incident.get("casualties_killed", 0)
        kidnapped = incident.get("kidnapped_count", 0)
        bandits = incident.get("estimated_bandits") or 0

        if killed >= 20:
            score += 3
        elif killed >= 5:
            score += 2
        elif killed >= 1:
            score += 1

        if kidnapped >= 50:
            score += 2
        elif kidnapped >= 10:
            score += 1

        if bandits >= 100:
            score += 2
        elif bandits >= 30:
            score += 1

        severity_types = {"attack_on_security_forces", "village_raid", "market_attack"}
        if incident.get("incident_type") in severity_types:
            score += 1

        return min(5, max(1, score))
