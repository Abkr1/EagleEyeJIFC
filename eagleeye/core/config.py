"""
Central configuration for EagleEye platform.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env file if present (must happen before reading env vars)
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass
DATA_DIR = BASE_DIR / "data"
HISTORICAL_DIR = DATA_DIR / "historical"
GEOGRAPHIC_DIR = DATA_DIR / "geographic"
MODELS_DIR = DATA_DIR / "models"

# Database
DATABASE_URL = os.getenv("EAGLEEYE_DB_URL", f"sqlite:///{BASE_DIR / 'eagleeye.db'}")

# Target states in Northern Nigeria
TARGET_STATES = [
    "Zamfara",
    "Kaduna",
    "Katsina",
    "Sokoto",
    "Niger",
    "Kebbi",
]

# Incident categories observed in the region
INCIDENT_TYPES = [
    "village_raid",
    "kidnapping",
    "cattle_rustling",
    "highway_ambush",
    "market_attack",
    "attack_on_security_forces",
    "mining_site_attack",
    "reprisal_attack",
    "arms_smuggling",
    "camp_sighting",
    "displacement",
    "negotiation_ransom",
    "inter_gang_clash",
    "infrastructure_attack",
]

# Threat levels
THREAT_LEVELS = {
    "CRITICAL": 5,
    "HIGH": 4,
    "ELEVATED": 3,
    "MODERATE": 2,
    "LOW": 1,
}

# Temporal analysis windows
ANALYSIS_WINDOWS = {
    "short_term": 7,      # days
    "medium_term": 30,     # days
    "long_term": 90,       # days
    "seasonal": 365,       # days
}

# Feature extraction settings for ML models
ML_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "min_samples_per_region": 10,
    "prediction_horizon_days": 14,
    "confidence_threshold": 0.6,
}

# Keywords for NLP extraction from security reports
BANDIT_KEYWORDS = [
    "bandits", "armed men", "gunmen", "terrorists", "kidnappers",
    "cattle rustlers", "rustlers", "armed gang", "criminal elements",
    "armed herdsmen", "militia", "insurgents", "assailants",
    "suspected gunmen", "unknown gunmen", "hoodlums", "marauders",
    "outlaws", "gang members", "criminal gang", "armed robbers",
    "armed attackers", "attackers",
]

ACTIVITY_KEYWORDS = [
    "attack", "raid", "kidnap", "abduct", "ambush", "kill",
    "loot", "burn", "displace", "ransom", "rustl", "shoot",
    "invade", "overrun", "surround", "block", "seize",
    "murder", "sack", "storm", "slaughter", "massacre", "destroy",
    "raze", "abduct", "intercept", "waylay", "bomb", "detonate",
    "maim", "wound", "injure", "rob", "plunder", "pillage",
    "terrorize", "terrorise", "slay", "butcher",
]

LOCATION_KEYWORDS = [
    "village", "community", "town", "local government", "LGA",
    "highway", "road", "market", "school", "mosque", "church",
    "forest", "bush", "camp", "mining site",
]

# News and data source configuration
NEWS_SOURCES = {
    "premium_times": "https://www.premiumtimesng.com",
    "daily_trust": "https://dailytrust.com",
    "the_cable": "https://www.thecable.ng",
    "sahara_reporters": "https://saharareporters.com",
    "punch": "https://punchng.com",
    "vanguard": "https://www.vanguardngr.com",
    "channels_tv": "https://www.channelstv.com",
}

# Claude AI configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
CLAUDE_MAX_TOKENS = 4096
CLAUDE_TEMPERATURE = 0.3

# Ollama local AI configuration (primary)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-v3.1:671b-cloud")
OLLAMA_MAX_TOKENS = 4096
OLLAMA_TEMPERATURE = 0.3

# HuggingFace fallback AI configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-32B-Instruct")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MAX_TOKENS = 4096
HF_TEMPERATURE = 0.3

# ACLED-style conflict data fields
ACLED_EVENT_TYPES = [
    "Battles",
    "Violence against civilians",
    "Explosions/Remote violence",
    "Strategic developments",
]
