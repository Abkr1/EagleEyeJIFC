# EagleEye

AI-powered security intelligence platform for predicting and anticipating armed bandit activities in Northern Nigeria.

**Target Region:** Zamfara, Kaduna, Katsina, Sokoto, Niger, and Kebbi States.

## What It Does

EagleEye ingests historical and real-time security reports, extracts structured incident data using NLP, trains ML models on attack patterns, and generates predictions and alerts to help security forces anticipate bandit operations.

### Core Capabilities

- **Data Collection** — Automated collection from Nigerian news sources and ACLED conflict database
- **Report Parsing** — NLP-based extraction of incidents, locations, casualties, weapons, and group details from unstructured text
- **Temporal Analysis** — Seasonal patterns, attack frequency trends, surge detection, day-of-week analysis
- **Spatial Analysis** — Hotspot identification, movement corridors, geographic clustering, cross-border tracking
- **Correlation Analysis** — Attack sequence patterns, escalation indicators, retaliatory pattern detection, vulnerability windows
- **Network Analysis** — Territory mapping, operational profiling, supply route identification, activity timelines
- **ML Prediction** — Ensemble models predicting incident types, locations, and threat levels
- **Threat Scoring** — Composite threat scores for states and LGAs based on weighted factors
- **Anomaly Detection** — Flags unusual patterns that may indicate new tactics or imminent operations
- **Alert System** — Automated alerts for high-threat predictions, escalation, pattern breaks, and surges
- **Intelligence Reports** — Automated SITREP generation, threat briefings, and predictive outlooks

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Initialize and Load Data

```bash
# Initialize database and ingest sample historical data
python scripts/train_model.py
```

### Launch the Web App

```bash
PYTHONPATH=. python -m eagleeye.cli serve
```

Open your browser to **http://localhost:8000** to access the full dashboard. API docs are available at **http://localhost:8000/docs**.

You can also specify a custom port:

```bash
PYTHONPATH=. python -m eagleeye.cli serve --port 9000
```

### CLI Commands

```bash
# View current threat levels
PYTHONPATH=. python -m eagleeye.cli threats

# Train prediction model
PYTHONPATH=. python -m eagleeye.cli train

# Generate predictions
PYTHONPATH=. python -m eagleeye.cli predict --state Zamfara --days 14

# Generate situation report
PYTHONPATH=. python -m eagleeye.cli sitrep --days 30

# Run full analysis from the command line
python scripts/run_analysis.py --state Zamfara --output report.json
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/incidents` | POST | Record a new incident |
| `/api/incidents` | GET | List incidents (filter by state, type, days) |
| `/api/intel/report` | POST | Submit a raw intel report for parsing |
| `/api/intel/collect` | POST | Trigger news collection cycle |
| `/api/analysis/temporal` | GET | Temporal pattern analysis |
| `/api/analysis/spatial` | GET | Geographic pattern analysis |
| `/api/analysis/correlation` | GET | Correlation analysis |
| `/api/analysis/network` | GET | Network/group analysis |
| `/api/threat/scores` | GET | Threat scores for all states |
| `/api/threat/state/{state}` | GET | Detailed state threat assessment |
| `/api/predict/train` | POST | Train prediction model |
| `/api/predict/forecast` | POST | Generate activity forecast |
| `/api/anomaly/fit` | POST | Fit anomaly detector |
| `/api/anomaly/detect` | GET | Detect anomalies |
| `/api/alerts` | GET | Get active alerts |
| `/api/reports/sitrep` | GET | Generate situation report |
| `/api/reports/threat-briefing/{state}` | GET | State threat briefing |
| `/api/reports/predictive-outlook` | GET | Predictive outlook report |

## Web Interface

EagleEye ships with a full web dashboard built on a dark-themed intelligence UI. All pages are served directly by the FastAPI backend — no separate frontend build step is required.

### Pages

| Page | URL | Description |
|---|---|---|
| **Dashboard** | `/` | Main overview — live threat level cards for all 6 states, incident timeline chart (Chart.js), interactive map of recent incidents (Leaflet.js), summary statistics, and active alert count |
| **Incidents** | `/incidents` | Browse, filter, and add incidents. Three tabs: **List** (filterable by state, type, and time range), **Add Incident** (full-field form for manual entry), and **Map** (all incidents plotted on an interactive map) |
| **Analysis** | `/analysis` | Four analysis tabs, each running the corresponding backend engine: **Temporal** (monthly trends, seasonal patterns, surge detection), **Spatial** (hotspots, state comparison, geographic clusters), **Correlation** (attack sequences, escalation indicators, vulnerability windows), **Network** (territory maps, operational profiles, supply routes) |
| **Predictions** | `/predictions` | Train the ML model, then generate multi-day forecasts. Displays a threat level forecast chart, predicted incident type breakdown (doughnut chart), day-by-day forecast cards, and recommended actions |
| **Alerts** | `/alerts` | View and manage security alerts. Filter by state and minimum severity. Each alert shows type, severity bar, description, and an acknowledge button |
| **Intelligence** | `/intel` | Submit raw intelligence reports for automated NLP parsing — the system extracts location, incident type, casualties, weapons, and group estimates from unstructured text. Also provides a button to trigger automated news collection from configured sources |
| **Reports** | `/reports` | Generate three report types: **Situation Report (SITREP)** summarizing recent activity, **Threat Briefing** for a specific state, and **Predictive Outlook** combining model forecasts with analyst-ready formatting |

### UI Features

- **Dark theme** designed for operational/intelligence use with low-glare color scheme
- **Real-time clock** and system status indicator in the top bar
- **Interactive maps** powered by Leaflet.js with OpenStreetMap tiles centered on Northern Nigeria
- **Charts** powered by Chart.js for timelines, bar charts, doughnut breakdowns, and threat forecasts
- **Toast notifications** for success/error feedback on all user actions
- **Responsive layout** with collapsible sidebar navigation
- **Threat level color coding** — LOW (green), MODERATE (blue), ELEVATED (yellow), HIGH (orange), CRITICAL (red)
- **No build step** — all frontend libraries loaded via CDN (Chart.js, Leaflet.js), templates rendered server-side with Jinja2

## Example: Submit an Intel Report

```bash
curl -X POST http://localhost:8000/api/intel/report \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Armed bandits attacked Badarawa village in Anka LGA of Zamfara State on Monday. The gunmen numbering about 50 arrived on motorcycles and killed 12 people, kidnapped 25 residents and rustled 200 cattle.",
    "source": "Field Report"
  }'
```

The system will automatically extract: location (Zamfara, Anka), incident type (village_raid), casualties (12 killed, 25 kidnapped), weapons (motorcycles), and estimated group size (50).

## Project Structure

```
EagleEye/
├── eagleeye/
│   ├── core/           # Database models, configuration
│   ├── data/           # Collection, processing, schemas
│   ├── analysis/       # Temporal, spatial, correlation, network analysis
│   ├── models/         # ML predictor, threat scorer, anomaly detector
│   ├── intel/          # Report parser, news monitor
│   ├── alerts/         # Alert engine, report generator
│   ├── api/            # FastAPI application and REST endpoints
│   ├── web/            # Web UI layer
│   │   ├── routes.py       # HTML page routes
│   │   ├── templates/      # Jinja2 HTML templates (7 pages)
│   │   └── static/         # CSS stylesheet and shared JS utilities
│   └── cli.py          # Command-line interface
├── data/
│   ├── historical/     # Historical incident data (38 sample incidents)
│   ├── geographic/     # Geographic reference data (6 states, LGA coordinates)
│   └── models/         # Trained model storage
├── scripts/            # Utility scripts
└── tests/              # Test suite (59 tests)
```

## How the Prediction Works

1. **Feature Engineering** — Incidents are transformed into features: temporal (month, season, day), spatial (state, LGA), rolling windows (incidents last 7/14/30 days), severity metrics, and contextual factors
2. **Ensemble Models** — GradientBoosting for incident type prediction, RandomForest for location prediction
3. **Threat Scoring** — Weighted composite of frequency, severity, trend, geographic spread, tactic escalation, and recency
4. **Anomaly Detection** — Isolation Forest identifies deviations from baseline patterns
5. **Alert Generation** — Predictions exceeding confidence thresholds trigger alerts with actionable recommendations

## Data Sources

The platform is designed to ingest data from:

- **Nigerian news outlets** — Premium Times, Daily Trust, The Cable, Sahara Reporters, Punch, Vanguard, Channels TV
- **ACLED** — Armed Conflict Location & Event Data Project (requires API key)
- **Manual field reports** — Direct submission via API or CLI
- **Custom sources** — Extensible collector architecture

## Running Tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

## Configuration

Key settings are in `eagleeye/core/config.py`:

- `TARGET_STATES` — States under surveillance
- `INCIDENT_TYPES` — Recognized incident categories
- `ML_CONFIG` — Model hyperparameters and thresholds
- `NEWS_SOURCES` — Configured news source URLs
- `BANDIT_KEYWORDS` / `ACTIVITY_KEYWORDS` — NLP extraction keywords

Environment variables:
- `EAGLEEYE_DB_URL` — Database connection URL (default: SQLite)
