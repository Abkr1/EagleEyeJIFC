"""
Central AI engine for EagleEye platform.
Provides structured intelligence capabilities across all features.

Fallback chain is configurable via Settings UI.
Default priority: Ollama (local) -> Claude (cloud) -> HuggingFace (cloud)
"""

import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

VALID_ENGINES = ("ollama", "claude", "huggingface")
DEFAULT_ENGINE_ORDER = ["ollama", "claude", "huggingface"]


def _get_ollama():
    from eagleeye.llm.ollama_engine import ollama_engine
    return ollama_engine


def _get_hf():
    from eagleeye.llm.huggingface_engine import hf_engine
    return hf_engine


class _ClaudeBackend:
    """Internal Claude API backend."""

    _client = None
    _available = None

    @property
    def is_available(self) -> bool:
        if _ClaudeBackend._available is not None:
            return _ClaudeBackend._available
        try:
            from eagleeye.core.config import ANTHROPIC_API_KEY
            if not ANTHROPIC_API_KEY:
                _ClaudeBackend._available = False
                return False
            import anthropic  # noqa: F401
            _ClaudeBackend._available = True
            return True
        except ImportError:
            _ClaudeBackend._available = False
            return False

    def _get_client(self):
        if _ClaudeBackend._client is None:
            from eagleeye.core.config import ANTHROPIC_API_KEY
            import anthropic
            _ClaudeBackend._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return _ClaudeBackend._client

    def parse_report(self, text, source="", published_date=None):
        from eagleeye.core.config import CLAUDE_MODEL, CLAUDE_MAX_TOKENS
        from eagleeye.llm.prompts import REPORT_PARSING_SYSTEM, REPORT_PARSING_TOOL

        client = self._get_client()
        date_context = ""
        if published_date:
            date_context = f"\n\nThis report was published on {published_date.strftime('%A, %d %B %Y')}. Use this date to resolve relative references like 'last Monday', 'yesterday', etc."

        user_msg = f"Extract all security incidents from this report.{date_context}\n\nSource: {source}\n\nReport text:\n{text[:8000]}"

        response = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS, temperature=0.1,
            system=REPORT_PARSING_SYSTEM, tools=[REPORT_PARSING_TOOL],
            tool_choice={"type": "tool", "name": "extract_incidents"},
            messages=[{"role": "user", "content": user_msg}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_incidents":
                incidents = block.input.get("incidents", [])
                logger.info(f"Claude extracted {len(incidents)} incident(s)")
                for inc in incidents:
                    inc["source"] = source
                    inc.setdefault("casualties_killed", 0)
                    inc.setdefault("casualties_injured", 0)
                    inc.setdefault("kidnapped_count", 0)
                    inc.setdefault("cattle_stolen", 0)
                    inc.setdefault("confidence_score", 0.8)
                    inc.setdefault("threat_level", 2)
                return incidents
        return None

    def narrate_analysis(self, analysis_type, data):
        from eagleeye.core.config import CLAUDE_MODEL, CLAUDE_MAX_TOKENS, CLAUDE_TEMPERATURE
        from eagleeye.llm.prompts import ANALYSIS_NARRATION_SYSTEM, ANALYSIS_NARRATION_PROMPT
        client = self._get_client()
        data_str = json.dumps(data, indent=2, default=str)[:6000]
        prompt = ANALYSIS_NARRATION_PROMPT.format(analysis_type=analysis_type, data=data_str)
        response = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS, temperature=CLAUDE_TEMPERATURE,
            system=ANALYSIS_NARRATION_SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def generate_report_narrative(self, report_type, data):
        from eagleeye.core.config import CLAUDE_MODEL, CLAUDE_MAX_TOKENS, CLAUDE_TEMPERATURE
        from eagleeye.llm.prompts import REPORT_GENERATION_SYSTEM, REPORT_NARRATIVE_PROMPT
        client = self._get_client()
        data_str = json.dumps(data, indent=2, default=str)[:6000]
        prompt = REPORT_NARRATIVE_PROMPT.format(report_type=report_type, data=data_str)
        response = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS, temperature=CLAUDE_TEMPERATURE,
            system=REPORT_GENERATION_SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def generate_intelligence_brief(self, context):
        from eagleeye.core.config import CLAUDE_MODEL, CLAUDE_MAX_TOKENS, CLAUDE_TEMPERATURE
        from eagleeye.llm.prompts import INTELLIGENCE_BRIEF_SYSTEM, INTELLIGENCE_BRIEF_PROMPT
        client = self._get_client()
        context_str = json.dumps(context, indent=2, default=str)[:4000]
        prompt = INTELLIGENCE_BRIEF_PROMPT.format(context=context_str)
        response = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS, temperature=CLAUDE_TEMPERATURE,
            system=INTELLIGENCE_BRIEF_SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def generate_templated_sitrep(self, template, data, context_texts=None):
        from eagleeye.core.config import CLAUDE_MODEL, CLAUDE_MAX_TOKENS
        from eagleeye.llm.prompts import TEMPLATED_SITREP_SYSTEM, TEMPLATED_SITREP_PROMPT
        client = self._get_client()
        data_str = json.dumps(data, indent=2, default=str)[:6000]
        context_section = ""
        if context_texts:
            parts, budget = [], 4000
            for i, txt in enumerate(context_texts, 1):
                chunk = txt[:budget]
                parts.append(f"--- Context Report {i} ---\n{chunk}")
                budget -= len(chunk)
                if budget <= 0:
                    break
            context_section = (
                "\n\nThe following field reports were uploaded as additional context. "
                "Use this information to enrich the SITREP with specific details, "
                "events, and ground-truth observations:\n\n" + "\n\n".join(parts)
            )
        prompt = TEMPLATED_SITREP_PROMPT.format(template=template[:3000], data=data_str) + context_section
        response = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS, temperature=0.3,
            system=TEMPLATED_SITREP_SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def generate_terrain_threat_briefing(self, area, state, terrain_data, incident_data):
        from eagleeye.core.config import CLAUDE_MODEL, CLAUDE_MAX_TOKENS
        from eagleeye.llm.prompts import TERRAIN_THREAT_BRIEFING_SYSTEM, TERRAIN_THREAT_BRIEFING_PROMPT
        client = self._get_client()
        terrain_str = json.dumps(terrain_data, indent=2, default=str)[:5000]
        incident_str = json.dumps(incident_data.get("incidents", {}), indent=2, default=str)[:3000]
        threat_str = json.dumps(incident_data.get("threat_scores", {}), indent=2, default=str)[:1000]
        prompt = TERRAIN_THREAT_BRIEFING_PROMPT.format(
            area_name=area, state=state, terrain_data=terrain_str,
            incident_data=incident_str, threat_scores=threat_str,
        )
        response = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=CLAUDE_MAX_TOKENS, temperature=0.2,
            system=TERRAIN_THREAT_BRIEFING_SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def analyze_report(self, report_text):
        from eagleeye.core.config import CLAUDE_MODEL
        from eagleeye.llm.prompts import REPORT_ANALYSIS_SYSTEM, REPORT_ANALYSIS_PROMPT
        client = self._get_client()
        prompt = REPORT_ANALYSIS_PROMPT.format(report_text=report_text[:12000])
        response = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=16000, temperature=0.2,
            system=REPORT_ANALYSIS_SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def generate_sitrep_from_report(self, report_text, template=None):
        from eagleeye.core.config import CLAUDE_MODEL
        from eagleeye.llm.prompts import SITREP_FROM_REPORT_SYSTEM
        client = self._get_client()
        if template:
            from eagleeye.llm.prompts import SITREP_FROM_REPORT_TEMPLATED_PROMPT
            prompt = SITREP_FROM_REPORT_TEMPLATED_PROMPT.format(
                template=template[:3000], report_text=report_text[:12000])
        else:
            from eagleeye.llm.prompts import SITREP_FROM_REPORT_PROMPT
            prompt = SITREP_FROM_REPORT_PROMPT.format(report_text=report_text[:12000])
        response = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=16000, temperature=0.2,
            system=SITREP_FROM_REPORT_SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def enhance_alert_description(self, alert_type, raw_desc, context=""):
        from eagleeye.core.config import CLAUDE_MODEL, CLAUDE_TEMPERATURE
        from eagleeye.llm.prompts import ALERT_ENHANCEMENT_SYSTEM, ALERT_ENHANCEMENT_PROMPT
        client = self._get_client()
        prompt = ALERT_ENHANCEMENT_PROMPT.format(
            alert_type=alert_type, raw_desc=raw_desc[:1000], context=context[:1000],
        )
        response = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=512, temperature=CLAUDE_TEMPERATURE,
            system=ALERT_ENHANCEMENT_SYSTEM, messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class ClaudeEngine:
    """Central AI engine. Routes through user-configured engine priority list."""

    def __init__(self):
        self._claude = _ClaudeBackend()
        self._enabled_engines = None  # cached list

    # --- Config management ---

    def _load_enabled_engines(self) -> list[str]:
        """Read enabled engines from DB. Returns ordered list. Caches result."""
        if self._enabled_engines is not None:
            return self._enabled_engines
        try:
            from eagleeye.core.database import get_session, AppSetting
            session = get_session()
            try:
                setting = session.query(AppSetting).filter_by(key="ai_engines").first()
                if setting:
                    engines = json.loads(setting.value)
                    # Validate
                    engines = [e for e in engines if e in VALID_ENGINES]
                    if engines:
                        self._enabled_engines = engines
                        return self._enabled_engines
            finally:
                session.close()
        except Exception as e:
            logger.debug(f"Could not load ai_engines setting: {e}")
        self._enabled_engines = list(DEFAULT_ENGINE_ORDER)
        return self._enabled_engines

    def reload_config(self):
        """Clear cached config so next call re-reads from DB."""
        self._enabled_engines = None

    def _get_backend(self, name: str):
        """Return the backend object for a given engine name."""
        if name == "ollama":
            return _get_ollama()
        elif name == "claude":
            return self._claude
        elif name == "huggingface":
            return _get_hf()
        return None

    def _get_engine_chain(self) -> list[tuple[str, object]]:
        """Return list of (name, engine_object) for enabled engines only."""
        enabled = self._load_enabled_engines()
        chain = []
        for name in enabled:
            backend = self._get_backend(name)
            if backend is not None:
                chain.append((name, backend))
        return chain

    def get_engine_status(self) -> dict[str, bool]:
        """Check actual connectivity for all 3 engines (regardless of enabled state)."""
        return {
            "ollama": _get_ollama().is_available,
            "claude": self._claude.is_available,
            "huggingface": _get_hf().is_available,
        }

    @property
    def is_available(self) -> bool:
        """Check if the first enabled engine is available."""
        chain = self._get_engine_chain()
        return chain[0][1].is_available if chain else False

    @property
    def any_ai_available(self) -> bool:
        """Check if any enabled AI backend is available."""
        for _name, backend in self._get_engine_chain():
            if backend.is_available:
                return True
        return False

    @property
    def active_backend(self) -> str:
        """Return which AI backend is currently primary (first available in enabled list)."""
        for name, backend in self._get_engine_chain():
            if backend.is_available:
                return name
        return "none"

    # --- Dispatch through engine chain ---

    def _dispatch(self, method_name: str, *args, **kwargs):
        """Try each enabled engine in priority order."""
        for name, backend in self._get_engine_chain():
            if not backend.is_available:
                continue
            try:
                func = getattr(backend, method_name)
                result = func(*args, **kwargs)
                if result is not None:
                    return result
                logger.warning(f"{name} {method_name} returned None, trying next")
            except Exception as e:
                logger.error(f"{name} {method_name} failed: {e}, trying next")
        return None

    def parse_report(self, text: str, source: str = "",
                     published_date: Optional[datetime] = None) -> Optional[list[dict]]:
        return self._dispatch("parse_report", text, source, published_date)

    def narrate_analysis(self, analysis_type: str, data: dict) -> Optional[str]:
        return self._dispatch("narrate_analysis", analysis_type, data)

    def generate_report_narrative(self, report_type: str, data: dict) -> Optional[str]:
        return self._dispatch("generate_report_narrative", report_type, data)

    def generate_intelligence_brief(self, context: dict) -> Optional[str]:
        return self._dispatch("generate_intelligence_brief", context)

    def generate_templated_sitrep(self, template: str, data: dict,
                                    context_texts: list[str] = None) -> Optional[str]:
        return self._dispatch("generate_templated_sitrep", template, data, context_texts)

    def generate_terrain_threat_briefing(self, area: str, state: str,
                                          terrain_data: dict,
                                          incident_data: dict) -> Optional[str]:
        return self._dispatch("generate_terrain_threat_briefing", area, state, terrain_data, incident_data)

    def analyze_report(self, report_text: str) -> Optional[str]:
        return self._dispatch("analyze_report", report_text)

    def generate_sitrep_from_report(self, report_text: str, template: str = None) -> Optional[str]:
        return self._dispatch("generate_sitrep_from_report", report_text, template)

    def enhance_alert_description(self, alert_type: str, raw_desc: str,
                                   context: str = "") -> Optional[str]:
        return self._dispatch("enhance_alert_description", alert_type, raw_desc, context)


# Singleton instance for use across the application
engine = ClaudeEngine()
