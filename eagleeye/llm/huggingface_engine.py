"""
HuggingFace fallback AI engine for EagleEye platform.
Uses Qwen/Qwen2.5-72B-Instruct via the HuggingFace Inference API.
Activated automatically when Claude is unavailable or hits rate limits.
"""

import json
import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class HuggingFaceEngine:
    """Fallback AI engine using HuggingFace Inference API."""

    def __init__(self):
        self._available = None

    @property
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        from eagleeye.core.config import HF_API_TOKEN
        if not HF_API_TOKEN:
            logger.info("HuggingFace: HF_API_TOKEN not set — fallback disabled")
            self._available = False
            return False
        self._available = True
        return True

    def _chat(self, system: str, user: str, temperature: float = 0.3,
              max_tokens: int = 4096) -> Optional[str]:
        """Send a chat completion request to HuggingFace Inference API."""
        from eagleeye.core.config import HF_API_TOKEN, HF_MODEL

        from eagleeye.core.config import HF_API_URL
        url = HF_API_URL

        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": HF_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text
        except requests.exceptions.Timeout:
            logger.error("HuggingFace API request timed out")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HuggingFace API HTTP error: {e} — {resp.text[:500]}")
            return None
        except Exception as e:
            logger.error(f"HuggingFace API request failed: {e}")
            return None

    def parse_report(self, text: str, source: str = "",
                     published_date=None) -> Optional[list[dict]]:
        """Parse a security report using HuggingFace model with JSON output."""
        if not self.is_available:
            return None

        from eagleeye.llm.prompts import REPORT_PARSING_SYSTEM

        date_context = ""
        if published_date:
            date_context = f"\n\nThis report was published on {published_date.strftime('%A, %d %B %Y')}."

        system = REPORT_PARSING_SYSTEM + """

You MUST respond with ONLY valid JSON in this exact format (no markdown, no explanation):
{"incidents": [{"date": "YYYY-MM-DD", "state": "...", "lga": "...", "location_name": "...", "incident_type": "...", "casualties_killed": 0, "casualties_injured": 0, "kidnapped_count": 0, "cattle_stolen": 0, "description": "...", "confidence_score": 0.8, "threat_level": 2}]}

Valid incident_type values: village_raid, kidnapping, cattle_rustling, highway_ambush, market_attack, attack_on_security_forces, mining_site_attack, reprisal_attack, arms_smuggling, camp_sighting, displacement, negotiation_ransom, inter_gang_clash, infrastructure_attack
Valid state values: Zamfara, Kaduna, Katsina, Sokoto, Niger, Kebbi, Unknown
threat_level: 1=LOW, 2=MODERATE, 3=ELEVATED, 4=HIGH, 5=CRITICAL"""

        user_msg = f"Extract all security incidents from this report.{date_context}\n\nSource: {source}\n\nReport text:\n{text[:6000]}"

        try:
            result = self._chat(system, user_msg, temperature=0.1)
            if not result:
                return None

            # Try to extract JSON from the response
            parsed = self._extract_json(result)
            if parsed is None:
                logger.warning("HuggingFace: could not parse JSON from report extraction")
                return None

            incidents = parsed.get("incidents", [])
            logger.info(f"HuggingFace extracted {len(incidents)} incident(s) from report")

            for inc in incidents:
                inc["source"] = source
                inc.setdefault("casualties_killed", 0)
                inc.setdefault("casualties_injured", 0)
                inc.setdefault("kidnapped_count", 0)
                inc.setdefault("cattle_stolen", 0)
                inc.setdefault("confidence_score", 0.7)
                inc.setdefault("threat_level", 2)
            return incidents

        except Exception as e:
            logger.error(f"HuggingFace report parsing failed: {e}")
            return None

    def narrate_analysis(self, analysis_type: str, data: dict) -> Optional[str]:
        if not self.is_available:
            return None
        try:
            from eagleeye.llm.prompts import ANALYSIS_NARRATION_SYSTEM, ANALYSIS_NARRATION_PROMPT
            data_str = json.dumps(data, indent=2, default=str)[:6000]
            prompt = ANALYSIS_NARRATION_PROMPT.format(analysis_type=analysis_type, data=data_str)
            result = self._chat(ANALYSIS_NARRATION_SYSTEM, prompt)
            if result:
                logger.info(f"HuggingFace generated {analysis_type} narrative ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"HuggingFace analysis narration failed: {e}")
            return None

    def generate_report_narrative(self, report_type: str, data: dict) -> Optional[str]:
        if not self.is_available:
            return None
        try:
            from eagleeye.llm.prompts import REPORT_GENERATION_SYSTEM, REPORT_NARRATIVE_PROMPT
            data_str = json.dumps(data, indent=2, default=str)[:6000]
            prompt = REPORT_NARRATIVE_PROMPT.format(report_type=report_type, data=data_str)
            result = self._chat(REPORT_GENERATION_SYSTEM, prompt)
            if result:
                logger.info(f"HuggingFace generated {report_type} report narrative ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"HuggingFace report narrative failed: {e}")
            return None

    def generate_intelligence_brief(self, context: dict) -> Optional[str]:
        if not self.is_available:
            return None
        try:
            from eagleeye.llm.prompts import INTELLIGENCE_BRIEF_SYSTEM, INTELLIGENCE_BRIEF_PROMPT
            context_str = json.dumps(context, indent=2, default=str)[:4000]
            prompt = INTELLIGENCE_BRIEF_PROMPT.format(context=context_str)
            result = self._chat(INTELLIGENCE_BRIEF_SYSTEM, prompt)
            if result:
                logger.info(f"HuggingFace generated intelligence brief ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"HuggingFace intelligence brief failed: {e}")
            return None

    def generate_templated_sitrep(self, template: str, data: dict,
                                  context_texts: list[str] = None) -> Optional[str]:
        if not self.is_available:
            return None
        try:
            from eagleeye.llm.prompts import TEMPLATED_SITREP_SYSTEM, TEMPLATED_SITREP_PROMPT
            data_str = json.dumps(data, indent=2, default=str)[:6000]

            context_section = ""
            if context_texts:
                parts = []
                budget = 4000
                for i, txt in enumerate(context_texts, 1):
                    chunk = txt[:budget]
                    parts.append(f"--- Context Report {i} ---\n{chunk}")
                    budget -= len(chunk)
                    if budget <= 0:
                        break
                context_section = (
                    "\n\nThe following field reports were uploaded as additional context. "
                    "Use this information to enrich the SITREP with specific details, "
                    "events, and ground-truth observations:\n\n"
                    + "\n\n".join(parts)
                )

            prompt = TEMPLATED_SITREP_PROMPT.format(
                template=template[:3000], data=data_str,
            ) + context_section

            result = self._chat(TEMPLATED_SITREP_SYSTEM, prompt)
            if result:
                logger.info(f"HuggingFace generated templated SITREP ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"HuggingFace templated SITREP failed: {e}")
            return None

    def generate_terrain_threat_briefing(self, area: str, state: str,
                                         terrain_data: dict,
                                         incident_data: dict) -> Optional[str]:
        if not self.is_available:
            return None
        try:
            from eagleeye.llm.prompts import (
                TERRAIN_THREAT_BRIEFING_SYSTEM, TERRAIN_THREAT_BRIEFING_PROMPT,
            )
            terrain_str = json.dumps(terrain_data, indent=2, default=str)[:5000]
            incident_str = json.dumps(incident_data.get("incidents", {}),
                                      indent=2, default=str)[:3000]
            threat_str = json.dumps(incident_data.get("threat_scores", {}),
                                    indent=2, default=str)[:1000]

            prompt = TERRAIN_THREAT_BRIEFING_PROMPT.format(
                area_name=area, state=state,
                terrain_data=terrain_str, incident_data=incident_str,
                threat_scores=threat_str,
            )

            result = self._chat(TERRAIN_THREAT_BRIEFING_SYSTEM, prompt, temperature=0.2)
            if result:
                logger.info(f"HuggingFace generated terrain threat briefing ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"HuggingFace terrain threat briefing failed: {e}")
            return None

    def analyze_report(self, report_text: str) -> Optional[str]:
        if not self.is_available:
            return None
        try:
            from eagleeye.llm.prompts import REPORT_ANALYSIS_SYSTEM, REPORT_ANALYSIS_PROMPT
            prompt = REPORT_ANALYSIS_PROMPT.format(report_text=report_text[:12000])
            result = self._chat(REPORT_ANALYSIS_SYSTEM, prompt, temperature=0.2, max_tokens=16000)
            if result:
                logger.info(f"HuggingFace generated report analysis ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"HuggingFace report analysis failed: {e}")
            return None

    def generate_sitrep_from_report(self, report_text: str, template: str = None) -> Optional[str]:
        if not self.is_available:
            return None
        try:
            from eagleeye.llm.prompts import SITREP_FROM_REPORT_SYSTEM
            if template:
                from eagleeye.llm.prompts import SITREP_FROM_REPORT_TEMPLATED_PROMPT
                prompt = SITREP_FROM_REPORT_TEMPLATED_PROMPT.format(
                    template=template[:3000], report_text=report_text[:12000])
            else:
                from eagleeye.llm.prompts import SITREP_FROM_REPORT_PROMPT
                prompt = SITREP_FROM_REPORT_PROMPT.format(report_text=report_text[:12000])
            result = self._chat(SITREP_FROM_REPORT_SYSTEM, prompt, temperature=0.2, max_tokens=16000)
            if result:
                logger.info(f"HuggingFace generated SITREP from report ({len(result)} chars)")
            return result
        except Exception as e:
            logger.error(f"HuggingFace SITREP from report failed: {e}")
            return None

    def enhance_alert_description(self, alert_type: str, raw_desc: str,
                                  context: str = "") -> Optional[str]:
        if not self.is_available:
            return None
        try:
            from eagleeye.llm.prompts import ALERT_ENHANCEMENT_SYSTEM, ALERT_ENHANCEMENT_PROMPT
            prompt = ALERT_ENHANCEMENT_PROMPT.format(
                alert_type=alert_type,
                raw_desc=raw_desc[:1000],
                context=context[:1000],
            )
            result = self._chat(ALERT_ENHANCEMENT_SYSTEM, prompt, max_tokens=512)
            return result
        except Exception as e:
            logger.error(f"HuggingFace alert enhancement failed: {e}")
            return None

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract JSON from model output that may contain markdown or extra text."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None


# Singleton
hf_engine = HuggingFaceEngine()
