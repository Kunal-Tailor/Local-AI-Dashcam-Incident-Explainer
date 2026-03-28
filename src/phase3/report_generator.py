"""
Phase 3 — Report Generator
Takes the structured JSON from vlm_narrator.py and uses a local Ollama LLM
(Mistral) to generate a human-readable insurance-style incident report paragraph.

Usage:
    from src.phase3.report_generator import ReportGenerator
    gen = ReportGenerator()
    prose = gen.generate(incident_json)
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

def _get_ollama():
    if os.environ.get("DASHCAM_MOCK_OLLAMA", "0") == "1":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src import ollama_mock
        mock = types.ModuleType("ollama")
        mock.chat = ollama_mock.chat
        mock.list = ollama_mock.list
        print("[ReportGen] Using Ollama MOCK")
        return mock
    try:
        import ollama as _real_ollama
        _real_ollama.list()
        return _real_ollama
    except Exception:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src import ollama_mock
        mock = types.ModuleType("ollama")
        mock.chat = ollama_mock.chat
        mock.list = ollama_mock.list
        print("[ReportGen] Ollama not reachable — using mock")
        return mock

ollama = _get_ollama()


REPORT_SYSTEM_PROMPT = """You are a professional automotive insurance claims adjuster.
Your task is to write a formal, concise incident report based on a structured JSON analysis of a dashcam-recorded traffic incident.
Write 150–200 words in a single paragraph. Use formal, factual language. Do not repeat the JSON, just write flowing prose.
Include: date placeholder [DATE], vehicles involved, fault assessment, road/weather conditions, estimated damage severity, and recommended follow-up action."""


def json_to_prompt(data: dict) -> str:
    vehicles = ", ".join(data.get("vehicles", [])) or "Unknown"
    fault = data.get("fault_analysis", "Not determined.")
    cond = data.get("conditions", {})
    road = cond.get("road", "Unknown")
    weather = cond.get("weather", "Unknown")
    visibility = cond.get("visibility", "Unknown")
    severity = data.get("severity", "Unknown")
    timeline = " | ".join(data.get("timeline", []))
    flag = "⚠ Marked for human review." if data.get("confidence_flag") else ""

    return (
        f"Incident Data:\n"
        f"- Vehicles observed: {vehicles}\n"
        f"- Fault analysis: {fault}\n"
        f"- Road: {road} | Weather: {weather} | Visibility: {visibility}\n"
        f"- Severity: {severity}\n"
        f"- Timeline: {timeline}\n"
        f"{flag}\n\n"
        f"Write the formal insurance incident report paragraph now:"
    )


class ReportGenerator:
    def __init__(self, model: str = "mistral"):
        self.model = model
        print(f"[ReportGen] Using LLM: {self.model}")

    def generate(self, incident_json: dict) -> str:
        """Generate prose report from incident JSON. Returns string."""
        user_prompt = json_to_prompt(incident_json)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.3, "num_predict": 400},
            )
            prose = response["message"]["content"].strip()
            print(f"[ReportGen] ✓ Report generated ({len(prose.split())} words)")
            return prose
        except Exception as e:
            print(f"[ReportGen] Error: {e}")
            return (
                "Incident report could not be generated automatically. "
                f"See structured data: severity={incident_json.get('severity')}, "
                f"vehicles={incident_json.get('vehicles')}. "
                "Manual review required."
            )
