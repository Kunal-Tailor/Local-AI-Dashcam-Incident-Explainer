"""
Phase 3 — VLM Narrator
Feeds sequential dashcam keyframes to a local VLM (MiniCPM-V via Ollama)
and extracts structured JSON incident analysis.

Output JSON schema:
{
  "vehicles": ["..."],            # vehicle types observed
  "fault_analysis": "...",        # who appears at fault and why
  "conditions": {                 # road and weather conditions
      "road": "...",
      "weather": "...",
      "visibility": "..."
  },
  "severity": "minor|moderate|severe",
  "timeline": ["pre", "impact", "post"],   # per-frame descriptions
  "confidence_flag": true|false   # true = uncertain, needs human review
}

Usage:
    python3 vlm_narrator.py --frames outputs/keyframes/kf1.jpg kf2.jpg kf3.jpg
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

import os

def _get_ollama():
    """Auto-detect Ollama availability; fall back to mock if server isn't running."""
    if os.environ.get("DASHCAM_MOCK_OLLAMA", "0") == "1":
        from src.ollama_mock import chat, list
        import types
        mock = types.ModuleType("ollama")
        mock.chat = chat
        mock.list = list
        print("[VLM] Using Ollama MOCK (DASHCAM_MOCK_OLLAMA=1)")
        return mock
    try:
        import ollama as _real_ollama
        # Quick ping to check if server is live
        try:
            _real_ollama.list()
            print("[VLM] Connected to real Ollama server")
            return _real_ollama
        except Exception:
            print("[VLM] Ollama server not reachable — falling back to mock")
            from src import ollama_mock
            import types
            mock = types.ModuleType("ollama")
            mock.chat = ollama_mock.chat
            mock.list = ollama_mock.list
            return mock
    except ImportError:
        print("[ERROR] ollama package not found. Run: pip install ollama", file=sys.stderr)
        sys.exit(1)

ollama = _get_ollama()


# ─── Hedging language checker ─────────────────────────────────────────────────

HEDGING_WORDS = [
    "possibly", "maybe", "unclear", "unsure", "might", "could be",
    "difficult to determine", "hard to tell", "not certain", "uncertain",
    "appears to", "seems to", "it is possible", "cannot confirm",
    "it's possible", "i cannot", "i can't"
]


def has_hedging_language(text: str) -> bool:
    lower = text.lower()
    return any(word in lower for word in HEDGING_WORDS)


# ─── Base64 encoder for frames ────────────────────────────────────────────────

def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR numpy frame to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def path_to_base64(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert traffic incident analyst reviewing dashcam footage.
You will be shown sequential frames from a dashcam video showing a traffic incident.
Analyse the frames carefully and respond ONLY with a valid JSON object using this exact schema:

{
  "vehicles": ["list of vehicle types you can identify, e.g. sedan, SUV, motorcycle, truck"],
  "fault_analysis": "one paragraph describing who appears at fault and the reasoning",
  "conditions": {
    "road": "description of road type and surface condition",
    "weather": "clear/rain/fog/night/etc",
    "visibility": "good/moderate/poor"
  },
  "severity": "one of: minor, moderate, severe",
  "timeline": ["description of frame 1 (pre-incident)", "description of frame 2 (impact/mid)", "description of frame 3 (post-incident)"],
  "confidence_flag": false
}

Set confidence_flag to true if you are uncertain about any major aspect.
Do NOT include any text outside the JSON object."""


FALLBACK_PROMPT = """Please look at these dashcam images and provide a JSON response about the traffic incident.
Use this exact format:
{"vehicles": [], "fault_analysis": "", "conditions": {"road": "", "weather": "", "visibility": ""}, "severity": "moderate", "timeline": [], "confidence_flag": true}"""


# ─── Narrator ─────────────────────────────────────────────────────────────────

class VLMNarrator:
    def __init__(
        self,
        model: str = "minicpm-v",
        max_retries: int = 2,
        timeout: int = 120,
    ):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        print(f"[VLM] Using model: {self.model}")

    def _build_messages(
        self, frames_b64: list[str], system_prompt: str
    ) -> list[dict]:
        """Build Ollama multimodal chat message list."""
        user_content = (
            f"I am showing you {len(frames_b64)} sequential dashcam frames "
            f"from a traffic incident. Frame 1 is the pre-incident scene, "
            f"frame {len(frames_b64)} is post-incident."
        )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content,
                "images": frames_b64,
            }
        ]

    def _extract_json(self, text: str) -> dict | None:
        """Try to extract valid JSON from model response."""
        # Direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Find first {...} block
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    def narrate(
        self,
        frames: list[np.ndarray | str],  # numpy BGR frames OR image paths
    ) -> dict:
        """
        Run VLM narration on a list of keyframes.

        Args:
            frames: list of numpy BGR arrays OR image file paths

        Returns:
            Parsed JSON dict with incident analysis
        """
        # Encode frames
        frames_b64: list[str] = []
        for f in frames:
            if isinstance(f, str):
                frames_b64.append(path_to_base64(f))
            else:
                frames_b64.append(frame_to_base64(f))

        prompts = [SYSTEM_PROMPT, FALLBACK_PROMPT]

        for attempt, prompt in enumerate(prompts[:self.max_retries]):
            print(f"[VLM] Attempt {attempt + 1} with {len(frames_b64)} frames...")
            try:
                messages = self._build_messages(frames_b64, prompt)
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": 0.1, "num_predict": 800},
                )
                raw_text = response["message"]["content"]
                parsed = self._extract_json(raw_text)

                if parsed is not None:
                    # Ensure required fields exist
                    parsed.setdefault("vehicles", [])
                    parsed.setdefault("fault_analysis", "")
                    parsed.setdefault("conditions", {})
                    parsed.setdefault("severity", "moderate")
                    parsed.setdefault("timeline", [])

                    # Auto-set confidence flag if hedging detected
                    hedging = has_hedging_language(
                        parsed.get("fault_analysis", "") +
                        " ".join(parsed.get("timeline", []))
                    )
                    parsed["confidence_flag"] = parsed.get("confidence_flag", False) or hedging

                    print(f"[VLM] ✓ JSON parsed. Severity: {parsed.get('severity')}, "
                          f"confidence_flag: {parsed['confidence_flag']}")
                    return parsed
                else:
                    print(f"[VLM] JSON parse failed on attempt {attempt + 1}. Raw:\n{raw_text[:300]}")

            except Exception as e:
                print(f"[VLM] Error on attempt {attempt + 1}: {e}")

        # Final fallback
        print("[VLM] All attempts failed. Returning error struct.")
        return {
            "vehicles": [],
            "fault_analysis": "Analysis failed — VLM did not return parseable JSON.",
            "conditions": {"road": "unknown", "weather": "unknown", "visibility": "unknown"},
            "severity": "unknown",
            "timeline": [],
            "confidence_flag": True,
        }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VLM Incident Narrator")
    parser.add_argument("--frames", nargs="+", required=True,
                        help="Paths to keyframe images (in order: pre, mid, post)")
    parser.add_argument("--model", default="minicpm-v",
                        help="Ollama VLM model name (default: minicpm-v)")
    args = parser.parse_args()

    for fp in args.frames:
        if not Path(fp).is_file():
            print(f"[ERROR] Frame not found: {fp}", file=sys.stderr)
            sys.exit(1)

    narrator = VLMNarrator(model=args.model)
    result = narrator.narrate(args.frames)

    print("\n" + "=" * 55)
    print("  VLM Incident Analysis")
    print("=" * 55)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
