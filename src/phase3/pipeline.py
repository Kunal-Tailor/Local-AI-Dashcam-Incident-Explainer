"""
Phase 3 — End-to-End Pipeline Runner
Orchestrates: video → incident detection → keyframe extraction → VLM narration → report

Usage:
    python3 src/phase3/pipeline.py --video data/samples/test.mp4 --keyframes 3

Output (in outputs/reports/):
    report_001.json   — structured incident analysis
    report_001.txt    — human-readable prose report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Make imports work from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase1.extract_keyframes import extract_uniform
from src.phase2.incident_extractor import IncidentExtractor
from src.phase3.keyframe_sampler import sample_keyframes
from src.phase3.vlm_narrator import VLMNarrator
from src.phase3.report_generator import ReportGenerator


def run_pipeline(
    video_path: str,
    n_keyframes: int = 3,
    vlm_model: str = "minicpm-v",
    llm_model: str = "mistral",
    skip_detection: bool = False,
    output_dir: str = "outputs/reports",
) -> list[dict]:
    """
    Full pipeline: video → incidents → VLM → prose.

    Args:
        video_path     : path to dashcam video
        n_keyframes    : keyframes per incident (1, 3, or 5)
        vlm_model      : Ollama VLM model
        llm_model      : Ollama LLM for prose generation
        skip_detection : if True, treat the whole video as one incident window
        output_dir     : where to save JSON + TXT reports

    Returns:
        list of result dicts (one per incident)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    narrator = VLMNarrator(model=vlm_model)
    reporter = ReportGenerator(model=llm_model)

    if skip_detection:
        # Treat entire video as incident
        print("[Pipeline] Skip-detection mode: using full video as incident")
        incident_clips = [video_path]
    else:
        # Phase 2: detect incidents
        print("[Pipeline] ▶ Phase 2: Detecting incidents...")
        extractor = IncidentExtractor()
        incident_dir = "outputs/incidents"
        windows = extractor.process_video(video_path, incident_dir)

        if not windows:
            print("[Pipeline] No incidents detected. Treating full video as incident.")
            incident_clips = [video_path]
        else:
            # Collect saved clip paths
            incident_clips = sorted(Path(incident_dir).glob("incident_*.mp4"))
            incident_clips = [str(p) for p in incident_clips]

    # Phase 3: VLM narration for each incident
    for idx, clip_path in enumerate(incident_clips, 1):
        print(f"\n[Pipeline] ▶ Phase 3: Incident {idx} → {clip_path}")

        # Sample keyframes
        kf_out = f"outputs/keyframes/incident_{idx:03d}"
        keyframes = sample_keyframes(clip_path, n=n_keyframes, output_dir=kf_out)

        if not keyframes:
            print(f"[Pipeline] No keyframes extracted from {clip_path}, skipping.")
            continue

        # Extract frame arrays
        frame_arrays = [frame for _, frame in keyframes]

        # VLM analysis
        t0 = time.time()
        incident_json = narrator.narrate(frame_arrays)
        t1 = time.time()
        incident_json["clip_path"] = str(clip_path)
        incident_json["keyframe_count"] = n_keyframes
        incident_json["vlm_model"] = vlm_model
        incident_json["inference_time_s"] = round(t1 - t0, 2)
        incident_json["timestamp"] = ts

        # Prose report
        prose = reporter.generate(incident_json)
        incident_json["prose_report"] = prose

        # Replace [DATE] placeholder
        prose = prose.replace("[DATE]", datetime.now().strftime("%d %B %Y"))
        incident_json["prose_report"] = prose

        # Save outputs
        json_path = os.path.join(output_dir, f"report_{idx:03d}_{ts}.json")
        txt_path = os.path.join(output_dir, f"report_{idx:03d}_{ts}.txt")

        with open(json_path, "w") as f:
            json.dump(incident_json, f, indent=2)
        with open(txt_path, "w") as f:
            f.write(f"DASHCAM INCIDENT REPORT\n")
            f.write(f"{'='*60}\n")
            f.write(f"Date        : {datetime.now().strftime('%d %B %Y, %H:%M')}\n")
            f.write(f"Video       : {video_path}\n")
            f.write(f"Incident    : {idx}\n")
            f.write(f"Severity    : {incident_json.get('severity', 'N/A').upper()}\n")
            f.write(f"Confidence  : {'⚠ REVIEW NEEDED' if incident_json.get('confidence_flag') else '✓ HIGH'}\n")
            f.write(f"{'='*60}\n\n")
            f.write(prose)
            f.write(f"\n\n{'─'*60}\n")
            f.write(f"Structured Data:\n{json.dumps(incident_json, indent=2)}\n")

        print(f"[Pipeline] ✓ Report saved → {json_path}")
        print(f"[Pipeline] ✓ Prose saved  → {txt_path}")
        results.append(incident_json)

    print(f"\n[Pipeline] ✅ Done. {len(results)} incident(s) processed.")
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Local AI Dashcam Incident Explainer — Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on a dashcam clip with 3 keyframes
  python3 src/phase3/pipeline.py --video data/samples/test.mp4 --keyframes 3

  # Skip incident detection (treat whole video as incident)
  python3 src/phase3/pipeline.py --video data/samples/clip.mp4 --skip-detection

  # Use different models
  python3 src/phase3/pipeline.py --video clip.mp4 --vlm internvl2 --llm llama3
        """
    )
    parser.add_argument("--video", required=True, help="Path to dashcam video")
    parser.add_argument("--keyframes", type=int, default=3, choices=[1, 3, 5],
                        help="Number of keyframes per incident (default: 3)")
    parser.add_argument("--vlm", default="minicpm-v",
                        help="VLM model for image analysis (default: minicpm-v)")
    parser.add_argument("--llm", default="mistral",
                        help="LLM model for prose report (default: mistral)")
    parser.add_argument("--skip-detection", action="store_true",
                        help="Skip incident detection; treat whole video as incident")
    parser.add_argument("--output", default="outputs/reports",
                        help="Output directory for reports")
    args = parser.parse_args()

    if not Path(args.video).is_file():
        print(f"[ERROR] Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  Local AI Dashcam Incident Explainer")
    print(f"  Video     : {args.video}")
    print(f"  Keyframes : {args.keyframes}")
    print(f"  VLM       : {args.vlm}")
    print(f"  LLM       : {args.llm}")
    print("=" * 60)

    results = run_pipeline(
        video_path=args.video,
        n_keyframes=args.keyframes,
        vlm_model=args.vlm,
        llm_model=args.llm,
        skip_detection=args.skip_detection,
        output_dir=args.output,
    )

    if results:
        for i, r in enumerate(results, 1):
            flag = "⚠ REVIEW" if r.get("confidence_flag") else "✓ OK"
            print(f"  Incident {i}: severity={r.get('severity')} | {flag} | "
                  f"inference={r.get('inference_time_s')}s")


if __name__ == "__main__":
    main()
