"""
Phase 4 — Gradio Web UI
Full drag-and-drop dashcam incident analysis interface.

Features:
  - Video upload → real-time processing log
  - Keyframe count selection (1 / 3 / 5)
  - VLM model dropdown (minicpm-v / internvl2)
  - Annotated keyframes gallery (privacy-blurred)
  - JSON viewer + prose report display
  - PDF download button
  - Confidence flag highlight

Run:
    python3 src/phase4/app.py
Then open:  http://localhost:7860
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import base64
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# Make imports work from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr

from src.phase2.detector import DashcamDetector
from src.phase2.incident_extractor import IncidentExtractor
from src.phase3.keyframe_sampler import sample_keyframes
from src.phase3.vlm_narrator import VLMNarrator
from src.phase3.report_generator import ReportGenerator
from src.phase4.privacy import PrivacyBlurrer
from src.phase4.pdf_exporter import PDFExporter


# ─── Global singletons (loaded once) ─────────────────────────────────────────

_detector: DashcamDetector | None = None
_privacy: PrivacyBlurrer | None = None
_pdf_exporter: PDFExporter | None = None


def get_singletons():
    global _detector, _privacy, _pdf_exporter
    if _detector is None:
        _detector = DashcamDetector()
    if _privacy is None:
        _privacy = PrivacyBlurrer()
    if _pdf_exporter is None:
        _pdf_exporter = PDFExporter()
    return _detector, _privacy, _pdf_exporter


def _img_to_data_uri(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def _img_tag(data_uri: str | None, cls: str, alt: str) -> str:
    if data_uri:
        return f'<img class="{cls}" src="{data_uri}" alt="{alt}" />'
    return f'<div class="{cls} placeholder" aria-label="{alt}"></div>'


def _pipeline_svg() -> str:
    return """
    <svg class="media-svg" viewBox="0 0 720 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Pipeline schematic">
      <defs>
        <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#0f766e"/>
          <stop offset="100%" stop-color="#f97316"/>
        </linearGradient>
      </defs>
      <rect x="10" y="20" width="700" height="200" rx="16" fill="rgba(255,255,255,0.85)" stroke="rgba(20,32,51,0.12)" />
      <g font-family="Space Grotesk, sans-serif" font-size="13" fill="#142033">
        <rect x="30" y="60" width="120" height="72" rx="12" fill="rgba(15,118,110,0.10)" stroke="rgba(15,118,110,0.3)"/>
        <text x="90" y="92" text-anchor="middle" font-weight="700">Keyframes</text>
        <text x="90" y="112" text-anchor="middle" fill="#2b3a52">Motion sampling</text>

        <rect x="180" y="60" width="120" height="72" rx="12" fill="rgba(15,118,110,0.10)" stroke="rgba(15,118,110,0.3)"/>
        <text x="240" y="92" text-anchor="middle" font-weight="700">Detection</text>
        <text x="240" y="112" text-anchor="middle" fill="#2b3a52">YOLO + SORT</text>

        <rect x="330" y="60" width="120" height="72" rx="12" fill="rgba(15,118,110,0.10)" stroke="rgba(15,118,110,0.3)"/>
        <text x="390" y="92" text-anchor="middle" font-weight="700">VLM JSON</text>
        <text x="390" y="112" text-anchor="middle" fill="#2b3a52">Scene logic</text>

        <rect x="480" y="60" width="120" height="72" rx="12" fill="rgba(15,118,110,0.10)" stroke="rgba(15,118,110,0.3)"/>
        <text x="540" y="92" text-anchor="middle" font-weight="700">Privacy</text>
        <text x="540" y="112" text-anchor="middle" fill="#2b3a52">Blur filters</text>

        <rect x="630" y="60" width="70" height="72" rx="12" fill="rgba(15,118,110,0.10)" stroke="rgba(15,118,110,0.3)"/>
        <text x="665" y="92" text-anchor="middle" font-weight="700">PDF</text>
        <text x="665" y="112" text-anchor="middle" fill="#2b3a52">Export</text>
      </g>
      <g stroke="url(#g1)" stroke-width="2" fill="none">
        <path d="M150 96 L180 96"/>
        <path d="M300 96 L330 96"/>
        <path d="M450 96 L480 96"/>
        <path d="M600 96 L630 96"/>
      </g>
      <g fill="#0f766e">
        <circle cx="150" cy="96" r="4"/>
        <circle cx="300" cy="96" r="4"/>
        <circle cx="450" cy="96" r="4"/>
        <circle cx="600" cy="96" r="4"/>
      </g>
      <text x="36" y="170" font-family="IBM Plex Mono, monospace" font-size="11" fill="#667085">
        Offline pipeline: ingest → detect → narrate → anonymize → export
      </text>
    </svg>
    """


# ─── Processing pipeline ─────────────────────────────────────────────────────

def process_video(
    video_file,
    n_keyframes: int,
    vlm_model: str,
    llm_model: str,
    skip_detection: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Main Gradio processing function.
    Yields (log_text, gallery_images, json_text, prose_text, pdf_path, clip_path)
    """
    if video_file is None:
        yield "❌ Please upload a dashcam video.", [], "{}", "", None, None
        return

    video_path = video_file if isinstance(video_file, str) else video_file.name
    log_lines = [f"▶ Processing: {Path(video_path).name}"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/reports/{ts}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        log_lines.append(msg)
        return "\n".join(log_lines)

    detector, privacy, pdf_exp = get_singletons()

    # ── Step 1: Incident Detection ──────────────────────────────────────────
    yield log("🔍 Step 1/4: Detecting incidents..."), [], "{}", "", None, None

    incident_clips: list[str] = []
    if skip_detection:
        incident_clips = [video_path]
        log("  ℹ Skip-detection mode: using full video")
    else:
        extractor = IncidentExtractor()
        inc_dir = f"outputs/incidents/{ts}"
        windows = extractor.process_video(video_path, inc_dir)
        if windows:
            clips = sorted(Path(inc_dir).glob("incident_*.mp4"))
            incident_clips = [str(c) for c in clips]
            log(f"  ✓ Found {len(windows)} incident(s)")
        else:
            incident_clips = [video_path]
            log("  ℹ No incidents detected — processing full video")

    yield "\n".join(log_lines), [], "{}", "", None, None

    all_json: list[dict] = []
    all_gallery_images: list[np.ndarray] = []

    for idx, clip_path in enumerate(incident_clips, 1):
        current_clip = clip_path
        yield log(f"\n🎞  Incident {idx}: {Path(clip_path).name}"), [], "{}", "", None, current_clip

        # ── Step 2: Keyframe Sampling ────────────────────────────────────
        log(f"  📷 Step 2/4: Sampling {n_keyframes} keyframes...")
        yield "\n".join(log_lines), [], "{}", "", None, current_clip

        kf_out = f"outputs/keyframes/{ts}_incident{idx}"
        keyframes = sample_keyframes(clip_path, n=n_keyframes, output_dir=kf_out)
        if not keyframes:
            log("  ⚠  No keyframes extracted, skipping.")
            continue

        frame_arrays = [f for _, f in keyframes]

        # YOLO annotate + privacy blur
        blurred_frames: list[np.ndarray] = []
        vehicle_bboxes: list[tuple] = []

        for frame in frame_arrays:
            dets = detector.detect_frame(frame)
            # Collect vehicle bboxes for LP blurring
            vbboxes = [d.bbox for d in dets
                       if d.class_name in ("car", "truck", "bus", "motorcycle")]
            vehicle_bboxes.extend(vbboxes)

            # Annotate
            annotated = detector.annotate_frame(frame, dets)
            # Privacy blur
            blurred = privacy.blur(annotated, vehicle_bboxes=vbboxes)
            blurred_frames.append(blurred)

        all_gallery_images.extend(blurred_frames)

        # Convert for Gradio gallery (RGB)
        gallery_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in all_gallery_images]

        log(f"  ✓ {len(blurred_frames)} frames annotated & privacy-blurred")
        yield "\n".join(log_lines), gallery_rgb, "{}", "", None, current_clip

        # ── Step 3: VLM Narration ────────────────────────────────────────
        log(f"  🤖 Step 3/4: VLM narration ({vlm_model})...")
        yield "\n".join(log_lines), gallery_rgb, "{}", "", None, current_clip

        narrator = VLMNarrator(model=vlm_model)
        t0 = time.time()
        incident_json = narrator.narrate(blurred_frames)
        t1 = time.time()

        incident_json.update({
            "clip_path": str(clip_path),
            "keyframe_count": n_keyframes,
            "vlm_model": vlm_model,
            "inference_time_s": round(t1 - t0, 2),
            "timestamp": ts,
        })

        log(f"  ✓ VLM done in {incident_json['inference_time_s']}s | "
            f"severity={incident_json.get('severity')} | "
            f"flag={'⚠' if incident_json.get('confidence_flag') else '✓'}")
        yield "\n".join(log_lines), gallery_rgb, json.dumps(incident_json, indent=2), "", None, current_clip

        # ── Step 4: Report Generation + PDF ─────────────────────────────
        log(f"  📝 Step 4/4: Generating report ({llm_model})...")
        yield "\n".join(log_lines), gallery_rgb, json.dumps(incident_json, indent=2), "", None, current_clip

        reporter = ReportGenerator(model=llm_model)
        prose = reporter.generate(incident_json)
        prose = prose.replace("[DATE]", datetime.now().strftime("%d %B %Y"))
        incident_json["prose_report"] = prose

        # Save JSON
        json_path = os.path.join(output_dir, f"report_{idx:03d}.json")
        with open(json_path, "w") as f:
            json.dump(incident_json, f, indent=2)

        # Save PDF
        pdf_path = os.path.join(output_dir, f"report_{idx:03d}.pdf")
        kf_paths = sorted(Path(kf_out).glob("*.jpg")) if Path(kf_out).exists() else []
        pdf_exp.export(
            incident_json,
            keyframe_paths=[str(p) for p in kf_paths],
            output_path=pdf_path,
            keyframe_arrays=blurred_frames,
        )

        log(f"  ✓ PDF: {pdf_path}")
        all_json.append(incident_json)

        best_json_text = json.dumps(all_json[0] if all_json else {}, indent=2)
        best_prose = all_json[0].get("prose_report", "") if all_json else ""

        yield "\n".join(log_lines), gallery_rgb, best_json_text, best_prose, pdf_path, current_clip

    final_log = log(f"\n✅ Done! {len(all_json)} incident(s) processed.")
    final_json = json.dumps(all_json[0] if all_json else {}, indent=2)
    final_prose = all_json[0].get("prose_report", "") if all_json else ""
    final_pdf = None
    final_clip = incident_clips[-1] if incident_clips else None
    # find latest pdf
    pdfs = sorted(Path(output_dir).glob("*.pdf"))
    if pdfs:
        final_pdf = str(pdfs[-1])

    gallery_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in all_gallery_images]
    yield final_log, gallery_rgb, final_json, final_prose, final_pdf, final_clip


# ─── Premium CSS ─────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg-0: #f7f3ea;
    --bg-1: #eef5f2;
    --ink-0: #142033;
    --ink-1: #2b3a52;
    --muted: #667085;
    --card: rgba(255, 255, 255, 0.78);
    --stroke: rgba(20, 32, 51, 0.12);
    --shadow: 0 18px 40px rgba(20, 32, 51, 0.12);
    --accent: #0f766e;
    --accent-2: #f97316;
    --accent-3: #16a34a;
    --accent-4: #38bdf8;
    --accent-5: #facc15;
    --glow: 0 24px 60px rgba(20, 32, 51, 0.18);
}

* {
    box-sizing: border-box;
}

/* ── Base ──────────────────────────────────────────────────────── */
body, .gradio-container {
    font-family: 'Space Grotesk', 'Segoe UI', Tahoma, sans-serif !important;
    background:
      radial-gradient(1200px 800px at 10% -10%, #fff6df 0%, transparent 60%),
      radial-gradient(900px 700px at 110% 10%, #e5f4ff 0%, transparent 55%),
      linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%) !important;
    color: var(--ink-0) !important;
}
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 22px 24px 44px !important;
    width: 100% !important;
    min-height: 92vh !important;
    position: relative;
}
.contain, .gap, footer { background: transparent !important; }
.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
      radial-gradient(900px 420px at 18% -10%, rgba(15,118,110,0.16), transparent 70%),
      radial-gradient(820px 520px at 120% 16%, rgba(249,115,22,0.16), transparent 70%),
      repeating-linear-gradient(90deg, rgba(20,32,51,0.04) 0 1px, transparent 1px 120px),
      repeating-linear-gradient(0deg, rgba(20,32,51,0.04) 0 1px, transparent 1px 120px),
      radial-gradient(600px 220px at 50% 100%, rgba(56,189,248,0.18), transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: gridshift 28s ease-in-out infinite;
}
.gradio-container::after {
    content: "";
    position: fixed;
    inset: -10% -10% -10% -10%;
    background:
      radial-gradient(520px 420px at 12% 30%, rgba(15,118,110,0.18), transparent 70%),
      radial-gradient(520px 420px at 88% 25%, rgba(249,115,22,0.18), transparent 70%),
      radial-gradient(520px 420px at 50% 78%, rgba(56,189,248,0.18), transparent 70%);
    filter: blur(10px);
    opacity: 0.7;
    pointer-events: none;
    z-index: 0;
    animation: blobdrift 22s ease-in-out infinite;
}
.gradio-container > * {
    position: relative;
    z-index: 1;
}

/* ── Hero ──────────────────────────────────────────────────────── */
#hero {
    background:
      linear-gradient(135deg, rgba(15,118,110,0.18), rgba(249,115,22,0.18)),
      var(--hero-bg, none),
      rgba(255,255,255,0.86);
    background-size: 100% 100%, cover, auto;
    background-position: center, center, center;
    background-blend-mode: screen, normal, normal;
    border: 1px solid var(--stroke);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 26px;
    position: relative;
    overflow: hidden;
    display: grid;
    grid-template-columns: 1.2fr 0.8fr;
    gap: 22px;
    box-shadow: var(--glow);
    backdrop-filter: blur(6px);
}
#hero::before {
    content: "";
    position: absolute;
    top: -140px;
    left: -140px;
    width: 320px;
    height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(15,118,110,0.18) 0%, transparent 70%);
    animation: float 12s ease-in-out infinite;
    pointer-events: none;
}
#hero::after {
    content: "";
    position: absolute;
    bottom: -160px;
    right: -120px;
    width: 340px;
    height: 340px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(249,115,22,0.18) 0%, transparent 70%);
    animation: float 14s ease-in-out infinite;
    pointer-events: none;
}
.hero-kicker {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(15,118,110,0.25);
    padding: 4px 10px;
    border-radius: 999px;
}
.hero-text {
    position: relative;
    z-index: 1;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--ink-0);
    margin: 8px 0 10px;
    letter-spacing: -0.4px;
}
.hero-sub {
    color: var(--ink-1);
    font-size: 0.98rem;
    line-height: 1.6;
    margin-bottom: 16px;
}
.hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.chip {
    background: linear-gradient(135deg, rgba(15,118,110,0.16), rgba(249,115,22,0.12));
    border: 1px solid rgba(15,118,110,0.25);
    color: var(--ink-1);
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
    box-shadow: 0 10px 20px rgba(20,32,51,0.08);
}
.hero-card {
    background: rgba(255,255,255,0.9);
    border: 1px solid var(--stroke);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 12px 30px rgba(15, 118, 110, 0.08);
    position: relative;
    z-index: 1;
}
.hero-side {
    display: grid;
    gap: 12px;
}
.hero-media {
    background: rgba(255,255,255,0.92);
    border: 1px solid var(--stroke);
    border-radius: 16px;
    padding: 12px;
    box-shadow: var(--shadow);
    display: grid;
    gap: 10px;
    position: relative;
    overflow: hidden;
    z-index: 0;
}
.hero-media > * { position: relative; z-index: 1; }
.hero-media::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 20% 0%, rgba(56,189,248,0.18) 0%, transparent 55%),
                radial-gradient(circle at 80% 100%, rgba(249,115,22,0.18) 0%, transparent 55%);
    pointer-events: none;
}
.hero-media .media-figure {
    border: none;
    background: transparent;
    overflow: visible;
}
.hero-media .media-figure::after { display: none; }
.hero-stack {
    position: relative;
    min-height: 230px;
    display: grid;
    align-items: center;
}
.hero-frame {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--stroke);
    background: rgba(255,255,255,0.75);
    box-shadow: 0 18px 45px rgba(20,32,51,0.18);
}
.hero-frame-main .media-img,
.hero-frame-main .placeholder {
    height: 210px;
}
.hero-frame-float {
    position: absolute;
    right: -8px;
    bottom: -12px;
    width: 60%;
    transform: rotate(2.8deg);
    box-shadow: 0 22px 50px rgba(249,115,22,0.22);
}
.hero-frame-float .media-img,
.hero-frame-float .placeholder {
    height: 140px;
}
.hero-media-title {
    font-size: 0.78rem;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 700;
}
.hero-media-caption {
    font-size: 0.82rem;
    color: var(--ink-1);
    line-height: 1.55;
}
.hero-media-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.hero-badge {
    background: linear-gradient(135deg, rgba(15,118,110,0.18), rgba(56,189,248,0.18));
    border: 1px solid rgba(15,118,110,0.35);
    color: #0b544f;
    border-radius: 999px;
    padding: 4px 8px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.9px;
    text-transform: uppercase;
}
.hero-card-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}
.hero-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px dashed rgba(20,32,51,0.12);
    color: var(--ink-1);
    font-size: 0.9rem;
}
.hero-item:last-child { border-bottom: none; }
.dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    flex-shrink: 0;
    box-shadow: 0 0 0 0 rgba(15,118,110,0.45);
    animation: pulse 3s ease-in-out infinite;
}

/* ── Two-column row ─────────────────────────────────────────────── */
.main-row { gap: 18px !important; align-items: flex-start !important; }
.left-col  { display: flex; flex-direction: column; gap: 14px; }
.right-col { display: flex; flex-direction: column; gap: 16px; }

/* ── Cards ─────────────────────────────────────────────────────── */
.card {
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: var(--shadow);
}
.card .wrap { background: transparent !important; }
.card,
.feature-card,
.media-card,
.kpi-card,
.signal-card,
.detail-card,
.use-card,
.cta-band,
.hero-card,
.hero-media {
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.feature-card:hover,
.media-card:hover,
.kpi-card:hover,
.signal-card:hover,
.detail-card:hover,
.use-card:hover,
.hero-card:hover,
.hero-media:hover {
    transform: translateY(-4px);
    box-shadow: 0 22px 50px rgba(20, 32, 51, 0.18);
    border-color: rgba(15,118,110,0.35);
}

/* ── Sections ──────────────────────────────────────────────────── */
.section {
    margin: 10px 0 22px;
}
.section-head {
    margin-bottom: 12px;
}
.section-kicker {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 700;
}
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--ink-0);
    margin: 6px 0;
    letter-spacing: -0.3px;
}
.section-sub {
    color: var(--ink-1);
    font-size: 0.94rem;
    line-height: 1.6;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
}
.feature-card {
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.feature-meta {
    font-size: 0.68rem;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 700;
}
.feature-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--ink-0);
}
.feature-desc {
    color: var(--ink-1);
    font-size: 0.88rem;
    line-height: 1.55;
}
.feature-tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--accent);
}

.media-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
}
.media-card {
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 12px;
    box-shadow: var(--shadow);
    display: grid;
    gap: 8px;
    position: relative;
    overflow: hidden;
}
.media-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(15,118,110,0.12), rgba(249,115,22,0.12));
    opacity: 0;
    transition: opacity 0.25s ease;
    pointer-events: none;
}
.media-card:hover::before { opacity: 0.2; }
.media-figure {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--stroke);
    background: rgba(255,255,255,0.75);
    position: relative;
}
.media-figure::after {
    content: "";
    position: absolute;
    inset: 0;
    background:
      linear-gradient(180deg, rgba(255,255,255,0.0) 0%, rgba(20,32,51,0.12) 100%),
      repeating-linear-gradient(0deg, rgba(255,255,255,0.08) 0 2px, transparent 2px 6px);
    opacity: 0.35;
    pointer-events: none;
}
.media-img {
    width: 100%;
    height: 180px;
    object-fit: cover;
    display: block;
    filter: saturate(1.1) contrast(1.05);
}
.media-svg {
    width: 100%;
    height: 180px;
    display: block;
}
.media-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--ink-0);
}
.media-desc {
    font-size: 0.86rem;
    color: var(--ink-1);
    line-height: 1.55;
}
.placeholder {
    width: 100%;
    height: 180px;
    border-radius: 12px;
    border: 1px dashed rgba(15,118,110,0.35);
    background: linear-gradient(135deg, rgba(15,118,110,0.15), rgba(249,115,22,0.18));
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-top: 8px;
}
.kpi-card {
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 12px;
    padding: 12px 10px;
    text-align: center;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-4));
    opacity: 0.75;
}
.kpi-val {
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--accent);
    margin-bottom: 4px;
}
.kpi-lbl {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
}
.signal-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 10px;
}
.signal-card {
    background: rgba(255,255,255,0.85);
    border: 1px solid var(--stroke);
    border-radius: 12px;
    padding: 10px 12px;
    display: grid;
    gap: 6px;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.signal-card::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-2), var(--accent), var(--accent-5));
    opacity: 0.7;
}
.signal-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--ink-0);
}
.signal-desc {
    font-size: 0.82rem;
    color: var(--ink-1);
    line-height: 1.5;
}

.detail-grid {
    display: grid;
    grid-template-columns: 1.2fr 0.8fr;
    gap: 12px;
}
.detail-card {
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: var(--shadow);
}
.detail-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--ink-0);
    margin-bottom: 8px;
}
.detail-desc {
    color: var(--ink-1);
    font-size: 0.92rem;
    line-height: 1.6;
}
.detail-list {
    display: grid;
    gap: 8px;
    margin-top: 12px;
}
.detail-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    color: var(--ink-1);
    font-size: 0.88rem;
}
.detail-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-top: 6px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    flex-shrink: 0;
}
.mini-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 14px;
}
.mini-stat {
    background: rgba(15,118,110,0.08);
    border: 1px solid rgba(15,118,110,0.2);
    border-radius: 12px;
    padding: 10px 12px;
    text-align: center;
}
.mini-val {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent);
}
.mini-lbl {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    font-weight: 700;
}

.use-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}
.use-card {
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: var(--shadow);
}
.use-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--ink-0);
    margin-bottom: 6px;
}
.use-desc {
    color: var(--ink-1);
    font-size: 0.86rem;
    line-height: 1.55;
}
.use-tag {
    display: inline-block;
    margin-top: 8px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--accent);
}

.cta-band {
    background: linear-gradient(135deg, rgba(15,118,110,0.12), rgba(249,115,22,0.12));
    border: 1px solid var(--stroke);
    border-radius: 16px;
    padding: 16px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    box-shadow: var(--shadow);
    margin: 10px 0 18px;
}
.cta-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: var(--ink-0);
}
.cta-sub {
    color: var(--ink-1);
    font-size: 0.9rem;
    margin-top: 4px;
}
.cta-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.cta-pill {
    background: rgba(15,118,110,0.1);
    border: 1px solid rgba(15,118,110,0.25);
    color: var(--ink-1);
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

.timeline {
    display: grid;
    gap: 10px;
}
.tl-step {
    display: grid;
    grid-template-columns: 38px 1fr;
    gap: 12px;
    align-items: start;
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: var(--shadow);
}
.tl-num {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #fff;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
}
.tl-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--ink-0);
}
.tl-desc {
    color: var(--ink-1);
    font-size: 0.88rem;
    line-height: 1.55;
    margin-top: 4px;
}
.tl-meta {
    color: var(--muted);
    font-size: 0.75rem;
    margin-top: 6px;
}

.split-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
}
.code-card {
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: var(--shadow);
}
.code-title {
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 700;
    margin-bottom: 8px;
}
.code-block {
    background: #0b1220;
    color: #dbeafe;
    border-radius: 10px;
    padding: 12px;
    font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
    font-size: 0.8rem;
    line-height: 1.6;
    border: 1px solid rgba(15,118,110,0.25);
    overflow-x: auto;
}

.faq {
    display: grid;
    gap: 10px;
}
.faq details {
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 12px;
    padding: 10px 12px;
    box-shadow: var(--shadow);
}
.faq summary {
    cursor: pointer;
    font-weight: 700;
    color: var(--ink-0);
    font-size: 0.92rem;
}
.faq p {
    color: var(--ink-1);
    font-size: 0.88rem;
    line-height: 1.6;
    margin: 8px 0 0;
}

.compare-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
}
.compare-card {
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: var(--shadow);
}
.compare-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--ink-0);
    margin-bottom: 6px;
}
.compare-desc {
    color: var(--ink-1);
    font-size: 0.88rem;
    line-height: 1.55;
}
.compare-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-top: 10px;
}
.compare-table th, .compare-table td {
    border-bottom: 1px solid rgba(20,32,51,0.12);
    padding: 8px 6px;
    text-align: left;
    color: var(--ink-1);
}
.compare-table th {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
}
.compare-pill {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 999px;
    background: rgba(15,118,110,0.1);
    border: 1px solid rgba(15,118,110,0.25);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: var(--accent);
}

/* ── Section heading ────────────────────────────────────────────── */
.sec-head {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(20,32,51,0.08);
    margin-bottom: 4px;
}

/* ── Video widget ───────────────────────────────────────────────── */
.vid-wrap video { border-radius: 10px !important; }

/* ── Slider ─────────────────────────────────────────────────────── */
input[type=range] { accent-color: var(--accent) !important; }

/* ── Dropdown ───────────────────────────────────────────────────── */
select, .multiselect { border-radius: 10px !important; }

/* ── Accordion ──────────────────────────────────────────────────── */
.accordion {
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 14px;
    box-shadow: var(--shadow);
    padding: 6px 10px 10px;
}

/* ── Analyze button ─────────────────────────────────────────────── */
#btn-analyze {
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    color: #fff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.6px !important;
    padding: 12px 0 !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 8px 24px rgba(15,118,110,0.22) !important;
    transition: box-shadow .2s, transform .2s !important;
    width: 100% !important;
    margin-top: 6px !important;
}
#btn-analyze:hover {
    box-shadow: 0 12px 30px rgba(15,118,110,0.32) !important;
    transform: translateY(-2px) !important;
}

/* ── Callout ───────────────────────────────────────────────────── */
.callout {
    background: rgba(15,118,110,0.08);
    border: 1px solid rgba(15,118,110,0.2);
    border-radius: 12px;
    padding: 10px 12px;
    font-size: 0.85rem;
    color: var(--ink-1);
}

/* ── How it works ───────────────────────────────────────────────── */
.howitworks {
    background: rgba(255,255,255,0.9);
    border: 1px solid var(--stroke);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: var(--shadow);
}
.hiw-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}
.hiw-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px dashed rgba(20,32,51,0.12);
    font-size: 0.86rem;
    color: var(--ink-1);
}
.hiw-step:last-child { border-bottom: none; }
.hiw-num {
    flex-shrink: 0;
    width: 20px; height: 20px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: white;
    font-size: 0.63rem;
    font-weight: 800;
    display: flex; align-items: center; justify-content: center;
}

/* ── Stat cards ─────────────────────────────────────────────────── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 12px;
}
.scard {
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: 12px;
    padding: 12px 10px;
    text-align: center;
    box-shadow: var(--shadow);
}
.scard-val {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1.1;
    margin-bottom: 4px;
}
.scard-lbl {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: var(--muted);
}
.status-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 6px;
}
.status-pill {
    background: rgba(15,118,110,0.08);
    border: 1px solid rgba(15,118,110,0.22);
    color: var(--ink-1);
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── Processing log ─────────────────────────────────────────────── */
#log-box textarea {
    font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace !important;
    font-size: 11.5px !important;
    line-height: 1.6 !important;
    background: #0b1220 !important;
    color: #b6f3d1 !important;
    border: 1px solid rgba(15,118,110,0.25) !important;
    border-radius: 10px !important;
    padding: 11px 13px !important;
}

/* ── Gallery ────────────────────────────────────────────────────── */
.gallery-wrap {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--stroke);
    background: rgba(255,255,255,0.8);
}

/* ── Tabs ───────────────────────────────────────────────────────── */
.tabs > .tab-nav > button {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: var(--ink-1) !important;
}
.tabs > .tab-nav > button.selected {
    color: #fff !important;
    border-bottom: none !important;
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    box-shadow: 0 8px 20px rgba(15,118,110,0.25) !important;
}
#top-tabs > .tab-nav {
    background: rgba(255,255,255,0.78) !important;
    border: 1px solid var(--stroke) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 6px !important;
    box-shadow: var(--shadow) !important;
    position: sticky !important;
    top: 10px;
    z-index: 5;
}
#top-tabs > .tab-nav > button {
    border-radius: 10px !important;
    padding: 8px 14px !important;
}
#top-tabs > .tabitem {
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
    min-height: 74vh !important;
}

/* ── Report textbox ─────────────────────────────────────────────── */
#report-box textarea {
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    background: rgba(255,255,255,0.9) !important;
    color: var(--ink-0) !important;
    border: 1px solid var(--stroke) !important;
    border-radius: 10px !important;
    padding: 13px 15px !important;
}

/* ── Code / JSON ────────────────────────────────────────────────── */
.code-wrap pre {
    background: #0e1524 !important;
    color: #dbeafe !important;
    font-size: 12px !important;
    border-radius: 10px !important;
}

/* ── PDF widget ─────────────────────────────────────────────────── */
.pdf-info {
    font-size: 0.8rem;
    line-height: 1.7;
    color: var(--muted);
    padding: 10px 0 0 8px;
}

/* ── Custom scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(15,118,110,0.35); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(15,118,110,0.6); }

/* ── Global label style ─────────────────────────────────────────── */
label > span, .label-wrap span {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--ink-1) !important;
    letter-spacing: 0.2px !important;
}

/* ── Animations ─────────────────────────────────────────────────── */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(12px); }
    100% { transform: translateY(0px); }
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(15,118,110,0.35); }
    70% { box-shadow: 0 0 0 10px rgba(15,118,110,0); }
    100% { box-shadow: 0 0 0 0 rgba(15,118,110,0); }
}
@keyframes blobdrift {
    0% { transform: translate3d(0, 0, 0) scale(1); }
    33% { transform: translate3d(2%, -2%, 0) scale(1.02); }
    66% { transform: translate3d(-2%, 2%, 0) scale(0.98); }
    100% { transform: translate3d(0, 0, 0) scale(1); }
}
@keyframes gridshift {
    0% { background-position: 0 0, 0 0, 0 0, 0 0, 0 0; }
    50% { background-position: -40px 20px, 30px -20px, 20px 10px, -20px -10px, 0 0; }
    100% { background-position: 0 0, 0 0, 0 0, 0 0, 0 0; }
}

/* ── Responsive ─────────────────────────────────────────────────── */
@media (max-width: 980px) {
    #hero { grid-template-columns: 1fr; }
    .hero-stack { min-height: unset; }
    .hero-frame-float {
        position: relative;
        right: 0;
        bottom: 0;
        width: 100%;
        transform: none;
        margin-top: 10px;
    }
    .stats-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .feature-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .media-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .split-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .cta-band { flex-direction: column; align-items: flex-start; }
    .detail-grid { grid-template-columns: 1fr; }
    .mini-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .use-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .signal-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
    .stats-row { grid-template-columns: 1fr; }
    .hero-frame-main .media-img,
    .hero-frame-main .placeholder { height: 180px; }
    .hero-frame-float .media-img,
    .hero-frame-float .placeholder { height: 140px; }
    .feature-grid { grid-template-columns: 1fr; }
    .media-grid { grid-template-columns: 1fr; }
    .split-grid { grid-template-columns: 1fr; }
    .tl-step { grid-template-columns: 1fr; }
    .tl-num { margin-bottom: 6px; }
    .mini-row { grid-template-columns: 1fr; }
    .use-grid { grid-template-columns: 1fr; }
    .kpi-grid { grid-template-columns: 1fr; }
    .signal-row { grid-template-columns: 1fr; }
}
"""


def build_ui():
    _proj_root = Path(__file__).resolve().parents[2]
    _shots = sorted(_proj_root.glob("Screenshot*.png"))
    _shot_a = _img_to_data_uri(_shots[0]) if len(_shots) > 0 else None
    _shot_b = _img_to_data_uri(_shots[1]) if len(_shots) > 1 else None
    _hero_img = _shot_b or _shot_a
    hero_media_img = _img_tag(_hero_img, "media-img", "Gradio UI preview")
    tour_img_a = _img_tag(_shot_a, "media-img", "Dashboard overview")
    tour_img_b = _img_tag(_shot_b, "media-img", "Report and keyframe preview")
    pipeline_svg = _pipeline_svg()
    hero_bg_style = f"--hero-bg: url('{_hero_img}')" if _hero_img else "--hero-bg: none"
    hero_stack_float = tour_img_a if _shot_a else hero_media_img
    hero_media_stack = f"""
    <div class="hero-stack">
      <div class="hero-frame hero-frame-main">{hero_media_img}</div>
      <div class="hero-frame hero-frame-float">{hero_stack_float}</div>
    </div>
    """

    with gr.Blocks(title="🚗 Local AI Dashcam Incident Explainer") as demo:
        with gr.Tabs(elem_id="top-tabs"):
            with gr.Tab("Overview"):
                # ── Hero ─────────────────────────────────────────────────
                gr.HTML(f"""
                <div id="hero" style="{hero_bg_style}">
                  <div class="hero-text">
                    <div class="hero-kicker">Local AI | Private by design | Offline</div>
                    <div class="hero-title">Local AI Dashcam Incident Explainer</div>
                    <div class="hero-sub">
                      Insurance-grade incident analysis with on-device vision and language models.
                      Upload footage and receive structured JSON, a narrative report, and a PDF package.
                    </div>
                    <div class="hero-chips">
                      <span class="chip">YOLO11 detection</span>
                      <span class="chip">SORT tracking</span>
                      <span class="chip">Optical flow</span>
                      <span class="chip">MiniCPM-V VLM</span>
                      <span class="chip">Mistral report</span>
                      <span class="chip">Privacy blur</span>
                    </div>
                  </div>
                  <div class="hero-side">
                    <div class="hero-card">
                      <div class="hero-card-title">Outputs you get</div>
                      <div class="hero-item"><span class="dot"></span>Annotated keyframes with privacy blur</div>
                      <div class="hero-item"><span class="dot"></span>Structured JSON with severity and fault analysis</div>
                      <div class="hero-item"><span class="dot"></span>Readable insurance narrative</div>
                      <div class="hero-item"><span class="dot"></span>Exportable PDF report</div>
                    </div>
                    <div class="hero-media">
                      <div class="hero-media-title">Interface preview</div>
                      <div class="media-figure">{hero_media_stack}</div>
                      <div class="hero-media-caption">
                        A live look at the local Gradio interface that produces the keyframes, JSON, and report
                        bundle without leaving your device.
                      </div>
                      <div class="hero-media-badges">
                        <span class="hero-badge">Local only</span>
                        <span class="hero-badge">PDF ready</span>
                        <span class="hero-badge">Privacy blur</span>
                      </div>
                    </div>
                  </div>
                </div>
                """)

                gr.HTML(f"""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">Visual tour</div>
                    <div class="section-title">See the pipeline at a glance</div>
                    <div class="section-sub">
                      A quick snapshot of the UI, the report outputs, and the offline pipeline
                      that connects the entire workflow end to end.
                    </div>
                  </div>
                  <div class="media-grid">
                    <div class="media-card">
                      <div class="media-figure">{tour_img_a}</div>
                      <div class="media-title">Dashboard overview</div>
                      <div class="media-desc">
                        Upload a clip, tune keyframe sampling, and monitor the pipeline in real time.
                      </div>
                    </div>
                    <div class="media-card">
                      <div class="media-figure">{tour_img_b}</div>
                      <div class="media-title">Report-ready outputs</div>
                      <div class="media-desc">
                        A clear narrative report, structured JSON, and an exportable PDF package.
                      </div>
                    </div>
                    <div class="media-card">
                      <div class="media-figure">{pipeline_svg}</div>
                      <div class="media-title">Offline pipeline</div>
                      <div class="media-desc">
                        Each phase runs locally, passing context from detection to narration to export.
                      </div>
                    </div>
                  </div>
                  <div class="kpi-grid">
                    <div class="kpi-card">
                      <div class="kpi-val">3</div>
                      <div class="kpi-lbl">Phases of AI</div>
                    </div>
                    <div class="kpi-card">
                      <div class="kpi-val">0</div>
                      <div class="kpi-lbl">Cloud calls</div>
                    </div>
                    <div class="kpi-card">
                      <div class="kpi-val">100%</div>
                      <div class="kpi-lbl">Local compute</div>
                    </div>
                    <div class="kpi-card">
                      <div class="kpi-val">PDF</div>
                      <div class="kpi-lbl">Evidence pack</div>
                    </div>
                  </div>
                  <div class="signal-row">
                    <div class="signal-card">
                      <div class="signal-title">Signals captured</div>
                      <div class="signal-desc">
                        Motion spikes, vehicle trajectories, and contextual cues are fused into one
                        structured narrative.
                      </div>
                    </div>
                    <div class="signal-card">
                      <div class="signal-title">Human readable</div>
                      <div class="signal-desc">
                        The report is designed for reviewers, instructors, and claims teams to read quickly.
                      </div>
                    </div>
                    <div class="signal-card">
                      <div class="signal-title">Audit friendly</div>
                      <div class="signal-desc">
                        JSON outputs preserve metadata, severity, and fault analysis for later checks.
                      </div>
                    </div>
                  </div>
                </section>
                """)

                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">Project details</div>
                    <div class="section-title">Built for local incident analysis</div>
                    <div class="section-sub">
                      A complete on-device pipeline that turns raw dashcam footage into a structured
                      incident narrative and a downloadable report.
                    </div>
                  </div>
                  <div class="detail-grid">
                    <div class="detail-card">
                      <div class="detail-title">What this system delivers</div>
                      <div class="detail-desc">
                        The explainer detects the incident window, samples keyframes, runs a vision-language
                        model for structured analysis, and generates a human-readable report with a PDF export.
                      </div>
                      <div class="detail-list">
                        <div class="detail-item"><span class="detail-dot"></span>Offline by default, no cloud APIs</div>
                        <div class="detail-item"><span class="detail-dot"></span>Privacy-first outputs with blur filters</div>
                        <div class="detail-item"><span class="detail-dot"></span>Audit-friendly JSON + narrative report</div>
                      </div>
                    </div>
                    <div class="detail-card">
                      <div class="detail-title">Core guarantees</div>
                      <div class="detail-desc">
                        Designed for coursework and real-world review workflows that demand local processing
                        and repeatable results.
                      </div>
                      <div class="mini-row">
                        <div class="mini-stat">
                          <div class="mini-val">100%</div>
                          <div class="mini-lbl">Local</div>
                        </div>
                        <div class="mini-stat">
                          <div class="mini-val">0</div>
                          <div class="mini-lbl">Cloud APIs</div>
                        </div>
                        <div class="mini-stat">
                          <div class="mini-val">PDF</div>
                          <div class="mini-lbl">Export</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </section>
                """)

                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">System overview</div>
                    <div class="section-title">End-to-end local pipeline</div>
                    <div class="section-sub">
                      Every phase runs on-device, from keyframe sampling to PDF export.
                      Designed for fast review, auditability, and privacy-sensitive workflows.
                    </div>
                  </div>
                  <div class="feature-grid">
                    <div class="feature-card">
                      <div class="feature-meta">Phase 1</div>
                      <div class="feature-title">Keyframe intelligence</div>
                      <div class="feature-desc">
                        Smart motion sampling reduces redundant frames while preserving the critical
                        pre-impact and post-impact context.
                      </div>
                      <div class="feature-tag">FFmpeg + motion score</div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-meta">Phase 2</div>
                      <div class="feature-title">Incident extraction</div>
                      <div class="feature-desc">
                        YOLO11 detection, SORT tracking, and optical flow spikes combine to identify
                        the exact incident window.
                      </div>
                      <div class="feature-tag">YOLO11 + SORT + flow</div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-meta">Phase 3</div>
                      <div class="feature-title">Vision-language narration</div>
                      <div class="feature-desc">
                        Sequential keyframes are summarized into structured JSON including severity,
                        conditions, and fault analysis.
                      </div>
                      <div class="feature-tag">MiniCPM-V</div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-meta">Phase 4</div>
                      <div class="feature-title">Privacy filters</div>
                      <div class="feature-desc">
                        Faces and license plates are blurred before export to protect identities
                        in the final evidence package.
                      </div>
                      <div class="feature-tag">OpenCV + heuristics</div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-meta">Phase 4</div>
                      <div class="feature-title">Insurance report export</div>
                      <div class="feature-desc">
                        ReportLab assembles the narrative, severity badge, and keyframes into a
                        professional PDF report.
                      </div>
                      <div class="feature-tag">ReportLab PDF</div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-meta">Phase 5</div>
                      <div class="feature-title">Evaluation toolkit</div>
                      <div class="feature-desc">
                        BLEU and ROUGE scoring provide benchmarking for generated narratives
                        against ground truth datasets.
                      </div>
                      <div class="feature-tag">BLEU + ROUGE</div>
                    </div>
                  </div>
                </section>
                """)

                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">Use cases</div>
                    <div class="section-title">Where this UI fits best</div>
                    <div class="section-sub">
                      Designed for anyone who needs a fast, repeatable way to summarize incidents
                      without uploading sensitive footage.
                    </div>
                  </div>
                  <div class="use-grid">
                    <div class="use-card">
                      <div class="use-title">Insurance review</div>
                      <div class="use-desc">
                        Generate standardized incident summaries and attach PDF evidence packages.
                      </div>
                      <div class="use-tag">Claims</div>
                    </div>
                    <div class="use-card">
                      <div class="use-title">Fleet safety</div>
                      <div class="use-desc">
                        Quickly triage incident clips and provide structured analysis for managers.
                      </div>
                      <div class="use-tag">Operations</div>
                    </div>
                    <div class="use-card">
                      <div class="use-title">Research & coursework</div>
                      <div class="use-desc">
                        Evaluate CV pipelines and generate narratives for academic reports.
                      </div>
                      <div class="use-tag">Academia</div>
                    </div>
                    <div class="use-card">
                      <div class="use-title">Driver training</div>
                      <div class="use-desc">
                        Review events with anonymized visuals and clear descriptions.
                      </div>
                      <div class="use-tag">Training</div>
                    </div>
                  </div>
                </section>
                """)

                gr.HTML("""
                <div class="cta-band">
                  <div>
                    <div class="cta-title">Run a full incident analysis locally</div>
                    <div class="cta-sub">
                      Switch to the Analyze tab to upload a dashcam clip and generate structured JSON,
                      a narrative report, and a downloadable PDF package.
                    </div>
                  </div>
                  <div class="cta-actions">
                    <span class="cta-pill">No cloud</span>
                    <span class="cta-pill">Offline models</span>
                    <span class="cta-pill">Audit-friendly</span>
                  </div>
                </div>
                """)

                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">Evidence package</div>
                    <div class="section-title">What you export per incident</div>
                    <div class="section-sub">
                      Each run produces a consistent bundle of artifacts that can be shared with
                      reviewers, instructors, or claims teams.
                    </div>
                  </div>
                  <div class="split-grid">
                    <div class="feature-card">
                      <div class="feature-title">Incident clip</div>
                      <div class="feature-desc">
                        A short, focused window around the event, extracted automatically from the
                        original footage.
                      </div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-title">Structured JSON</div>
                      <div class="feature-desc">
                        Machine-readable output with severity, conditions, fault analysis, and
                        timeline metadata.
                      </div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-title">PDF report</div>
                      <div class="feature-desc">
                        A professional narrative report with keyframes, severity badge, and
                        summary findings.
                      </div>
                    </div>
                  </div>
                </section>
                """)

            with gr.Tab("Architecture"):
                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">Architecture</div>
                    <div class="section-title">Five phases, one pipeline</div>
                    <div class="section-sub">
                      A practical flow from raw dashcam footage to a complete, privacy-safe
                      report package.
                    </div>
                  </div>
                  <div class="timeline">
                    <div class="tl-step">
                      <div class="tl-num">1</div>
                      <div>
                        <div class="tl-title">Keyframe extraction</div>
                        <div class="tl-desc">
                          Extract uniform or motion-triggered frames to reduce compute cost while
                          preserving the decisive moments.
                        </div>
                        <div class="tl-meta">Outputs: keyframes, motion score</div>
                      </div>
                    </div>
                    <div class="tl-step">
                      <div class="tl-num">2</div>
                      <div>
                        <div class="tl-title">Incident detection</div>
                        <div class="tl-desc">
                          YOLO11 and SORT track vehicles while optical flow detects sudden velocity spikes.
                        </div>
                        <div class="tl-meta">Outputs: incident clip window</div>
                      </div>
                    </div>
                    <div class="tl-step">
                      <div class="tl-num">3</div>
                      <div>
                        <div class="tl-title">VLM narration</div>
                        <div class="tl-desc">
                          MiniCPM-V produces structured JSON with severity, fault, and scene conditions.
                        </div>
                        <div class="tl-meta">Outputs: structured JSON</div>
                      </div>
                    </div>
                    <div class="tl-step">
                      <div class="tl-num">4</div>
                      <div>
                        <div class="tl-title">Privacy filters</div>
                        <div class="tl-desc">
                          Faces and license plates are blurred before any visual output is shown or exported.
                        </div>
                        <div class="tl-meta">Outputs: anonymized keyframes</div>
                      </div>
                    </div>
                    <div class="tl-step">
                      <div class="tl-num">5</div>
                      <div>
                        <div class="tl-title">Report export</div>
                        <div class="tl-desc">
                          Mistral generates the narrative and ReportLab assembles the final PDF package.
                        </div>
                        <div class="tl-meta">Outputs: PDF report</div>
                      </div>
                    </div>
                  </div>
                </section>
                """)

                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">Model stack</div>
                    <div class="section-title">CV + VLM + LLM layers</div>
                    <div class="section-sub">
                      Classical CV handles detection and motion. Vision-language models add semantic
                      understanding. A local LLM generates the report.
                    </div>
                  </div>
                  <div class="split-grid">
                    <div class="feature-card">
                      <div class="feature-title">YOLO11 detector</div>
                      <div class="feature-desc">
                        Object detection on vehicles and pedestrians to anchor the incident window.
                      </div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-title">SORT tracking</div>
                      <div class="feature-desc">
                        Frame-to-frame tracking and IoU-based association to detect interactions.
                      </div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-title">MiniCPM-V + Mistral</div>
                      <div class="feature-desc">
                        Structured JSON from the VLM and prose narrative from the local LLM.
                      </div>
                    </div>
                  </div>
                </section>
                """)

                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">Evaluation</div>
                    <div class="section-title">Quality and reliability</div>
                    <div class="section-sub">
                      Outputs can be benchmarked using BLEU and ROUGE to compare against ground-truth
                      narratives.
                    </div>
                  </div>
                  <div class="feature-grid">
                    <div class="feature-card">
                      <div class="feature-meta">Metric</div>
                      <div class="feature-title">BLEU-4</div>
                      <div class="feature-desc">
                        Measures n-gram overlap between generated and reference narratives.
                      </div>
                      <div class="feature-tag">Precision</div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-meta">Metric</div>
                      <div class="feature-title">ROUGE-L</div>
                      <div class="feature-desc">
                        Captures longest common subsequence for recall-oriented scoring.
                      </div>
                      <div class="feature-tag">Recall</div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-meta">Study</div>
                      <div class="feature-title">Ablation tests</div>
                      <div class="feature-desc">
                        Compare 1/3/5 keyframe strategies to validate the tradeoff.
                      </div>
                      <div class="feature-tag">Robustness</div>
                    </div>
                  </div>
                </section>
                """)

            with gr.Tab("Privacy"):
                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">Privacy and safety</div>
                    <div class="section-title">Designed for sensitive footage</div>
                    <div class="section-sub">
                      The pipeline blurs identities, keeps data local, and produces evidence that can be
                      shared without exposing raw footage.
                    </div>
                  </div>
                  <div class="split-grid">
                    <div class="feature-card">
                      <div class="feature-title">Face blur</div>
                      <div class="feature-desc">
                        OpenCV DNN detects driver and pedestrian faces for automatic anonymization.
                      </div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-title">License plate blur</div>
                      <div class="feature-desc">
                        Vehicle ROIs are scanned to mask plates without an external ALPR model.
                      </div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-title">Offline guarantee</div>
                      <div class="feature-desc">
                        All inference is local. No cloud APIs or network calls are required.
                      </div>
                    </div>
                  </div>
                </section>
                """)

                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">Threat model</div>
                    <div class="section-title">Protecting sensitive identities</div>
                    <div class="section-sub">
                      Privacy filtering is applied before any visual output is displayed or exported.
                    </div>
                  </div>
                  <div class="use-grid">
                    <div class="use-card">
                      <div class="use-title">Face anonymization</div>
                      <div class="use-desc">
                        DNN-based face detection with Gaussian blur to protect drivers and pedestrians.
                      </div>
                      <div class="use-tag">Faces</div>
                    </div>
                    <div class="use-card">
                      <div class="use-title">Plate masking</div>
                      <div class="use-desc">
                        Vehicle ROI heuristics to blur plates without external ALPR calls.
                      </div>
                      <div class="use-tag">Plates</div>
                    </div>
                    <div class="use-card">
                      <div class="use-title">Local storage</div>
                      <div class="use-desc">
                        All outputs are stored locally under outputs/ with timestamped folders.
                      </div>
                      <div class="use-tag">Storage</div>
                    </div>
                    <div class="use-card">
                      <div class="use-title">No telemetry</div>
                      <div class="use-desc">
                        No analytics or external logging is required for operation.
                      </div>
                      <div class="use-tag">Offline</div>
                    </div>
                  </div>
                </section>
                """)

            with gr.Tab("Workflow"):
                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">Workflow</div>
                    <div class="section-title">CLI and file outputs</div>
                    <div class="section-sub">
                      The UI is ideal for quick review, while the CLI is suited for batch runs and
                      repeatable experiments.
                    </div>
                  </div>
                  <div class="split-grid">
                    <div class="code-card">
                      <div class="code-title">Quick start</div>
                      <div class="code-block">
                        source .venv/bin/activate<br/>
                        python3 src/phase4/app.py<br/>
                        # open http://localhost:7860
                      </div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-title">Generated assets</div>
                      <div class="feature-desc">
                        - Incident clips<br/>
                        - Annotated keyframes<br/>
                        - Structured JSON report<br/>
                        - Insurance narrative<br/>
                        - PDF export package
                      </div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-title">Model stack</div>
                      <div class="feature-desc">
                        - YOLO11 detection<br/>
                        - SORT tracking<br/>
                        - MiniCPM-V narration<br/>
                        - Mistral report writing
                      </div>
                    </div>
                  </div>
                </section>
                """)

            with gr.Tab("Research Compare"):
                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">Research comparison</div>
                    <div class="section-title">How this project compares</div>
                    <div class="section-sub">
                      This section contrasts your end-to-end local pipeline against research that
                      focuses on detection or anticipation. It highlights where your system is
                      strongest: local processing, privacy filters, and report-ready outputs.
                    </div>
                  </div>
                  <div class="compare-grid">
                    <div class="compare-card">
                      <div class="compare-title">Your system strengths</div>
                      <div class="compare-desc">
                        Your project is designed for complete incident analysis workflows, not just
                        detection. The emphasis is on local processing and shareable evidence.
                      </div>
                      <div class="detail-list">
                        <div class="detail-item"><span class="detail-dot"></span>Fully local pipeline (no cloud APIs)</div>
                        <div class="detail-item"><span class="detail-dot"></span>Privacy blurring built into outputs</div>
                        <div class="detail-item"><span class="detail-dot"></span>Structured JSON + PDF report export</div>
                        <div class="detail-item"><span class="detail-dot"></span>Evaluated with BLEU/ROUGE metrics</div>
                      </div>
                    </div>
                    <div class="compare-card">
                      <div class="compare-title">Why it is best (context-specific)</div>
                      <div class="compare-desc">
                        Compared with research papers that focus on anomaly detection or accident anticipation,
                        your system provides end-to-end, privacy-preserving reporting. That makes it better for
                        real-world review workflows where a report must be generated, shared, and archived.
                      </div>
                      <div class="detail-list">
                        <div class="detail-item"><span class="detail-dot"></span>Produces analyst-ready PDF + JSON outputs</div>
                        <div class="detail-item"><span class="detail-dot"></span>Works fully offline on local hardware</div>
                        <div class="detail-item"><span class="detail-dot"></span>Includes privacy safeguards by default</div>
                        <div class="detail-item"><span class="detail-dot"></span>Extends beyond detection to narrative reporting</div>
                      </div>
                    </div>
                  </div>

                  <div class="compare-card" style="margin-top:12px;">
                    <div class="compare-title">Comparison matrix (fill this in)</div>
                    <div class="compare-desc">
                      The entries below list common research baselines and how they differ from your
                      end-to-end pipeline. Update the evaluation column with your measured results
                      if you run direct comparisons.
                    </div>
                    <table class="compare-table">
                      <thead>
                        <tr>
                          <th>System</th>
                          <th>Offline</th>
                          <th>Privacy</th>
                          <th>Outputs</th>
                          <th>Evaluation</th>
                          <th>Notes</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td><span class="compare-pill">Your system</span></td>
                          <td>Yes</td>
                          <td>Yes</td>
                          <td>JSON + PDF + clip</td>
                          <td>BLEU/ROUGE</td>
                          <td>End-to-end local pipeline</td>
                        </tr>
                        <tr>
                          <td>
                            <a href="https://doi.org/10.1109/TPAMI.2022.3150763" target="_blank" rel="noopener">
                              DoTA: Unsupervised Detection of Traffic Anomaly (TPAMI 2023)
                            </a>
                          </td>
                          <td>Not specified</td>
                          <td>No</td>
                          <td>Anomaly localization</td>
                          <td>STAUC (spatio-temporal)</td>
                          <td>Focuses on anomaly detection, not report generation</td>
                        </tr>
                        <tr>
                          <td>
                            <a href="https://dl.acm.org/doi/10.1145/3394171.3413827" target="_blank" rel="noopener">
                              Uncertainty-based Accident Anticipation (MM 2020)
                            </a>
                          </td>
                          <td>Not specified</td>
                          <td>No</td>
                          <td>Accident probability / anticipation</td>
                          <td>Accident anticipation metrics</td>
                          <td>Predicts accidents early; no privacy or PDF outputs</td>
                        </tr>
                        <tr>
                          <td>
                            <a href="https://arxiv.org/abs/1904.12634" target="_blank" rel="noopener">
                              DADA-2000 Attention Benchmark (ITSC 2019)
                            </a>
                          </td>
                          <td>Not specified</td>
                          <td>No</td>
                          <td>Attention + accident prediction</td>
                          <td>Attention + anticipation benchmarks</td>
                          <td>Dataset + attention focus, not end-to-end reporting</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </section>
                """)

                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">Command line</div>
                    <div class="section-title">Batch-friendly workflows</div>
                    <div class="section-sub">
                      Automate evaluation or run the full pipeline on multiple clips.
                    </div>
                  </div>
                  <div class="split-grid">
                    <div class="code-card">
                      <div class="code-title">Full pipeline</div>
                      <div class="code-block">
                        python3 src/phase3/pipeline.py \\
                        --video data/samples/test.mp4 \\
                        --keyframes 3
                      </div>
                    </div>
                    <div class="code-card">
                      <div class="code-title">Evaluation</div>
                      <div class="code-block">
                        python3 src/phase5/evaluate.py ablation \\
                        --video data/samples/test.mp4 \\
                        --vlm minicpm-v
                      </div>
                    </div>
                    <div class="feature-card">
                      <div class="feature-title">Folder outputs</div>
                      <div class="feature-desc">
                        Each run writes to outputs/ by timestamp for consistent auditing.
                      </div>
                    </div>
                  </div>
                </section>
                """)

            with gr.Tab("FAQ"):
                gr.HTML("""
                <section class="section">
                  <div class="section-head">
                    <div class="section-kicker">FAQ</div>
                    <div class="section-title">Common questions</div>
                    <div class="section-sub">
                      Answers for setup, performance, and privacy.
                    </div>
                  </div>
                  <div class="faq">
                    <details>
                      <summary>Do I need internet access to run this?</summary>
                      <p>No. All models run locally once downloaded by Ollama.</p>
                    </details>
                    <details>
                      <summary>What hardware is recommended?</summary>
                      <p>8 GB RAM minimum, 16 GB recommended. MPS or CUDA is used when available.</p>
                    </details>
                    <details>
                      <summary>Can I process long videos?</summary>
                      <p>Yes. The incident extractor trims the relevant window so only short clips are analyzed.</p>
                    </details>
                    <details>
                      <summary>Where are outputs stored?</summary>
                      <p>Reports and keyframes are written under the outputs/ directory by timestamp.</p>
                    </details>
                    <details>
                      <summary>Which models are supported?</summary>
                      <p>MiniCPM-V for vision-language and Mistral for prose by default, with optional alternatives.</p>
                    </details>
                    <details>
                      <summary>Can I skip incident detection?</summary>
                      <p>Yes. Use the skip detection toggle to process the full video directly.</p>
                    </details>
                  </div>
                </section>
                """)

            with gr.Tab("Analyze"):
                # ── Main two-column layout ───────────────────────────────
                with gr.Row(elem_classes="main-row"):

                    # ═══ LEFT COLUMN ═════════════════════════════════════
                    with gr.Column(scale=4, elem_classes="left-col"):

                        # Section heading
                        gr.HTML('<div class="sec-head">⚙️ &nbsp;Configuration</div>')

                        _real  = _proj_root / "data/samples/real_dashcam.mp4"
                        _synth = _proj_root / "data/samples/test_dashcam_h264.mp4"
                        _sample = str(_real if _real.exists() else _synth if _synth.exists() else "")

                        video_input = gr.Video(
                            label="🎬 Dashcam Video",
                            height=220,
                            value=_sample or None,
                            elem_classes=["vid-wrap", "card"],
                        )

                        n_keyframes = gr.Slider(
                            label="📷 Keyframes per Incident",
                            minimum=1, maximum=5, step=2, value=3,
                            info="1 = fast | 3 = balanced | 5 = detailed",
                            elem_classes="card",
                        )

                        examples = []
                        if _real.exists():
                            examples.append([str(_real)])
                        if _synth.exists():
                            examples.append([str(_synth)])
                        if examples:
                            gr.Examples(
                                examples=examples,
                                inputs=[video_input],
                                label="Try a sample clip",
                            )

                        with gr.Accordion("Advanced Settings", open=False, elem_classes="accordion"):
                            vlm_model = gr.Dropdown(
                                label="🔭 Vision Model (VLM)",
                                choices=["minicpm-v", "internvl2", "llava", "bakllava"],
                                value="minicpm-v",
                                info="Must be pulled via Ollama",
                            )

                            llm_model = gr.Dropdown(
                                label="✍️ Report LLM",
                                choices=["mistral", "llama3", "gemma2", "phi3"],
                                value="mistral",
                                info="For prose report generation",
                            )

                            skip_det = gr.Checkbox(
                                label="⏭  Skip incident detection - process full video",
                                value=False,
                            )

                        gr.HTML("""
                        <div class="callout">
                          Fully local by default. No data leaves your device.
                        </div>
                        """)

                        submit_btn = gr.Button(
                            "🔍  Analyze Incident",
                            variant="primary",
                            elem_id="btn-analyze",
                            size="lg",
                        )

                        # How it works
                        gr.HTML("""
                        <div class="howitworks">
                          <div class="hiw-title">How It Works</div>
                          <div class="hiw-step"><div class="hiw-num">1</div>Upload your dashcam video</div>
                          <div class="hiw-step"><div class="hiw-num">2</div>YOLO11 + optical flow detects incidents</div>
                          <div class="hiw-step"><div class="hiw-num">3</div>Keyframes sampled: pre / impact / post</div>
                          <div class="hiw-step"><div class="hiw-num">4</div>Local VLM analyses frames → structured JSON</div>
                          <div class="hiw-step"><div class="hiw-num">5</div>Mistral LLM writes insurance prose report</div>
                          <div class="hiw-step"><div class="hiw-num">6</div>Faces and licence plates blurred</div>
                          <div class="hiw-step"><div class="hiw-num">7</div>PDF exported fully offline</div>
                        </div>
                        """)

                    # ═══ RIGHT COLUMN ═══════════════════════════════════
                    with gr.Column(scale=7, elem_classes="right-col"):

                        # Section heading
                        gr.HTML('<div class="sec-head">📊 &nbsp;Analysis Results</div>')

                        # Stat cards
                        gr.HTML("""
                        <div class="stats-row">
                          <div class="scard">
                            <div class="scard-val">—</div>
                            <div class="scard-lbl">Incidents</div>
                          </div>
                          <div class="scard">
                            <div class="scard-val">—</div>
                            <div class="scard-lbl">Max Severity</div>
                          </div>
                          <div class="scard">
                            <div class="scard-val">—</div>
                            <div class="scard-lbl">Inference Time</div>
                          </div>
                          <div class="scard">
                            <div class="scard-val">—</div>
                            <div class="scard-lbl">Confidence</div>
                          </div>
                        </div>
                        <div class="status-strip">
                          <span class="status-pill">Offline mode</span>
                          <span class="status-pill">Privacy blur</span>
                          <span class="status-pill">PDF export</span>
                          <span class="status-pill">No cloud</span>
                        </div>
                        """)

                        # Live log
                        log_box = gr.Textbox(
                            label="⚡ Processing Log",
                            lines=7,
                            max_lines=10,
                            elem_id="log-box",
                            elem_classes="card",
                            placeholder="Pipeline output will appear here in real-time...",
                        )

                        # Keyframe gallery
                        gallery = gr.Gallery(
                            label="🖼  Annotated Keyframes (Privacy-Blurred)",
                            show_label=True,
                            height=260,
                            columns=3,
                            object_fit="contain",
                            preview=True,
                            elem_classes="gallery-wrap",
                        )

                        incident_clip = gr.Video(
                            label="🎞  Incident Clip Preview",
                            height=220,
                            interactive=False,
                            elem_classes="card",
                        )

                        # Output tabs
                        with gr.Tabs():
                            with gr.Tab("📝 Incident Report"):
                                prose_output = gr.Textbox(
                                    label="Insurance Narrative",
                                    lines=10,
                                    elem_id="report-box",
                                    elem_classes="card",
                                    placeholder="The AI-generated insurance report will appear here after analysis...",
                                )
                            with gr.Tab("🗂 Structured JSON"):
                                json_output = gr.Code(
                                    label="VLM Structured Analysis",
                                    language="json",
                                    lines=18,
                                    elem_classes="card",
                                )

                        # PDF download + info
                        with gr.Row():
                            pdf_download = gr.File(
                                label="📄 Download PDF Report",
                                interactive=False,
                                scale=3,
                                elem_classes="card",
                            )
                            gr.HTML("""
                            <div class="pdf-info">
                              PDF includes:<br>
                              - Severity badge<br>
                              - Annotated keyframes<br>
                              - Fault and liability analysis<br>
                              - Full prose narrative
                            </div>
                            """)

        # ── Footer ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; margin-top:32px;
                    color:rgba(110,128,168,0.4); font-size:0.73rem; letter-spacing:0.3px;">
          100% local | No data leaves your device | Built with YOLO11, Ollama and Gradio
        </div>
        """)

        # ── Event binding ──────────────────────────────────────────────────
        submit_btn.click(
            fn=process_video,
            inputs=[video_input, n_keyframes, vlm_model, llm_model, skip_det],
            outputs=[log_box, gallery, json_output, prose_output, pdf_download, incident_clip],
            show_progress="full",
        )

    return demo


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.green,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Space Grotesk"),
        ),
    )
