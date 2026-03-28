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
    Yields (log_text, gallery_images, json_text, prose_text, pdf_path)
    """
    if video_file is None:
        yield "❌ Please upload a dashcam video.", [], "{}", "", None
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
    yield log("🔍 Step 1/4: Detecting incidents..."), [], "{}", "", None

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

    yield "\n".join(log_lines), [], "{}", "", None

    all_json: list[dict] = []
    all_gallery_images: list[np.ndarray] = []

    for idx, clip_path in enumerate(incident_clips, 1):
        yield log(f"\n🎞  Incident {idx}: {Path(clip_path).name}"), [], "{}", "", None

        # ── Step 2: Keyframe Sampling ────────────────────────────────────
        log(f"  📷 Step 2/4: Sampling {n_keyframes} keyframes...")
        yield "\n".join(log_lines), [], "{}", "", None

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
        yield "\n".join(log_lines), gallery_rgb, "{}", "", None

        # ── Step 3: VLM Narration ────────────────────────────────────────
        log(f"  🤖 Step 3/4: VLM narration ({vlm_model})...")
        yield "\n".join(log_lines), gallery_rgb, "{}", "", None

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
        yield "\n".join(log_lines), gallery_rgb, json.dumps(incident_json, indent=2), "", None

        # ── Step 4: Report Generation + PDF ─────────────────────────────
        log(f"  📝 Step 4/4: Generating report ({llm_model})...")
        yield "\n".join(log_lines), gallery_rgb, json.dumps(incident_json, indent=2), "", None

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

        yield "\n".join(log_lines), gallery_rgb, best_json_text, best_prose, pdf_path

    final_log = log(f"\n✅ Done! {len(all_json)} incident(s) processed.")
    final_json = json.dumps(all_json[0] if all_json else {}, indent=2)
    final_prose = all_json[0].get("prose_report", "") if all_json else ""
    final_pdf = None
    # find latest pdf
    pdfs = sorted(Path(output_dir).glob("*.pdf"))
    if pdfs:
        final_pdf = str(pdfs[-1])

    gallery_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in all_gallery_images]
    yield final_log, gallery_rgb, final_json, final_prose, final_pdf


# ─── Premium CSS ─────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ──────────────────────────────────────────────────────── */
body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #07090f !important;
}
.gradio-container {
    max-width: 1360px !important;
    margin: 0 auto !important;
    padding: 20px 24px 40px !important;
}
/* Remove Gradio's default white/light background from inner containers */
.contain, .gap, footer { background: transparent !important; }

/* ── Hero Header ───────────────────────────────────────────────── */
#hero-header {
    background: linear-gradient(135deg, #0b1023 0%, #0e1535 50%, #180924 100%);
    border: 1px solid rgba(130,70,230,0.22);
    border-radius: 18px;
    padding: 36px 48px 30px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    text-align: center;
}
#hero-header::before {
    content:''; position:absolute; top:-80px; left:-60px;
    width:240px; height:240px; border-radius:50%;
    background: radial-gradient(circle, rgba(233,69,96,.16) 0%, transparent 70%);
    pointer-events:none;
}
#hero-header::after {
    content:''; position:absolute; bottom:-90px; right:-50px;
    width:280px; height:280px; border-radius:50%;
    background: radial-gradient(circle, rgba(88,101,242,.13) 0%, transparent 70%);
    pointer-events:none;
}
#hero-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #e94560;           /* solid fallback */
    background: linear-gradient(90deg, #e94560 0%, #c084fc 55%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.4px;
    margin-bottom: 8px;
    line-height: 1.2;
}
#hero-header .sub {
    color: rgba(175,190,225,0.7);
    font-size: 0.92rem;
    letter-spacing: 0.2px;
    margin-bottom: 18px;
}
.badge-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
}
.tbadge {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    color: rgba(195,210,240,0.88);
    border-radius: 20px;
    padding: 4px 13px;
    font-size: 0.77rem;
    font-weight: 500;
    white-space: nowrap;
}

/* ── Two-column row ─────────────────────────────────────────────── */
/* Gradio .gap class controls the row gap */
.main-row { gap: 20px !important; align-items: flex-start !important; }
.left-col  { display: flex; flex-direction: column; gap: 14px; }
.right-col { display: flex; flex-direction: column; gap: 16px; }

/* ── Panel wrapper (replaces gr.Group which breaks ID) ─────────── */
.panel {
    background: rgba(11,16,32,0.92);
    border: 1px solid rgba(90,100,240,0.18);
    border-radius: 14px;
    padding: 20px 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.panel-right {
    background: rgba(9,13,26,0.88);
    border: 1px solid rgba(120,60,220,0.15);
    border-radius: 14px;
    padding: 20px 20px;
}

/* ── Section heading ────────────────────────────────────────────── */
.sec-head {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: rgba(140,158,200,0.55);
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 4px;
}

/* ── Video widget ───────────────────────────────────────────────── */
.vid-wrap video { border-radius: 10px !important; }

/* ── Slider ─────────────────────────────────────────────────────── */
input[type=range] { accent-color: #e94560 !important; }

/* ── Dropdown ───────────────────────────────────────────────────── */
select, .multiselect { border-radius: 8px !important; }

/* ── Analyze button ─────────────────────────────────────────────── */
#btn-analyze {
    background: linear-gradient(135deg, #e94560, #b5172f) !important;
    color: #fff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.6px !important;
    padding: 13px 0 !important;
    border-radius: 11px !important;
    border: none !important;
    box-shadow: 0 4px 22px rgba(233,69,96,.38) !important;
    transition: box-shadow .2s, transform .2s !important;
    width: 100% !important;
    margin-top: 4px !important;
}
#btn-analyze:hover {
    box-shadow: 0 7px 28px rgba(233,69,96,.55) !important;
    transform: translateY(-2px) !important;
}

/* ── How it works ───────────────────────────────────────────────── */
.howitworks {
    background: rgba(8,12,24,0.7);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px 16px;
}
.hiw-title {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: rgba(140,158,200,0.5);
    margin-bottom: 10px;
}
.hiw-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.82rem;
    color: rgba(180,192,220,0.82);
}
.hiw-step:last-child { border-bottom: none; }
.hiw-num {
    flex-shrink: 0;
    width: 20px; height: 20px;
    border-radius: 50%;
    background: linear-gradient(135deg, #e94560, #7c3aed);
    color: white;
    font-size: 0.63rem;
    font-weight: 800;
    display: flex; align-items: center; justify-content: center;
}

/* ── Stat cards ─────────────────────────────────────────────────── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 14px;
}
.scard {
    background: rgba(11,16,32,0.94);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 11px;
    padding: 13px 10px;
    text-align: center;
}
.scard-val {
    font-size: 1.55rem;
    font-weight: 800;
    color: #e94560;   /* solid fallback so text shows on all browsers */
    line-height: 1.1;
    margin-bottom: 4px;
}
.scard-lbl {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: rgba(130,148,190,0.55);
}

/* ── Processing log ─────────────────────────────────────────────── */
#log-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 11.5px !important;
    line-height: 1.6 !important;
    background: #01030a !important;
    color: #7ee787 !important;
    border: 1px solid rgba(0,200,80,0.14) !important;
    border-radius: 9px !important;
    padding: 11px 13px !important;
}

/* ── Gallery ────────────────────────────────────────────────────── */
.gallery-wrap { border-radius: 10px; overflow: hidden; }

/* ── Tabs ───────────────────────────────────────────────────────── */
.tabs > .tab-nav > button {
    font-size: 0.84rem !important;
    font-weight: 600 !important;
}
.tabs > .tab-nav > button.selected {
    color: #e94560 !important;
    border-bottom: 2px solid #e94560 !important;
    background: rgba(233,69,96,0.08) !important;
}

/* ── Report textbox ─────────────────────────────────────────────── */
#report-box textarea {
    font-size: 0.9rem !important;
    line-height: 1.8 !important;
    background: rgba(4,7,16,0.8) !important;
    color: rgba(210,222,245,0.9) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 9px !important;
    padding: 13px 15px !important;
}

/* ── Code / JSON ────────────────────────────────────────────────── */
.code-wrap pre {
    background: #01030a !important;
    font-size: 12px !important;
    border-radius: 9px !important;
}

/* ── PDF widget ─────────────────────────────────────────────────── */
.pdf-info {
    font-size: 0.77rem;
    line-height: 1.7;
    color: rgba(140,155,195,0.5);
    padding: 10px 0 0 8px;
}

/* ── Custom scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(233,69,96,.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(233,69,96,.55);}

/* ── Global label style ─────────────────────────────────────────── */
label > span, .label-wrap span {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: rgba(168,182,218,0.75) !important;
    letter-spacing: 0.2px !important;
}
"""


def build_ui():
    with gr.Blocks(title="🚗 Local AI Dashcam Incident Explainer") as demo:

        # ── Hero ───────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="hero-header">
          <h1>🚗 Local AI Dashcam Incident Explainer</h1>
          <div class="sub">100% on-device · Privacy-preserving · Insurance-grade reports</div>
          <div class="badge-row">
            <span class="tbadge">⚡ YOLO11</span>
            <span class="tbadge">🔀 SORT Tracker</span>
            <span class="tbadge">🌊 Optical Flow</span>
            <span class="tbadge">🤖 MiniCPM-V VLM</span>
            <span class="tbadge">📝 Mistral LLM</span>
            <span class="tbadge">🔒 Privacy Blurring</span>
          </div>
        </div>
        """)

        # ── Main two-column layout ─────────────────────────────────────────
        with gr.Row(elem_classes="main-row"):

            # ═══ LEFT COLUMN ══════════════════════════════════════════════
            with gr.Column(scale=4, elem_classes="left-col"):

                # Section heading
                gr.HTML('<div class="sec-head">⚙️ &nbsp;Configuration</div>')

                # Config card
                gr.HTML('<div class="panel">', visible=False)   # open marker only for reference

                _proj_root = Path(__file__).resolve().parents[2]
                _real  = _proj_root / "data/samples/real_dashcam.mp4"
                _synth = _proj_root / "data/samples/test_dashcam_h264.mp4"
                _sample = str(_real if _real.exists() else _synth if _synth.exists() else "")

                video_input = gr.Video(
                    label="🎬 Dashcam Video",
                    height=220,
                    value=_sample or None,
                    elem_classes="vid-wrap",
                )

                n_keyframes = gr.Slider(
                    label="📷 Keyframes per Incident",
                    minimum=1, maximum=5, step=2, value=3,
                    info="1 = fast  ·  3 = balanced  ·  5 = detailed",
                )

                # Stacked dropdowns (NOT side-by-side → fixes cramping)
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
                    label="⏭  Skip incident detection — process full video",
                    value=False,
                )

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
                  <div class="hiw-step"><div class="hiw-num">3</div>Keyframes sampled: pre · impact · post</div>
                  <div class="hiw-step"><div class="hiw-num">4</div>Local VLM analyses frames → structured JSON</div>
                  <div class="hiw-step"><div class="hiw-num">5</div>Mistral LLM writes insurance prose report</div>
                  <div class="hiw-step"><div class="hiw-num">6</div>Faces &amp; licence plates blurred</div>
                  <div class="hiw-step"><div class="hiw-num">7</div>PDF exported — fully offline</div>
                </div>
                """)

            # ═══ RIGHT COLUMN ═════════════════════════════════════════════
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
                    <div class="scard-lbl">Infer Time</div>
                  </div>
                  <div class="scard">
                    <div class="scard-val">—</div>
                    <div class="scard-lbl">Confidence</div>
                  </div>
                </div>
                """)

                # Live log
                log_box = gr.Textbox(
                    label="⚡ Processing Log",
                    lines=7,
                    max_lines=10,
                    elem_id="log-box",
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

                # Output tabs
                with gr.Tabs():
                    with gr.Tab("📝 Incident Report"):
                        prose_output = gr.Textbox(
                            label="Insurance Narrative",
                            lines=10,
                            elem_id="report-box",
                            placeholder="The AI-generated insurance report will appear here after analysis...",
                        )
                    with gr.Tab("🗂 Structured JSON"):
                        json_output = gr.Code(
                            label="VLM Structured Analysis",
                            language="json",
                            lines=18,
                        )

                # PDF download + info
                with gr.Row():
                    pdf_download = gr.File(
                        label="📄 Download PDF Report",
                        interactive=False,
                        scale=3,
                    )
                    gr.HTML("""
                    <div class="pdf-info">
                      PDF includes:<br>
                      • Severity badge<br>
                      • Annotated keyframes<br>
                      • Fault &amp; liability analysis<br>
                      • Full prose narrative
                    </div>
                    """)

        # ── Footer ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; margin-top:32px;
                    color:rgba(110,128,168,0.4); font-size:0.73rem; letter-spacing:0.3px;">
          🔒 100% local &nbsp;·&nbsp; No data leaves your device &nbsp;·&nbsp;
          Built with YOLO11, Ollama &amp; Gradio
        </div>
        """)

        # ── Event binding ──────────────────────────────────────────────────
        submit_btn.click(
            fn=process_video,
            inputs=[video_input, n_keyframes, vlm_model, llm_model, skip_det],
            outputs=[log_box, gallery, json_output, prose_output, pdf_download],
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
            primary_hue=gr.themes.colors.red,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
    )
