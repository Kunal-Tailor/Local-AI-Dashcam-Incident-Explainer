"""
Phase 4 — PDF Report Exporter
Generates a professional, structured PDF incident report using ReportLab.

Includes:
  - Cover section with incident metadata + severity badge
  - Annotated keyframe thumbnails (YOLO overlays if present)
  - Structured data table (vehicles, conditions, fault)
  - Prose report paragraph
  - Confidence flag banner (if VLM uncertain)

Usage:
    from src.phase4.pdf_exporter import PDFExporter
    exporter = PDFExporter()
    exporter.export(incident_json, keyframe_paths, output_path="outputs/reports/report.pdf")
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


# ─── Colour palette ──────────────────────────────────────────────────────────

DARK_BG    = colors.HexColor("#1a1a2e")
ACCENT     = colors.HexColor("#e94560")
LIGHT_GREY = colors.HexColor("#f0f0f0")
MID_GREY   = colors.HexColor("#888888")
WARN_AMBER = colors.HexColor("#ff9800")
OK_GREEN   = colors.HexColor("#4caf50")
TEXT_DARK  = colors.HexColor("#1a1a1a")
WHITE      = colors.white


# ─── Severity badge colours ──────────────────────────────────────────────────

SEVERITY_COLORS = {
    "minor":    colors.HexColor("#4caf50"),
    "moderate": colors.HexColor("#ff9800"),
    "severe":   colors.HexColor("#f44336"),
    "unknown":  colors.HexColor("#888888"),
}


def _frame_to_image_obj(frame: np.ndarray, max_width: float, max_height: float) -> Image:
    """Convert a BGR numpy frame to a ReportLab Image object."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Encode to JPEG in memory
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    # Maintain aspect ratio within bounds
    h, w = frame.shape[:2]
    ratio = min(max_width / w, max_height / h)
    return Image(buf, width=w * ratio, height=h * ratio)


class PDFExporter:
    def __init__(self, page_size=A4):
        self.page_size = page_size
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        s = self.styles
        base = s["Normal"]

        s.add(ParagraphStyle(
            "ReportTitle",
            parent=base,
            fontSize=22,
            textColor=WHITE,
            fontName="Helvetica-Bold",
            alignment=TA_CENTER,
            spaceAfter=4,
        ))
        s.add(ParagraphStyle(
            "Subtitle",
            parent=base,
            fontSize=11,
            textColor=colors.HexColor("#cccccc"),
            fontName="Helvetica",
            alignment=TA_CENTER,
            spaceAfter=2,
        ))
        s.add(ParagraphStyle(
            "SectionHeader",
            parent=base,
            fontSize=13,
            textColor=ACCENT,
            fontName="Helvetica-Bold",
            spaceBefore=12,
            spaceAfter=4,
        ))
        s.add(ParagraphStyle(
            "BodyText2",
            parent=base,
            fontSize=10,
            textColor=TEXT_DARK,
            fontName="Helvetica",
            alignment=TA_JUSTIFY,
            leading=16,
        ))
        s.add(ParagraphStyle(
            "WarnBanner",
            parent=base,
            fontSize=11,
            textColor=WHITE,
            fontName="Helvetica-Bold",
            alignment=TA_CENTER,
        ))
        s.add(ParagraphStyle(
            "TableCell",
            parent=base,
            fontSize=9,
            fontName="Helvetica",
            textColor=TEXT_DARK,
        ))

    def _header_footer(self, canvas, doc):
        """Draw header + footer on every page."""
        canvas.saveState()
        pw, ph = self.page_size

        # Header bar
        canvas.setFillColor(DARK_BG)
        canvas.rect(0, ph - 2.2 * cm, pw, 2.2 * cm, fill=True, stroke=False)
        canvas.setFillColor(ACCENT)
        canvas.rect(0, ph - 2.2 * cm, pw, 3, fill=True, stroke=False)

        canvas.setFont("Helvetica-Bold", 13)
        canvas.setFillColor(WHITE)
        canvas.drawString(1.5 * cm, ph - 1.5 * cm, "🚗  DASHCAM INCIDENT REPORT")

        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#aaaaaa"))
        canvas.drawRightString(pw - 1.5 * cm, ph - 1.5 * cm,
                               f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}")

        # Footer
        canvas.setFillColor(LIGHT_GREY)
        canvas.rect(0, 0, pw, 1.2 * cm, fill=True, stroke=False)
        canvas.setFillColor(MID_GREY)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(1.5 * cm, 0.45 * cm,
                          "Confidential — Generated by Local AI Dashcam Incident Explainer")
        canvas.drawRightString(pw - 1.5 * cm, 0.45 * cm, f"Page {doc.page}")

        canvas.restoreState()

    def export(
        self,
        incident_json: dict,
        keyframe_paths: list[str],  # ordered image paths (privacy-blurred versions)
        output_path: str,
        keyframe_arrays: Optional[list[np.ndarray]] = None,
    ) -> str:
        """
        Build and save the PDF report.

        Args:
            incident_json   : parsed VLM output dict
            keyframe_paths  : image file paths for thumbnails
            output_path     : where to save the PDF
            keyframe_arrays : numpy BGR frames (used if paths unavailable)

        Returns:
            Absolute path to saved PDF
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pw, ph = self.page_size

        doc = BaseDocTemplate(
            output_path,
            pagesize=self.page_size,
            topMargin=2.8 * cm,
            bottomMargin=1.8 * cm,
            leftMargin=1.5 * cm,
            rightMargin=1.5 * cm,
        )

        frame = Frame(
            doc.leftMargin, doc.bottomMargin,
            pw - doc.leftMargin - doc.rightMargin,
            ph - doc.topMargin - doc.bottomMargin,
            id="main"
        )
        template = PageTemplate(
            id="main",
            frames=[frame],
            onPage=self._header_footer,
        )
        doc.addPageTemplates([template])

        story = []

        # ── Confidence Flag Banner ─────────────────────────────────────────
        if incident_json.get("confidence_flag"):
            banner = Table(
                [[Paragraph("⚠  REQUIRES HUMAN REVIEW — VLM confidence was low",
                            self.styles["WarnBanner"])]],
                colWidths=[pw - 3 * cm],
            )
            banner.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), WARN_AMBER),
                ("ROUNDEDCORNERS", [6]),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]))
            story.append(banner)
            story.append(Spacer(1, 0.4 * cm))

        # ── Incident Overview Table ────────────────────────────────────────
        severity = incident_json.get("severity", "unknown").lower()
        sev_color = SEVERITY_COLORS.get(severity, MID_GREY)
        sev_para = Paragraph(
            f'<font color="white"><b>{severity.upper()}</b></font>',
            ParagraphStyle("sev", fontSize=11, alignment=TA_CENTER)
        )

        cond = incident_json.get("conditions", {})
        overview_data = [
            ["Field", "Value"],
            ["Severity", sev_para],
            ["Vehicles", ", ".join(incident_json.get("vehicles", [])) or "N/A"],
            ["Road", cond.get("road", "N/A")],
            ["Weather", cond.get("weather", "N/A")],
            ["Visibility", cond.get("visibility", "N/A")],
            ["VLM Model", incident_json.get("vlm_model", "N/A")],
            ["Inference Time", f"{incident_json.get('inference_time_s', '?')}s"],
            ["Timestamp", incident_json.get("timestamp", "N/A")],
        ]

        overview_table = Table(overview_data, colWidths=[5 * cm, pw - 8 * cm])
        overview_style = TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), DARK_BG),
            ("TEXTCOLOR",    (0, 0), (-1, 0), WHITE),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0), 10),
            ("BACKGROUND",   (0, 2), (-1, -1), LIGHT_GREY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
            ("BACKGROUND",   (1, 1), (1, 1), sev_color),   # severity cell
            ("GRID",         (0, 0), (-1, -1), 0.5, MID_GREY),
            ("FONTNAME",     (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 1), (-1, -1), 9),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ])
        overview_table.setStyle(overview_style)
        story.append(Paragraph("Incident Overview", self.styles["SectionHeader"]))
        story.append(overview_table)
        story.append(Spacer(1, 0.5 * cm))

        # ── Fault Analysis ──────────────────────────────────────────────────
        story.append(Paragraph("Fault Analysis", self.styles["SectionHeader"]))
        story.append(Paragraph(
            incident_json.get("fault_analysis", "Not determined."),
            self.styles["BodyText2"]
        ))
        story.append(Spacer(1, 0.4 * cm))

        # ── Timeline ──────────────────────────────────────────────────────
        timeline = incident_json.get("timeline", [])
        if timeline:
            story.append(Paragraph("Event Timeline", self.styles["SectionHeader"]))
            labels = ["Pre-Incident", "Impact / Mid", "Post-Incident"]
            for i, desc in enumerate(timeline):
                label = labels[i] if i < len(labels) else f"Frame {i+1}"
                story.append(Paragraph(
                    f"<b>{label}:</b> {desc}",
                    self.styles["BodyText2"]
                ))
            story.append(Spacer(1, 0.4 * cm))

        # ── Prose Report ─────────────────────────────────────────────────────
        prose = incident_json.get("prose_report", "")
        if prose:
            story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph("Insurance Incident Report", self.styles["SectionHeader"]))
            story.append(Paragraph(prose, self.styles["BodyText2"]))
            story.append(Spacer(1, 0.5 * cm))

        # ── Keyframe Thumbnails ──────────────────────────────────────────────
        if keyframe_paths or keyframe_arrays:
            story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph("Evidence Keyframes", self.styles["SectionHeader"]))

            thumb_w = (pw - 3 * cm) / 3 - 0.3 * cm
            thumb_h = thumb_w * 0.6

            frames_to_use = keyframe_arrays or []
            if not frames_to_use and keyframe_paths:
                for p in keyframe_paths:
                    f = cv2.imread(p)
                    if f is not None:
                        frames_to_use.append(f)

            if frames_to_use:
                img_cells = []
                label_cells = []
                frame_labels = ["Pre-Incident", "Impact", "Post-Incident"]
                for i, frame in enumerate(frames_to_use[:5]):
                    img_obj = _frame_to_image_obj(frame, thumb_w, thumb_h)
                    img_cells.append(img_obj)
                    lbl = frame_labels[i] if i < len(frame_labels) else f"Frame {i+1}"
                    label_cells.append(Paragraph(
                        f"<b>{lbl}</b>",
                        ParagraphStyle("tc", fontSize=8, alignment=TA_CENTER)
                    ))

                # Pad to 3 columns
                while len(img_cells) < 3:
                    img_cells.append("")
                    label_cells.append("")

                thumb_table = Table(
                    [img_cells[:3], label_cells[:3]],
                    colWidths=[thumb_w + 0.3 * cm] * 3,
                )
                thumb_table.setStyle(TableStyle([
                    ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING",  (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                    ("GRID",        (0, 0), (-1, -1), 0.3, LIGHT_GREY),
                ]))
                story.append(thumb_table)

        # ── Build PDF ──────────────────────────────────────────────────────
        doc.build(story)
        print(f"[PDF] ✓ Report saved → {output_path}")
        return output_path
