"""
Quick PDF generation test — uses the existing report JSON to produce a PDF.
Run from project root:
    DASHCAM_MOCK_OLLAMA=1 .venv/bin/python3 tests/test_pdf.py
"""
import json
import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.phase4.pdf_exporter import PDFExporter
from src.phase3.keyframe_sampler import sample_keyframes

# Find the latest JSON report
reports = sorted(glob.glob("outputs/reports/*.json"))
if not reports:
    print("[ERROR] No reports found. Run pipeline first.")
    sys.exit(1)

with open(reports[-1]) as f:
    data = json.load(f)

print(f"[Test] Loaded report: {reports[-1]}")
print(f"[Test] Severity: {data.get('severity')} | Flag: {data.get('confidence_flag')}")

# Get keyframes from sample video
kf = sample_keyframes("data/samples/test_dashcam.mp4", n=3)
frames = [f for _, f in kf]

# Generate PDF
exporter = PDFExporter()
pdf_path = exporter.export(
    incident_json=data,
    keyframe_paths=[],
    output_path="outputs/reports/test_report.pdf",
    keyframe_arrays=frames,
)
size_kb = Path(pdf_path).stat().st_size // 1024
print(f"[Test] ✓ PDF generated: {pdf_path} ({size_kb} KB)")
