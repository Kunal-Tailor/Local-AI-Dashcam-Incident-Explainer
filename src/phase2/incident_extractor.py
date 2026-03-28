"""
Phase 2 — Incident Extractor
Combines YOLO11 detection + SORT tracking + optical flow spike detection
to automatically locate crash/incident windows and export ±5s clips.

Rule-based trigger:
  - Two tracked bboxes overlap (IoU > overlap_thresh)  AND
  - At least one bbox has a flow spike

Usage:
    python3 incident_extractor.py --video data/samples/test.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.phase2.detector import DashcamDetector, Detection
from src.phase2.tracker import SORTTracker
from src.phase2.optical_flow import (
    farneback_flow, bbox_flow_score, FlowSpikeDetector
)


# ─── IoU helper ──────────────────────────────────────────────────────────────

def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ─── Incident Window ─────────────────────────────────────────────────────────

class IncidentWindow:
    def __init__(self, trigger_frame: int, fps: float,
                 before_s: float = 5.0, after_s: float = 5.0):
        self.trigger_frame = trigger_frame
        self.fps = fps
        before_frames = int(fps * before_s)
        after_frames = int(fps * after_s)
        self.start_frame = max(0, trigger_frame - before_frames)
        self.end_frame = trigger_frame + after_frames
        self.start_sec = self.start_frame / fps
        self.end_sec = self.end_frame / fps

    def __repr__(self):
        return (f"IncidentWindow(trigger={self.trigger_frame}, "
                f"range=[{self.start_frame}–{self.end_frame}], "
                f"time=[{self.start_sec:.1f}s–{self.end_sec:.1f}s])")


# ─── Main Extractor ──────────────────────────────────────────────────────────

class IncidentExtractor:
    def __init__(
        self,
        overlap_thresh: float = 0.15,
        flow_spike_k: float = 2.5,
        min_flow_spike: float = 3.0,
        before_s: float = 5.0,
        after_s: float = 5.0,
        cooldown_frames: int = 60,  # prevent duplicate triggers
    ):
        self.overlap_thresh = overlap_thresh
        self.flow_spike_k = flow_spike_k
        self.min_flow_spike = min_flow_spike
        self.before_s = before_s
        self.after_s = after_s
        self.cooldown_frames = cooldown_frames

    def process_video(
        self,
        video_path: str,
        output_dir: str = "outputs/incidents",
    ) -> list[IncidentWindow]:
        """
        Scan a video end-to-end, detect incidents, and save clip windows.

        Returns list of detected IncidentWindows.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Extractor] {total_frames} frames @ {fps:.1f} FPS")

        detector = DashcamDetector()
        tracker = SORTTracker()
        spike_detectors: dict[int, FlowSpikeDetector] = {}  # per track_id

        incidents: list[IncidentWindow] = []
        last_trigger = -self.cooldown_frames

        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return incidents
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Detect + Track ──────────────────────────────────────────────
            raw_dets = detector.detect_frame(frame, frame_idx)
            det_input = [(d.bbox, d.class_name) for d in raw_dets]
            tracked = tracker.update(det_input)  # [(id, bbox, cls)]

            # ── Optical Flow ────────────────────────────────────────────────
            flow = farneback_flow(prev_gray, gray)
            flow_scores: dict[int, float] = {}
            for tid, bbox, _ in tracked:
                score = bbox_flow_score(flow, bbox)
                flow_scores[tid] = score

                if tid not in spike_detectors:
                    spike_detectors[tid] = FlowSpikeDetector(
                        spike_k=self.flow_spike_k,
                        min_spike=self.min_flow_spike,
                    )
                spike_detectors[tid].push(score)

            # ── Trigger Rule ────────────────────────────────────────────────
            if frame_idx - last_trigger < self.cooldown_frames:
                prev_gray = gray
                continue

            bboxes = [(tid, bbox) for tid, bbox, _ in tracked]
            flow_spike_any = any(
                spike_detectors[tid].push(flow_scores.get(tid, 0))
                for tid, _ in bboxes
                if tid in spike_detectors
            )

            # Check pairwise overlap
            overlap_found = False
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    if iou(bboxes[i][1], bboxes[j][1]) >= self.overlap_thresh:
                        overlap_found = True
                        break

            if overlap_found and flow_spike_any:
                window = IncidentWindow(frame_idx, fps, self.before_s, self.after_s)
                incidents.append(window)
                last_trigger = frame_idx
                print(f"[Extractor] ⚠  Incident detected at frame {frame_idx} "
                      f"({frame_idx/fps:.1f}s)")
                self._save_clip(video_path, window, output_dir, len(incidents))

            prev_gray = gray

        cap.release()
        print(f"[Extractor] Found {len(incidents)} incident(s).")
        return incidents

    def _save_clip(
        self, video_path: str, window: IncidentWindow,
        output_dir: str, idx: int
    ):
        """Export incident clip using FFmpeg trim."""
        duration = window.end_sec - window.start_sec
        out_path = os.path.join(output_dir, f"incident_{idx:03d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(window.start_sec),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
            out_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[Extractor] Clip saved → {out_path}")
        else:
            print(f"[WARN] FFmpeg clip export failed: {result.stderr[:200]}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dashcam Incident Extractor")
    parser.add_argument("--video", required=True, help="Path to dashcam video")
    parser.add_argument("--output", default="outputs/incidents",
                        help="Output dir for incident clips")
    parser.add_argument("--overlap-thresh", type=float, default=0.15)
    parser.add_argument("--flow-spike-k", type=float, default=2.5)
    parser.add_argument("--before", type=float, default=5.0,
                        help="Seconds before trigger to include in clip")
    parser.add_argument("--after", type=float, default=5.0,
                        help="Seconds after trigger to include in clip")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    extractor = IncidentExtractor(
        overlap_thresh=args.overlap_thresh,
        flow_spike_k=args.flow_spike_k,
        before_s=args.before,
        after_s=args.after,
    )
    incidents = extractor.process_video(args.video, args.output)

    print(f"\n{'='*50}")
    for i, w in enumerate(incidents, 1):
        print(f"  Incident {i}: {w}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
