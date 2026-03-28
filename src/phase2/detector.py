"""
Phase 2 — YOLO11 Object Detector
Wraps Ultralytics YOLO11 for vehicle, pedestrian, and motorcycle detection
on dashcam video frames.

Usage:
    python3 detector.py --image data/samples/frame.jpg
    python3 detector.py --video data/samples/test.mp4 --output outputs/detected
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ultralytics YOLO (pip install ultralytics)
from ultralytics import YOLO


# ─── COCO class IDs we care about ────────────────────────────────────────────
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    frame_idx: int
    track_id: int  # assigned by tracker; -1 before tracking
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    # Derived
    cx: float = field(init=False)
    cy: float = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


# ─── Detector ────────────────────────────────────────────────────────────────

class DashcamDetector:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.35,
        device: Optional[str] = None,
    ):
        print(f"[Detector] Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.conf = confidence_threshold
        # Auto-detect device: MPS (Apple Silicon) → CUDA → CPU
        if device is None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        print(f"[Detector] Using device: {self.device}")

    def detect_frame(self, frame: np.ndarray, frame_idx: int = 0) -> list[Detection]:
        """Run YOLO on a single frame; return filtered detections."""
        results = self.model(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(Detection(
                    frame_idx=frame_idx,
                    track_id=-1,
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=TARGET_CLASSES[cls_id],
                    confidence=conf,
                ))
        return detections

    def detect_video(
        self,
        video_path: str,
        skip_frames: int = 1,
    ) -> dict[int, list[Detection]]:
        """
        Detect objects on every `skip_frames`-th frame of a video.
        Returns dict: frame_idx → list[Detection]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
            sys.exit(1)

        all_detections: dict[int, list[Detection]] = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip_frames == 0:
                dets = self.detect_frame(frame, frame_idx)
                if dets:
                    all_detections[frame_idx] = dets
            frame_idx += 1

        cap.release()
        total = sum(len(v) for v in all_detections.values())
        print(f"[Detector] {frame_idx} frames processed, {total} target detections")
        return all_detections

    def annotate_frame(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw coloured bounding boxes + labels on frame."""
        COLOR_MAP = {
            "car": (0, 200, 50),
            "truck": (0, 120, 255),
            "bus": (0, 180, 180),
            "person": (255, 80, 0),
            "motorcycle": (200, 0, 200),
            "bicycle": (80, 80, 255),
        }
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = COLOR_MAP.get(det.class_name, (200, 200, 200))
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.track_id >= 0:
                label = f"#{det.track_id} {label}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return out


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Dashcam Detector")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image")
    group.add_argument("--video", help="Path to a video file")
    parser.add_argument("--output", default="outputs/detected",
                        help="Output directory for annotated frames")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold (default: 0.35)")
    args = parser.parse_args()

    detector = DashcamDetector(confidence_threshold=args.conf)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[ERROR] Cannot read image: {args.image}", file=sys.stderr)
            sys.exit(1)
        dets = detector.detect_frame(frame, 0)
        annotated = detector.annotate_frame(frame, dets)
        out_path = os.path.join(args.output, "detected.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"[Done] {len(dets)} detections. Saved → {out_path}")
        for d in dets:
            print(f"  {d.class_name} conf={d.confidence:.2f} bbox={d.bbox}")
    else:
        all_dets = detector.detect_video(args.video)
        print(f"[Done] Detections across {len(all_dets)} frames.")


if __name__ == "__main__":
    main()
