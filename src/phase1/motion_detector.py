"""
Phase 1 — Motion Detector
Analyses a dashcam video and outputs per-frame motion intensity scores.
Uses absolute frame differencing (fast, CPU-only).

Usage:
    python3 motion_detector.py --video path/to/video.mp4 --threshold 8.0
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def compute_motion_scores(video_path: str) -> tuple[list[float], float]:
    """
    Compute motion score for every frame transition.

    Returns:
        scores: list of float motion scores (length = total_frames - 1)
        fps: video FPS
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scores: list[float] = []

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return scores, fps

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        score = float(np.mean(diff))
        scores.append(score)
        prev_gray = gray

    cap.release()
    return scores, fps


def detect_high_motion_events(
    scores: list[float],
    fps: float,
    threshold: float,
    min_gap_s: float = 2.0
) -> list[dict]:
    """
    Identify continuous high-motion windows.

    Returns a list of event dicts with keys:
        start_frame, end_frame, start_sec, end_sec, peak_score
    """
    min_gap = int(fps * min_gap_s)
    events: list[dict] = []
    in_event = False
    start = 0
    peak = 0.0

    for i, score in enumerate(scores):
        if score >= threshold:
            if not in_event:
                in_event = True
                start = i
                peak = score
            else:
                peak = max(peak, score)
        else:
            if in_event:
                # Close event if we've been below threshold long enough
                if i - start >= min_gap or score < threshold:
                    events.append({
                        "start_frame": start,
                        "end_frame": i,
                        "start_sec": round(start / fps, 2),
                        "end_sec": round(i / fps, 2),
                        "peak_score": round(peak, 3)
                    })
                    in_event = False

    if in_event:
        events.append({
            "start_frame": start,
            "end_frame": len(scores),
            "start_sec": round(start / fps, 2),
            "end_sec": round(len(scores) / fps, 2),
            "peak_score": round(peak, 3)
        })

    return events


def print_summary(scores: list[float], fps: float, events: list[dict], threshold: float):
    total_frames = len(scores) + 1
    duration = len(scores) / fps
    mean_score = float(np.mean(scores)) if scores else 0.0

    print(f"\n{'='*55}")
    print(f"  Motion Analysis Summary")
    print(f"{'='*55}")
    print(f"  Total frames analysed : {total_frames}")
    print(f"  Video duration        : {duration:.1f}s  ({fps:.1f} FPS)")
    print(f"  Mean motion score     : {mean_score:.3f}")
    print(f"  Threshold             : {threshold}")
    print(f"  High-motion events    : {len(events)}")
    print(f"{'='*55}")
    for i, ev in enumerate(events, 1):
        print(f"  Event {i}: frames {ev['start_frame']}–{ev['end_frame']}  "
              f"({ev['start_sec']}s – {ev['end_sec']}s)  "
              f"peak={ev['peak_score']}")
    print(f"{'='*55}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dashcam Motion Detector")
    parser.add_argument("--video", required=True, help="Path to dashcam video")
    parser.add_argument("--threshold", type=float, default=8.0,
                        help="Motion score threshold (default: 8.0; lower = more sensitive)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    scores, fps = compute_motion_scores(args.video)
    events = detect_high_motion_events(scores, fps, args.threshold)
    print_summary(scores, fps, events, args.threshold)


if __name__ == "__main__":
    main()
