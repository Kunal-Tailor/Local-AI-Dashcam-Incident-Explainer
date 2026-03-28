"""
Phase 1 — Keyframe Extractor
Extracts frames from a dashcam video using FFmpeg.
Supports two modes:
  • uniform  : extract every N seconds
  • smart    : extract only during high-motion windows (uses motion_detector.py)

Usage:
    python3 extract_keyframes.py --video path/to/video.mp4 --fps 1 --mode smart --output outputs/keyframes
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# ─── Uniform Extraction ───────────────────────────────────────────────────────

def extract_uniform(video_path: str, fps: float, output_dir: str) -> list[str]:
    """Use FFmpeg to extract frames at a fixed FPS rate."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_pattern = os.path.join(output_dir, "frame_%05d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        out_pattern
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] FFmpeg failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    print(f"[Phase 1] Extracted {len(frames)} frames at {fps} FPS → {output_dir}")
    return [str(f) for f in frames]


# ─── Smart Extraction (Motion-Guided) ────────────────────────────────────────

def motion_scores(video_path: str) -> list[float]:
    """
    Compute per-frame motion score using absolute frame differencing.
    Returns a list of float scores, one per frame pair.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    scores = []
    ret, prev = cap.read()
    if not ret:
        return scores
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
    return scores


def extract_smart(video_path: str, motion_threshold: float, output_dir: str,
                  min_gap_frames: int = 5) -> list[str]:
    """
    Extract frames only where motion score exceeds threshold.
    Skips frames within min_gap_frames of a previously saved frame.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scores = motion_scores(video_path)

    cap = cv2.VideoCapture(video_path)
    saved_frames: list[str] = []
    last_saved = -min_gap_frames
    frame_idx = 0

    ret, _ = cap.read()  # skip first frame (no score)

    for idx, score in enumerate(scores):
        ret, frame = cap.read()
        if not ret:
            break
        if score >= motion_threshold and (idx - last_saved) >= min_gap_frames:
            out_path = os.path.join(output_dir, f"smart_{idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_frames.append(out_path)
            last_saved = idx
        frame_idx += 1

    cap.release()
    print(f"[Phase 1] Smart extraction: {len(saved_frames)} high-motion frames → {output_dir}")
    return saved_frames


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dashcam Keyframe Extractor")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="FPS for uniform mode (default: 1.0)")
    parser.add_argument("--mode", choices=["uniform", "smart"], default="smart",
                        help="Extraction mode: uniform or smart (default: smart)")
    parser.add_argument("--motion-threshold", type=float, default=8.0,
                        help="Motion score threshold for smart mode (default: 8.0)")
    parser.add_argument("--output", default="outputs/keyframes",
                        help="Output directory (default: outputs/keyframes)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "uniform":
        frames = extract_uniform(args.video, args.fps, args.output)
    else:
        frames = extract_smart(args.video, args.motion_threshold, args.output)

    print(f"[Done] {len(frames)} keyframes saved to: {args.output}")


if __name__ == "__main__":
    main()
