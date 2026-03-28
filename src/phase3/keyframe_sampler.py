"""
Phase 3 — Keyframe Sampler
Selects N representative frames from an incident window for VLM analysis.

Strategy: divide the window into N equal segments; pick one frame per segment.
  - Segment 0 = pre-incident context
  - Segment N//2 = impact moment (near trigger frame)
  - Segment N-1 = post-incident state

Usage:
    from src.phase3.keyframe_sampler import sample_keyframes
    frames = sample_keyframes("outputs/incidents/incident_001.mp4", n=3)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np


def sample_keyframes(
    video_path: str,
    n: int = 3,
    output_dir: str | None = None,
) -> list[tuple[int, np.ndarray]]:
    """
    Sample N evenly-spaced keyframes from a video clip.

    Args:
        video_path  : path to incident clip (or any video)
        n           : number of keyframes to extract (1, 3, or 5)
        output_dir  : if set, save JPEGs to this dir

    Returns:
        list of (frame_index, frame_bgr_numpy) sorted by frame index
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}", file=sys.stderr)
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    n = min(n, total)
    # Compute target frame indices (evenly spaced, including first & last)
    if n == 1:
        indices = [total // 2]
    else:
        step = (total - 1) / (n - 1)
        indices = [round(i * step) for i in range(n)]

    keyframes: list[tuple[int, np.ndarray]] = []
    saved_paths: list[str] = []

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        keyframes.append((idx, frame))

        if output_dir:
            basename = Path(video_path).stem
            out_path = os.path.join(output_dir, f"{basename}_kf{idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)

    cap.release()

    if saved_paths:
        print(f"[Sampler] {len(keyframes)} keyframes saved → {output_dir}")
    else:
        print(f"[Sampler] {len(keyframes)} keyframes extracted (in-memory)")

    return keyframes


def load_keyframes_from_dir(keyframe_dir: str) -> list[tuple[str, np.ndarray]]:
    """
    Load all .jpg frames from a directory.
    Returns list of (path, frame_bgr).
    """
    paths = sorted(Path(keyframe_dir).glob("*.jpg"))
    result = []
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            result.append((str(p), frame))
    return result
