"""
Phase 2 — Optical Flow Analyser
Uses Farneback dense optical flow to detect sudden velocity changes
(spikes) that indicate a potential collision or incident.

Provides:
  - Per-frame flow magnitude
  - Per-bbox flow score
  - Spike detection (flow > mean + k*std)
"""

from __future__ import annotations

import cv2
import numpy as np


# ─── Dense Optical Flow ───────────────────────────────────────────────────────

def farneback_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """
    Compute dense optical flow using Gunnar Farneback's algorithm.

    Returns:
        flow: (H, W, 2) float32 array of x/y flow vectors
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,   # pyramid scale
        levels=3,         # pyramid levels
        winsize=15,       # averaging window size
        iterations=3,     # iterations per level
        poly_n=5,         # polynomial expansion neighbourhood
        poly_sigma=1.2,   # Gaussian std for polynomial expansion
        flags=0,
    )
    return flow


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """Return per-pixel flow magnitude from (H, W, 2) flow array."""
    fx, fy = flow[..., 0], flow[..., 1]
    return np.sqrt(fx**2 + fy**2)


def bbox_flow_score(
    flow: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad: int = 4,
) -> float:
    """
    Compute mean flow magnitude inside a bounding box (with padding).
    Returns 0.0 if bbox is out of frame bounds.
    """
    h, w = flow.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    mag = flow_magnitude(flow[y1:y2, x1:x2])
    return float(np.mean(mag))


# ─── Spike Detector ───────────────────────────────────────────────────────────

class FlowSpikeDetector:
    """
    Tracks a rolling history of per-frame flow scores.
    Detects spikes as values > mean + k*std of the rolling window.

    Args:
        window_size : number of historical frames to keep
        spike_k     : spike sensitivity (higher = less sensitive)
        min_spike   : absolute minimum score to treat as a spike
    """

    def __init__(
        self,
        window_size: int = 30,
        spike_k: float = 2.5,
        min_spike: float = 3.0,
    ):
        self.window_size = window_size
        self.spike_k = spike_k
        self.min_spike = min_spike
        self._history: list[float] = []

    def push(self, score: float) -> bool:
        """
        Add a flow score observaton.
        Returns True if this score is a spike.
        """
        self._history.append(score)
        if len(self._history) > self.window_size:
            self._history.pop(0)

        if len(self._history) < 5:
            return False  # not enough data

        arr = np.array(self._history[:-1])  # exclude current for baseline
        threshold = arr.mean() + self.spike_k * arr.std()
        threshold = max(threshold, self.min_spike)
        return score >= threshold

    def reset(self):
        self._history.clear()


# ─── Video-level Flow Processor ──────────────────────────────────────────────

def compute_video_flow_scores(video_path: str) -> list[float]:
    """
    Compute global mean flow magnitude for every frame transition.

    Returns:
        scores: list of float, length = total_frames - 1
    """
    import sys
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
        return []

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return []

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    scores: list[float] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = farneback_flow(prev_gray, gray)
        mag = flow_magnitude(flow)
        scores.append(float(np.mean(mag)))
        prev_gray = gray

    cap.release()
    return scores
