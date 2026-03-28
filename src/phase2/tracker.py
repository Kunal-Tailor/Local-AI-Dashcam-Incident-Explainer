"""
Phase 2 — SORT-style Multi-Object Tracker
A lightweight IoU-based tracker that assigns consistent track IDs across frames.
Implements Hungarian algorithm assignment (via scipy) without needing the
`sort` package.

Designed to be used together with detector.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


# ─── Utils ───────────────────────────────────────────────────────────────────

def iou(boxA: tuple, boxB: tuple) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ─── Track ───────────────────────────────────────────────────────────────────

@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    class_name: str
    hits: int = 1
    age: int = 0         # frames since last match
    is_confirmed: bool = False

    # Simple velocity for Kalman-lite prediction
    _prev_bbox: Optional[tuple] = field(default=None, repr=False)
    vx: float = 0.0
    vy: float = 0.0

    def predict(self) -> tuple[int, int, int, int]:
        """Predict next position using constant-velocity model."""
        x1, y1, x2, y2 = self.bbox
        return (
            int(x1 + self.vx), int(y1 + self.vy),
            int(x2 + self.vx), int(y2 + self.vy)
        )

    def update(self, bbox: tuple[int, int, int, int]):
        """Update track with a new matched detection."""
        if self._prev_bbox is not None:
            ox1, oy1 = self._prev_bbox[0], self._prev_bbox[1]
            self.vx = (bbox[0] - ox1) * 0.5
            self.vy = (bbox[1] - oy1) * 0.5
        self._prev_bbox = self.bbox
        self.bbox = bbox
        self.hits += 1
        self.age = 0
        if self.hits >= 2:
            self.is_confirmed = True


# ─── Tracker ─────────────────────────────────────────────────────────────────

class SORTTracker:
    """
    Simple Online and Realtime Tracking (SORT) with IoU-based assignment.

    Args:
        max_age     : frames a track survives without a match before deletion
        min_hits    : frames a track needs before being confirmed
        iou_threshold: minimum IoU to consider a match
    """

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 2,
        iou_threshold: float = 0.25,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: list[Track] = []
        self._next_id = 1

    # ── Public API ───────────────────────────────────────────────────────────

    def update(
        self,
        detections: list[tuple[tuple[int, int, int, int], str]],
    ) -> list[tuple[int, tuple[int, int, int, int], str]]:
        """
        Update tracker with current frame's detections.

        Args:
            detections: list of (bbox, class_name) from detector

        Returns:
            list of (track_id, bbox, class_name) for confirmed tracks
        """
        # 1. Predict all track positions
        predicted_bboxes = [t.predict() for t in self._tracks]

        # 2. Build cost matrix (negative IoU)
        if self._tracks and detections:
            cost_matrix = np.zeros((len(self._tracks), len(detections)))
            for ti, pred_bbox in enumerate(predicted_bboxes):
                for di, (det_bbox, _) in enumerate(detections):
                    cost_matrix[ti, di] = -iou(pred_bbox, det_bbox)

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_track_ids: set[int] = set()
            matched_det_ids: set[int] = set()

            for ti, di in zip(row_ind, col_ind):
                if -cost_matrix[ti, di] >= self.iou_threshold:
                    self._tracks[ti].update(detections[di][0])
                    self._tracks[ti].class_name = detections[di][1]
                    matched_track_ids.add(ti)
                    matched_det_ids.add(di)
        else:
            matched_track_ids = set()
            matched_det_ids = set()

        # 3. Age unmatched tracks
        unmatched_tracks = [i for i in range(len(self._tracks)) if i not in matched_track_ids]
        for ti in unmatched_tracks:
            self._tracks[ti].age += 1

        # 4. Create new tracks for unmatched detections
        for di, (det_bbox, cls_name) in enumerate(detections):
            if di not in matched_det_ids:
                self._tracks.append(Track(
                    track_id=self._next_id,
                    bbox=det_bbox,
                    class_name=cls_name,
                ))
                self._next_id += 1

        # 5. Remove dead tracks
        self._tracks = [t for t in self._tracks if t.age <= self.max_age]

        # 6. Return confirmed tracks
        results = []
        for t in self._tracks:
            if t.is_confirmed:
                results.append((t.track_id, t.bbox, t.class_name))

        return results

    def get_all_tracks(self) -> list[Track]:
        return list(self._tracks)

    def reset(self):
        self._tracks = []
        self._next_id = 1
