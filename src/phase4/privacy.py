"""
Phase 4 — Privacy Module
Automatically blurs licence plates and faces in dashcam keyframes.

Techniques:
  - Licence plates: YOLO-based vehicle crop → aspect-ratio heuristic for LP region → Gaussian blur
  - Faces: OpenCV DNN face detector (Caffe model) with fallback to Haar cascade

Usage:
    from src.phase4.privacy import PrivacyBlurrer
    blurred = PrivacyBlurrer().blur(frame)
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ─── DNN face detector config ────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).resolve().parent / "models"
_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_WEIGHTS_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def _ensure_face_model() -> tuple[str, str]:
    """Download OpenCV DNN face detector if not present."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    proto = _MODEL_DIR / "deploy.prototxt"
    weights = _MODEL_DIR / "res10_ssd.caffemodel"

    if not proto.exists():
        print("[Privacy] Downloading face detector prototxt...")
        try:
            urllib.request.urlretrieve(_PROTO_URL, proto)
        except Exception as e:
            print(f"[Privacy] Could not download prototxt: {e}")
            return "", ""

    if not weights.exists():
        print("[Privacy] Downloading face detector weights (~10MB)...")
        try:
            urllib.request.urlretrieve(_WEIGHTS_URL, weights)
        except Exception as e:
            print(f"[Privacy] Could not download weights: {e}")
            return "", ""

    return str(proto), str(weights)


# ─── Blurring helpers ─────────────────────────────────────────────────────────

def gaussian_blur_region(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    ksize: int = 51,
) -> np.ndarray:
    """Apply Gaussian blur to a rectangular region in-place."""
    h, w = image.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return image
    roi = image[y1:y2, x1:x2]
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    image[y1:y2, x1:x2] = blurred
    return image


# ─── Licence Plate Blurrer ───────────────────────────────────────────────────

def _lp_heuristic_regions(
    vehicle_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]]:
    """
    Heuristic: estimate licence plate region as a wide, short strip
    at the bottom of a vehicle bounding box.

    Returns a list of (x1, y1, x2, y2) candidate regions.
    """
    x1, y1, x2, y2 = vehicle_bbox
    bw = x2 - x1
    bh = y2 - y1

    # Front plate: bottom-centre strip (~40% width, ~10% height from bottom)
    lp_h = max(20, int(bh * 0.12))
    lp_w = max(40, int(bw * 0.45))
    lp_x1 = x1 + int((bw - lp_w) / 2)
    lp_y1 = y2 - lp_h - 5
    front = (lp_x1, lp_y1, lp_x1 + lp_w, y2)

    # Rear plate candidate (upper-centre for bikes)
    rear_y1 = y1 + int(bh * 0.6)
    rear = (lp_x1, rear_y1, lp_x1 + lp_w, rear_y1 + lp_h)

    return [front, rear]


def blur_licence_plates(
    image: np.ndarray,
    vehicle_bboxes: list[tuple[int, int, int, int]],
    ksize: int = 45,
) -> np.ndarray:
    """Blur probable licence plate regions for a list of vehicle bboxes."""
    out = image.copy()
    for bbox in vehicle_bboxes:
        for lp_region in _lp_heuristic_regions(bbox):
            out = gaussian_blur_region(out, *lp_region, ksize=ksize)
    return out


# ─── Face Blurrer ─────────────────────────────────────────────────────────────

class FaceBlurrer:
    def __init__(self, confidence_thresh: float = 0.5):
        self.confidence_thresh = confidence_thresh
        self._net = None
        self._haar = None
        self._init()

    def _init(self):
        """Try to load DNN face detector; fall back to Haar cascade."""
        proto, weights = _ensure_face_model()
        if proto and weights:
            try:
                self._net = cv2.dnn.readNetFromCaffe(proto, weights)
                print("[Privacy] Loaded DNN face detector")
                return
            except Exception as e:
                print(f"[Privacy] DNN load failed: {e}, using Haar fallback")

        # Haar cascade fallback
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        if self._haar.empty():
            print("[Privacy] WARNING: Haar cascade also failed to load")
            self._haar = None
        else:
            print("[Privacy] Loaded Haar cascade face detector")

    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (x1, y1, x2, y2) face bboxes."""
        faces = []
        h, w = image.shape[:2]

        if self._net is not None:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
            self._net.setInput(blob)
            detections = self._net.forward()
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf < self.confidence_thresh:
                    continue
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append((x1, y1, x2, y2))

        elif self._haar is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            for (x, y, fw, fh) in rects:
                faces.append((x, y, x + fw, y + fh))

        return faces

    def blur(self, image: np.ndarray, ksize: int = 51) -> np.ndarray:
        """Detect and blur all faces in the image."""
        out = image.copy()
        faces = self.detect_faces(image)
        for bbox in faces:
            out = gaussian_blur_region(out, *bbox, ksize=ksize)
        if faces:
            print(f"[Privacy] Blurred {len(faces)} face(s)")
        return out


# ─── Combined Blurrer ─────────────────────────────────────────────────────────

class PrivacyBlurrer:
    """One-stop privacy blurrer: licence plates + faces."""

    def __init__(self):
        self.face_blurrer = FaceBlurrer()

    def blur(
        self,
        image: np.ndarray,
        vehicle_bboxes: list[tuple[int, int, int, int]] | None = None,
    ) -> np.ndarray:
        """
        Apply all privacy blurring in sequence.

        Args:
            image          : BGR numpy frame
            vehicle_bboxes : if provided, blur LP regions inside these bboxes

        Returns:
            Privacy-blurred image (copy)
        """
        out = image.copy()
        if vehicle_bboxes:
            out = blur_licence_plates(out, vehicle_bboxes)
        out = self.face_blurrer.blur(out)
        return out
