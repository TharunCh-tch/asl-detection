"""Hand landmark extraction using MediaPipe Tasks (HandLandmarker).

MediaPipe's HandLandmarker detects 21 3D landmarks per hand (wrist, and four
joints per finger). We convert those landmarks into a fixed-length, 63-dim
feature vector (21 landmarks x [x, y, z]) that is translation- and
scale-invariant, so the same gesture looks (numerically) the same regardless
of where the hand is in the frame or how close it is to the camera:

  1. Translate: subtract the wrist landmark (index 0) from every landmark.
  2. Scale: divide by the distance from the wrist to the middle-finger MCP
     joint (index 9), a stable reference "bone length" for a given hand.

This is the standard feature representation used in most landmark-based sign
language recognition projects (as opposed to feeding raw pixels into a CNN).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

NUM_LANDMARKS = 21
NUM_COORDS = 3
FEATURE_DIM = NUM_LANDMARKS * NUM_COORDS  # 63

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "mediapipe", "hand_landmarker.task"
)
_DEFAULT_MODEL_PATH = os.path.normpath(_DEFAULT_MODEL_PATH)

WRIST = 0
MIDDLE_MCP = 9


def normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    """Translate + scale a (21, 3) landmark array to a flat 63-dim vector.

    ``raw`` holds MediaPipe's per-landmark (x, y, z) coordinates, already
    normalized to [0, 1] relative to image width/height by MediaPipe itself.
    This function removes the remaining position/scale dependence.
    """
    raw = np.asarray(raw, dtype=np.float32).reshape(NUM_LANDMARKS, NUM_COORDS)
    origin = raw[WRIST]
    translated = raw - origin
    scale = np.linalg.norm(translated[MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1e-6
    normalized = translated / scale
    return normalized.reshape(-1).astype(np.float32)


@dataclass
class LandmarkResult:
    features: np.ndarray  # (63,) normalized feature vector
    raw_landmarks: np.ndarray  # (21, 3) raw MediaPipe coordinates
    handedness: Optional[str] = None
    confidence: Optional[float] = None


class HandLandmarkExtractor:
    """Thin wrapper around MediaPipe's HandLandmarker (Tasks API, IMAGE mode)."""

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe hand landmarker model not found at {model_path}. "
                "Run `python scripts/download_mediapipe_model.py` first."
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def extract(self, image_rgb: np.ndarray) -> Optional[LandmarkResult]:
        """Run detection on an RGB uint8 numpy image (H, W, 3).

        Returns None if no hand was detected.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must be an (H, W, 3) RGB array")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None

        hand = result.hand_landmarks[0]  # single-hand mode
        raw = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
        features = normalize_landmarks(raw)

        handedness = None
        confidence = None
        if result.handedness:
            top = result.handedness[0][0]
            handedness = top.category_name
            confidence = float(top.score)

        return LandmarkResult(
            features=features,
            raw_landmarks=raw,
            handedness=handedness,
            confidence=confidence,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
