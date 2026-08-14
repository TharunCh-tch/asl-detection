import numpy as np
import pytest

from asl_detection.infer import predict_image
from asl_detection.landmarks import FEATURE_DIM, LandmarkResult


class _StubClassifier:
    classes = ["A", "B", "C"]

    def predict_proba(self, X):
        probs = np.array([[0.1, 0.7, 0.2]], dtype=np.float32)
        return probs


class _StubExtractorHandFound:
    def extract(self, image_rgb):
        return LandmarkResult(
            features=np.zeros(FEATURE_DIM, dtype=np.float32),
            raw_landmarks=np.zeros((21, 3), dtype=np.float32),
            handedness="Left",
            confidence=0.8,
        )


class _StubExtractorNoHand:
    def extract(self, image_rgb):
        return None


def test_predict_image_returns_top_prediction_and_topk():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = predict_image(image, _StubClassifier(), _StubExtractorHandFound(), top_k=2)
    assert result["detected"] is True
    assert result["letter"] == "B"  # highest prob class
    assert result["confidence"] == pytest.approx(0.7, abs=1e-6)
    assert len(result["top_k"]) == 2
    assert result["top_k"][0]["letter"] == "B"
    assert result["top_k"][1]["letter"] == "C"
    assert result["handedness"] == "Left"


def test_predict_image_no_hand_detected():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = predict_image(image, _StubClassifier(), _StubExtractorNoHand())
    assert result["detected"] is False
    assert "message" in result
