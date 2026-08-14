import numpy as np
import pytest

from asl_detection.landmarks import FEATURE_DIM, WRIST, normalize_landmarks


def test_normalize_landmarks_output_shape(synthetic_raw_landmarks):
    features = normalize_landmarks(synthetic_raw_landmarks)
    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32


def test_normalize_landmarks_is_translation_invariant(synthetic_raw_landmarks):
    shifted = synthetic_raw_landmarks + np.array([0.1, -0.2, 0.05], dtype=np.float32)
    f1 = normalize_landmarks(synthetic_raw_landmarks)
    f2 = normalize_landmarks(shifted)
    np.testing.assert_allclose(f1, f2, atol=1e-5)


def test_normalize_landmarks_is_scale_invariant(synthetic_raw_landmarks):
    origin = synthetic_raw_landmarks[WRIST]
    scaled = origin + (synthetic_raw_landmarks - origin) * 2.0
    f1 = normalize_landmarks(synthetic_raw_landmarks)
    f2 = normalize_landmarks(scaled.astype(np.float32))
    np.testing.assert_allclose(f1, f2, atol=1e-4)


def test_normalize_landmarks_wrist_maps_to_origin(synthetic_raw_landmarks):
    features = normalize_landmarks(synthetic_raw_landmarks)
    wrist_features = features.reshape(21, 3)[WRIST]
    np.testing.assert_allclose(wrist_features, np.zeros(3), atol=1e-6)


def test_normalize_landmarks_handles_degenerate_scale():
    # wrist and middle-MCP coincide -> scale would be ~0; must not divide by zero / NaN
    raw = np.zeros((21, 3), dtype=np.float32)
    features = normalize_landmarks(raw)
    assert np.all(np.isfinite(features))


class _FakeCategory:
    def __init__(self, name, score):
        self.category_name = name
        self.score = score


class _FakeLandmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _FakeDetectionResult:
    def __init__(self, hand_landmarks, handedness):
        self.hand_landmarks = hand_landmarks
        self.handedness = handedness


def test_extractor_extract_returns_none_when_no_hand(monkeypatch, synthetic_raw_landmarks):
    """Mock MediaPipe's HandLandmarker so this test needs no model file / real inference."""
    from asl_detection import landmarks as lm_module

    class _FakeLandmarker:
        def detect(self, mp_image):
            return _FakeDetectionResult(hand_landmarks=[], handedness=[])

        def close(self):
            pass

    extractor = object.__new__(lm_module.HandLandmarkExtractor)
    extractor._landmarker = _FakeLandmarker()
    result = extractor.extract(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result is None


def test_extractor_extract_returns_features_when_hand_present(synthetic_raw_landmarks):
    from asl_detection import landmarks as lm_module

    fake_hand = [_FakeLandmark(*row) for row in synthetic_raw_landmarks]
    fake_handedness = [[_FakeCategory("Right", 0.98)]]

    class _FakeLandmarker:
        def detect(self, mp_image):
            return _FakeDetectionResult(hand_landmarks=[fake_hand], handedness=fake_handedness)

        def close(self):
            pass

    extractor = object.__new__(lm_module.HandLandmarkExtractor)
    extractor._landmarker = _FakeLandmarker()
    result = extractor.extract(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result is not None
    assert result.features.shape == (FEATURE_DIM,)
    assert result.handedness == "Right"
    assert result.confidence == pytest.approx(0.98)


def test_extractor_extract_rejects_bad_shape():
    from asl_detection import landmarks as lm_module

    extractor = object.__new__(lm_module.HandLandmarkExtractor)
    with pytest.raises(ValueError):
        extractor.extract(np.zeros((10, 10), dtype=np.uint8))
