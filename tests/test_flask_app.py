import io

import numpy as np
import pytest
from PIL import Image

from asl_detection.app_flask import create_app


class _StubClassifier:
    classes = ["A", "B", "C"]

    def predict_proba(self, X):
        n = X.shape[0]
        probs = np.zeros((n, 3), dtype=np.float32)
        probs[:, 0] = 1.0  # always confidently predicts "A"
        return probs


class _StubExtractorHandFound:
    def extract(self, image_rgb):
        from asl_detection.landmarks import FEATURE_DIM, LandmarkResult

        return LandmarkResult(
            features=np.zeros(FEATURE_DIM, dtype=np.float32),
            raw_landmarks=np.zeros((21, 3), dtype=np.float32),
            handedness="Right",
            confidence=0.9,
        )


class _StubExtractorNoHand:
    def extract(self, image_rgb):
        return None


def _make_test_image_bytes():
    img = Image.new("RGB", (64, 64), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def client_with_hand():
    app = create_app(classifier=_StubClassifier(), extractor=_StubExtractorHandFound())
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def client_no_hand():
    app = create_app(classifier=_StubClassifier(), extractor=_StubExtractorNoHand())
    app.config["TESTING"] = True
    return app.test_client()


def test_health_endpoint(client_with_hand):
    resp = client_with_hand.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_missing_image_field(client_with_hand):
    resp = client_with_hand.post("/predict", data={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_with_detected_hand(client_with_hand):
    img_bytes = _make_test_image_bytes()
    resp = client_with_hand.post(
        "/predict",
        data={"image": (img_bytes, "test.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["detected"] is True
    assert body["letter"] == "A"
    assert body["confidence"] == pytest.approx(1.0)
    assert len(body["top_k"]) == 3


def test_predict_no_hand_detected(client_no_hand):
    img_bytes = _make_test_image_bytes()
    resp = client_no_hand.post(
        "/predict",
        data={"image": (img_bytes, "test.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["detected"] is False


def test_predict_invalid_image_bytes(client_with_hand):
    bad = io.BytesIO(b"not an image")
    resp = client_with_hand.post(
        "/predict",
        data={"image": (bad, "bad.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
