"""Flask backend serving ASL letter predictions from an uploaded image.

Endpoints:
    GET  /health   -> {"status": "ok", "model_loaded": bool}
    POST /predict  -> multipart/form-data with an "image" file field
                       {"detected": bool, "letter": str, "confidence": float, "top_k": [...]}

Run directly for local dev:
    python -m asl_detection.app_flask
(defaults to http://127.0.0.1:8000)

NOTE ON LIVE WEBCAM: this backend only handles *uploaded* image files. It
has been tested end-to-end with real uploaded images (see tests/ and the
README "Verified vs unverified" section) since this sandbox has no camera
or GPU. It does not itself open a webcam; the Streamlit UI's optional
camera widget is the (unverified-in-this-sandbox) live-capture path.
"""
from __future__ import annotations

import io
import os

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from asl_detection.infer import predict_image
from asl_detection.landmarks import HandLandmarkExtractor
from asl_detection.model import ASLClassifier

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "models", "asl_landmark_mlp.pt")


def create_app(classifier: ASLClassifier | None = None, extractor: HandLandmarkExtractor | None = None) -> Flask:
    """App factory. Pass in ``classifier``/``extractor`` (e.g. mocks) for testing;
    if omitted, the real trained model + MediaPipe extractor are lazy-loaded
    on first request so importing this module never requires them."""
    app = Flask(__name__)
    app.config["CLASSIFIER"] = classifier
    app.config["EXTRACTOR"] = extractor

    def get_classifier() -> ASLClassifier:
        if app.config["CLASSIFIER"] is None:
            app.config["CLASSIFIER"] = ASLClassifier.load(DEFAULT_MODEL_PATH)
        return app.config["CLASSIFIER"]

    def get_extractor() -> HandLandmarkExtractor:
        if app.config["EXTRACTOR"] is None:
            app.config["EXTRACTOR"] = HandLandmarkExtractor()
        return app.config["EXTRACTOR"]

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": app.config["CLASSIFIER"] is not None})

    @app.post("/predict")
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "missing 'image' file field"}), 400
        file = request.files["image"]
        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"could not read image: {exc}"}), 400

        image_rgb = np.array(image)
        result = predict_image(image_rgb, get_classifier(), get_extractor())
        return jsonify(result)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=8000, debug=False)
