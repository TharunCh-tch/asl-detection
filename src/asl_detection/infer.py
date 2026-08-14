"""CLI inference: predict an ASL letter from a single image file.

Usage:
    python -m asl_detection.infer --image path/to/hand.jpg
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from asl_detection.landmarks import HandLandmarkExtractor
from asl_detection.model import ASLClassifier

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "models", "asl_landmark_mlp.pt")


def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def predict_image(
    image_rgb: np.ndarray,
    classifier: ASLClassifier,
    extractor: HandLandmarkExtractor,
    top_k: int = 3,
) -> dict:
    """Run landmark extraction + classification on one RGB image array.

    Returns a dict with the predicted letter, confidence, and top-k
    predictions, or a dict with ``"detected": False`` if no hand was found.
    """
    result = extractor.extract(image_rgb)
    if result is None:
        return {"detected": False, "message": "No hand detected in image."}

    probs = classifier.predict_proba(result.features[np.newaxis, :])[0]
    order = np.argsort(probs)[::-1]
    top = [{"letter": classifier.classes[i], "confidence": float(probs[i])} for i in order[:top_k]]

    return {
        "detected": True,
        "letter": top[0]["letter"],
        "confidence": top[0]["confidence"],
        "top_k": top,
        "handedness": result.handedness,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to an input image file.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model checkpoint.")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    classifier = ASLClassifier.load(args.model)
    with HandLandmarkExtractor() as extractor:
        image_rgb = load_image_rgb(args.image)
        result = predict_image(image_rgb, classifier, extractor, top_k=args.top_k)

    if not result["detected"]:
        print(result["message"])
        return

    print(f"Predicted letter: {result['letter']}  (confidence: {result['confidence']:.3f})")
    print("Top predictions:")
    for entry in result["top_k"]:
        print(f"  {entry['letter']}: {entry['confidence']:.3f}")


if __name__ == "__main__":
    main()
