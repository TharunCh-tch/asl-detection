"""Download the MediaPipe HandLandmarker model file (not committed to git).

Usage:
    python scripts/download_mediapipe_model.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from asl_detection.landmarks import DEFAULT_MODEL_PATH as DEST  # noqa: E402
from asl_detection.landmarks import MODEL_URL, ensure_model_downloaded  # noqa: E402


def main() -> None:
    if os.path.exists(DEST):
        print(f"Already present: {DEST}")
        return
    print(f"Downloading {MODEL_URL} -> {DEST}")
    ensure_model_downloaded(DEST)
    size_mb = os.path.getsize(DEST) / (1024 * 1024)
    print(f"Done ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
