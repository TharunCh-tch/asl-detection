"""Download the MediaPipe HandLandmarker model file (not committed to git).

Usage:
    python scripts/download_mediapipe_model.py
"""
import os
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEST = os.path.join(os.path.dirname(__file__), "..", "models", "mediapipe", "hand_landmarker.task")
DEST = os.path.normpath(DEST)


def main() -> None:
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if os.path.exists(DEST):
        print(f"Already present: {DEST}")
        return
    print(f"Downloading {MODEL_URL} -> {DEST}")
    urllib.request.urlretrieve(MODEL_URL, DEST)
    size_mb = os.path.getsize(DEST) / (1024 * 1024)
    print(f"Done ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
