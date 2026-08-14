"""Streamlit UI for live-ish ASL-to-text translation.

Run with:
    streamlit run src/asl_detection/app_streamlit.py

Two input modes:
  1. File upload (JPG/PNG) -- fully tested end-to-end in this sandbox
     (no display/camera available here, but the inference path underneath
     is identical for both modes and is covered by the CLI + Flask tests).
  2. Browser camera snapshot via st.camera_input -- captures a single frame
     from the *viewer's* browser camera (not a server-side webcam loop).
     This is the closest this project gets to "live" capture without a
     physical camera in this sandbox; it has NOT been visually verified
     here since no camera/browser session is available. See README
     "Limitations" for the honest verified/unverified breakdown.
"""
from __future__ import annotations

import os

import numpy as np
import streamlit as st
from PIL import Image

from asl_detection.infer import predict_image
from asl_detection.landmarks import HandLandmarkExtractor
from asl_detection.model import ASLClassifier

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "models", "asl_landmark_mlp.pt")


@st.cache_resource
def load_classifier() -> ASLClassifier:
    return ASLClassifier.load(DEFAULT_MODEL_PATH)


@st.cache_resource
def load_extractor() -> HandLandmarkExtractor:
    return HandLandmarkExtractor()


def render_prediction(image_rgb: np.ndarray) -> None:
    classifier = load_classifier()
    extractor = load_extractor()
    result = predict_image(image_rgb, classifier, extractor)

    if not result["detected"]:
        st.warning(result["message"])
        return

    st.success(f"Predicted letter: **{result['letter']}**  "
               f"(confidence: {result['confidence']:.1%})")
    st.write("Top predictions:")
    st.bar_chart({entry["letter"]: entry["confidence"] for entry in result["top_k"]})


def main() -> None:
    st.set_page_config(page_title="ASL Detection", page_icon="🤟")
    st.title("ASL Alphabet Detection")
    st.caption(
        "MediaPipe hand-landmark extraction + a PyTorch MLP classifier. "
        "Upload a photo of a single hand sign, or (if your browser has a "
        "camera) capture a live snapshot below."
    )

    tab_upload, tab_camera = st.tabs(["Upload image", "Camera snapshot (unverified in dev sandbox)"])

    with tab_upload:
        uploaded = st.file_uploader("Upload a hand-sign image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Input image", width=300)
            render_prediction(np.array(image))

    with tab_camera:
        st.info(
            "This captures a single frame from your browser's camera via "
            "`st.camera_input`. It has not been visually verified in the "
            "build sandbox (no camera device there) -- the underlying "
            "inference call is the same one covered by the automated tests."
        )
        snapshot = st.camera_input("Take a picture of a hand sign")
        if snapshot is not None:
            image = Image.open(snapshot).convert("RGB")
            render_prediction(np.array(image))


if __name__ == "__main__":
    main()
