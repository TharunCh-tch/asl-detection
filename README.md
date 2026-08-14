# ASL Alphabet Detection

A real, from-scratch American Sign Language (ASL) alphabet recognition system: MediaPipe hand-landmark
extraction feeding a small PyTorch MLP classifier, served through a Flask API and a Streamlit UI, with a
CLI for the simplest possible end-to-end check (image in, predicted letter out).

This project revisits work originally done during my MS CS/AI-ML at SUNY Buffalo (Sep-Dec 2024), which
was previously only described on LinkedIn and never published as code. This repository is a genuine
rebuild: real dataset, real training run, real measured numbers (see [Results](#results)) -- not a
reproduction of any specific historical benchmark.

## Problem

Translate a static photo of a single ASL hand sign (the 26 letters A-Z) into the letter it represents,
fast enough and reliably enough to plug into a live "ASL-to-text" style interface.

## Architecture

```
image --> MediaPipe HandLandmarker --> 21 (x, y, z) landmarks --> normalize --> 63-dim vector --> MLP --> letter
```

**1. Hand landmark extraction (MediaPipe Tasks `HandLandmarker`)**
Each image is run through MediaPipe's `HandLandmarker` (the modern MediaPipe Tasks API; the older
`mp.solutions.hands` API has been removed in current MediaPipe releases). It returns 21 3D landmarks per
detected hand (wrist + 4 joints x 5 fingers).

**2. Feature normalization (`src/asl_detection/landmarks.py`)**
Raw landmark coordinates depend on where the hand sits in the frame and how close it is to the camera.
Two normalization steps remove that dependence so the same gesture produces (numerically) the same
feature vector regardless of framing:
- **Translate**: subtract the wrist landmark so the wrist becomes the origin.
- **Scale**: divide by the wrist-to-middle-finger-MCP distance, a stable per-hand reference length.

This produces a fixed 63-dim vector (21 landmarks x 3 coordinates) per image.

**3. Classifier (`src/asl_detection/model.py`)**
A small PyTorch MLP: `63 -> 128 -> 64 -> 26`, with BatchNorm + Dropout(0.3) after each hidden layer,
trained with Adam + `ReduceLROnPlateau` LR scheduling and early stopping on validation loss.

**Why landmarks + a small MLP instead of a CNN over raw pixels** (the original LinkedIn description
mentioned a CNN trained in TensorFlow/Keras): MediaPipe's landmark extraction is itself a strong,
well-established feature extractor for hand pose. Once you have 63 clean numbers describing hand shape,
a lightweight MLP (or a tree ensemble) is enough to classify them well -- there's no need for a
multi-million-parameter CNN scanning raw pixels, which would need far more data and compute (i.e. a GPU)
to train competitively. This is a deliberate, documented architecture change from the original project,
not an attempt to reproduce its numbers -- see [Limitations](#limitations). This is still genuinely a
deep-learning model (a real multi-layer neural network trained end-to-end), just correctly sized for a
63-dimensional input rather than an oversized CNN.

**4. Serving**
- `src/asl_detection/infer.py` -- CLI: one image in, one predicted letter out.
- `src/asl_detection/app_flask.py` -- Flask backend, `POST /predict` (multipart image upload) and
  `GET /health`.
- `src/asl_detection/app_streamlit.py` -- Streamlit UI with an image-upload tab and a browser-camera
  snapshot tab (`st.camera_input`) for live-ish ASL-to-text translation.

## Dataset

**Source**: [`Marxulia/asl_sign_languages_alphabets_v03`](https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03)
on the Hugging Face Hub -- 10,873 RGB photographs of hand signs across all 26 English-alphabet letters
(A-Z), loaded with the `datasets` library, no API key required. It mirrors the style of the well-known
Kaggle "ASL Alphabet" datasets.

**License / provenance note**: the HF dataset card does not state an explicit license for the images
themselves. This repo therefore does **not** redistribute the raw images, treats the dataset as
research/educational use only, and documents the source dataset ID rather than committing the images.
The `MIT` license in `LICENSE` covers only the original code in this repository.

**What's actually used for training is not the raw images** -- it's the landmark vectors extracted from
them. `src/asl_detection/dataset.py` downloads the source dataset, runs MediaPipe's `HandLandmarker` over
every image, keeps only images where a hand was actually detected, and writes the resulting 63-dim
feature vectors + labels to `data/landmarks.parquet`. Exact counts (images attempted, images with a
detected hand, per-class breakdown) are written to `data/extraction_report.json` from the real extraction
run used for this repo's results, and summarized in [Results](#results) below.

Note: static images can't fully capture `J` and `Z`, which are signed with motion in real ASL. Like the
source dataset itself, this project treats them as static single-frame classes (the dataset's own posed
photos for those letters); this is a known simplification of real ASL, not a claim that motion-based
letters are being modeled dynamically.

To regenerate `data/landmarks.parquet` yourself:
```bash
python scripts/download_mediapipe_model.py     # fetches hand_landmarker.task (~7.5 MB, not committed)
python -m asl_detection.dataset                 # downloads dataset + runs extraction (~10-15 min on CPU)
```

## How to run

### Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/download_mediapipe_model.py
```

### CLI inference (simplest entry point)
```bash
python -m asl_detection.infer --image path/to/hand_photo.jpg
```
Prints the predicted letter, confidence, and top-3 alternatives.

### Train from scratch
```bash
python -m asl_detection.dataset     # build data/landmarks.parquet (see Dataset section)
python -m asl_detection.train       # trains the MLP, writes models/asl_landmark_mlp.pt, results.json, results.md
```

### Flask API
```bash
python -m asl_detection.app_flask
# POST an image to http://127.0.0.1:8000/predict as multipart/form-data field "image"
# GET  http://127.0.0.1:8000/health
```

### Streamlit UI
```bash
streamlit run src/asl_detection/app_streamlit.py
```
Upload an image, or (browser-permitting) take a live camera snapshot.

### Tests
```bash
pytest tests/ -v
```

## Results

See [`results.md`](results.md) / [`results.json`](results.json) for the full numbers from the actual
training run used in this repository (exact dataset size, per-class precision/recall/F1, confusion
matrix). Summary from that real run:

- **Landmark extraction**: 10,873 source images attempted -> hand detected in **8,399 (77.2%)**. The
  remaining ~23% are images MediaPipe couldn't find a confident hand in (cropped/occluded hands, unusual
  angles) and were correctly dropped rather than fed to the classifier as noise.
- **Split**: 8,399 samples -> 5,879 train / 1,260 val / 1,260 test (stratified 70/15/15).
- **Training**: 25.2s on CPU, 60 epochs, early stopping on validation loss.
- **Test set (1,260 held-out samples, never seen during training)**:

  | Metric | Value |
  |---|---|
  | Accuracy | **0.9183** |
  | Macro precision | 0.9180 |
  | Macro recall | 0.9164 |
  | Macro F1 | **0.9160** |

  Weakest classes were letters that are visually similar in this dataset's hand poses (`U` P=0.833/R=0.816,
  `Z` P=0.875/R=0.761, `P` P=0.841/R=0.787); strongest were visually distinct shapes (`A`, `L`, `Y` all
  >=0.97 F1). Full per-class table and confusion matrix: `results.md` / `results_confusion_matrix.png`.
- **End-to-end verification**: the CLI (`python -m asl_detection.infer`) and the Flask `/predict` endpoint
  were both run against real held-out sample photos from the source dataset (not synthetic data) during
  development and correctly classified them with high confidence -- e.g. a real "B" photo posted to
  `/predict` returned `{"letter": "B", "confidence": 0.998, ...}`.

## Limitations

- **No GPU in this build environment.** Training was run on CPU only. This is exactly why a lightweight
  landmark-based MLP was chosen over a raw-pixel CNN -- it's realistic to train well on CPU in minutes,
  where a CNN over full images would not be.
- **No webcam in this build/sandbox environment.** The image-upload inference path (CLI, Flask
  `/predict`, and the Streamlit "Upload image" tab) is exercised end-to-end with real images and covered
  by automated tests. The Streamlit "Camera snapshot" tab (`st.camera_input`, which captures from the
  *viewer's browser*, not a server-side camera loop) is implemented but has **not been visually verified**
  in this sandbox, since there is no camera or display here to test it against. Same honesty pattern as
  other repos in this portfolio that flag things verified in CI/description but not locally.
- **Static images only, no motion.** `J` and `Z` are signed with motion in real ASL; this project (like
  its source dataset) classifies static posed photos for all 26 letters, which is a known simplification.
- **No attempt to match the original LinkedIn-described numbers.** That project reported ">95% accuracy"
  and specific relative-improvement percentages (18% precision, 30% training time, 40% latency) against a
  particular baseline CNN run that no longer exists and wasn't preserved. This rebuild uses a different
  architecture (landmarks + MLP instead of a raw-pixel CNN), a different dataset, and reports whatever
  this actual run measured -- see `results.md` -- rather than reproducing those figures.
- **Dataset size and provenance.** ~10.9K source images across 26 classes from a single Hugging Face
  mirror dataset with no explicit license; not a large-scale or professionally curated corpus. Real-world
  generalization (different skin tones, lighting, hand sizes, backgrounds, camera angles not represented
  in this dataset) is untested.
