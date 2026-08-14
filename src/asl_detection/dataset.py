"""Build a landmark-feature dataset from a real ASL alphabet image dataset.

Source dataset: "Marxulia/asl_sign_languages_alphabets_v03" on the Hugging
Face Hub (public, no auth required) -- 10,873 RGB photos of hand gestures
for the 26 English-alphabet letters (A-Z). It mirrors the well-known
Kaggle "ASL Alphabet" style datasets. No explicit license file was found on
the HF dataset card, so this project treats it as research/educational-use
only and does not redistribute the raw images (see README "Dataset" section).

This script downloads that dataset via `datasets.load_dataset` (no API key
needed) and runs MediaPipe's HandLandmarker over every image, keeping only
the images where a hand was actually detected. The resulting per-image
normalized 63-dim landmark vectors + labels are what the classifier trains
on -- NOT raw pixels.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset

from asl_detection.landmarks import FEATURE_DIM, HandLandmarkExtractor

DATASET_ID = "Marxulia/asl_sign_languages_alphabets_v03"

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
LANDMARKS_PATH = os.path.join(DATA_DIR, "landmarks.parquet")
EXTRACTION_REPORT_PATH = os.path.join(DATA_DIR, "extraction_report.json")


def build_landmark_dataset(
    max_per_class: int | None = None,
    seed: int = 42,
    min_hand_detection_confidence: float = 0.5,
) -> pd.DataFrame:
    """Download the source image dataset and extract hand landmarks.

    Parameters
    ----------
    max_per_class:
        If set, cap the number of *source images attempted* per class
        (before hand-detection filtering) so extraction runs in a bounded
        amount of time on a CPU-only sandbox. None = use every image.
    """
    print(f"Loading source dataset '{DATASET_ID}' from Hugging Face Hub...")
    ds = load_dataset(DATASET_ID)["train"]
    label_names = ds.features["label"].names
    print(f"Loaded {len(ds)} images across {len(label_names)} classes: {label_names}")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(ds))
    if max_per_class is not None:
        by_label: dict[int, list[int]] = {}
        for i, lbl in enumerate(ds["label"]):
            by_label.setdefault(lbl, []).append(i)
        selected = []
        for lbl, idxs in by_label.items():
            idxs = np.array(idxs)
            rng.shuffle(idxs)
            selected.extend(idxs[:max_per_class].tolist())
        indices = np.array(sorted(selected))
    print(f"Attempting landmark extraction on {len(indices)} images...")

    extractor = HandLandmarkExtractor(min_hand_detection_confidence=min_hand_detection_confidence)

    rows = []
    detected = 0
    attempted_per_class = Counter()
    detected_per_class = Counter()
    t0 = time.time()
    try:
        for n, i in enumerate(indices):
            ex = ds[int(i)]
            label_idx = ex["label"]
            label = label_names[label_idx]
            attempted_per_class[label] += 1
            img = ex["image"].convert("RGB")
            arr = np.array(img)
            result = extractor.extract(arr)
            if result is None:
                continue
            detected += 1
            detected_per_class[label] += 1
            row = {f"f{j}": v for j, v in enumerate(result.features)}
            row["label"] = label
            rows.append(row)

            if (n + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  {n + 1}/{len(indices)} processed, {detected} hands detected, {elapsed:.0f}s elapsed")
    finally:
        extractor.close()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Hand detected in {detected}/{len(indices)} images "
          f"({100 * detected / len(indices):.1f}%).")

    df = pd.DataFrame(rows)
    assert df.shape[1] == FEATURE_DIM + 1, f"expected {FEATURE_DIM + 1} columns, got {df.shape[1]}"

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(LANDMARKS_PATH, index=False)

    report = {
        "source_dataset": DATASET_ID,
        "classes": label_names,
        "num_classes": len(label_names),
        "images_attempted": int(len(indices)),
        "images_with_hand_detected": int(detected),
        "hand_detection_rate": detected / len(indices),
        "attempted_per_class": dict(attempted_per_class),
        "detected_per_class": dict(detected_per_class),
        "extraction_seconds": elapsed,
        "feature_dim": FEATURE_DIM,
        "seed": seed,
        "max_per_class_cap": max_per_class,
    }
    with open(EXTRACTION_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved landmarks to {LANDMARKS_PATH}")
    print(f"Saved extraction report to {EXTRACTION_REPORT_PATH}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap source images attempted per class (default: use all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()
    build_landmark_dataset(
        max_per_class=args.max_per_class,
        seed=args.seed,
        min_hand_detection_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
