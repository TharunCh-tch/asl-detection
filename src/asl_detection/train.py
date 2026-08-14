"""Train the landmark MLP classifier and write results.json / results.md.

Usage:
    python -m asl_detection.train
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from asl_detection.evaluate import compute_metrics, plot_confusion_matrix
from asl_detection.landmarks import FEATURE_DIM
from asl_detection.model import ASLClassifier

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
LANDMARKS_PATH = os.path.join(REPO_ROOT, "data", "landmarks.parquet")
EXTRACTION_REPORT_PATH = os.path.join(REPO_ROOT, "data", "extraction_report.json")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "asl_landmark_mlp.pt")
RESULTS_JSON_PATH = os.path.join(REPO_ROOT, "results.json")
RESULTS_MD_PATH = os.path.join(REPO_ROOT, "results.md")
CONFUSION_MATRIX_PNG = os.path.join(REPO_ROOT, "results_confusion_matrix.png")


def load_landmark_dataset(path: str = LANDMARKS_PATH) -> tuple[np.ndarray, list[str], list[str]]:
    df = pd.read_parquet(path)
    feature_cols = [f"f{i}" for i in range(FEATURE_DIM)]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].tolist()
    classes = sorted(set(y))
    return X, y, classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, classes = load_landmark_dataset()
    print(f"Loaded {len(X)} landmark samples across {len(classes)} classes.")

    # 70/15/15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=args.seed, stratify=y_temp
    )
    print(f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    clf = ASLClassifier(classes=classes)
    t0 = time.time()
    history = clf.fit(X_train, y_train, X_val, y_val, epochs=args.epochs)
    train_seconds = time.time() - t0
    print(f"Training took {train_seconds:.1f}s")

    clf.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    y_pred = clf.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, labels=classes)
    print(f"Test accuracy={metrics['accuracy']:.4f}  "
          f"macro_precision={metrics['macro_precision']:.4f}  "
          f"macro_recall={metrics['macro_recall']:.4f}  "
          f"macro_f1={metrics['macro_f1']:.4f}")

    plot_confusion_matrix(metrics["confusion_matrix"], classes, CONFUSION_MATRIX_PNG,
                           title="ASL Landmark MLP - Test Set Confusion Matrix")

    extraction_report = {}
    if os.path.exists(EXTRACTION_REPORT_PATH):
        with open(EXTRACTION_REPORT_PATH) as f:
            extraction_report = json.load(f)

    results = {
        "model": "ASLMLP (63 -> 128 -> 64 -> num_classes, BatchNorm + Dropout(0.3), Adam + ReduceLROnPlateau)",
        "framework": "PyTorch",
        "feature_representation": "MediaPipe HandLandmarker 21 landmarks x (x,y,z), wrist-centered, "
                                   "scaled by wrist-to-middle-MCP distance (63-dim vector)",
        "dataset_source": extraction_report.get("source_dataset"),
        "num_classes": len(classes),
        "classes": classes,
        "dataset_sizes": {
            "total_landmark_samples": len(X),
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "extraction_report": extraction_report,
        "train_seconds": train_seconds,
        "epochs_run": len(history["train_loss"]),
        "training_history": history,
        "test_metrics": metrics,
        "seed": args.seed,
    }
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {RESULTS_JSON_PATH}")

    write_results_md(results)
    print(f"Saved {RESULTS_MD_PATH}")


def write_results_md(results: dict) -> None:
    m = results["test_metrics"]
    lines = []
    lines.append("# Results\n")
    lines.append(f"- **Model**: {results['model']}")
    lines.append(f"- **Framework**: {results['framework']}")
    lines.append(f"- **Feature representation**: {results['feature_representation']}")
    lines.append(f"- **Source dataset**: {results['dataset_source']}")
    lines.append(f"- **Classes ({results['num_classes']})**: {', '.join(results['classes'])}")
    ds = results["dataset_sizes"]
    lines.append(f"- **Landmark samples**: {ds['total_landmark_samples']} total "
                 f"(train {ds['train']} / val {ds['val']} / test {ds['test']}, 70/15/15 stratified split)")
    er = results.get("extraction_report") or {}
    if er:
        lines.append(f"- **Hand-detection rate during extraction**: "
                      f"{er.get('images_with_hand_detected')}/{er.get('images_attempted')} "
                      f"({100 * er.get('hand_detection_rate', 0):.1f}%)")
    lines.append(f"- **Training time**: {results['train_seconds']:.1f}s on CPU "
                 f"over {results['epochs_run']} epochs (early stopping on val loss)\n")

    lines.append("## Test set metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Accuracy | {m['accuracy']:.4f} |")
    lines.append(f"| Macro precision | {m['macro_precision']:.4f} |")
    lines.append(f"| Macro recall | {m['macro_recall']:.4f} |")
    lines.append(f"| Macro F1 | {m['macro_f1']:.4f} |")
    lines.append(f"| Test samples | {m['num_samples']} |\n")

    lines.append("## Per-class report (test set)\n")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for c in results["classes"]:
        r = m["per_class_report"].get(c, {})
        lines.append(f"| {c} | {r.get('precision', 0):.3f} | {r.get('recall', 0):.3f} | "
                      f"{r.get('f1-score', 0):.3f} | {int(r.get('support', 0))} |")
    lines.append("")
    lines.append("Confusion matrix image: `results_confusion_matrix.png`. Full raw numbers: `results.json`.\n")

    with open(RESULTS_MD_PATH, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
