"""Metrics computation, decoupled from training so it's independently unit-testable."""
from __future__ import annotations

from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict:
    """Compute accuracy + macro precision/recall/F1 + confusion matrix.

    ``labels`` fixes the class ordering used for the confusion matrix and
    per-class report, so results are stable/reproducible regardless of which
    classes happen to appear in a given batch.
    """
    labels = list(labels)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
        "num_samples": len(y_true),
    }


def plot_confusion_matrix(cm, labels: Sequence[str], out_path: str, title: str = "Confusion Matrix") -> None:
    """Save a confusion-matrix heatmap PNG. Imports matplotlib lazily so
    importing this module doesn't require a display backend for tests."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
