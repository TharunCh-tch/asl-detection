from asl_detection.evaluate import compute_metrics


def test_compute_metrics_perfect_predictions():
    labels = ["A", "B", "C"]
    y_true = ["A", "A", "B", "B", "C", "C"]
    y_pred = ["A", "A", "B", "B", "C", "C"]
    m = compute_metrics(y_true, y_pred, labels)
    assert m["accuracy"] == 1.0
    assert m["macro_precision"] == 1.0
    assert m["macro_recall"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["num_samples"] == 6
    # confusion matrix should be diagonal
    cm = m["confusion_matrix"]
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            assert val == (2 if i == j else 0)


def test_compute_metrics_known_values():
    labels = ["A", "B"]
    # A: 1 correct, 1 wrong (predicted B). B: 2 correct.
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    m = compute_metrics(y_true, y_pred, labels)
    assert m["accuracy"] == 0.75
    # precision_A = 1/1 = 1.0 (1 correct A pred out of 1 A pred)
    # precision_B = 2/3
    # macro precision = (1.0 + 2/3) / 2
    assert abs(m["macro_precision"] - (1.0 + 2 / 3) / 2) < 1e-6
    # recall_A = 1/2, recall_B = 2/2 = 1.0
    assert abs(m["macro_recall"] - (0.5 + 1.0) / 2) < 1e-6


def test_compute_metrics_handles_missing_class_in_batch():
    # class "C" never appears in this batch but is still in labels
    labels = ["A", "B", "C"]
    y_true = ["A", "B"]
    y_pred = ["A", "B"]
    m = compute_metrics(y_true, y_pred, labels)
    assert m["accuracy"] == 1.0
    assert len(m["confusion_matrix"]) == 3
