import numpy as np

from asl_detection.landmarks import FEATURE_DIM
from asl_detection.model import ASLMLP, ASLClassifier, FeatureScaler


def test_mlp_forward_shape():
    model = ASLMLP(num_classes=5)
    model.eval()
    import torch

    x = torch.randn(8, FEATURE_DIM)
    out = model(x)
    assert out.shape == (8, 5)


def test_feature_scaler_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=3.0, scale=2.0, size=(100, FEATURE_DIM)).astype(np.float32)
    scaler = FeatureScaler().fit(X)
    transformed = scaler.transform(X)
    assert np.allclose(transformed.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(transformed.std(axis=0), 1, atol=1e-4)

    d = scaler.to_dict()
    scaler2 = FeatureScaler.from_dict(d)
    np.testing.assert_allclose(scaler.mean, scaler2.mean)
    np.testing.assert_allclose(scaler.std, scaler2.std)


def test_classifier_fit_predict_on_separable_data(synthetic_landmark_dataset, classes):
    X, y = synthetic_landmark_dataset
    clf = ASLClassifier(classes=classes)
    clf.fit(X, y, X, y, epochs=25, verbose=False)

    preds = clf.predict(X)
    accuracy = sum(p == t for p, t in zip(preds, y)) / len(y)
    # Synthetic classes are well-separated (cluster centers 5.0 apart, std 0.1) so
    # a correctly-wired train/predict loop should fit them almost perfectly.
    assert accuracy > 0.9


def test_classifier_predict_proba_sums_to_one(synthetic_landmark_dataset, classes):
    X, y = synthetic_landmark_dataset
    clf = ASLClassifier(classes=classes)
    clf.fit(X, y, epochs=5, verbose=False)
    probs = clf.predict_proba(X[:10])
    assert probs.shape == (10, len(classes))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_classifier_save_and_load_roundtrip(tmp_path, synthetic_landmark_dataset, classes):
    X, y = synthetic_landmark_dataset
    clf = ASLClassifier(classes=classes)
    clf.fit(X, y, epochs=5, verbose=False)
    preds_before = clf.predict(X[:20])

    path = str(tmp_path / "model.pt")
    clf.save(path)
    loaded = ASLClassifier.load(path)
    preds_after = loaded.predict(X[:20])

    assert preds_before == preds_after
    assert loaded.classes == classes
