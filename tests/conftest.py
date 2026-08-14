import numpy as np
import pytest

from asl_detection.landmarks import FEATURE_DIM, NUM_LANDMARKS


@pytest.fixture
def synthetic_raw_landmarks() -> np.ndarray:
    """A plausible (21, 3) raw MediaPipe landmark array (values in [0, 1] for x/y)."""
    rng = np.random.default_rng(0)
    raw = rng.uniform(low=0.2, high=0.8, size=(NUM_LANDMARKS, 3)).astype(np.float32)
    raw[0] = [0.5, 0.5, 0.0]  # wrist
    raw[9] = [0.5, 0.3, 0.0]  # middle MCP, distinct from wrist so scale != 0
    return raw


@pytest.fixture
def classes() -> list[str]:
    return ["A", "B", "C"]


@pytest.fixture
def synthetic_landmark_dataset(classes):
    """A tiny, cleanly-separable synthetic landmark dataset for fast model tests."""
    rng = np.random.default_rng(0)
    n_per_class = 40
    X, y = [], []
    for i, c in enumerate(classes):
        center = np.full(FEATURE_DIM, i * 5.0, dtype=np.float32)
        samples = center + rng.normal(scale=0.1, size=(n_per_class, FEATURE_DIM)).astype(np.float32)
        X.append(samples)
        y.extend([c] * n_per_class)
    X = np.concatenate(X, axis=0)
    return X, y
