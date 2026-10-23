"""ONNX parity tests — Phase 6.

Tests:
    test_feature_schema_valid     — feature_schema.json is well-formed
    test_model_joblib_predict     — sklearn pipeline predict_proba works
    test_onnx_file_exists         — model.onnx produced by export_onnx.py
    test_onnx_parity              — max |sklearn − onnx| < 1e-5
    test_threshold_json_valid     — threshold.json has required keys

Run from project root:
    pytest tests/test_onnx_parity.py -v

Environment note:
    test_onnx_file_exists and test_onnx_parity are auto-skipped when
    skl2onnx / onnxruntime are not installed (CI-safe).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
ONNX_PATH = MODELS_DIR / "model.onnx"
SPLIT_PATH = MODELS_DIR / "split_indices.npz"


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_data():
    """Load the test split (X, y) once per session."""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data_loader import load_and_clean, make_target
    from src.preprocess import prepare_features

    df = load_and_clean()
    X = prepare_features(df)
    y = make_target(df)
    idx = np.load(SPLIT_PATH)
    return X.iloc[idx["test"]], y.iloc[idx["test"]]


@pytest.fixture(scope="session")
def sklearn_pipe():
    return joblib.load(MODELS_DIR / "model.joblib")


# ------------------------------------------------------------------
# Tests — always run
# ------------------------------------------------------------------

def test_threshold_json_valid():
    """threshold.json must have the four required numeric keys."""
    assert THRESHOLD_PATH.exists(), "models/threshold.json missing"
    data = json.loads(THRESHOLD_PATH.read_text())
    for key in ("threshold", "recall", "precision", "f1"):
        assert key in data, f"Key '{key}' missing from threshold.json"
        assert isinstance(data[key], float), f"'{key}' must be float"
    assert 0 < data["threshold"] < 1
    assert 0 < data["precision"] <= 1
    assert data["precision"] >= 0.30, (
        f"Precision {data['precision']:.4f} < 0.30 — violates clinical rule")


def test_model_joblib_predict(sklearn_pipe, test_data):
    """sklearn pipeline must return valid probabilities on the test set."""
    X_te, y_te = test_data
    proba = sklearn_pipe.predict_proba(X_te)[:, 1]
    assert proba.shape == (len(X_te),)
    assert np.all(proba >= 0) and np.all(proba <= 1), "Probs out of [0,1]"
    # PR-AUC sanity (should be >> baseline ~0.1135)
    from sklearn.metrics import average_precision_score
    pr_auc = average_precision_score(y_te, proba)
    assert pr_auc > 0.18, f"PR-AUC {pr_auc:.4f} suspiciously low"


def test_feature_schema_valid():
    """feature_schema.json must exist with required keys."""
    assert SCHEMA_PATH.exists(), (
        "models/feature_schema.json missing — run: python -m src.export_onnx")
    schema = json.loads(SCHEMA_PATH.read_text())
    for key in ("n_onnx_features", "threshold", "feature_groups"):
        assert key in schema, f"Key '{key}' missing from feature_schema.json"
    assert schema["n_onnx_features"] > 0
    assert 0 < schema["threshold"] < 1


# ------------------------------------------------------------------
# Tests — skip gracefully when ONNX toolchain is absent
# ------------------------------------------------------------------

skl2onnx_available = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("skl2onnx"),
    reason="skl2onnx not installed",
)
onnxruntime_available = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("onnxruntime"),
    reason="onnxruntime not installed",
)


@skl2onnx_available
def test_onnx_file_exists():
    """model.onnx must exist and be non-trivially sized (> 50 KB)."""
    assert ONNX_PATH.exists(), (
        "models/model.onnx missing — run: python -m src.export_onnx")
    size_kb = ONNX_PATH.stat().st_size / 1024
    assert size_kb > 50, f"model.onnx suspiciously small: {size_kb:.1f} KB"


@skl2onnx_available
@onnxruntime_available
def test_onnx_parity(sklearn_pipe, test_data):
    """max |sklearn_proba − onnx_proba| must be < 1e-5."""
    if not ONNX_PATH.exists():
        pytest.skip("model.onnx not found — run export_onnx first")

    import onnxruntime as rt

    X_te, _ = test_data
    sk_proba = sklearn_pipe.predict_proba(X_te)[:, 1]

    sess = rt.InferenceSession(str(ONNX_PATH))
    input_names = [inp.name for inp in sess.get_inputs()]
    feed = {}
    for name in input_names:
        col = X_te[name]
        if col.dtype == object:
            feed[name] = col.values.reshape(-1, 1).astype(bytes)
        else:
            feed[name] = col.values.reshape(-1, 1).astype(np.float32)

    onnx_out = sess.run(None, feed)
    onnx_proba = onnx_out[1]
    if isinstance(onnx_proba, list):
        onnx_proba = np.array([d[1] for d in onnx_proba])
    else:
        onnx_proba = onnx_proba[:, 1]

    max_diff = np.abs(sk_proba - onnx_proba).max()
    assert max_diff < 1e-5, (
        f"ONNX parity FAIL: max |sklearn − onnx| = {max_diff:.2e} ≥ 1e-5")
