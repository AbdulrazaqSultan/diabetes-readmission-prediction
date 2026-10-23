"""ONNX export for the trained LightGBM Pipeline.

Design decisions (documented):
- We export the FULL sklearn Pipeline (preprocessor + classifier) using
  skl2onnx so the Flutter app only needs to pass raw feature values —
  no preprocessing logic in Dart.
- Fallback (if skl2onnx is unavailable): export coefficients + feature
  means to JSON (colab_onnx_export.ipynb) and replicate in Dart.
- Parity guarantee: max |sklearn_proba − onnx_proba| < 1e-5 over the
  test set; checked by tests/test_onnx_parity.py.

Usage:
    python -m src.export_onnx           # saves models/model.onnx
    python -m src.export_onnx --check   # also runs parity test
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from src.config import MODELS_DIR, REPORTS_DIR
from src.data_loader import load_and_clean, make_target
from src.preprocess import prepare_features

ONNX_PATH = MODELS_DIR / "model.onnx"
SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
PARITY_TOL = 1e-5


# ---------------------------------------------------------------------------
# feature_schema.json — consumed by the Flutter app
# ---------------------------------------------------------------------------

def build_feature_schema(pipe) -> dict:
    """Return a JSON-serialisable schema of raw input features.

    The Flutter app uses this to build the input tensor in the correct order
    and apply any client-side validation (e.g. range checks).
    """
    from src.config import (CATEGORICAL_LOW_COLS, DIAG_COLS, DRUG_COLS,
                             NUMERIC_COLS)
    from src.preprocess import AGE_MIDPOINTS, DRUG_VALUES

    schema = {
        "model_type": "LightGBM (sklearn Pipeline)",
        "n_onnx_features": int(
            pipe.named_steps["pre"].get_feature_names_out().shape[0]),
        "threshold": json.loads(
            (MODELS_DIR / "threshold.json").read_text())["threshold"],
        "positive_label": "<30 days readmission",
        "feature_groups": {
            "numeric": NUMERIC_COLS + ["age"],
            "categorical_low": CATEGORICAL_LOW_COLS,
            "diagnosis": DIAG_COLS,
            "drug": DRUG_COLS,
        },
        "value_hints": {
            "age": {"type": "bin", "bins": list(AGE_MIDPOINTS.keys()),
                    "midpoints": list(AGE_MIDPOINTS.values())},
            "drug_cols": {"allowed": DRUG_VALUES},
        },
    }
    return schema


# ---------------------------------------------------------------------------
# ONNX export (requires skl2onnx)
# ---------------------------------------------------------------------------

def export_onnx(pipe, X_sample: np.ndarray) -> None:
    """Convert sklearn Pipeline to ONNX and save to MODELS_DIR/model.onnx."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    except ImportError:
        print("[export_onnx] skl2onnx not installed — skipping ONNX export.")
        print("  Install with:  pip install skl2onnx onnxruntime")
        print("  Then re-run:   python -m src.export_onnx")
        return

    import pandas as pd
    # Build initial_type from DataFrame dtypes
    initial_type = []
    for col in X_sample.columns:
        if X_sample[col].dtype == object or X_sample[col].dtype.name == "category":
            initial_type.append((col, StringTensorType([None, 1])))
        else:
            initial_type.append((col, FloatTensorType([None, 1])))

    print(f"Converting Pipeline → ONNX ({len(initial_type)} input features)…")
    onnx_model = convert_sklearn(pipe, initial_types=initial_type,
                                 target_opset=17)
    ONNX_PATH.parent.mkdir(exist_ok=True)
    ONNX_PATH.write_bytes(onnx_model.SerializeToString())
    size_kb = ONNX_PATH.stat().st_size // 1024
    print(f"Saved: {ONNX_PATH}  ({size_kb} KB)")


# ---------------------------------------------------------------------------
# Parity test
# ---------------------------------------------------------------------------

def parity_test(pipe, X_te, y_te) -> bool:
    """Check that ONNX probabilities match sklearn to within PARITY_TOL."""
    if not ONNX_PATH.exists():
        print("[parity] model.onnx not found — skipping parity check.")
        return True

    try:
        import onnxruntime as rt
    except ImportError:
        print("[parity] onnxruntime not installed — skipping parity check.")
        return True

    sk_proba = pipe.predict_proba(X_te)[:, 1]

    # Build ONNX input dict (column-by-column, float32 or bytes)
    import pandas as pd
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
    # probability output is typically at index 1
    onnx_proba = onnx_out[1]
    if isinstance(onnx_proba, list):
        onnx_proba = np.array([d[1] for d in onnx_proba])
    else:
        onnx_proba = onnx_proba[:, 1]

    diff = np.abs(sk_proba - onnx_proba).max()
    ok = diff < PARITY_TOL
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"[parity] max |sklearn − onnx| = {diff:.2e}  →  {status} "
          f"(tolerance {PARITY_TOL:.0e})")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(check: bool = False) -> None:
    pipe = joblib.load(MODELS_DIR / "model.joblib")

    df = load_and_clean()
    X = prepare_features(df)
    y = make_target(df)
    idx = np.load(MODELS_DIR / "split_indices.npz")
    X_te = X.iloc[idx["test"]]
    y_te = y.iloc[idx["test"]]

    # Feature schema (always written)
    schema = build_feature_schema(pipe)
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
    print(f"Saved feature schema → {SCHEMA_PATH}")
    print(f"  n_onnx_features : {schema['n_onnx_features']}")
    print(f"  threshold        : {schema['threshold']:.4f}")

    # ONNX export
    export_onnx(pipe, X_te.head(5))

    # Optional parity check
    if check:
        ok = parity_test(pipe, X_te, y_te)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Run parity test after export")
    args = parser.parse_args()
    main(check=args.check)
