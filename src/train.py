"""Train candidate models and compare them on the test set.

- GroupShuffleSplit on patient_nbr (leakage prevention)
- Primary metric: PR-AUC (average precision); tie-break: lower Brier
- Bootstrap 95% CI for PR-AUC / ROC-AUC (robustness of differences)
- Saves: models/model.joblib (best full Pipeline), models/split_indices.npz,
         reports/metrics.md
"""
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.config import (MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE)
from src.data_loader import load_and_clean, make_target
from src.preprocess import build_preprocessor, prepare_features

N_BOOTSTRAP = 1000


def get_models():
    """Candidate models (spec models 1-4 + approved LightGBM addition)."""
    return {
        "LogReg_baseline": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "LogReg_L2_C0.1": LogisticRegression(
            penalty="l2", C=0.1, class_weight="balanced",
            max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced", random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
            verbose=-1),
    }


def bootstrap_ci(y_true, proba, n=N_BOOTSTRAP, seed=RANDOM_STATE):
    """95% percentile bootstrap CI for PR-AUC and ROC-AUC."""
    rng = np.random.default_rng(seed)
    n_test = len(y_true)
    y_true = np.asarray(y_true)
    pr, roc = [], []
    for _ in range(n):
        idx = rng.integers(0, n_test, n_test)
        if y_true[idx].sum() == 0:
            continue
        pr.append(average_precision_score(y_true[idx], proba[idx]))
        roc.append(roc_auc_score(y_true[idx], proba[idx]))
    return (np.percentile(pr, 2.5), np.percentile(pr, 97.5),
            np.percentile(roc, 2.5), np.percentile(roc, 97.5))


def eval_fitted(pipe, X_te, y_te):
    """Metrics for an already-fitted estimator (Pipeline or calibrated)."""
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    pr_lo, pr_hi, roc_lo, roc_hi = bootstrap_ci(y_te, proba)
    return proba, {
        "pr_auc": average_precision_score(y_te, proba),
        "pr_auc_ci": f"[{pr_lo:.3f}, {pr_hi:.3f}]",
        "roc_auc": roc_auc_score(y_te, proba),
        "roc_auc_ci": f"[{roc_lo:.3f}, {roc_hi:.3f}]",
        "recall@0.5": recall_score(y_te, pred),
        "precision@0.5": precision_score(y_te, pred, zero_division=0),
        "f1@0.5": f1_score(y_te, pred, zero_division=0),
        "brier": brier_score_loss(y_te, proba),
    }


def evaluate_model(model, X_tr, y_tr, X_te, y_te):
    """Fit a classifier inside a fresh Pipeline, then evaluate."""
    pipe = Pipeline([("pre", build_preprocessor()), ("clf", model)])
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    proba, metrics = eval_fitted(pipe, X_te, y_te)
    metrics["fit_seconds"] = round(fit_s, 1)
    return pipe, proba, metrics


def main():
    df = load_and_clean()
    X = prepare_features(df)
    y = make_target(df)

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(X, y, groups=df["patient_nbr"]))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    MODELS_DIR.mkdir(exist_ok=True)
    np.savez(MODELS_DIR / "split_indices.npz", train=tr_idx, test=te_idx)
    print(f"Split: train={len(tr_idx):,} test={len(te_idx):,} "
          f"pos_rate(test)={y_te.mean():.2%}")

    results, pipes = {}, {}
    for name, model in get_models().items():
        pipe, proba, metrics = evaluate_model(model, X_tr, y_tr, X_te, y_te)
        results[name], pipes[name] = metrics, pipe
        print(f"[{name}] PR-AUC={metrics['pr_auc']:.4f} "
              f"ROC-AUC={metrics['roc_auc']:.4f} "
              f"Brier={metrics['brier']:.4f} "
              f"({metrics['fit_seconds']}s)", flush=True)

    # Calibrated version of the best base model (approved addition)
    best_name = max(results,
                    key=lambda k: (results[k]["pr_auc"],
                                   -results[k]["brier"]))
    print(f"\nBest base model: {best_name} -> fitting CalibratedClassifierCV(cv=3)")
    calib = CalibratedClassifierCV(estimator=pipes[best_name], cv=3)
    t0 = time.time()
    calib.fit(X_tr, y_tr)  # raw features: inner estimator is a full Pipeline
    print(f"calibration fit: {time.time() - t0:.0f}s")
    proba, metrics = eval_fitted(calib, X_te, y_te)
    metrics["fit_seconds"] = round(time.time() - t0, 1)
    results[f"{best_name}+Calibrated"] = metrics
    pipes[f"{best_name}+Calibrated"] = calib

    res_df = (pd.DataFrame(results).T
              .sort_values("pr_auc", ascending=False))
    best = res_df.index[0]

    joblib.dump(pipes[best], MODELS_DIR / "model.joblib")
    np.save(MODELS_DIR / "feature_names.npy",
            pipes[best].named_steps["pre"].get_feature_names_out())
    print(f"\nSaved best model: {best} -> models/model.joblib")

    md = ["# Model Comparison (test set)", "",
          f"- Split: GroupShuffleSplit on `patient_nbr` "
          f"(test_size={TEST_SIZE}, seed={RANDOM_STATE})",
          f"- n_train={len(tr_idx):,}, n_test={len(te_idx):,}, "
          f"positive rate (test)={y_te.mean():.2%}",
          "- Metrics at threshold 0.5; PR-AUC is the primary metric.",
          "- Bootstrap: 1,000 resamples of the test set (95% percentile CI).",
          "", res_df.to_markdown(floatfmt=".4f"), "",
          f"**Selected model:** `{best}` with PR-AUC = "
          f"{res_df.loc[best, 'pr_auc']:.4f}", ""]
    (REPORTS_DIR / "metrics.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
