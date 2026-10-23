"""Threshold selection + evaluation figures for the selected model.

Rule (clinical): highest Recall subject to Precision >= 0.30.
Appends the decision to reports/metrics.md and saves the threshold
to models/threshold.json (needed by the Flutter app).
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.config import (FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR)
from src.data_loader import load_and_clean, make_target
from src.preprocess import prepare_features

MIN_PRECISION = 0.30


def load_test_set():
    df = load_and_clean()
    X = prepare_features(df)
    y = make_target(df)
    idx = np.load(MODELS_DIR / "split_indices.npz")
    return X.iloc[idx["test"]], y.iloc[idx["test"]]


def select_threshold(y_true, proba, n_grid=100):
    """Highest recall with precision >= MIN_PRECISION over 0.05..0.90."""
    grid = np.linspace(0.05, 0.90, n_grid)
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        rows.append({"threshold": t, "precision": p, "recall": r, "f1": f})
    df = pd.DataFrame(rows)
    feasible = df[df["precision"] >= MIN_PRECISION]
    if feasible.empty:  # fallback: precision closest to MIN_PRECISION
        best = df.iloc[(df["precision"] - MIN_PRECISION).abs().argsort()[:1]]
        best = df.loc[best.index[0]]
    else:
        best = feasible.sort_values(["recall", "precision"],
                                    ascending=False).iloc[0]
    return float(best["threshold"]), df


def main():
    plt.rcParams["figure.dpi"] = 120
    pipe = joblib.load(MODELS_DIR / "model.joblib")
    X_te, y_te = load_test_set()
    proba = pipe.predict_proba(X_te)[:, 1]

    pr_auc = average_precision_score(y_te, proba)
    roc_auc = roc_auc_score(y_te, proba)
    brier = brier_score_loss(y_te, proba)
    print(f"PR-AUC={pr_auc:.4f} ROC-AUC={roc_auc:.4f} Brier={brier:.4f}")

    threshold, sweep = select_threshold(y_te.values, proba)
    pred = (proba >= threshold).astype(int)
    rec = recall_score(y_te, pred)
    prec = precision_score(y_te, pred)
    f1 = f1_score(y_te, pred)
    cm = confusion_matrix(y_te, pred)
    print(f"Selected threshold={threshold:.3f} | Recall={rec:.3f} "
          f"Precision={prec:.3f} F1={f1:.3f} | TP={cm[1,1]} FP={cm[0,1]} "
          f"FN={cm[1,0]} TN={cm[0,0]}")

    (MODELS_DIR / "threshold.json").write_text(json.dumps({
        "threshold": threshold, "rule": "max recall s.t. precision >= 0.30",
        "recall": rec, "precision": prec, "f1": f1}, indent=2))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. PR curve with baseline
    p_curve, r_curve, _ = precision_recall_curve(y_te, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r_curve, p_curve, color="#c0392b", lw=2,
            label=f"PR-AUC = {pr_auc:.3f}")
    ax.axhline(y_te.mean(), color="#7f8c8d", ls="--",
               label=f"Baseline = {y_te.mean():.3f}")
    ax.axvline(rec, color="#2980b9", ls=":", label=f"Chosen threshold "
               f"{threshold:.2f} (recall {rec:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=120); plt.close(fig)

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_te, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"AUROC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#7f8c8d", ls="--", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=120); plt.close(fig)

    # 3. Confusion matrix at selected threshold
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]:,}\n({pct:.1%})", ha="center",
                    va="center", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred: no", "pred: <30"])
    ax.set_yticklabels(["true: no", "true: <30"])
    ax.set_title(f"Confusion Matrix @ threshold = {threshold:.3f}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120); plt.close(fig)

    # 4. Calibration curve
    frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10,
                                            strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="#7f8c8d", label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", color="#27ae60", lw=2,
            label="Model (quantile bins)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title(f"Calibration Curve (Brier = {brier:.4f})")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=120); plt.close(fig)

    md = ["", "## Threshold Selection", "",
          f"- Grid: 100 thresholds in [0.05, 0.90]; seed = {RANDOM_STATE}.",
          f"- Rule: **highest Recall subject to Precision >= {MIN_PRECISION}** "
          "(clinical cost asymmetry: a missed readmission is worse than a "
          "false alert).", "",
          f"**Selected threshold = {threshold:.4f}** — at this threshold: "
          f"Recall = {rec:.4f}, Precision = {prec:.4f}, F1 = {f1:.4f}.", "",
          f"| | count | row % |", f"|---|---|---|",
          f"| TP | {cm[1,1]:,} | of positives: {cm[1,1]/cm[1].sum():.1%} |",
          f"| FN | {cm[1,0]:,} | of positives: {cm[1,0]/cm[1].sum():.1%} |",
          f"| FP | {cm[0,1]:,} | of negatives: {cm[0,1]/cm[0].sum():.1%} |",
          f"| TN | {cm[0,0]:,} | of negatives: {cm[0,0]/cm[0].sum():.1%} |", ""]
    with open(REPORTS_DIR / "metrics.md", "a") as f:
        f.write("\n".join(md))
    print("\nFigures saved to reports/figures/: pr_curve, roc_curve, "
          "confusion_matrix, calibration")


if __name__ == "__main__":
    main()
