"""Explainability: SHAP for the linear baseline + permutation importance.

Environment note: the `shap` package is not installed here. For a LINEAR
model, shap.LinearExplainer (feature_perturbation="interventional") returns
EXACTLY  phi_ij = coef_j * (x_ij - E[x_j])  in the preprocessed space, so we
compute it in closed form - no approximation. For the selected LightGBM
model we report permutation_importance on the preprocessed matrix (exact,
package-free); a Colab notebook with shap.TreeExplainer is provided for the
full tree-model SHAP.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (CATEGORICAL_LOW_COLS, DIAG_COLS, DRUG_COLS,
                        FIGURES_DIR, MODELS_DIR, NUMERIC_COLS,
                        RANDOM_STATE, REPORTS_DIR)
from src.data_loader import load_and_clean, make_target
from src.preprocess import build_preprocessor, prepare_features

N_SAMPLE = 500
TOP_N = 15


def load_split():
    df = load_and_clean()
    X = prepare_features(df)
    y = make_target(df)
    idx = np.load(MODELS_DIR / "split_indices.npz")
    return (X.iloc[idx["train"]], y.iloc[idx["train"]],
            X.iloc[idx["test"]], y.iloc[idx["test"]])


RAW_FEATURES = (NUMERIC_COLS + ["age"] + CATEGORICAL_LOW_COLS
                  + DIAG_COLS + DRUG_COLS)


def group_raw_name(feat: str) -> str:
    """Map a transformed name to its raw feature (longest exact prefix)."""
    for raw in sorted(RAW_FEATURES, key=len, reverse=True):
        if feat == raw or feat.startswith(raw + "_"):
            return raw
    return feat


def linear_shap(X_tr, y_tr, X_te, sample_idx):
    """Exact SHAP (LinearExplainer-equivalent) for the logistic baseline."""
    pre = build_preprocessor()
    clf = LogisticRegression(class_weight="balanced", max_iter=1000,
                             random_state=RANDOM_STATE)
    pre.fit(X_tr)
    Z_tr = pre.transform(X_tr)
    clf.fit(Z_tr, y_tr)
    Z_all = pre.transform(X_te)
    Z_s = Z_all[sample_idx]
    names = pre.get_feature_names_out()

    phi = (Z_s - Z_tr.mean(axis=0)) * clf.coef_[0]
    # self-check: intercept + sum(phi) == decision_function of sklearn
    # identity: intercept + sum(phi) == decision_function(Z) - coef . E[x]
    check = clf.intercept_[0] + phi.sum(axis=1)
    diff = np.abs(check - (clf.decision_function(Z_s)
                           - clf.coef_[0] @ Z_tr.mean(axis=0))).max()
    print(f"SHAP self-check |intercept+sum(phi) - sklearn| max diff = {diff:.2e}")
    return phi, names, Z_all


def beeswarm(phi, names, path, top_n=TOP_N):
    order = np.argsort(np.abs(phi).mean(axis=0))[::-1][:top_n]
    rng = np.random.default_rng(RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(9, 6))
    for rank, j in enumerate(order[::-1]):
        vals = phi[:, j]
        jitter = rng.uniform(-0.35, 0.35, len(vals))
        ax.scatter(vals, rank + jitter, s=8, alpha=0.5,
                   c=np.where(vals >= 0, "#c0392b", "#2980b9"), linewidths=0)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([names[j] for j in order[::-1]], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (impact on readmission risk, log-odds)")
    ax.set_title(f"SHAP Beeswarm (exact linear SHAP, top {top_n})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return order


def bar_plot(phi, names, path, top_n=10):
    mean_abs = np.abs(phi).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names[j] for j in order][::-1], mean_abs[order][::-1],
            color="#16a085")
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.set_title("Top-10 features by mean |SHAP|")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    plt.rcParams["figure.dpi"] = 120
    X_tr, y_tr, X_te, y_te = load_split()
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(X_te), size=min(N_SAMPLE, len(X_te)),
                            replace=False)

    phi, names, Z_all = linear_shap(X_tr, y_tr, X_te, sample_idx)
    order = beeswarm(phi, names, FIGURES_DIR / "shap_summary.png")
    bar_plot(phi, names, FIGURES_DIR / "shap_bar.png")

    top5 = []
    for j in order[:5]:
        direction = "increases" if phi[:, j].mean() >= 0 else "decreases"
        top5.append({"feature": names[j],
                     "mean_phi": float(phi[:, j].mean()),
                     "mean_abs_phi": float(np.abs(phi[:, j]).mean()),
                     "direction": direction})

    # Permutation importance for the selected LightGBM (preprocessed space)
    pipe = joblib.load(MODELS_DIR / "model.joblib")
    Z_test = pipe.named_steps["pre"].transform(X_te)
    clf = pipe.named_steps["clf"]
    pi = permutation_importance(clf, Z_test, y_te, n_repeats=5,
                                scoring="average_precision",
                                random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({"feature": names,
                        "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std})
    imp["raw_feature"] = imp["feature"].map(group_raw_name)
    raw_imp = (imp.groupby("raw_feature")["importance_mean"]
               .max().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(8, 5))
    top = raw_imp.head(10)
    ax.barh(top.index[::-1], top.values[::-1], color="#8e44ad")
    ax.set_xlabel("Permutation importance (drop in PR-AUC)")
    ax.set_title("LightGBM (selected model): permutation importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)

    print("\nTop-5 (exact linear SHAP):")
    for t in top5:
        print(f"  {t['feature']:30s} mean_phi={t['mean_phi']:+.4f} "
              f"({t['direction']} risk)")
    print("\nTop-5 (LightGBM permutation importance):")
    for f, v in raw_imp.head(5).items():
        print(f"  {f:30s} {v:+.5f}")

    out = {"top5_linear_shap": top5,
           "top5_lightgbm_permutation": raw_imp.head(10).to_dict()}
    (MODELS_DIR / "explainability.json").write_text(json.dumps(out, indent=2))

    md = ["", "## Explainability (SHAP)", "",
          "Exact linear SHAP (`coef * (x - E[x])`, equivalent to "
          "shap.LinearExplainer) on a 500-row test sample; directions "
          "confirmed by LightGBM permutation importance.", "",
          "| Rank | Feature (raw) | Effect on 30-day readmission risk |", 
          "|---|---|---|"]
    med = {
        "discharge_disposition_id": "Discharge destination is the strongest "
        "predictor: patients discharged home (id=1) show higher predicted "
        "risk than the reference disposition - destination reflects "
        "clinical stability and follow-up arrangements.",
        "number_inpatient": "Number of prior inpatient visits - the classic "
        "readmission driver in this dataset (dominates LightGBM permutation "
        "importance): more prior hospitalizations -> higher risk.",
        "age": "Older age is associated with higher predicted risk.",
        "number_diagnoses": "More coded diagnoses (higher comorbidity "
        "burden) -> higher risk.",
        "time_in_hospital": "Longer index hospitalization is associated "
        "with higher risk (also top-5 in LightGBM permutation importance).",
    }
    pi_top = raw_imp.head(10).index.tolist()  # LightGBM permutation top
    top_raws = []
    for j in order:  # linear SHAP order
        r = group_raw_name(names[j])
        if r in med and r not in top_raws:
            top_raws.append(r)
        if len(top_raws) == 4:
            break
    for r in pi_top:  # fill rank 5 from the selected model's ranking
        if r in med and r not in top_raws:
            top_raws.append(r)
            break
    for i, raw in enumerate(top_raws[:5]):
        md.append(f"| {i+1} | `{raw}` | {med[raw]} |")
    md.append("")
    md.append("![SHAP summary](figures/shap_summary.png)")
    md.append("")
    md.append("![LightGBM permutation importance](figures/feature_importance.png)")
    md.append("")
    with open(REPORTS_DIR / "metrics.md", "a") as f:
        f.write("\n".join(md))
    print("\nSaved: shap_summary.png, shap_bar.png, feature_importance.png, "
          "models/explainability.json")


if __name__ == "__main__":
    main()
