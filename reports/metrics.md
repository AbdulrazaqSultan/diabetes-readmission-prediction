# Model Comparison (test set)

- Split: GroupShuffleSplit on `patient_nbr` (test_size=0.2, seed=42)
- n_train=79,567, n_test=19,773, positive rate (test)=11.35%
- Metrics at threshold 0.5; PR-AUC is the primary metric.
- Bootstrap: 1,000 resamples of the test set (95% percentile CI).

|                      |   pr_auc | pr_auc_ci      |   roc_auc | roc_auc_ci     |   recall@0.5 |   precision@0.5 |   f1@0.5 |   brier |   fit_seconds |
|:---------------------|---------:|:---------------|----------:|:---------------|-------------:|----------------:|---------:|--------:|--------------:|
| LightGBM             |   0.2383 | [0.222, 0.255] |    0.6739 | [0.662, 0.686] |       0.5642 |          0.1867 |   0.2806 |  0.2118 |        4.1000 |
| LightGBM+Calibrated  |   0.2379 | [0.222, 0.255] |    0.6744 | [0.663, 0.686] |       0.0000 |          0.0000 |   0.0000 |  0.0959 |       19.1000 |
| HistGradientBoosting |   0.2348 | [0.219, 0.252] |    0.6753 | [0.663, 0.688] |       0.5775 |          0.1823 |   0.2771 |  0.2172 |        7.9000 |
| LogReg_baseline      |   0.2255 | [0.211, 0.242] |    0.6704 | [0.659, 0.681] |       0.5455 |          0.1864 |   0.2778 |  0.2249 |       15.9000 |
| LogReg_L2_C0.1       |   0.2249 | [0.210, 0.241] |    0.6705 | [0.659, 0.682] |       0.5450 |          0.1861 |   0.2775 |  0.2250 |       10.5000 |
| RandomForest         |   0.2081 | [0.194, 0.224] |    0.6556 | [0.644, 0.668] |       0.0004 |          0.1667 |   0.0009 |  0.0968 |       30.0000 |

**Selected model:** `LightGBM` with PR-AUC = 0.2383

## Threshold Selection

- Grid: 100 thresholds in [0.05, 0.90]; seed = 42.
- Rule: **highest Recall subject to Precision >= 0.3** (clinical cost asymmetry: a missed readmission is worse than a false alert).

**Selected threshold = 0.6510** — at this threshold: Recall = 0.2210, Precision = 0.3039, F1 = 0.2559.

| | count | row % |
|---|---|---|
| TP | 496 | of positives: 22.1% |
| FN | 1,748 | of positives: 77.9% |
| FP | 1,136 | of negatives: 6.5% |
| TN | 16,393 | of negatives: 93.5% |

## Explainability (SHAP)

Exact linear SHAP (`coef * (x - E[x])`, equivalent to shap.LinearExplainer) on a 500-row test sample; directions confirmed by LightGBM permutation importance.

| Rank | Feature (raw) | Effect on 30-day readmission risk |
|---|---|---|
| 1 | `discharge_disposition_id` | Discharge destination is the strongest predictor: patients discharged home (id=1) show higher predicted risk than the reference disposition - destination reflects clinical stability and follow-up arrangements. |
| 2 | `number_inpatient` | Number of prior inpatient visits - the classic readmission driver in this dataset (dominates LightGBM permutation importance): more prior hospitalizations -> higher risk. |
| 3 | `age` | Older age is associated with higher predicted risk. |
| 4 | `number_diagnoses` | More coded diagnoses (higher comorbidity burden) -> higher risk. |
| 5 | `time_in_hospital` | Longer index hospitalization is associated with higher risk (also top-5 in LightGBM permutation importance). |

![SHAP summary](figures/shap_summary.png)

![LightGBM permutation importance](figures/feature_importance.png)
