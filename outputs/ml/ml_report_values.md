# Nilai ML untuk laporan

| Engine | Model | Hiperparameter | ROC-AUC | PR-AUC | Precision | Recall | F1 | Recall@5% | Lift@5% | Waktu latih (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Spark MLlib | Logistic Regression | regParam=0.0 | 0,7973 | 0,1991 | 0,562 | 0,180 | 0,272 | 28,9% | 5,8x | 7,1 |
| Spark MLlib | Random Forest | maxDepth=8 | 0,8342 | 0,1768 | 0,415 | 0,164 | 0,235 | 35,8% | 7,2x | 11,5 |
| Spark MLlib | Gradient Boosting (GBTClassifier) | maxDepth=6 | 0,8190 | 0,2117 | 0,415 | 0,219 | 0,287 | 37,2% | 7,4x | 26,2 |
| scikit-learn | Logistic Regression | C=100.0 | 0,7972 | 0,1936 | 0,561 | 0,181 | 0,274 | 29,0% | 5,8x | 0,6 |
| scikit-learn | Random Forest | max_depth=12 | 0,8560 | 0,2344 | 0,301 | 0,250 | 0,273 | 40,4% | 8,1x | 2,0 |
| scikit-learn | Gradient Boosting (HistGB) | max_depth=6 | 0,8431 | 0,2622 | 0,447 | 0,255 | 0,325 | 39,7% | 7,9x | 1,2 |

```json
{
  "best_spark": "gbt",
  "best_sklearn": "gbt",
  "spearman_spark_vs_sklearn": {
    "logistic_regression": 0.9999642052659621,
    "random_forest": 0.9441871169399305,
    "gbt": 0.9331675226828049
  },
  "spark_oot": {
    "model": "gbt",
    "params": {
      "maxDepth": 6
    },
    "cutoff": "2024-03-01",
    "train_rows": 184183,
    "test_rows": 47502,
    "roc_auc": 0.6008009551227759,
    "pr_auc": 0.11123709972203881,
    "base_rate": 0.035493242389794115,
    "recall_at_5pct": 0.11743772241992882,
    "lift_at_5pct": 2.3488533433227197
  },
  "sklearn_oot": {
    "model": "gbt",
    "params": {
      "max_depth": 6
    },
    "cutoff": "2024-03-01",
    "train_rows": 184183,
    "test_rows": 47502,
    "roc_auc": 0.5969849988949557,
    "pr_auc": 0.12128764498054306,
    "base_rate": 0.035493242389794115,
    "recall_at_5pct": 0.14887307236061684,
    "lift_at_5pct": 2.977586814010114
  },
  "spark_version": "4.2.0",
  "sklearn_version": "1.9.1",
  "spark_prep_fit_sec": 1.6609910839999884,
  "sklearn_load_sec": 0.3456257089999326
}
```