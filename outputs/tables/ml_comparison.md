| Engine | Model | Hiperparameter | ROC-AUC | PR-AUC | Precision | Recall | F1 | Recall@5% | Lift@5% | Waktu latih (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Spark MLlib | Logistic Regression | regParam=0.0 | 0,7973 | 0,1991 | 0,562 | 0,180 | 0,272 | 28,9% | 5,8x | 7,1 |
| Spark MLlib | Random Forest | maxDepth=8 | 0,8342 | 0,1768 | 0,415 | 0,164 | 0,235 | 35,8% | 7,2x | 11,5 |
| Spark MLlib | Gradient Boosting (GBTClassifier) | maxDepth=6 | 0,8190 | 0,2117 | 0,415 | 0,219 | 0,287 | 37,2% | 7,4x | 26,2 |
| scikit-learn | Logistic Regression | C=100.0 | 0,7972 | 0,1936 | 0,561 | 0,181 | 0,274 | 29,0% | 5,8x | 0,6 |
| scikit-learn | Random Forest | max_depth=12 | 0,8560 | 0,2344 | 0,301 | 0,250 | 0,273 | 40,4% | 8,1x | 2,0 |
| scikit-learn | Gradient Boosting (HistGB) | max_depth=6 | 0,8431 | 0,2622 | 0,447 | 0,255 | 0,325 | 39,7% | 7,9x | 1,2 |