import itertools
import time

from common import LAKE_DIR, get_spark, lake_path
from ml_common import (BIN_FEATURES, CAT_FEATURES, LABEL, LOAD_FEATURES, ML_DATASET, ML_DIR,
                       OOT_CUTOFF, PRED_DIR, SEED, best_f1_threshold, evaluate, save_json)
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, SQLTransformer, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

IDX = [f"{c}_idx" for c in CAT_FEATURES]
OHE = [f"{c}_ohe" for c in CAT_FEATURES]
LOG = [f"log_{c}" for c in LOAD_FEATURES]
TREE_INPUTS = IDX + BIN_FEATURES + LOG
LINEAR_INPUTS = OHE + BIN_FEATURES + LOG

GRID = {
    "logistic_regression": {"regParam": [0.0, 0.001, 0.01]},
    "random_forest": {"maxDepth": [8, 12]},
    "gbt": {"maxDepth": [4, 6]},
}


def prep_pipeline():
    log_sql = "SELECT *, " + ", ".join(f"log1p({c}) AS log_{c}" for c in LOAD_FEATURES) + " FROM __THIS__"
    return Pipeline(stages=[
        SQLTransformer(statement=log_sql),
        StringIndexer(inputCols=CAT_FEATURES, outputCols=IDX, handleInvalid="keep",
                      stringOrderType="frequencyDesc"),
        OneHotEncoder(inputCols=IDX, outputCols=OHE, handleInvalid="keep", dropLast=True),
        VectorAssembler(inputCols=LINEAR_INPUTS, outputCol="features_linear"),
        VectorAssembler(inputCols=TREE_INPUTS, outputCol="features_tree"),
    ])


def make_model(name, params):
    common = dict(labelCol=LABEL, weightCol="w")
    if name == "logistic_regression":
        return LogisticRegression(featuresCol="features_linear", maxIter=100, elasticNetParam=0.0,
                                  standardization=True, **common, **params)
    if name == "random_forest":
        return RandomForestClassifier(featuresCol="features_tree", numTrees=100, maxBins=64,
                                      featureSubsetStrategy="sqrt", minInstancesPerNode=5, seed=SEED,
                                      **common, **params)
    return GBTClassifier(featuresCol="features_tree", maxIter=60, stepSize=0.1, maxBins=64,
                         seed=SEED, **common, **params)


def add_weights(df):
    # Bobot 'balanced' = n / (2 * n_kelas), sama dengan class_weight='balanced' di scikit-learn.
    c = {int(r[LABEL]): r["count"] for r in df.groupBy(LABEL).count().collect()}
    n = sum(c.values())
    w0, w1 = n / (2 * c[0]), n / (2 * c[1])
    return df.withColumn("w", F.when(F.col(LABEL) == 1, F.lit(w1)).otherwise(F.lit(w0))), (w0, w1)


def scores(model, df):
    pdf = (model.transform(df)
           .select("case_number", LABEL, vector_to_array("probability")[1].alias("p")).toPandas())
    return pdf


def main():
    ML_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    spark = get_spark("10_ml_spark")
    data = spark.read.parquet(lake_path("gold", ML_DATASET))

    t0 = time.perf_counter()
    prep = prep_pipeline().fit(data.where("split = 'train'"))
    prep_sec = time.perf_counter() - t0
    feat = prep.transform(data)
    train, wts = add_weights(feat.where("split = 'train'"))
    train = train.cache()
    valid = feat.where("split = 'valid'").cache()
    test = feat.where("split = 'test'").cache()
    print(f"[INFO] train={train.count():,} valid={valid.count():,} test={test.count():,}; "
          f"bobot kelas w0={wts[0]:.3f} w1={wts[1]:.3f}; prep {prep_sec:.1f}s")
    n_linear = len(train.first()["features_linear"])

    results, tuning = {}, []
    for name, grid in GRID.items():
        best = None
        keys = list(grid)
        for combo in itertools.product(*grid.values()):
            params = dict(zip(keys, combo))
            t0 = time.perf_counter()
            m = make_model(name, params).fit(train)
            fit_sec = time.perf_counter() - t0
            v = scores(m, valid)
            ev = evaluate(v[LABEL], v["p"], 0.5)
            tuning.append({"engine": "spark", "model": name, **params, "fit_sec": fit_sec,
                           "valid_pr_auc": ev["pr_auc"], "valid_roc_auc": ev["roc_auc"]})
            print(f"  [{name}] {params} fit={fit_sec:.1f}s valid PR-AUC={ev['pr_auc']:.4f} "
                  f"ROC-AUC={ev['roc_auc']:.4f}")
            if best is None or ev["pr_auc"] > best["valid_pr_auc"]:
                best = {"params": params, "model": m, "fit_sec": fit_sec, "valid": v,
                        "valid_pr_auc": ev["pr_auc"]}

        m = best["model"]
        thr = best_f1_threshold(best["valid"][LABEL], best["valid"]["p"])
        t0 = time.perf_counter()
        te = scores(m, test)
        pred_sec = time.perf_counter() - t0
        te.to_csv(PRED_DIR / f"spark_{name}.csv", index=False)
        metrics = evaluate(te[LABEL], te["p"], thr)
        native_auc = BinaryClassificationEvaluator(labelCol=LABEL, rawPredictionCol="rawPrediction",
                                                   metricName="areaUnderROC").evaluate(m.transform(test))
        results[name] = {"engine": "Spark MLlib", "best_params": best["params"],
                         "fit_sec": best["fit_sec"], "predict_sec": pred_sec,
                         "valid_pr_auc": best["valid_pr_auc"], "spark_native_test_roc_auc": native_auc,
                         "test": metrics}
        m.write().overwrite().save(str(LAKE_DIR / "models" / f"spark_{name}"))
        print(f"[TEST] {name}: ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
              f"Recall@5%={metrics['recall_at_5pct']:.3f}")

        # importance / koefisien
        if name == "logistic_regression":
            attrs = sorted(test.schema["features_linear"].metadata["ml_attr"]["attrs"].get("binary", [])
                           + test.schema["features_linear"].metadata["ml_attr"]["attrs"].get("numeric", []),
                           key=lambda a: a["idx"])
            coefs = m.coefficients.toArray()
            pairs = sorted(((attrs[i]["name"], float(coefs[i])) for i in range(len(coefs))),
                           key=lambda x: -abs(x[1]))
            results[name]["top_coefficients"] = pairs[:20]
        else:
            imp = m.featureImportances.toArray()
            raw = CAT_FEATURES + BIN_FEATURES + LOAD_FEATURES
            results[name]["feature_importance"] = dict(sorted(
                {raw[i]: float(imp[i]) for i in range(len(raw))}.items(), key=lambda x: -x[1]))

    # uji out-of-time untuk model terbaik (berdasarkan PR-AUC validasi)
    best_name = max(results, key=lambda k: results[k]["valid_pr_auc"])
    params = results[best_name]["best_params"]
    tr_raw = data.where(F.col("created_ts") < F.lit(OOT_CUTOFF))
    te_raw = data.where(F.col("created_ts") >= F.lit(OOT_CUTOFF))
    prep_oot = prep_pipeline().fit(tr_raw)
    tr_oot, _ = add_weights(prep_oot.transform(tr_raw))
    m_oot = make_model(best_name, params).fit(tr_oot)
    s_oot = scores(m_oot, prep_oot.transform(te_raw))
    ev_oot = evaluate(s_oot[LABEL], s_oot["p"], 0.5)
    oot = {"model": best_name, "params": params, "cutoff": OOT_CUTOFF,
           "train_rows": tr_oot.count(), "test_rows": len(s_oot),
           "roc_auc": ev_oot["roc_auc"], "pr_auc": ev_oot["pr_auc"], "base_rate": ev_oot["base_rate"],
           "recall_at_5pct": ev_oot["recall_at_5pct"], "lift_at_5pct": ev_oot["lift_at_5pct"]}
    s_oot.to_csv(PRED_DIR / f"spark_{best_name}_oot.csv", index=False)
    print(f"[OOT] {best_name}: ROC-AUC={oot['roc_auc']:.4f} PR-AUC={oot['pr_auc']:.4f} "
          f"(base {oot['base_rate']:.4f})")

    save_json(ML_DIR / "metrics_spark.json", {
        "engine": "Spark MLlib", "spark_version": spark.version,
        "master": spark.sparkContext.master, "default_parallelism": spark.sparkContext.defaultParallelism,
        "prep_fit_sec": prep_sec, "class_weights": {"w0": wts[0], "w1": wts[1]},
        "n_linear_features": n_linear, "n_tree_features": len(TREE_INPUTS),
        "models": results, "best_by_valid": best_name, "out_of_time": oot, "tuning": tuning})
    print(f"[OK] metrik -> {ML_DIR / 'metrics_spark.json'}")
    spark.stop()


if __name__ == "__main__":
    main()
