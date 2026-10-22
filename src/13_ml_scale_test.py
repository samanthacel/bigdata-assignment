import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import FIG_DIR, LAKE_DIR, get_spark, lake_path
from ml_common import LABEL, ML_DATASET, ML_DIR, RAW_FEATURES, save_json

FACTORS = [int(x) for x in os.environ.get("SCALE_FACTORS", "1,2,4,8").split(",")]


def main():
    spec_sp = __import__("10_ml_spark")
    spec_sk = __import__("11_ml_sklearn")
    spark = get_spark("13_ml_scale_test")
    base = spark.read.parquet(lake_path("gold", ML_DATASET)).where("split = 'train'")
    pdf = pd.read_parquet(LAKE_DIR / "gold" / ML_DATASET)
    pdf = pdf[pdf.split == "train"]
    for c in spec_sp.CAT_FEATURES:
        pdf[c] = pdf[c].astype(str)
    out = []
    for k in FACTORS:
        # Spark MLlib (prep pipeline + LR)
        df = base.crossJoin(spark.range(k).withColumnRenamed("id", "rep")).drop("rep")
        df, _ = spec_sp.add_weights(df)
        df = df.cache()
        n = df.count()
        t0 = time.perf_counter()
        prep = spec_sp.prep_pipeline().fit(df)
        spec_sp.make_model("logistic_regression", {"regParam": 0.0}).fit(prep.transform(df))
        t_spark = time.perf_counter() - t0
        df.unpersist()
        # scikit-learn (pipeline sama: OHE + LR)
        big = pd.concat([pdf] * k, ignore_index=True)
        mem = big.memory_usage(deep=True).sum() / 2**20
        t0 = time.perf_counter()
        spec_sk.make_model("logistic_regression", {"C": 100.0}).fit(big[RAW_FEATURES], big[LABEL].astype(int))
        t_sk = time.perf_counter() - t0
        del big
        out.append({"factor": k, "rows": n, "spark_sec": t_spark, "sklearn_sec": t_sk, "pandas_mb": mem})
        print(f"[SCALE] x{k}: {n:,} baris | Spark {t_spark:.1f}s | sklearn {t_sk:.1f}s | pandas {mem:.0f} MB")
    save_json(ML_DIR / "scale_test.json", {"cores": spark.sparkContext.defaultParallelism, "runs": out})
    d = pd.DataFrame(out)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(d.rows / 1e6, d.spark_sec, "o-", color="#1F4E79", label="Spark MLlib (local[*])")
    ax.plot(d.rows / 1e6, d.sklearn_sec, "s--", color="#ED7D31", label="scikit-learn (single-node)")
    ax.set(xlabel="Jumlah baris data latih (juta)", ylabel="Waktu latih (detik)",
           title="Uji Skalabilitas Logistic Regression")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_5_6_scale_test.png", dpi=200)
    plt.close(fig)
    spark.stop()


if __name__ == "__main__":
    main()
