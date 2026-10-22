import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATASET, FIG_DIR, TABLE_DIR, ensure_dirs, get_spark, lake_path, sync_layer
from ml_common import (CAT_FEATURES, LABEL, LEAKY_COLUMNS, LOAD_FEATURES, ML_DATASET, ML_DIR,
                       NUM_FEATURES, TRAIN_PCT, VALID_PCT, idn, save_json)
from pyspark.sql import Window as W
from pyspark.sql import functions as F

DAY = {1: "Min", 2: "Sen", 3: "Sel", 4: "Rab", 5: "Kam", 6: "Jum", 7: "Sab"}
C1, C2 = "#1F4E79", "#C00000"


def main():
    ensure_dirs()
    ML_DIR.mkdir(parents=True, exist_ok=True)
    spark = get_spark("09_ml_dataset")
    s = spark.read.parquet(lake_path("silver", DATASET))
    n_silver = s.count()

    df = (s.where(F.col("is_within_sla").isNotNull() & F.col("created_ts").isNotNull())
           .withColumn(LABEL, (1 - F.col("is_within_sla")).cast("double"))
           .withColumn("ts", F.col("created_ts").cast("long"))
           # fitur waktu
           .withColumn("is_weekend", F.col("created_dow").isin(1, 7).cast("int"))
           .withColumn("is_business_hour",
                       ((F.col("created_hour") >= 9) & (F.col("created_hour") < 17)
                        & ~F.col("created_dow").isin(1, 7)).cast("int")))

    # fitur beban kerja (window function; hanya melihat ke BELAKANG, tanpa melihat masa depan)
    one = F.lit(1)
    w_batch = W.partitionBy("created_ts")                                        # case dgn cap waktu sama
    w_team = W.partitionBy("assigned_to").orderBy("ts").rangeBetween(-86400, -1)  # tim, 24 jam terakhir
    w_1h = W.orderBy("ts").rangeBetween(-3600, -1)                               # seluruh case, 1 jam terakhir
    w_7d = W.orderBy("ts").rangeBetween(-7 * 86400, -1)                          # seluruh case, 7 hari terakhir
    df = (df.withColumn("batch_size", F.count(one).over(w_batch))
            .withColumn("team_load_24h", F.count(one).over(w_team))
            .withColumn("all_load_1h", F.count(one).over(w_1h))
            .withColumn("all_load_7d", F.count(one).over(w_7d)))

    # pembagian train/valid/test deterministik (CRC32 nomor case)
    bucket = F.pmod(F.crc32(F.col("case_number").cast("string")), F.lit(100))
    df = df.withColumn("split", F.when(bucket < TRAIN_PCT, "train")
                                 .when(bucket < TRAIN_PCT + VALID_PCT, "valid").otherwise("test"))

    out_cols = (["case_number", "created_ts", "created_ym"] + CAT_FEATURES + NUM_FEATURES
                + [LABEL, "split"])
    ml = df.select(*out_cols).fillna("Unknown", subset=CAT_FEATURES[:5])
    ml.repartition(8).write.mode("overwrite").parquet(lake_path("gold", ML_DATASET))
    sync_layer("gold", ML_DATASET)

    ml = spark.read.parquet(lake_path("gold", ML_DATASET)).cache()
    n = ml.count()
    by_split = {r["split"]: {"rows": r["rows"], "breach": int(r["breach"]), "rate": r["rate"]}
                for r in ml.groupBy("split").agg(F.count("*").alias("rows"),
                                                 F.sum(LABEL).alias("breach"),
                                                 F.avg(LABEL).alias("rate")).collect()}
    pos = int(ml.agg(F.sum(LABEL)).first()[0])
    loads = ml.select(*[F.expr(f"percentile_approx({c}, array(0.5, 0.99))").alias(c)
                        for c in LOAD_FEATURES] + [F.max(c).alias(f"{c}_max") for c in LOAD_FEATURES]).first()
    oot = ml.groupBy((F.col("created_ts") >= F.lit("2024-03-01")).alias("after")) \
            .agg(F.count("*").alias("rows"), F.avg(LABEL).alias("rate")).collect()
    summary = {
        "silver_rows": n_silver, "ml_rows": n, "positives": pos, "negatives": n - pos,
        "positive_rate": pos / n, "imbalance_ratio": (n - pos) / pos,
        "n_raw_features": len(CAT_FEATURES) + len(NUM_FEATURES),
        "cat_features": CAT_FEATURES, "num_features": NUM_FEATURES,
        "excluded_leaky_columns": LEAKY_COLUMNS, "splits": by_split,
        "load_feature_p50_p99_max": {c: list(loads[c]) + [loads[f"{c}_max"]] for c in LOAD_FEATURES},
        "out_of_time": {("after_2024_03" if r["after"] else "before_2024_03"): {"rows": r["rows"], "rate": r["rate"]}
                        for r in oot},
    }
    save_json(ML_DIR / "dataset_summary.json", summary)

    rows = ["| Split | Baris | Breach | Rasio breach |", "|---|---|---|---|"]
    for k in ["train", "valid", "test"]:
        v = by_split[k]
        rows.append(f"| {k} | {idn(v['rows'], 0)} | {idn(v['breach'], 0)} | {idn(100 * v['rate'])}% |")
    rows.append(f"| total | {idn(n, 0)} | {idn(pos, 0)} | {idn(100 * pos / n)}% |")
    (TABLE_DIR / "ml_dataset_split.md").write_text("\n".join(rows), encoding="utf-8")

    # rasio breach per fitur kandidat (motivasi feature engineering)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    h = ml.groupBy("created_hour").agg(F.avg(LABEL).alias("r")).orderBy("created_hour").toPandas()
    ax[0, 0].bar(h.created_hour, 100 * h.r, color=C1)
    ax[0, 0].set_title("Rasio breach per jam dibuat")
    ax[0, 0].set_xlabel("Jam")
    d = ml.groupBy("created_dow").agg(F.avg(LABEL).alias("r")).orderBy("created_dow").toPandas()
    ax[0, 1].bar(d.created_dow.map(DAY), 100 * d.r, color=C1)
    ax[0, 1].set_title("Rasio breach per hari dibuat")
    t = (ml.groupBy("assigned_to").agg(F.avg(LABEL).alias("r"), F.count("*").alias("n"))
           .where("n >= 100").orderBy("r").toPandas())
    ax[1, 0].barh(t.assigned_to, 100 * t.r, color=C1)
    ax[1, 0].set_title("Rasio breach per tim (tim dengan >= 100 case)")
    ax[1, 0].tick_params(axis="y", labelsize=7)
    b = (ml.withColumn("bin", F.when(F.col("all_load_7d") < 5000, "<5rb")
                               .when(F.col("all_load_7d") < 10000, "5-10rb")
                               .when(F.col("all_load_7d") < 20000, "10-20rb").otherwise(">=20rb"))
           .groupBy("bin").agg(F.avg(LABEL).alias("r"), F.count("*").alias("n")).toPandas()
           .set_index("bin").reindex(["<5rb", "5-10rb", "10-20rb", ">=20rb"]).reset_index())
    ax[1, 1].bar(b["bin"], 100 * b.r, color=C2)
    ax[1, 1].set_title("Rasio breach vs beban 7 hari terakhir (all_load_7d)")
    for a in ax.flat:
        a.grid(axis="y" if a is not ax[1, 0] else "x", alpha=.3)
        a.set_axisbelow(True)
    for a in (ax[0, 0], ax[0, 1], ax[1, 1]):
        a.set_ylabel("% breach")
    ax[1, 0].set_xlabel("% breach")
    fig.suptitle("Rasio Pelanggaran SLA terhadap Fitur Kandidat", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_5_1_breach_drivers.png", dpi=200)
    plt.close(fig)

    print(f"[OK] {n:,} baris -> {lake_path('gold', ML_DATASET)}; breach {pos:,} ({100 * pos / n:.2f}%)")
    for k in ["train", "valid", "test"]:
        print(f"     {k:5s}: {by_split[k]['rows']:,} baris, breach {by_split[k]['breach']:,}")
    spark.stop()


if __name__ == "__main__":
    main()
