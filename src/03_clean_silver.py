import re

from common import (DATASET, OUTPUT_DIR, REQUIRED, USE_S3A, get_spark, lake_path,
                    latest_manifest, norm_col, sync_layer, write_json)
from pyspark.sql import Window
from pyspark.sql import functions as F

CATEGORICAL = ["sla_status", "status", "channel", "category", "subcategory", "case_type", "assigned_to"]
NULL_TOKENS = ["", "null", "none", "nan", "n/a", "na", "-", "--"]


def detect_dayfirst(values):
    a = b = 0
    for v in values:
        m = re.match(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", v or "")
        if m:
            a += int(m.group(1)) > 12
            b += int(m.group(2)) > 12
    if a == 0 and b == 0:
        return True, "tidak ada bukti (ISO/ambigu) -> default dd/MM/yyyy"
    dayfirst = a >= b
    return dayfirst, f"{'dd/MM/yyyy' if dayfirst else 'MM/dd/yyyy'} (bagian1>12={a}, bagian2>12={b})"


def parse_ts(colname, dayfirst):
    iso = ["yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd'T'HH:mm:ss", "yyyy-MM-dd HH:mm", "yyyy-MM-dd"]
    dm = ["d/M/yyyy H:mm:ss", "d/M/yyyy H:mm", "d-M-yyyy H:mm:ss", "d-M-yyyy H:mm", "d/M/yyyy"]
    md = ["M/d/yyyy H:mm:ss", "M/d/yyyy H:mm", "M/d/yyyy h:mm:ss a", "M/d/yyyy h:mm a", "M/d/yyyy"]
    pats = iso + (dm + md if dayfirst else md + dm)
    return F.coalesce(*[F.to_timestamp(F.col(colname), p) for p in pats])


def canonical_casing(df, c):
    """Satukan variasi ejaan huruf besar/kecil -> ejaan yang paling sering muncul."""
    counts = (df.where(F.col(c).isNotNull())
                .groupBy(F.lower(F.col(c)).alias("_k"), F.col(c).alias("_v")).count())
    w = Window.partitionBy("_k").orderBy(F.desc("count"), F.asc("_v"))
    best = (counts.withColumn("_rn", F.row_number().over(w)).where("_rn = 1")
                  .select("_k", F.col("_v").alias("_canon")))
    changed = (counts.join(best, "_k").where(F.col("_v") != F.col("_canon"))
                     .agg(F.sum("count")).first()[0]) or 0
    out = (df.withColumn("_k", F.lower(F.col(c))).join(F.broadcast(best), "_k", "left")
             .withColumn(c, F.coalesce(F.col("_canon"), F.col(c))).drop("_k", "_canon"))
    return out, int(changed)


def main():
    spark = get_spark("03_clean_silver")
    m = latest_manifest()
    raw = (spark.read.option("header", True).option("multiLine", True).option("escape", '"')
                .csv(m["s3a_csv"] if USE_S3A else m["local_csv"]))
    n_raw = raw.count()
    df = raw.toDF(*[norm_col(c) for c in raw.columns])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Kolom wajib tidak ditemukan: {missing}\nKolom tersedia: {df.columns}")

    # [1] trim, rapikan spasi ganda, token null -> NULL
    for c in df.columns:
        v = F.trim(F.regexp_replace(F.col(c), r"\s+", " "))
        df = df.withColumn(c, F.when(v.isNull() | F.lower(v).isin(NULL_TOKENS),
                                     F.lit(None).cast("string")).otherwise(v))
    df = df.cache()
    null_raw = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in REQUIRED]).first().asDict()

    # [2] duplikat persis
    df_nodup = df.dropDuplicates()
    n_exact_dup = n_raw - df_nodup.count()
    df = df_nodup

    # [3] konversi tipe: case_number -> long, tanggal -> timestamp
    df = df.withColumn("case_number", F.regexp_replace("case_number", r"\.0+$", "").cast("long"))
    sample = [r[0] for r in df.select("created_date").dropna().limit(20000).collect()]
    sample += [r[0] for r in df.select("closed_date").dropna().limit(20000).collect()]
    dayfirst, date_note = detect_dayfirst(sample)
    df = (df.withColumn("created_ts", parse_ts("created_date", dayfirst))
            .withColumn("closed_ts", parse_ts("closed_date", dayfirst)))
    n_bad_created = df.where(F.col("created_date").isNotNull() & F.col("created_ts").isNull()).count()
    n_bad_closed = df.where(F.col("closed_date").isNotNull() & F.col("closed_ts").isNull()).count()

    # [4] karantina baris yang gagal validasi kritis
    critical = F.col("case_number").isNull() | F.col("created_ts").isNull()
    quarantine = df.where(critical).withColumn(
        "dq_reason", F.when(F.col("case_number").isNull(), "case_number_invalid")
                      .otherwise("created_date_invalid"))
    n_quarantine = quarantine.count()
    df = df.where(~critical)

    # [5] konsistensi ejaan nilai kategorikal
    casing_fixes = {}
    for c in CATEGORICAL:
        df, casing_fixes[c] = canonical_casing(df, c)

    # [6] standarisasi domain SLA & status
    k = F.lower(F.regexp_replace(F.col("sla_status"), "[^A-Za-z]", ""))
    df = (df.withColumn("sla_status_raw", F.col("sla_status"))
            .withColumn("sla_status",
                        F.when(F.col("sla_status").isNull(), F.lit(None).cast("string"))
                         .when(k.rlike("notwithin|breach|outside|exceed|miss|violat|overdue|late|beyond|outof"),
                               "Breached SLA")
                         .when(k.rlike("within|insla|ontime|^met"), "Within SLA")
                         .otherwise("Other"))
            .withColumn("is_within_sla", F.when(F.col("sla_status") == "Within SLA", 1)
                                          .when(F.col("sla_status") == "Breached SLA", 0))
            .withColumn("is_closed", F.when(F.lower(F.col("status"))
                                            .rlike("closed|resolved|completed|cancel"), 1).otherwise(0)))
    df = df.cache()
    df.count()

    # [7] duplikat case_number -> simpan versi terbaru
    w = Window.partitionBy("case_number").orderBy(F.col("closed_ts").desc_nulls_last(),
                                                  F.col("created_ts").desc())
    df = df.withColumn("_rn", F.row_number().over(w))
    n_dup_case = df.where("_rn > 1").count()
    df = df.where("_rn = 1").drop("_rn")

    # [8] fitur turunan + validasi logis
    hours = (F.unix_timestamp("closed_ts") - F.unix_timestamp("created_ts")) / 3600.0
    df = (df.withColumn("resolution_hours", F.round(hours, 4))
            .withColumn("flag_negative_duration",
                        F.coalesce((F.col("resolution_hours") < 0).cast("int"), F.lit(0)))
            .withColumn("flag_closed_without_date",
                        ((F.col("is_closed") == 1) & F.col("closed_ts").isNull()).cast("int"))
            .withColumn("flag_open_with_closed_date",
                        ((F.col("is_closed") == 0) & F.col("closed_ts").isNotNull()).cast("int"))
            .withColumn("resolution_hours_clean",
                        F.when(F.col("resolution_hours") >= 0, F.col("resolution_hours")))
            .withColumn("created_date_only", F.to_date("created_ts"))
            .withColumn("created_year", F.year("created_ts"))
            .withColumn("created_month", F.month("created_ts"))
            .withColumn("created_ym", F.date_format("created_ts", "yyyy-MM"))
            .withColumn("created_hour", F.hour("created_ts"))
            .withColumn("created_dow", F.dayofweek("created_ts"))
            .withColumn("created_day_name", F.date_format("created_ts", "EEEE")))
    q = df.approxQuantile("resolution_hours_clean", [0.99], 0.001)
    p99 = q[0] if q and q[0] == q[0] else None
    out_expr = (F.col("resolution_hours_clean") > F.lit(p99)) if p99 is not None else F.lit(False)
    df = df.withColumn("flag_outlier_resolution", F.coalesce(out_expr.cast("int"), F.lit(0)))
    df = df.withColumn("dq_flags", F.concat_ws(",",
        F.when(F.col("flag_negative_duration") == 1, F.lit("negative_duration")),
        F.when(F.col("flag_closed_without_date") == 1, F.lit("closed_without_date")),
        F.when(F.col("flag_open_with_closed_date") == 1, F.lit("open_with_closed_date")),
        F.when(F.col("flag_outlier_resolution") == 1, F.lit("outlier_resolution")),
        F.when(F.col("sla_status") == "Other", F.lit("sla_status_unmapped"))))

    base = ["case_number", "created_ts", "closed_ts", "created_date", "closed_date", "channel",
            "category", "subcategory", "case_type", "status", "is_closed", "sla_status",
            "sla_status_raw", "is_within_sla", "assigned_to", "resolution_hours",
            "resolution_hours_clean", "created_date_only", "created_year", "created_month",
            "created_ym", "created_hour", "created_dow", "created_day_name",
            "flag_negative_duration", "flag_closed_without_date", "flag_open_with_closed_date",
            "flag_outlier_resolution", "dq_flags"]
    extra = [c for c in df.columns if c not in base]
    silver = df.select(*base, *extra).cache()
    n_clean = silver.count()

    a = silver.agg(F.sum("flag_negative_duration").alias("neg"),
                   F.sum("flag_closed_without_date").alias("cnd"),
                   F.sum("flag_open_with_closed_date").alias("owd"),
                   F.sum("flag_outlier_resolution").alias("out"),
                   F.sum((F.col("sla_status") == "Other").cast("int")).alias("oth"),
                   F.sum((F.col("dq_flags") == "").cast("int")).alias("ok")).first()
    comp_cols = REQUIRED + ["created_ts", "closed_ts"]
    comp = silver.select([(F.count(c) / F.count(F.lit(1)) * 100).alias(c) for c in comp_cols]).first().asDict()
    sla_map = [r.asDict() for r in silver.groupBy("sla_status_raw", "sla_status").count().collect()]

    silver.write.mode("overwrite").partitionBy("created_ym").parquet(lake_path("silver", DATASET))
    quarantine.write.mode("overwrite").parquet(lake_path("quarantine", DATASET))
    sync_layer("silver", DATASET)
    sync_layer("quarantine", DATASET)

    dq = {
        "n_raw": n_raw, "null_raw": null_raw, "n_exact_duplicates": n_exact_dup,
        "date_format_detected": date_note, "n_created_unparseable": n_bad_created,
        "n_closed_unparseable": n_bad_closed, "n_quarantined": n_quarantine,
        "casing_fixes": casing_fixes, "n_duplicate_case_number": n_dup_case,
        "n_negative_duration": int(a["neg"] or 0), "n_closed_without_date": int(a["cnd"] or 0),
        "n_open_with_closed_date": int(a["owd"] or 0), "n_outlier_resolution": int(a["out"] or 0),
        "p99_resolution_hours": round(p99, 2) if p99 is not None else None,
        "n_sla_unmapped": int(a["oth"] or 0), "sla_mapping": sla_map, "n_clean": n_clean,
        "dq_score_pct": round(100.0 * (a["ok"] or 0) / n_clean, 2) if n_clean else 0,
        "completeness_pct": {k2: round(v or 0, 2) for k2, v in comp.items()},
    }
    write_json(OUTPUT_DIR / "dq_report.json", dq)
    print(f"[OK] Silver: {n_clean:,} baris bersih dari {n_raw:,} baris mentah")
    for key in ["date_format_detected", "n_exact_duplicates", "n_duplicate_case_number",
                "n_quarantined", "n_negative_duration", "n_closed_without_date",
                "n_outlier_resolution", "n_sla_unmapped", "dq_score_pct"]:
        print(f"     {key:28s}: {dq[key]}")
    print("     pemetaan SLA:", sla_map)


if __name__ == "__main__":
    main()
