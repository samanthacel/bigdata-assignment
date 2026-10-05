"""TAHAP 4 - Silver -> Gold: star schema (Parquet) + load ke PostgreSQL data warehouse."""
import psycopg

from common import (DATASET, PG, PG_JDBC_URL, POSTGRES_JDBC_PACKAGE, PROJECT_DIR, get_spark,
                    lake_path, sync_layer)
from pyspark.sql import Window
from pyspark.sql import functions as F

ORDER = ["dim_date", "dim_channel", "dim_category", "dim_team", "dim_case_status", "fact_case"]
FILL = ["channel", "category", "subcategory", "case_type", "assigned_to", "status", "sla_status"]


def with_key(df, key, cols):
    return df.withColumn(key, F.row_number().over(Window.orderBy(*cols)))


def main():
    spark = get_spark("04_build_gold", [POSTGRES_JDBC_PACKAGE])
    s = spark.read.parquet(lake_path("silver", DATASET)).fillna("Unknown", subset=FILL).cache()

    dim_channel = with_key(s.select(F.col("channel").alias("channel_name")).distinct(),
                           "channel_key", ["channel_name"])
    dim_category = with_key(s.select("category", "subcategory", "case_type").distinct(),
                            "category_key", ["category", "subcategory", "case_type"])
    dim_team = with_key(s.select(F.col("assigned_to").alias("team_name")).distinct(),
                        "team_key", ["team_name"])
    dim_case_status = with_key(s.select("status", "sla_status").distinct(),
                               "status_key", ["status", "sla_status"])

    bnd = s.select(F.least(F.min(F.to_date("created_ts")), F.min(F.to_date("closed_ts"))).alias("mn"),
                   F.greatest(F.max(F.to_date("created_ts")), F.max(F.to_date("closed_ts"))).alias("mx")).first()
    fd = F.col("full_date")
    dim_date = (spark.range(1)
                .select(F.explode(F.sequence(F.lit(bnd["mn"]), F.lit(bnd["mx"]),
                                             F.expr("interval 1 day"))).alias("full_date"))
                .select(F.date_format(fd, "yyyyMMdd").cast("int").alias("date_key"), fd,
                        F.dayofmonth(fd).alias("day"), F.month(fd).alias("month"),
                        F.date_format(fd, "MMMM").alias("month_name"), F.quarter(fd).alias("quarter"),
                        F.year(fd).alias("year"), F.dayofweek(fd).alias("day_of_week"),
                        F.date_format(fd, "EEEE").alias("day_name"),
                        F.dayofweek(fd).isin(1, 7).alias("is_weekend")))

    fact = (s.join(F.broadcast(dim_channel.withColumnRenamed("channel_name", "channel")), "channel")
             .join(F.broadcast(dim_category), ["category", "subcategory", "case_type"])
             .join(F.broadcast(dim_team.withColumnRenamed("team_name", "assigned_to")), "assigned_to")
             .join(F.broadcast(dim_case_status), ["status", "sla_status"])
             .select("case_number",
                     F.date_format("created_ts", "yyyyMMdd").cast("int").alias("created_date_key"),
                     F.date_format("closed_ts", "yyyyMMdd").cast("int").alias("closed_date_key"),
                     "channel_key", "category_key", "team_key", "status_key", "created_ts", "closed_ts",
                     F.col("created_hour").cast("int").alias("created_hour"),
                     F.round("resolution_hours_clean", 2).alias("resolution_hours"),
                     F.col("is_within_sla").cast("int").alias("is_within_sla"),
                     F.col("is_closed").cast("int").alias("is_closed"),
                     F.lit(1).alias("case_count")))

    tables = {"dim_date": dim_date, "dim_channel": dim_channel, "dim_category": dim_category,
              "dim_team": dim_team, "dim_case_status": dim_case_status, "fact_case": fact}
    for name in ORDER:
        tables[name].write.mode("overwrite").parquet(lake_path("gold", name))
        sync_layer("gold", name)

    ddl = (PROJECT_DIR / "sql" / "01_ddl_star_schema.sql").read_text(encoding="utf-8")
    with psycopg.connect(**PG, autocommit=True) as conn:
        conn.execute(ddl)
        conn.execute("TRUNCATE dw.fact_case, dw.dim_date, dw.dim_channel, dw.dim_category, "
                     "dw.dim_team, dw.dim_case_status")
    props = {"user": PG["user"], "password": PG["password"], "driver": "org.postgresql.Driver"}
    for name in ORDER:
        spark.read.parquet(lake_path("gold", name)).write.jdbc(
            PG_JDBC_URL, f"dw.{name}", mode="append", properties=props)

    with psycopg.connect(**PG) as conn:
        for name in ORDER:
            n = conn.execute(f"SELECT COUNT(*) FROM dw.{name}").fetchone()[0]
            print(f"[OK] dw.{name:16s}: {n:,} baris")
    print(f"     cek: silver={s.count():,} vs fact={fact.count():,} (harus sama)")


if __name__ == "__main__":
    main()
