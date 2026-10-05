"""(Opsional) Spark Structured Streaming: monitor kepatuhan SLA per menit + sink ke Bronze."""
import argparse

import pyspark

from common import KAFKA_BOOTSTRAP, KAFKA_TOPIC, LAKE_DIR, get_spark, jar_version, lake_path
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

FIELDS = ["case_number", "sla_status", "closed_date", "created_date", "channel", "category",
          "status", "subcategory", "case_type", "assigned_to", "event_time"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--threshold", type=float, default=90.0, help="target SLA (%)")
    a = ap.parse_args()
    scala = ".".join((jar_version("scala-library") or "2.13.0").split(".")[:2])
    spark = get_spark("08_stream_sla_monitor",
                      [f"org.apache.spark:spark-sql-kafka-0-10_{scala}:{pyspark.__version__}"])
    schema = StructType([StructField(c, StringType()) for c in FIELDS])
    sla = F.lower(F.col("sla_status"))
    ev = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
               .option("subscribe", KAFKA_TOPIC).option("startingOffsets", "earliest").load()
               .select(F.from_json(F.col("value").cast("string"), schema).alias("e")).select("e.*")
               .withColumn("event_ts", F.to_timestamp("event_time"))
               .withColumn("within", F.when(sla.contains("within") & ~sla.contains("not"), 1).otherwise(0)))
    (ev.writeStream.format("parquet").outputMode("append")
       .option("path", lake_path("bronze", "customer_care_stream"))
       .option("checkpointLocation", str(LAKE_DIR / "_checkpoints" / "stream_bronze")).start())
    agg = (ev.withWatermark("event_ts", "2 minutes")
             .groupBy(F.window("event_ts", "1 minute").alias("w"), "category")
             .agg(F.count("*").alias("cases"), F.round(F.avg("within") * 100, 1).alias("sla_pct"))
             .withColumn("alert", F.when(F.col("sla_pct") < a.threshold, "ALERT: di bawah target")
                                   .otherwise("OK"))
             .select(F.col("w.start").alias("window_start"), "category", "cases", "sla_pct", "alert"))
    (agg.writeStream.outputMode("update").format("console").option("truncate", False)
        .trigger(processingTime="10 seconds").start())
    spark.streams.awaitAnyTermination(a.seconds)
    for q in spark.streams.active:
        q.stop()
    print("[OK] streaming dihentikan")


if __name__ == "__main__":
    main()
