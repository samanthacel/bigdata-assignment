#!/usr/bin/env bash
set -euo pipefail
mkdir -p src sql data/raw lake outputs

cat > docker-compose.yml <<'EOF'
name: re-customer-care

services:
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    ports: ["9000:9000", "9001:9001"]
    volumes: [minio_data:/data]

  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: dw
      POSTGRES_PASSWORD: dw123
      POSTGRES_DB: re_dw
    ports: ["5433:5432"]
    volumes: [pg_data:/var/lib/postgresql/data]

  mongo:
    image: mongo:7
    environment:
      MONGO_INITDB_ROOT_USERNAME: root
      MONGO_INITDB_ROOT_PASSWORD: root123
    ports: ["27017:27017"]
    volumes: [mongo_data:/data/db]

  spark:
    build: .
    depends_on: [minio, postgres, mongo]
    environment:
      JUPYTER_TOKEN: bigdata
      PROJECT_DIR: /home/jovyan/work
      S3_ENDPOINT: http://minio:9000
      S3_ACCESS_KEY: minioadmin
      S3_SECRET_KEY: minioadmin123
      USE_S3A: "0"
      PG_HOST: postgres
      PG_PORT: "5432"
      PG_DB: re_dw
      PG_USER: dw
      PG_PASSWORD: dw123
      MONGO_URI: mongodb://root:root123@mongo:27017/?authSource=admin
      KAFKA_BOOTSTRAP: kafka:9092
      SPARK_DRIVER_MEMORY: 4g
    ports: ["8888:8888", "4040:4040"]
    volumes: ["./:/home/jovyan/work"]

  kafka:
    image: apache/kafka:3.9.0
    profiles: ["streaming"]
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"

volumes:
  minio_data:
  pg_data:
  mongo_data:
EOF

cat > Dockerfile <<'EOF'
FROM quay.io/jupyter/pyspark-notebook:latest
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
EOF

cat > requirements.txt <<'EOF'
boto3
pymongo
psycopg[binary]
openpyxl
tabulate
kafka-python>=2.1
EOF

cat > run_all.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
run() { echo -e "\n===== $1 ====="; docker compose exec -T -w /home/jovyan/work spark python "src/$1"; }
run 01_ingest_bronze.py
run 02_profile_raw.py
run 03_clean_silver.py
run 04_build_gold.py
run 05_load_mongo.py
run 06_analysis.py
echo -e "\nSelesai. Hasil ada di folder outputs/"
EOF

# ---------------------------------------------------------------- common.py
cat > src/common.py <<'EOF'
"""Konfigurasi & utilitas bersama untuk seluruh pipeline."""
import glob
import json
import os
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[1]))
RAW_DIR = PROJECT_DIR / "data" / "raw"
LAKE_DIR = PROJECT_DIR / "lake"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
DATASET = "customer_care"
BUCKETS = ["bronze", "silver", "gold", "quarantine"]

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin123")
USE_S3A = os.environ.get("USE_S3A", "0") == "1"

PG = {
    "host": os.environ.get("PG_HOST", "postgres"),
    "port": int(os.environ.get("PG_PORT", "5432")),
    "dbname": os.environ.get("PG_DB", "re_dw"),
    "user": os.environ.get("PG_USER", "dw"),
    "password": os.environ.get("PG_PASSWORD", "dw123"),
}
PG_JDBC_URL = f"jdbc:postgresql://{PG['host']}:{PG['port']}/{PG['dbname']}"
POSTGRES_JDBC_PACKAGE = "org.postgresql:postgresql:42.7.4"
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://root:root123@mongo:27017/?authSource=admin")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
KAFKA_TOPIC = "case-events"

# Nama kolom asli -> nama standar snake_case
RENAME = {
    "casenumber": "case_number", "slastatus": "sla_status", "closeddate": "closed_date",
    "createddate": "created_date", "channel": "channel", "category": "category",
    "status": "status", "subcategory": "subcategory", "type": "case_type",
    "assignedto": "assigned_to",
}
REQUIRED = list(RENAME.values())


def norm_col(name):
    clean = str(name).replace("\ufeff", "").strip()
    key = re.sub(r"[^a-z0-9]", "", clean.lower())
    return RENAME.get(key, re.sub(r"[^a-z0-9]+", "_", clean.lower()).strip("_"))


def ensure_dirs():
    for d in (RAW_DIR, LAKE_DIR, OUTPUT_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest_manifest():
    p = LAKE_DIR / "bronze" / DATASET / "_latest.json"
    if not p.exists():
        sys.exit("Manifest bronze belum ada. Jalankan dulu: python src/01_ingest_bronze.py")
    return read_json(p)


# ------------------------------------------------------------------ MinIO
def s3_client():
    import boto3
    from botocore.client import Config
    return boto3.client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=S3_ACCESS_KEY,
                        aws_secret_access_key=S3_SECRET_KEY, region_name="us-east-1",
                        config=Config(signature_version="s3v4", connect_timeout=5,
                                      retries={"max_attempts": 2}))


def minio_available():
    try:
        s3_client().list_buckets()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] MinIO tidak bisa diakses ({e.__class__.__name__}); lanjut mode lokal saja.")
        return False


def ensure_buckets(s3):
    existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    for b in BUCKETS:
        if b not in existing:
            s3.create_bucket(Bucket=b)


def upload_dir(local_dir, bucket, prefix):
    s3 = s3_client()
    n = 0
    for p in Path(local_dir).rglob("*"):
        if p.is_file() and not p.name.endswith(".crc"):
            s3.upload_file(str(p), bucket, f"{prefix}/{p.relative_to(local_dir).as_posix()}")
            n += 1
    return n


def sync_layer(layer, name):
    """Mode lokal: cerminkan lake/<layer>/<name> ke bucket MinIO <layer>/<name>/."""
    local = LAKE_DIR / layer / name
    if USE_S3A or not local.exists() or not minio_available():
        return
    s3 = s3_client()
    ensure_buckets(s3)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=layer, Prefix=f"{name}/"):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objs:
            s3.delete_objects(Bucket=layer, Delete={"Objects": objs})
    print(f"[SYNC] {upload_dir(local, layer, name)} file -> s3://{layer}/{name}/")


def lake_path(layer, name):
    return f"s3a://{layer}/{name}" if USE_S3A else str(LAKE_DIR / layer / name)


# ------------------------------------------------------------------ Spark
def _ensure_pyspark_importable():
    try:
        import pyspark  # noqa: F401
        return
    except ImportError:
        pass
    home = os.environ.get("SPARK_HOME", "/usr/local/spark")
    sys.path.insert(0, os.path.join(home, "python"))
    for z in glob.glob(os.path.join(home, "python", "lib", "py4j-*.zip")):
        sys.path.insert(0, z)


_ensure_pyspark_importable()


def jar_version(prefix):
    dirs = []
    if os.environ.get("SPARK_HOME"):
        dirs.append(Path(os.environ["SPARK_HOME"]) / "jars")
    try:
        import pyspark
        dirs.append(Path(pyspark.__file__).parent / "jars")
    except ImportError:
        pass
    for d in dirs:
        for f in sorted(d.glob(f"{prefix}-*.jar")):
            m = re.match(rf"{re.escape(prefix)}-(\d+\.\d+\.\d+)\.jar$", f.name)
            if m:
                return m.group(1)
    return None


def get_spark(app_name, packages=None):
    from pyspark.sql import SparkSession
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    pkgs = list(packages or [])
    b = (SparkSession.builder.appName(app_name)
         .master(os.environ.get("SPARK_MASTER", "local[*]"))
         .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.sql.ansi.enabled", "false")
         .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
         .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.jars.ivy", str(PROJECT_DIR / ".ivy2")))
    if USE_S3A:
        pkgs.append(f"org.apache.hadoop:hadoop-aws:{jar_version('hadoop-client-api') or '3.3.4'}")
        s3conf = {
            "spark.hadoop.fs.s3a.endpoint": S3_ENDPOINT,
            "spark.hadoop.fs.s3a.endpoint.region": "us-east-1",
            "spark.hadoop.fs.s3a.access.key": S3_ACCESS_KEY,
            "spark.hadoop.fs.s3a.secret.key": S3_SECRET_KEY,
            "spark.hadoop.fs.s3a.path.style.access": "true",
            "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
            "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
            "spark.hadoop.fs.s3a.aws.credentials.provider":
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        }
        for k, v in s3conf.items():
            b = b.config(k, v)
    if pkgs:
        b = b.config("spark.jars.packages", ",".join(pkgs))
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
EOF

# ---------------------------------------------------------------- 01 ingest
cat > src/01_ingest_bronze.py <<'EOF'
"""TAHAP 1 - Data ingestion (batch): file Kaggle -> Bronze layer (lokal + MinIO).
Bronze bersifat immutable: file asli disimpan apa adanya + salinan CSV UTF-8 + manifest."""
import hashlib
import shutil
import sys
from datetime import datetime, timezone

import pandas as pd

from common import (DATASET, LAKE_DIR, RAW_DIR, USE_S3A, ensure_buckets, ensure_dirs,
                    minio_available, s3_client, upload_dir, write_json)

SOURCE_URL = "https://www.kaggle.com/datasets/akshayahk/real-estate-customer-care-dataset"


def find_source_file():
    files = sorted((p for p in RAW_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}),
                   key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        sys.exit(f"Tidak ada file .csv/.xlsx di {RAW_DIR}. Download dataset Kaggle ke data/raw/.")
    if len(files) > 1:
        print(f"[INFO] Ada {len(files)} file, memakai yang terbesar: {files[0].name}")
    return files[0]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_any(path):
    if path.suffix.lower() == ".csv":
        for enc in ("utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc), enc
            except UnicodeDecodeError:
                continue
    return pd.read_excel(path, sheet_name=0, dtype=str).fillna(""), "excel"


def main():
    ensure_dirs()
    src = find_source_file()
    df, enc = read_any(src)
    ingest_date = datetime.now().strftime("%Y-%m-%d")
    base_rel = f"{DATASET}/ingest_date={ingest_date}"
    bronze = LAKE_DIR / "bronze" / base_rel
    (bronze / "original").mkdir(parents=True, exist_ok=True)
    (bronze / "data").mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, bronze / "original" / src.name)
    csv_name = src.stem.replace(" ", "_") + ".csv"
    csv_path = bronze / "data" / csv_name
    df.to_csv(csv_path, index=False, encoding="utf-8")

    manifest = {
        "dataset": DATASET, "source": "Kaggle", "source_url": SOURCE_URL,
        "original_file": src.name, "original_size_bytes": src.stat().st_size,
        "sha256": sha256_of(src), "encoding_detected": enc,
        "n_rows": int(len(df)), "n_cols": int(df.shape[1]), "columns": list(df.columns),
        "ingest_date": ingest_date, "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_csv": str(csv_path), "s3a_csv": f"s3a://bronze/{base_rel}/data/{csv_name}",
        "minio_uploaded": False,
    }
    write_json(bronze / "_manifest.json", manifest)
    if minio_available():
        s3 = s3_client()
        ensure_buckets(s3)
        n = upload_dir(bronze, "bronze", base_rel)
        manifest["minio_uploaded"] = True
        write_json(bronze / "_manifest.json", manifest)
        s3.upload_file(str(bronze / "_manifest.json"), "bronze", f"{base_rel}/_manifest.json")
        print(f"[OK] {n} file diunggah ke s3://bronze/{base_rel}/")
    elif USE_S3A:
        sys.exit("USE_S3A=1 tetapi MinIO tidak bisa diakses.")
    write_json(LAKE_DIR / "bronze" / DATASET / "_latest.json", manifest)

    print(f"[OK] Bronze siap: {manifest['n_rows']:,} baris x {manifest['n_cols']} kolom")
    print(f"     File   : {src.name} ({manifest['original_size_bytes']/1024:,.1f} KB), encoding={enc}")
    print(f"     SHA256 : {manifest['sha256']}")
    print(f"     Kolom  : {manifest['columns']}")


if __name__ == "__main__":
    main()
EOF

# ---------------------------------------------------------------- 02 profile
cat > src/02_profile_raw.py <<'EOF'
"""TAHAP 2 - Profiling data mentah: identifikasi elemen, tipe, format, kelengkapan."""
import re

import pandas as pd

from common import OUTPUT_DIR, ensure_dirs, latest_manifest, write_json


def infer_kind(nn):
    if nn.empty:
        return "kosong"
    if pd.to_numeric(nn, errors="coerce").notna().mean() > 0.98:
        return "numerik (ID/angka)"
    if nn.str.match(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}").mean() > 0.9:
        return "tanggal-waktu"
    if nn.nunique() <= 50 or nn.nunique() / len(nn) < 0.05:
        return "kategorikal"
    return "teks bebas"


def date_hint(nn):
    a = b = 0
    for v in nn.head(20000):
        m = re.match(r"^\s*(\d{1,2})[/-](\d{1,2})[/-]", v)
        if m:
            a += int(m.group(1)) > 12
            b += int(m.group(2)) > 12
    return {"bagian1_gt12": a, "bagian2_gt12": b, "contoh": nn.head(3).tolist()}


def main():
    ensure_dirs()
    m = latest_manifest()
    df = pd.read_csv(m["local_csv"], dtype=str, keep_default_na=False)
    rows, hints = [], {}
    for c in df.columns:
        s = df[c].str.strip()
        s = s.mask(s == "")
        nn = s.dropna()
        kind = infer_kind(nn)
        if kind == "tanggal-waktu":
            hints[c] = date_hint(nn)
        top = nn.value_counts().head(5)
        rows.append({
            "kolom": c, "jenis_terdeteksi": kind, "non_null": int(nn.size),
            "null": int(s.isna().sum()), "null_pct": round(float(s.isna().mean()) * 100, 2),
            "unik": int(nn.nunique()), "contoh": nn.iloc[0] if len(nn) else "",
            "top5": "; ".join(f"{k} ({v})" for k, v in top.items()),
        })
    prof = pd.DataFrame(rows)
    summary = {"n_rows": len(df), "n_cols": df.shape[1],
               "exact_duplicate_rows": int(df.duplicated().sum()),
               "columns": rows, "date_hints": hints}
    write_json(OUTPUT_DIR / "profile_raw.json", summary)
    md = (f"# Profil Data Mentah\n\nBaris: {len(df):,} | Kolom: {df.shape[1]} | "
          f"Duplikat persis: {summary['exact_duplicate_rows']:,}\n\n"
          + prof.to_markdown(index=False)
          + "\n\n## Petunjuk format tanggal\n\n" + str(hints) + "\n")
    (OUTPUT_DIR / "profile_raw.md").write_text(md, encoding="utf-8")
    print(prof[["kolom", "jenis_terdeteksi", "null", "unik", "contoh"]].to_string(index=False))
    print(f"\n[OK] outputs/profile_raw.md & profile_raw.json | petunjuk tanggal: {hints}")


if __name__ == "__main__":
    main()
EOF

# ---------------------------------------------------------------- 03 clean
cat > src/03_clean_silver.py <<'EOF'
"""TAHAP 3 - Pembersihan & organisasi data (Bronze -> Silver) dengan PySpark.
Output: lake/silver/customer_care (Parquet, partisi created_ym), lake/quarantine/customer_care,
        outputs/dq_report.json"""
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
EOF

# ---------------------------------------------------------------- 04 gold
cat > src/04_build_gold.py <<'EOF'
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
EOF

# ---------------------------------------------------------------- 05 mongo
cat > src/05_load_mongo.py <<'EOF'
"""TAHAP 5 - Model dokumen (MongoDB): satu dokumen 'Case 360' per case."""
from pymongo import ASCENDING, DESCENDING, MongoClient

from common import DATASET, MONGO_URI, OUTPUT_DIR, get_spark, lake_path, write_json


def to_doc(r):
    timeline = [{"event": "CREATED", "at": r["created_ts"], "via": r["channel"]},
                {"event": "ASSIGNED", "to": r["assigned_to"]}]
    if r["closed_ts"]:
        timeline.append({"event": "CLOSED", "at": r["closed_ts"]})
    return {
        "_id": int(r["case_number"]),
        "channel": r["channel"],
        "classification": {"category": r["category"], "subcategory": r["subcategory"],
                           "type": r["case_type"]},
        "assignment": {"team": r["assigned_to"]},
        "status": r["status"], "is_closed": bool(r["is_closed"]),
        "sla": {"status": r["sla_status"], "status_raw": r["sla_status_raw"],
                "within": None if r["is_within_sla"] is None else bool(r["is_within_sla"])},
        "created_at": r["created_ts"], "closed_at": r["closed_ts"],
        "metrics": {"resolution_hours": r["resolution_hours_clean"]},
        "timeline": timeline,
        "dq_flags": [f for f in (r["dq_flags"] or "").split(",") if f],
        "notes": [],  # di produksi: transkrip telepon, email, chat (data tidak terstruktur)
    }


def main():
    spark = get_spark("05_load_mongo")
    s = spark.read.parquet(lake_path("silver", DATASET))
    coll = MongoClient(MONGO_URI)["customer_care"]["cases"]
    coll.drop()
    batch, n = [], 0
    for row in s.toLocalIterator():
        batch.append(to_doc(row.asDict()))
        if len(batch) >= 2000:
            coll.insert_many(batch)
            n += len(batch)
            batch = []
    if batch:
        coll.insert_many(batch)
        n += len(batch)
    coll.create_index([("classification.category", ASCENDING), ("sla.within", ASCENDING)])
    coll.create_index([("created_at", DESCENDING)])
    coll.create_index([("assignment.team", ASCENDING)])

    pipeline = [
        {"$match": {"sla.within": {"$ne": None}}},
        {"$group": {"_id": "$classification.category", "cases": {"$sum": 1},
                    "within": {"$sum": {"$cond": ["$sla.within", 1, 0]}},
                    "avg_res_h": {"$avg": "$metrics.resolution_hours"}}},
        {"$project": {"cases": 1, "avg_res_h": {"$round": ["$avg_res_h", 2]},
                      "sla_compliance_pct": {"$round": [{"$multiply": [
                          {"$divide": ["$within", "$cases"]}, 100]}, 2]}}},
        {"$sort": {"cases": -1}},
    ]
    result = list(coll.aggregate(pipeline))
    write_json(OUTPUT_DIR / "mongo_aggregation_example.json",
               {"pipeline": pipeline, "result": result, "sample_document": coll.find_one()})
    print(f"[OK] {n:,} dokumen dimuat ke customer_care.cases")
    for r in result:
        print(f"     {r['_id']:30s} cases={r['cases']:>7,}  SLA={r['sla_compliance_pct']}%  avg={r['avg_res_h']} jam")


if __name__ == "__main__":
    main()
EOF

# ---------------------------------------------------------------- 06 analysis
cat > src/06_analysis.py <<'EOF'
"""TAHAP 6 - Analisis deskriptif, grafik, dan nilai placeholder untuk laporan Bab 4."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (DATASET, FIG_DIR, OUTPUT_DIR, REQUIRED, TABLE_DIR, ensure_dirs,  # noqa: E402
                    get_spark, lake_path, latest_manifest, read_json, write_json)
from pyspark.sql import functions as F  # noqa: E402

DIM = ["channel", "category", "subcategory", "case_type", "assigned_to", "status", "sla_status"]
DAY_ID = {1: "Minggu", 2: "Senin", 3: "Selasa", 4: "Rabu", 5: "Kamis", 6: "Jumat", 7: "Sabtu"}
C1, C2 = "#1F4E79", "#C00000"


def idn(x, d=2):
    if x is None:
        return "-"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def barh(pdf, label, value, title, xlabel, fname, fmt="{:,.0f}"):
    d = pdf.sort_values(value)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.42 * len(d) + 1.2)))
    ax.barh(d[label].astype(str), d[value], color=C1)
    for i, v in enumerate(d[value]):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=200)
    plt.close(fig)


def main():
    ensure_dirs()
    spark = get_spark("06_analysis")
    s = spark.read.parquet(lake_path("silver", DATASET)).fillna("Unknown", subset=DIM).cache()
    n = s.count()

    def dist(c):
        pdf = s.groupBy(c).count().orderBy(F.desc("count")).toPandas()
        pdf["pct"] = (100 * pdf["count"] / pdf["count"].sum()).round(2)
        pdf.to_csv(TABLE_DIR / f"dist_{c}.csv", index=False)
        (TABLE_DIR / f"dist_{c}.md").write_text(pdf.to_markdown(index=False), encoding="utf-8")
        return pdf

    d = {c: dist(c) for c in DIM}
    st = s.agg(F.avg("resolution_hours_clean").alias("avg"),
               F.percentile_approx("resolution_hours_clean", 0.5).alias("med"),
               F.percentile_approx("resolution_hours_clean", 0.9).alias("p90"),
               F.percentile_approx("resolution_hours_clean", 0.99).alias("p99"),
               F.avg("is_within_sla").alias("sla"), F.min("created_ts").alias("mn"),
               F.max("created_ts").alias("mx"), F.countDistinct("created_ym").alias("months"),
               F.sum(1 - F.col("is_closed")).alias("open")).first()
    sla_cat = (s.where(F.col("is_within_sla").isNotNull()).groupBy("category")
                .agg(F.count("*").alias("cases"), (F.avg("is_within_sla") * 100).alias("sla_pct"),
                     F.avg("resolution_hours_clean").alias("avg_hours"))
                .orderBy("sla_pct").toPandas())
    sla_cat.round(2).to_csv(TABLE_DIR / "sla_by_category.csv", index=False)
    (TABLE_DIR / "sla_by_category.md").write_text(sla_cat.round(2).to_markdown(index=False), encoding="utf-8")
    monthly = (s.groupBy("created_ym").agg(F.count("*").alias("cases"),
                                           (F.avg("is_within_sla") * 100).alias("sla_pct"))
                .orderBy("created_ym").toPandas())
    monthly.round(2).to_csv(TABLE_DIR / "monthly_trend.csv", index=False)
    hourly = s.groupBy("created_hour").count().orderBy("created_hour").toPandas()
    heat = s.groupBy("created_dow", "created_hour").count().toPandas()
    dow_top = s.groupBy("created_dow").count().orderBy(F.desc("count")).first()
    res = s.where(F.col("resolution_hours_clean").isNotNull()).select("resolution_hours_clean")
    if n > 300_000:
        res = res.sample(fraction=300_000 / n, seed=42)
    res = res.toPandas()["resolution_hours_clean"]

    # ---------------- grafik
    barh(d["sla_status"], "sla_status", "count", "Distribusi Status SLA", "Jumlah case", "fig_4_1_sla_status.png")
    barh(d["category"], "category", "count", "Distribusi Case per Kategori", "Jumlah case", "fig_4_2_category.png")
    barh(d["subcategory"].head(15), "subcategory", "count", "15 Subkategori Teratas", "Jumlah case",
         "fig_4_3_subcategory_top15.png")
    barh(d["channel"], "channel", "count", "Distribusi Case per Kanal", "Jumlah case", "fig_4_4_channel.png")
    barh(d["case_type"], "case_type", "count", "Distribusi Case per Tipe", "Jumlah case", "fig_4_5_type.png")
    if len(res):
        clip = st["p99"] or float(res.max())
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(res.clip(upper=clip), bins=50, color=C1, edgecolor="white")
        if st["med"] is not None:
            ax.axvline(st["med"], color=C2, ls="--", label=f"Median {st['med']:.1f} jam")
        if st["p90"] is not None:
            ax.axvline(st["p90"], color="#ED7D31", ls=":", label=f"P90 {st['p90']:.1f} jam")
        ax.set_title("Distribusi Waktu Penyelesaian (dipotong di P99)", fontweight="bold")
        ax.set_xlabel("Jam")
        ax.set_ylabel("Jumlah case")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_4_6_resolution_hist.png", dpi=200)
        plt.close(fig)
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(monthly["created_ym"], monthly["cases"], color=C1)
    ax1.set_ylabel("Jumlah case")
    ax2 = ax1.twinx()
    ax2.plot(monthly["created_ym"], monthly["sla_pct"], color=C2, marker="o")
    ax2.set_ylabel("Kepatuhan SLA (%)")
    ax2.set_ylim(0, 105)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_title("Tren Bulanan: Volume Case & Kepatuhan SLA", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_4_7_monthly_trend.png", dpi=200)
    plt.close(fig)
    barh(sla_cat, "category", "sla_pct", "Kepatuhan SLA per Kategori (%)", "% Within SLA",
         "fig_4_8_sla_by_category.png", fmt="{:.1f}%")
    mat = (heat.pivot_table(index="created_dow", columns="created_hour", values="count", fill_value=0)
               .reindex(index=range(1, 8), columns=range(24), fill_value=0))
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(mat.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(7))
    ax.set_yticklabels([DAY_ID[i] for i in range(1, 8)])
    ax.set_xticks(range(24))
    ax.set_xlabel("Jam dibuat")
    ax.set_title("Beban Case: Hari x Jam", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Jumlah case")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_4_9_heatmap_dow_hour.png", dpi=200)
    plt.close(fig)
    barh(d["assigned_to"], "assigned_to", "count", "Distribusi Case per Tim Penanggung Jawab",
         "Jumlah case", "fig_4_10_assigned_to.png")

    # ---------------- nilai placeholder
    man = latest_manifest()
    dq = read_json(OUTPUT_DIR / "dq_report.json")

    def top(c):
        return (str(d[c].iloc[0][c]), idn(d[c].iloc[0]["pct"])) if len(d[c]) else ("-", "-")

    valid = sla_cat[sla_cat["cases"] >= max(10, 0.01 * n)]
    valid = valid if len(valid) else sla_cat
    comp = {k: v for k, v in dq["completeness_pct"].items() if k in REQUIRED}
    min_col = min(comp, key=comp.get) if comp else "-"
    peak_hour = int(hourly.loc[hourly["count"].idxmax(), "created_hour"]) if len(hourly) else None
    sla_pct = st["sla"] * 100 if st["sla"] is not None else None
    v = {
        "FILE_NAME": man["original_file"], "FILE_SIZE_KB": idn(man["original_size_bytes"] / 1024, 1),
        "SHA256_SHORT": man["sha256"][:16] + "...", "INGEST_DATE": man["ingest_date"],
        "N_ROWS_RAW": idn(man["n_rows"], 0), "N_COLS_RAW": man["n_cols"],
        "DATE_FORMAT": dq["date_format_detected"], "N_EXACT_DUP": idn(dq["n_exact_duplicates"], 0),
        "N_DUP_CASE": idn(dq["n_duplicate_case_number"], 0), "N_QUARANTINE": idn(dq["n_quarantined"], 0),
        "N_BAD_CREATED": idn(dq["n_created_unparseable"], 0), "N_BAD_CLOSED": idn(dq["n_closed_unparseable"], 0),
        "N_NEG_DURATION": idn(dq["n_negative_duration"], 0), "N_CLOSED_NO_DATE": idn(dq["n_closed_without_date"], 0),
        "N_OPEN_WITH_DATE": idn(dq["n_open_with_closed_date"], 0), "N_OUTLIER": idn(dq["n_outlier_resolution"], 0),
        "P99_HOURS": idn(dq["p99_resolution_hours"]), "N_CASING_FIX": idn(sum(dq["casing_fixes"].values()), 0),
        "N_SLA_OTHER": idn(dq["n_sla_unmapped"], 0), "N_ROWS_CLEAN": idn(n, 0), "DQ_SCORE": idn(dq["dq_score_pct"]),
        "COMPLETENESS_MIN_COL": min_col, "COMPLETENESS_MIN_PCT": idn(comp.get(min_col)),
        "DATE_MIN": str(st["mn"])[:16], "DATE_MAX": str(st["mx"])[:16], "N_MONTHS": st["months"],
        "N_CATEGORY": len(d["category"]), "N_SUBCATEGORY": len(d["subcategory"]),
        "N_TYPE": len(d["case_type"]), "N_CHANNEL": len(d["channel"]), "N_TEAM": len(d["assigned_to"]),
        "TOP_CATEGORY": top("category")[0], "TOP_CATEGORY_PCT": top("category")[1],
        "TOP_SUBCATEGORY": top("subcategory")[0], "TOP_CHANNEL": top("channel")[0],
        "TOP_CHANNEL_PCT": top("channel")[1], "TOP_TEAM": top("assigned_to")[0],
        "TOP_TEAM_PCT": top("assigned_to")[1],
        "SLA_COMPLIANCE_PCT": idn(sla_pct), "SLA_BREACH_PCT": idn(100 - sla_pct if sla_pct is not None else None),
        "AVG_RES_HOURS": idn(st["avg"]), "MEDIAN_RES_HOURS": idn(st["med"]), "P90_RES_HOURS": idn(st["p90"]),
        "WORST_SLA_CATEGORY": str(valid.iloc[0]["category"]) if len(valid) else "-",
        "WORST_SLA_CATEGORY_PCT": idn(valid.iloc[0]["sla_pct"]) if len(valid) else "-",
        "BEST_SLA_CATEGORY": str(valid.iloc[-1]["category"]) if len(valid) else "-",
        "BEST_SLA_CATEGORY_PCT": idn(valid.iloc[-1]["sla_pct"]) if len(valid) else "-",
        "PEAK_HOUR": f"{peak_hour:02d}.00" if peak_hour is not None else "-",
        "PEAK_DAY": DAY_ID.get(dow_top["created_dow"], "-") if dow_top else "-",
        "OPEN_CASES": idn(st["open"] or 0, 0),
    }
    write_json(OUTPUT_DIR / "report_values.json", v)
    md = ["# Nilai untuk laporan (Find & Replace {{PLACEHOLDER}})", "", "| Placeholder | Nilai |", "|---|---|"]
    md += [f"| {{{{{k}}}}} | {val} |" for k, val in v.items()]
    md += ["", "## Gambar (outputs/figures/)"] + [f"- {p.name}" for p in sorted(FIG_DIR.glob("*.png"))]
    md += ["", "## Tabel (outputs/tables/)"] + [f"- {p.name}" for p in sorted(TABLE_DIR.glob("*.md"))]
    (OUTPUT_DIR / "report_values.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:len(v) + 4]))
    print("\n[OK] Grafik di outputs/figures, tabel di outputs/tables, nilai di outputs/report_values.md")


if __name__ == "__main__":
    main()
EOF

# ---------------------------------------------------------------- 07/08 streaming (opsional)
cat > src/07_stream_producer.py <<'EOF'
"""(Opsional) Simulasi sumber streaming: replay case sebagai event JSON ke Kafka."""
import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd
from kafka import KafkaProducer

from common import KAFKA_BOOTSTRAP, KAFKA_TOPIC, latest_manifest, norm_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=20, help="event per detik (0 = secepatnya)")
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args()
    df = pd.read_csv(latest_manifest()["local_csv"], dtype=str, keep_default_na=False)
    df.columns = [norm_col(c) for c in df.columns]
    prod = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP, linger_ms=20,
                         key_serializer=lambda k: str(k).encode(),
                         value_serializer=lambda v: json.dumps(v).encode())
    for i, rec in enumerate(df.head(a.limit).to_dict("records"), 1):
        rec["event_time"] = datetime.now(timezone.utc).isoformat()
        prod.send(KAFKA_TOPIC, key=rec.get("case_number"), value=rec)
        if i % 500 == 0:
            print(f"{i:,} event terkirim")
        if a.rate > 0:
            time.sleep(1.0 / a.rate)
    prod.flush()
    print("[OK] selesai mengirim event")


if __name__ == "__main__":
    main()
EOF

cat > src/08_stream_sla_monitor.py <<'EOF'
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
EOF

# ---------------------------------------------------------------- SQL
cat > sql/01_ddl_star_schema.sql <<'EOF'
CREATE SCHEMA IF NOT EXISTS dw;

CREATE TABLE IF NOT EXISTS dw.dim_date (
  date_key INTEGER PRIMARY KEY, full_date DATE NOT NULL, day INTEGER, month INTEGER,
  month_name VARCHAR(20), quarter INTEGER, year INTEGER, day_of_week INTEGER,
  day_name VARCHAR(20), is_weekend BOOLEAN
);
CREATE TABLE IF NOT EXISTS dw.dim_channel (
  channel_key INTEGER PRIMARY KEY, channel_name VARCHAR(100) NOT NULL
);
CREATE TABLE IF NOT EXISTS dw.dim_category (
  category_key INTEGER PRIMARY KEY, category VARCHAR(150), subcategory VARCHAR(150), case_type VARCHAR(100)
);
CREATE TABLE IF NOT EXISTS dw.dim_team (
  team_key INTEGER PRIMARY KEY, team_name VARCHAR(150)
);
CREATE TABLE IF NOT EXISTS dw.dim_case_status (
  status_key INTEGER PRIMARY KEY, status VARCHAR(50), sla_status VARCHAR(50)
);
CREATE TABLE IF NOT EXISTS dw.fact_case (
  case_number      BIGINT PRIMARY KEY,
  created_date_key INTEGER NOT NULL REFERENCES dw.dim_date(date_key),
  closed_date_key  INTEGER REFERENCES dw.dim_date(date_key),
  channel_key      INTEGER NOT NULL REFERENCES dw.dim_channel(channel_key),
  category_key     INTEGER NOT NULL REFERENCES dw.dim_category(category_key),
  team_key         INTEGER NOT NULL REFERENCES dw.dim_team(team_key),
  status_key       INTEGER NOT NULL REFERENCES dw.dim_case_status(status_key),
  created_ts       TIMESTAMP NOT NULL,
  closed_ts        TIMESTAMP,
  created_hour     INTEGER,
  resolution_hours NUMERIC(12,2),
  is_within_sla    INTEGER,
  is_closed        INTEGER,
  case_count       INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_fact_created  ON dw.fact_case(created_date_key);
CREATE INDEX IF NOT EXISTS ix_fact_category ON dw.fact_case(category_key);
CREATE INDEX IF NOT EXISTS ix_fact_channel  ON dw.fact_case(channel_key);
CREATE INDEX IF NOT EXISTS ix_fact_team     ON dw.fact_case(team_key);

CREATE OR REPLACE VIEW dw.v_case AS
SELECT f.*, d.full_date AS created_date, d.year, d.month, d.day_name,
       c.channel_name, k.category, k.subcategory, k.case_type, t.team_name, s.status, s.sla_status
FROM dw.fact_case f
JOIN dw.dim_date d        ON d.date_key = f.created_date_key
JOIN dw.dim_channel c     USING (channel_key)
JOIN dw.dim_category k    USING (category_key)
JOIN dw.dim_team t        USING (team_key)
JOIN dw.dim_case_status s USING (status_key);
EOF

cat > sql/02_kpi_queries.sql <<'EOF'
-- 1. KPI utama
SELECT COUNT(*) AS total_case,
       ROUND(100.0 * AVG(is_within_sla), 2) AS sla_compliance_pct,
       ROUND(AVG(resolution_hours), 2) AS avg_resolution_h,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resolution_hours) AS median_h,
       PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY resolution_hours) AS p90_h
FROM dw.fact_case;

-- 2. Roll-up / drill-down: kategori -> subkategori
SELECT k.category, k.subcategory, COUNT(*) AS cases,
       ROUND(100.0 * AVG(f.is_within_sla), 2) AS sla_pct, ROUND(AVG(f.resolution_hours), 2) AS avg_h
FROM dw.fact_case f JOIN dw.dim_category k USING (category_key)
GROUP BY ROLLUP (k.category, k.subcategory)
ORDER BY k.category NULLS LAST, cases DESC;

-- 3. Tren bulanan (slice waktu)
SELECT d.year, d.month, COUNT(*) AS cases, ROUND(100.0 * AVG(f.is_within_sla), 2) AS sla_pct
FROM dw.fact_case f JOIN dw.dim_date d ON d.date_key = f.created_date_key
GROUP BY d.year, d.month ORDER BY d.year, d.month;

-- 4. Dice: kanal x tipe
SELECT c.channel_name, k.case_type, COUNT(*) AS cases
FROM dw.fact_case f JOIN dw.dim_channel c USING (channel_key) JOIN dw.dim_category k USING (category_key)
GROUP BY GROUPING SETS ((c.channel_name, k.case_type), (c.channel_name), ())
ORDER BY 1 NULLS LAST, 3 DESC;

-- 5. Beban kerja per hari & jam (perencanaan staf)
SELECT d.day_name, f.created_hour, COUNT(*) AS cases
FROM dw.fact_case f JOIN dw.dim_date d ON d.date_key = f.created_date_key
GROUP BY d.day_name, f.created_hour ORDER BY cases DESC LIMIT 20;

-- 6. Top-3 subkategori per kategori (window function)
SELECT * FROM (
  SELECT k.category, k.subcategory, COUNT(*) AS n,
         RANK() OVER (PARTITION BY k.category ORDER BY COUNT(*) DESC) AS rk
  FROM dw.fact_case f JOIN dw.dim_category k USING (category_key)
  GROUP BY k.category, k.subcategory) x
WHERE rk <= 3 ORDER BY category, rk;

-- 7. Tim dengan pelanggaran SLA terbanyak
SELECT t.team_name, COUNT(*) FILTER (WHERE f.is_within_sla = 0) AS breached,
       ROUND(100.0 * AVG(f.is_within_sla), 2) AS sla_pct
FROM dw.fact_case f JOIN dw.dim_team t USING (team_key)
GROUP BY t.team_name ORDER BY breached DESC;

-- 8. Backlog case terbuka
SELECT k.category, COUNT(*) AS open_cases
FROM dw.fact_case f JOIN dw.dim_category k USING (category_key)
WHERE f.is_closed = 0 GROUP BY k.category ORDER BY open_cases DESC;
EOF

echo "Project siap. Taruh dataset di data/raw/ lalu: docker compose build && docker compose up -d"
