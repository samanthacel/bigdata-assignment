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
