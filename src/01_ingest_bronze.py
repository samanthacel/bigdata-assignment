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
