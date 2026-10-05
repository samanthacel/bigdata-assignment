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
