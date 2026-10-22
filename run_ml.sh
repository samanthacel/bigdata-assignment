#!/usr/bin/env bash
# Assignment 2: jalankan setelah run_all.sh (Assignment I) selesai.
set -euo pipefail
cd "$(dirname "$0")"
run() { echo -e "\n===== $1 =====" ; docker compose exec -T -w /home/jovyan/work spark python "src/$1"; }
run 09_ml_dataset.py
run 10_ml_spark.py
run 11_ml_sklearn.py
run 12_ml_compare.py
run 13_ml_scale_test.py
run 15_bootstrap_ci.py
run 14_graph_analytics.py
echo -e "\nSelesai. Hasil: outputs/ml, outputs/graph, outputs/tables, outputs/figures"
