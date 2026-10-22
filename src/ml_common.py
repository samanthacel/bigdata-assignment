import json
import zlib
from pathlib import Path

import numpy as np

from common import OUTPUT_DIR

ML_DIR = OUTPUT_DIR / "ml"
PRED_DIR = ML_DIR / "predictions"
ML_DATASET = "ml_sla_features"          # lake/gold/ml_sla_features
LABEL = "label"                         # 1 = Breached SLA, 0 = Within SLA
SEED = 42

# Fitur kategorikal (di-encode one-hot / ordinal)
CAT_FEATURES = ["channel", "category", "subcategory", "case_type", "assigned_to",
                "created_hour", "created_dow"]
# Fitur biner waktu
BIN_FEATURES = ["is_weekend", "is_business_hour"]
# Fitur beban kerja (hitungan) -> ditransformasi log1p sebelum masuk model
LOAD_FEATURES = ["batch_size", "team_load_24h", "all_load_1h", "all_load_7d"]
NUM_FEATURES = BIN_FEATURES + LOAD_FEATURES
RAW_FEATURES = CAT_FEATURES + NUM_FEATURES

# Kolom yang SENGAJA tidak dipakai karena baru diketahui setelah case selesai (kebocoran label)
LEAKY_COLUMNS = ["closed_ts", "closed_date", "resolution_hours", "resolution_hours_clean",
                 "status", "is_closed", "sla_status", "sla_status_raw", "flag_outlier_resolution"]

# Pembagian deterministik berbasis CRC32(case_number): identik di Spark (F.crc32) dan Python (zlib)
TRAIN_PCT, VALID_PCT = 70, 10           # sisanya (20%) = test
OOT_CUTOFF = "2024-03-01"               # uji out-of-time: latih < cutoff, uji >= cutoff
TOP_K_PCT = 0.05                        # kapasitas tinjau supervisor: 5% case berisiko tertinggi


def split_of(case_number):
    b = zlib.crc32(str(int(case_number)).encode()) % 100
    return "train" if b < TRAIN_PCT else ("valid" if b < TRAIN_PCT + VALID_PCT else "test")


def best_f1_threshold(y, p):
    """Ambang probabilitas yang memaksimalkan F1 (dipilih pada data validasi)."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def evaluate(y, p, threshold, top_k=TOP_K_PCT):
    """Metrik evaluasi yang sama untuk semua model (Spark maupun scikit-learn)."""
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    k = max(1, int(round(top_k * len(p))))
    top = np.argsort(-p, kind="stable")[:k]
    base = y.mean()
    prec_k = y[top].mean()
    return {
        "n": int(len(y)), "positives": int(y.sum()), "base_rate": float(base),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "accuracy": float((yhat == y).mean()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        f"precision_at_{int(top_k*100)}pct": float(prec_k),
        f"recall_at_{int(top_k*100)}pct": float(y[top].sum() / max(1, y.sum())),
        f"lift_at_{int(top_k*100)}pct": float(prec_k / base) if base > 0 else None,
    }


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def idn(x, d=2):
    """Format angka gaya Indonesia: 1.234,56"""
    if x is None:
        return "-"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
