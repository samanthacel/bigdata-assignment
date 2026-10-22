"""TAHAP 12 - Perbandingan model Spark MLlib vs scikit-learn: tabel & grafik untuk Subbab 5.5-5.6.

Membaca outputs/ml/metrics_*.json dan outputs/ml/predictions/*.csv (tidak butuh Spark).
Keluaran: outputs/tables/ml_comparison.(md|csv), outputs/figures/fig_5_2..fig_5_5, outputs/ml/ml_report_values.md
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

from common import FIG_DIR, TABLE_DIR  # noqa: E402
from ml_common import LABEL, ML_DIR, PRED_DIR, idn  # noqa: E402

NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
         "gbt": "Gradient Boosting"}
COL = {"logistic_regression": "#1F4E79", "random_forest": "#548235", "gbt": "#C00000"}


def main():
    sp = json.loads((ML_DIR / "metrics_spark.json").read_text())
    sk = json.loads((ML_DIR / "metrics_sklearn.json").read_text())
    rows = []
    for eng, js, short in (("Spark MLlib", sp, "spark"), ("scikit-learn", sk, "sklearn")):
        for k, r in js["models"].items():
            t = r["test"]
            label = NAMES[k] + (" (GBTClassifier)" if eng == "Spark MLlib" and k == "gbt" else
                                " (HistGB)" if k == "gbt" else "")
            rows.append({"engine": eng, "key": k, "short": short, "model": label,
                         "params": ", ".join(f"{a}={b}" for a, b in r["best_params"].items()),
                         "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"], "precision": t["precision"],
                         "recall": t["recall"], "f1": t["f1"], "threshold": t["threshold"],
                         "recall_at_5pct": t["recall_at_5pct"], "precision_at_5pct": t["precision_at_5pct"],
                         "lift_at_5pct": t["lift_at_5pct"], "accuracy": t["accuracy"],
                         "fit_sec": r["fit_sec"], "predict_sec": r["predict_sec"]})
    cmp = pd.DataFrame(rows)
    cmp.drop(columns=["key", "short"]).round(4).to_csv(TABLE_DIR / "ml_comparison.csv", index=False)
    md = ["| Engine | Model | Hiperparameter | ROC-AUC | PR-AUC | Precision | Recall | F1 | Recall@5% | Lift@5% | Waktu latih (s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['engine']} | {r['model']} | {r['params']} | {idn(r['roc_auc'], 4)} | {idn(r['pr_auc'], 4)} | "
                  f"{idn(r['precision'], 3)} | {idn(r['recall'], 3)} | {idn(r['f1'], 3)} | "
                  f"{idn(100 * r['recall_at_5pct'], 1)}% | {idn(r['lift_at_5pct'], 1)}x | {idn(r['fit_sec'], 1)} |")
    (TABLE_DIR / "ml_comparison.md").write_text("\n".join(md), encoding="utf-8")
    abl = ["| Kelompok fitur | Jumlah fitur | ROC-AUC | PR-AUC | Recall@5% |", "|---|---|---|---|---|"]
    for a in sk.get("ablation", []):
        abl.append(f"| {a['set']} | {a['n_features']} | {idn(a['roc_auc'], 4)} | {idn(a['pr_auc'], 4)} | "
                   f"{idn(100 * a['recall_at_5pct'], 1)}% |")
    (TABLE_DIR / "ml_ablation.md").write_text("\n".join(abl), encoding="utf-8")

    preds = {(r["short"], r["key"]): pd.read_csv(PRED_DIR / f"{r['short']}_{r['key']}.csv") for r in rows}

    # --- Gambar 5.2: ROC & PR
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for (eng, k), p in preds.items():
        ls = "-" if eng == "spark" else "--"
        lab = f"{NAMES[k]} - {'Spark' if eng == 'spark' else 'sklearn'}"
        fpr, tpr, _ = roc_curve(p[LABEL], p["p"])
        pr, rc, _ = precision_recall_curve(p[LABEL], p["p"])
        ax[0].plot(fpr, tpr, ls, color=COL[k], lw=1.4, label=lab)
        ax[1].plot(rc, pr, ls, color=COL[k], lw=1.4, label=lab)
    base = preds[("spark", "gbt")][LABEL].mean()
    ax[0].plot([0, 1], [0, 1], ":", color="gray", lw=1)
    ax[1].axhline(base, ls=":", color="gray", lw=1, label=f"Acak (base rate {100 * base:.2f}%)")
    ax[0].set(title="Kurva ROC (data uji)", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].set(title="Kurva Precision-Recall (data uji)", xlabel="Recall", ylabel="Precision")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_5_2_roc_pr_curves.png", dpi=200)
    plt.close(fig)

    # --- Gambar 5.3: perbandingan metrik & waktu latih
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(3)
    for j, (eng, off) in enumerate((("Spark MLlib", -0.2), ("scikit-learn", 0.2))):
        d = cmp[cmp.engine == eng].set_index("key").reindex(list(NAMES))
        c = "#1F4E79" if j == 0 else "#ED7D31"
        for i, (m, t) in enumerate((("pr_auc", "PR-AUC"), ("recall_at_5pct", "Recall@5% (tinjau 5% teratas)"),
                                    ("fit_sec", "Waktu latih model terpilih (detik)"))):
            ax[i].bar(x + off, d[m], width=0.4, color=c, label=eng)
            for xi, v in zip(x + off, d[m]):
                ax[i].text(xi, v, f"{v:.3f}" if m != "fit_sec" else f"{v:.1f}", ha="center",
                           va="bottom", fontsize=7.5)
            ax[i].set_title(t, fontsize=10)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels([NAMES[k] for k in NAMES], fontsize=8)
        a.grid(axis="y", alpha=.3)
        a.set_axisbelow(True)
    ax[0].axhline(base, ls=":", color="gray")
    ax[1].axhline(0.05, ls=":", color="gray")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_5_3_metric_comparison.png", dpi=200)
    plt.close(fig)

    # --- Gambar 5.4: kepentingan fitur
    fi_sp = pd.Series(sp["models"]["gbt"]["feature_importance"]).sort_values()
    fi_sk = pd.Series(sk["models"]["gbt"]["permutation_importance_pr_auc"]).sort_values()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].barh(fi_sp.index, fi_sp.values, color="#1F4E79")
    ax[0].set_title("Spark GBTClassifier: featureImportances", fontsize=10)
    ax[1].barh(fi_sk.index, fi_sk.values, color="#ED7D31")
    ax[1].set_title("scikit-learn HistGB: permutation importance (penurunan PR-AUC)", fontsize=10)
    for a in ax:
        a.grid(axis="x", alpha=.3)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_5_4_feature_importance.png", dpi=200)
    plt.close(fig)

    # --- Gambar 5.5: cumulative gain + confusion matrix model terbaik Spark
    best_sp = sp["best_by_valid"]
    p = preds[("spark", best_sp)].sort_values("p", ascending=False).reset_index(drop=True)
    gain = p[LABEL].cumsum() / p[LABEL].sum()
    frac = (np.arange(len(p)) + 1) / len(p)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].plot(100 * frac, 100 * gain, color="#C00000", label=f"Spark {NAMES[best_sp]}")
    pk = preds[("sklearn", sk["best_by_valid"])].sort_values("p", ascending=False)
    ax[0].plot(100 * frac, 100 * pk[LABEL].cumsum().values / pk[LABEL].sum(), "--", color="#ED7D31",
               label=f"sklearn {NAMES[sk['best_by_valid']]}")
    ax[0].plot([0, 100], [0, 100], ":", color="gray", label="Acak")
    for k in (5, 10, 20):
        g = 100 * gain.iloc[int(k / 100 * len(p)) - 1]
        ax[0].annotate(f"{k}% -> {g:.0f}%", (k, g), textcoords="offset points", xytext=(8, -12), fontsize=8)
    ax[0].set(title="Cumulative gain (data uji)", xlabel="% case ditinjau (urut risiko tertinggi)",
              ylabel="% pelanggaran SLA tertangkap")
    ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=8)
    thr = sp["models"][best_sp]["test"]["threshold"]
    cm = confusion_matrix(p[LABEL], (p["p"] >= thr).astype(int), labels=[0, 1])
    ax[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[1].text(j, i, f"{v:,}".replace(",", "."), ha="center", va="center",
                   color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax[1].set_xticks([0, 1], ["Prediksi Within", "Prediksi Breach"])
    ax[1].set_yticks([0, 1], ["Aktual Within", "Aktual Breach"])
    ax[1].set_title(f"Confusion matrix Spark {NAMES[best_sp]} (ambang {thr:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_5_5_gain_confusion.png", dpi=200)
    plt.close(fig)

    # --- konsistensi skor antar engine (algoritma sama)
    agree = {}
    for k in NAMES:
        a = preds[("spark", k)].set_index("case_number")["p"]
        b = preds[("sklearn", k)].set_index("case_number")["p"].reindex(a.index)
        agree[k] = float(spearmanr(a, b).statistic)

    vals = {"best_spark": best_sp, "best_sklearn": sk["best_by_valid"], "spearman_spark_vs_sklearn": agree,
            "spark_oot": sp["out_of_time"], "sklearn_oot": sk["out_of_time"],
            "spark_version": sp.get("spark_version"), "sklearn_version": sk.get("sklearn_version"),
            "spark_prep_fit_sec": sp.get("prep_fit_sec"), "sklearn_load_sec": sk.get("load_sec")}
    (ML_DIR / "ml_report_values.json").write_text(json.dumps(vals, indent=2), encoding="utf-8")
    lines = ["# Nilai ML untuk laporan", "", "\n".join(md), "", "```json", json.dumps(vals, indent=2), "```"]
    (ML_DIR / "ml_report_values.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(md))
    print("\n".join(abl))
    print(json.dumps(vals, indent=2))


if __name__ == "__main__":
    main()
