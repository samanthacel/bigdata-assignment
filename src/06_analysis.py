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
