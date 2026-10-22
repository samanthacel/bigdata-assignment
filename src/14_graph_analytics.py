import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from common import DATASET, FIG_DIR, OUTPUT_DIR, TABLE_DIR, get_spark, lake_path
from ml_common import idn, save_json
from pyspark.sql import functions as F

GRAPH_DIR = OUTPUT_DIR / "graph"
DAMPING, ITER = 0.85, 40
MIN_EDGE_CASES = 20


def wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - adj) / den


def spark_pagerank(spark, edges_pdf, weight):
    """PageRank terbobot pada graf tak-berarah dengan Spark DataFrame (tanpa library graf)."""
    e = spark.createDataFrame(edges_pdf[["src", "dst", weight]].rename(columns={weight: "w"}))
    e = e.unionByName(e.select(F.col("dst").alias("src"), F.col("src").alias("dst"), "w")).where("w > 0")
    out = e.groupBy("src").agg(F.sum("w").alias("out_w"))
    e = e.join(out, "src").withColumn("p", F.col("w") / F.col("out_w")).select("src", "dst", "p").cache()
    nodes = e.select(F.col("src").alias("id")).union(e.select("dst")).distinct().cache()
    n = nodes.count()
    rank = nodes.withColumn("rank", F.lit(1.0 / n))
    deltas = []
    for i in range(ITER):
        contrib = (e.join(rank.withColumnRenamed("id", "src"), "src")
                    .groupBy("dst").agg(F.sum(F.col("rank") * F.col("p")).alias("c")))
        new = (nodes.join(contrib.withColumnRenamed("dst", "id"), "id", "left")
                    .select("id", ((1 - DAMPING) / n + DAMPING * F.coalesce("c", F.lit(0.0))).alias("rank")))
        new = new.localCheckpoint()                          # potong lineage tiap iterasi
        d = (new.join(rank.withColumnRenamed("rank", "old"), "id")
                .agg(F.sum(F.abs(F.col("rank") - F.col("old")))).first()[0])
        deltas.append(d)
        rank = new
        if d < 1e-10:
            break
    return rank.toPandas().set_index("id")["rank"], deltas


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    spark = get_spark("14_graph_analytics")
    s = (spark.read.parquet(lake_path("silver", DATASET))
         .withColumn("breach", (1 - F.col("is_within_sla")).cast("int")))

    # nodes & edges
    e_ts = (s.groupBy("assigned_to", "subcategory")
             .agg(F.count("*").alias("cases"), F.sum("breach").alias("breaches"),
                  F.avg("resolution_hours_clean").alias("avg_hours"))
             .withColumn("breach_rate", F.col("breaches") / F.col("cases"))).toPandas()
    e_sc = s.groupBy("subcategory", "category").agg(F.count("*").alias("cases")).toPandas()
    v_team = (s.groupBy("assigned_to").agg(F.count("*").alias("cases"), F.sum("breach").alias("breaches"))
               .toPandas())
    v_sub = (s.groupBy("subcategory", "category").agg(F.count("*").alias("cases"),
                                                      F.sum("breach").alias("breaches")).toPandas())
    total_cases, total_breach = int(v_team.cases.sum()), int(v_team.breaches.sum())
    base_rate = total_breach / total_cases

    e_ts["src"] = "T:" + e_ts.assigned_to
    e_ts["dst"] = "S:" + e_ts.subcategory
    nodes = pd.concat([
        pd.DataFrame({"id": "T:" + v_team.assigned_to, "label": v_team.assigned_to, "type": "team",
                      "category": "", "cases": v_team.cases, "breaches": v_team.breaches}),
        pd.DataFrame({"id": "S:" + v_sub.subcategory, "label": v_sub.subcategory, "type": "subcategory",
                      "category": v_sub.category, "cases": v_sub.cases, "breaches": v_sub.breaches}),
        pd.DataFrame({"id": "C:" + e_sc.category.unique(), "label": e_sc.category.unique(),
                      "type": "category", "category": e_sc.category.unique(), "cases": 0, "breaches": 0}),
    ], ignore_index=True)
    nodes["breach_rate"] = nodes.breaches / nodes.cases.replace(0, np.nan)
    edges = pd.concat([
        e_ts.assign(relation="HANDLES")[["src", "dst", "relation", "cases", "breaches", "breach_rate", "avg_hours"]],
        pd.DataFrame({"src": "S:" + e_sc.subcategory, "dst": "C:" + e_sc.category, "relation": "BELONGS_TO",
                      "cases": e_sc.cases, "breaches": np.nan, "breach_rate": np.nan, "avg_hours": np.nan}),
    ], ignore_index=True)
    nodes.to_csv(GRAPH_DIR / "nodes.csv", index=False)
    edges.to_csv(GRAPH_DIR / "edges.csv", index=False)

    # PageRank Spark + validasi
    pr_breach_sp, deltas = spark_pagerank(spark, e_ts, "breaches")
    pr_vol_sp, _ = spark_pagerank(spark, e_ts, "cases")
    B = nx.Graph()
    for r in e_ts.itertuples():
        B.add_edge(r.src, r.dst, cases=r.cases, breaches=r.breaches, breach_rate=r.breach_rate)
    for n_ in B:
        B.nodes[n_]["bipartite"] = 0 if n_.startswith("T:") else 1
    pr_breach_nx = pd.Series(nx.pagerank(B, alpha=DAMPING, weight="breaches", tol=1e-12, max_iter=500))
    pr_vol_nx = pd.Series(nx.pagerank(B, alpha=DAMPING, weight="cases", tol=1e-12, max_iter=500))
    max_diff = float((pr_breach_sp - pr_breach_nx.reindex(pr_breach_sp.index)).abs().max())

    # analitik NetworkX
    teams = [n_ for n_ in B if n_.startswith("T:")]
    subs = [n_ for n_ in B if n_.startswith("S:")]
    deg = dict(B.degree())
    wdeg = dict(B.degree(weight="cases"))
    btw = nx.betweenness_centrality(B, normalized=True)
    comms = nx.community.louvain_communities(B, weight="cases", seed=42)
    modularity = nx.community.modularity(B, comms, weight="cases")
    comm_of = {n_: i for i, c in enumerate(comms) for n_ in c}

    # Graf lengkap 3 lapis (Tim-Subkategori-Kategori) untuk statistik keseluruhan
    G = B.copy()
    for r in e_sc.itertuples():
        G.add_edge("S:" + r.subcategory, "C:" + r.category, cases=r.cases)

    metrics = nodes.set_index("id").copy()
    metrics["degree"] = pd.Series(deg)
    metrics["weighted_degree_cases"] = pd.Series(wdeg)
    metrics["betweenness"] = pd.Series(btw)
    metrics["pagerank_breach_spark"] = pr_breach_sp
    metrics["pagerank_volume_spark"] = pr_vol_sp
    metrics["pagerank_breach_nx"] = pr_breach_nx
    metrics["risk_concentration"] = metrics.pagerank_breach_spark / metrics.pagerank_volume_spark
    metrics["community"] = pd.Series(comm_of)
    metrics = metrics.reset_index()
    metrics.to_csv(GRAPH_DIR / "node_metrics.csv", index=False)

    # Kemiripan profil subkategori antar tim (cosine)
    mat = e_ts.pivot_table(index="assigned_to", columns="subcategory", values="cases", fill_value=0)
    mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
    unit = mat.values / np.linalg.norm(mat.values, axis=1, keepdims=True)
    cos = pd.DataFrame(unit @ unit.T, index=mat.index, columns=mat.index)
    big = mat.index[mat.sum(axis=1) >= 1000]
    cos_big = cos.loc[big, big].values[np.triu_indices(len(big), 1)]

    # Hotspot: sisi tim-subkategori dengan breach signifikan di atas base rate
    hs = e_ts[e_ts.cases >= MIN_EDGE_CASES].copy()
    hs["wilson_low"] = [wilson_lower(k, n_) for k, n_ in zip(hs.breaches, hs.cases)]
    hs = hs[hs.wilson_low > base_rate].sort_values(["wilson_low", "breaches"], ascending=False)
    hs[["assigned_to", "subcategory", "cases", "breaches", "breach_rate", "wilson_low", "avg_hours"]] \
        .to_csv(GRAPH_DIR / "hotspot_edges.csv", index=False)
    hs_breach = int(hs.breaches.sum())

    # Rangkuman komunitas
    comm_rows = []
    for i, c in enumerate(comms):
        t = sorted(x[2:] for x in c if x.startswith("T:"))
        sb = [x for x in c if x.startswith("S:")]
        cat_mix = nodes.set_index("id").loc[sb, "category"].value_counts() if sb else pd.Series(dtype=int)
        comm_rows.append({"community": i, "n_team": len(t), "n_subcategory": len(sb), "teams": "; ".join(t),
                          "top_categories": "; ".join(f"{k}({v})" for k, v in cat_mix.head(3).items())})
    pd.DataFrame(comm_rows).to_csv(GRAPH_DIR / "communities.csv", index=False)

    tm = metrics[metrics.type == "team"].sort_values("pagerank_breach_spark", ascending=False)
    sm = metrics[metrics.type == "subcategory"].sort_values("pagerank_breach_spark", ascending=False)
    summary = {
        "nodes_total": G.number_of_nodes(), "edges_total": G.number_of_edges(),
        "n_team": len(teams), "n_subcategory": len(subs), "n_category": int(e_sc.category.nunique()),
        "edges_team_subcategory": B.number_of_edges(), "edges_subcategory_category": len(e_sc),
        "bipartite_density": B.number_of_edges() / (len(teams) * len(subs)),
        "connected_components": nx.number_connected_components(G),
        "teams_full_coverage_50": int(sum(deg[t] == len(subs) for t in teams)),
        "team_degree_min_max": [int(min(deg[t] for t in teams)), int(max(deg[t] for t in teams))],
        "sub_degree_min_max": [int(min(deg[x] for x in subs)), int(max(deg[x] for x in subs))],
        "louvain_communities": len(comms), "modularity": modularity,
        "cosine_big_teams_min_mean": [float(cos_big.min()), float(cos_big.mean())], "n_big_teams": len(big),
        "pagerank_iterations": len(deltas), "pagerank_last_delta": deltas[-1],
        "pagerank_max_abs_diff_spark_vs_networkx": max_diff,
        "base_breach_rate": base_rate, "hotspot_edges": len(hs), "hotspot_breaches": hs_breach,
        "hotspot_cases": int(hs.cases.sum()), "hotspot_share_of_breaches": hs_breach / total_breach,
        "hotspot_share_of_cases": int(hs.cases.sum()) / total_cases,
        "top_team_pagerank_breach": tm[["label", "cases", "breaches", "breach_rate", "pagerank_breach_spark",
                                        "pagerank_volume_spark", "risk_concentration", "degree",
                                        "betweenness"]].head(12).to_dict("records"),
        "top_sub_pagerank_breach": sm[["label", "category", "cases", "breaches", "breach_rate",
                                       "pagerank_breach_spark", "risk_concentration"]].head(10).to_dict("records"),
        "top_hotspots": hs.head(15)[["assigned_to", "subcategory", "cases", "breaches", "breach_rate",
                                     "wilson_low"]].to_dict("records"),
        "communities": comm_rows,
    }
    save_json(GRAPH_DIR / "graph_summary.json", summary)

    md = ["| Tim | Case | Breach | % Breach | PageRank breach | PageRank volume | Konsentrasi risiko | Derajat |",
          "|---|---|---|---|---|---|---|---|"]
    for r in tm.head(12).itertuples():
        md.append(f"| {r.label} | {idn(r.cases, 0)} | {idn(r.breaches, 0)} | {idn(100 * r.breach_rate)}% | "
                  f"{idn(r.pagerank_breach_spark, 4)} | {idn(r.pagerank_volume_spark, 4)} | "
                  f"{idn(r.risk_concentration)}x | {int(r.degree)} |")
    (TABLE_DIR / "graph_team_centrality.md").write_text("\n".join(md), encoding="utf-8")

    # jaringan bipartit penuh
    fig, ax = plt.subplots(figsize=(13, 9))
    pos = nx.spring_layout(B, weight=None, seed=7, k=0.35)
    ew = np.array([B[u][v]["cases"] for u, v in B.edges()])
    nx.draw_networkx_edges(B, pos, ax=ax, width=0.3 + 2.5 * np.sqrt(ew / ew.max()), alpha=0.18,
                           edge_color="#7F7F7F")
    tsize = [60 + 2400 * math.sqrt(B.degree(t, weight="cases") / total_cases) for t in teams]
    trate = [metrics.set_index("id").loc[t, "breach_rate"] for t in teams]
    sc = nx.draw_networkx_nodes(B, pos, nodelist=teams, node_size=tsize, node_color=trate, cmap="Reds",
                                vmin=0, vmax=0.5, edgecolors="black", linewidths=0.6, ax=ax)
    nx.draw_networkx_nodes(B, pos, nodelist=subs, node_size=45, node_color="#9DC3E6", node_shape="s", ax=ax)
    nx.draw_networkx_labels(B, pos, labels={t: t[2:] for t in teams}, font_size=7, ax=ax)
    plt.colorbar(sc, ax=ax, shrink=0.6, label="Rasio breach tim")
    ax.set_title("Graf Bipartit Tim (lingkaran) - Subkategori (kotak biru); ukuran = volume case",
                 fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_6_1_bipartite_network.png", dpi=200)
    plt.close(fig)

    # PageRank breach vs volume (tim)
    d = tm.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(d))
    ax.barh(y + 0.2, d.pagerank_volume_spark, height=0.4, color="#9DC3E6", label="PageRank volume (bobot case)")
    ax.barh(y - 0.2, d.pagerank_breach_spark, height=0.4, color="#C00000", label="PageRank breach (bobot pelanggaran)")
    for yi, r in zip(y, d.itertuples()):
        ax.text(max(r.pagerank_breach_spark, r.pagerank_volume_spark), yi, f"  {r.risk_concentration:.1f}x",
                va="center", fontsize=8)
    ax.set_yticks(y, d.label, fontsize=8)
    ax.set_xlabel("Skor PageRank (Spark)")
    ax.set_title("PageRank Tim: Volume vs Pelanggaran SLA (angka = konsentrasi risiko)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_6_2_pagerank_teams.png", dpi=200)
    plt.close(fig)

    # heatmap kemiripan cosine antar tim
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cos.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cos)), cos.index, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cos)), cos.index, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Cosine similarity profil subkategori")
    ax.set_title("Kemiripan Profil Pekerjaan antar Tim (urut volume case)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_6_3_team_similarity.png", dpi=200)
    plt.close(fig)

    # subgraf hotspot
    H = nx.Graph()
    for r in hs.itertuples():
        H.add_edge("T:" + r.assigned_to, "S:" + r.subcategory, w=r.breaches, rate=r.breach_rate)
    if H.number_of_edges():
        fig, ax = plt.subplots(figsize=(12, 8))
        posh = nx.spring_layout(H, seed=3, k=0.9)
        ht = [n_ for n_ in H if n_.startswith("T:")]
        hsb = [n_ for n_ in H if n_.startswith("S:")]
        ewh = [H[u][v]["rate"] for u, v in H.edges()]
        ec = nx.draw_networkx_edges(H, posh, ax=ax, width=[0.8 + 0.12 * H[u][v]["w"] for u, v in H.edges()],
                                    edge_color=ewh, edge_cmap=plt.cm.Reds, edge_vmin=0, edge_vmax=1)
        nx.draw_networkx_nodes(H, posh, nodelist=ht, node_color="#F4B183", node_size=900, edgecolors="black", ax=ax)
        nx.draw_networkx_nodes(H, posh, nodelist=hsb, node_color="#9DC3E6", node_shape="s", node_size=300, ax=ax)
        nx.draw_networkx_labels(H, posh, labels={n_: n_[2:] for n_ in H}, font_size=7, ax=ax)
        plt.colorbar(ec, ax=ax, shrink=0.6, label="Rasio breach sisi")
        ax.set_title(f"Subgraf Hotspot: {len(hs)} pasangan Tim-Subkategori dengan breach signifikan "
                     f"(n >= {MIN_EDGE_CASES}, Wilson LB > {100 * base_rate:.2f}%)", fontsize=10, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_6_4_hotspot_subgraph.png", dpi=200)
        plt.close(fig)

    print(f"[GRAPH] simpul={summary['nodes_total']} sisi={summary['edges_total']} "
          f"densitas bipartit={summary['bipartite_density']:.3f} komunitas={len(comms)} Q={modularity:.4f}")
    print(f"[GRAPH] PageRank Spark {len(deltas)} iterasi; selisih maks vs NetworkX = {max_diff:.2e}")
    print(f"[GRAPH] hotspot={len(hs)} sisi, {hs_breach} breach "
          f"({100 * hs_breach / total_breach:.1f}% breach dari {100 * hs.cases.sum() / total_cases:.1f}% case)")
    spark.stop()


if __name__ == "__main__":
    main()
