#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cluster 183 papers and compare SPECTER2 vs TF-IDF K-Means (plus 2D and HDBSCAN).

Usage (from project root):
    python3 clustering.py
    python3 clustering.py --clusters 6
    python3 clustering.py --no-hdbscan

Inputs:
    reference/features/paper_embeddings.npy
    reference/features/tfidf_matrix.npz
    reference/features/analysis_pmids.csv
    results/reduction/{pca,umap,localmap}_2d.csv
    reference/features/papers_metadata_reviewed.csv  (optional)
    reference/features/tfidf_top_terms.csv          (optional)

Outputs under results/clustering/:
    kmeans_k_search.csv, clustering_metrics.csv, cluster_labels.csv,
    cluster_summary.csv, cluster_summary_tfidf.csv, clustering_settings.json
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import issparse, load_npz
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_pmids(path):
    if not path.is_file():
        raise FileNotFoundError(f"找不到 PMID 顺序文件：{path}")
    rows = read_csv(path)
    if not rows or "pmid" not in rows[0]:
        raise ValueError("analysis_pmids.csv 必须包含 pmid 列")
    pmids = [str(r["pmid"]).strip() for r in rows]
    if any(not p for p in pmids) or len(set(pmids)) != len(pmids):
        raise ValueError("analysis_pmids.csv 存在空 PMID 或重复 PMID")
    for i, row in enumerate(rows):
        if "row_index_0based" in row and str(row["row_index_0based"]).strip() != str(i):
            raise ValueError(f"analysis_pmids.csv 行索引有误：第 {i} 行")
    return pmids


def load_2d(path, pmids):
    if not path.is_file():
        raise FileNotFoundError(f"缺少降维结果：{path}\n请先运行 dimensionality_reduction.py")
    rows = read_csv(path)
    if len(rows) != len(pmids):
        raise ValueError(f"{path.name} 的行数与 PMID 数量不匹配")
    if not rows or not {"pmid", "x", "y"}.issubset(rows[0]):
        raise ValueError(f"{path.name} 必须包含 pmid、x、y 列")
    for i, (row, pmid) in enumerate(zip(rows, pmids)):
        if str(row["pmid"]).strip() != pmid:
            raise ValueError(f"{path.name} 第 {i} 行 PMID 错位，请检查 analysis_pmids.csv")
        if "row_index_0based" in row and str(row["row_index_0based"]).strip() != str(i):
            raise ValueError(f"{path.name} 第 {i} 行 row_index_0based 错误")
    x = np.array([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"{path.name} 含 NaN 或 Inf")
    return x


def cluster_metrics(x, labels):
    """For HDBSCAN, exclude noise (-1) when computing internal metrics."""
    labels = np.asarray(labels)
    valid = labels != -1
    y = labels[valid]
    x_valid = x[valid]
    n_clusters = len(np.unique(y))
    n_scored = len(y)
    scores = {
        "n_clusters": n_clusters,
        "n_noise": int((~valid).sum()),
        "n_scored": n_scored,
        "silhouette": "",
        "davies_bouldin": "",
        "calinski_harabasz": "",
    }
    if 2 <= n_clusters < n_scored:
        scores["silhouette"] = round(float(silhouette_score(x_valid, y)), 6)
        # Davies–Bouldin / Calinski–Harabasz require dense arrays in scikit-learn.
        # 183 x 12000 float64 is about 18 MB, so converting only for scoring is safe.
        dense_valid = x_valid.toarray() if issparse(x_valid) else x_valid
        scores["davies_bouldin"] = round(float(davies_bouldin_score(dense_valid, y)), 6)
        scores["calinski_harabasz"] = round(float(calinski_harabasz_score(dense_valid, y)), 6)
    return scores


def cluster_sizes(labels):
    sizes = [count for label, count in Counter(labels).items() if int(label) >= 0]
    return (min(sizes), max(sizes)) if sizes else (0, 0)


def main():
    parser = argparse.ArgumentParser(description="K-Means / HDBSCAN 文献聚类与降维对照")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--k-min", type=int, default=2, help="K-Means 扫描最小聚类数")
    parser.add_argument("--k-max", type=int, default=12, help="K-Means 扫描最大聚类数")
    parser.add_argument("--clusters", type=int, default=None, help="手动指定 K-Means 聚类数；默认按原始特征轮廓系数选")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN 最小簇大小")
    parser.add_argument("--min-samples", type=int, default=5, help="HDBSCAN 核心点邻域样本数")
    parser.add_argument("--no-hdbscan", action="store_true", help="只运行 K-Means")
    args = parser.parse_args()

    project = args.project.expanduser().resolve()
    features_dir = project / "reference" / "features"
    reduction_dir = project / "results" / "reduction"
    output_dir = project / "results" / "clustering"

    pmids = load_pmids(features_dir / "analysis_pmids.csv")
    raw = np.load(features_dir / "paper_embeddings.npy", allow_pickle=False)
    if raw.ndim != 2 or len(raw) != len(pmids) or not np.isfinite(raw).all():
        raise ValueError("SPECTER2 矩阵维度、行数或数值不正确")
    if np.any(np.linalg.norm(raw, axis=1) == 0):
        raise ValueError("SPECTER2 中存在全零向量")
    if not 2 <= args.k_min <= args.k_max < len(pmids):
        raise ValueError("需要满足 2 <= --k-min <= --k-max < 文献数量")
    if args.clusters is not None and not 2 <= args.clusters < len(pmids):
        raise ValueError("--clusters 必须在 2 到文献数量-1 之间")
    if args.min_cluster_size < 2 or args.min_samples < 1:
        raise ValueError("HDBSCAN 参数不合法")

    # Match dimensionality_reduction.py preprocessing exactly.
    x_high = normalize(raw.astype(np.float32), norm="l2", axis=1).astype(np.float64)

    # TF-IDF was already extracted; it has the SAME row order as analysis_pmids.csv.
    tfidf_path = features_dir / "tfidf_matrix.npz"
    if not tfidf_path.is_file():
        raise FileNotFoundError(f"缺少 TF-IDF 特征：{tfidf_path}；请先运行特征提取脚本")
    x_tfidf = load_npz(tfidf_path).tocsr()
    if x_tfidf.ndim != 2 or x_tfidf.shape[0] != len(pmids):
        raise ValueError(f"TF-IDF 矩阵行数 {x_tfidf.shape[0]} 与 PMID 数量 {len(pmids)} 不匹配")
    if not np.isfinite(x_tfidf.data).all() or np.any(x_tfidf.getnnz(axis=1) == 0):
        raise ValueError("TF-IDF 特征包含 NaN/Inf 或全零文献")
    x_tfidf = normalize(x_tfidf.astype(np.float64), norm="l2", axis=1)

    datasets = {
        "specter2": x_high,
        "pca": load_2d(reduction_dir / "pca_2d.csv", pmids),
        "umap": load_2d(reduction_dir / "umap_2d.csv", pmids),
        "localmap": load_2d(reduction_dir / "localmap_2d.csv", pmids),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"读取 {len(pmids)} 篇文献；SPECTER2：{raw.shape[1]} 维；TF-IDF：{x_tfidf.shape[1]} 维", flush=True)

    # 1) Choose k using the ORIGINAL 768D embedding, not the 2D projections.
    print(f"\n[1/3] 在原始语义空间扫描 K-Means 聚类数 {args.k_min}~{args.k_max}", flush=True)
    search_rows = []
    for k in range(args.k_min, args.k_max + 1):
        start = time.perf_counter()
        labels = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(x_high)
        scores = cluster_metrics(x_high, labels)
        lo, hi = cluster_sizes(labels)
        row = {
            "k": k, "silhouette": scores["silhouette"],
            "davies_bouldin": scores["davies_bouldin"],
            "calinski_harabasz": scores["calinski_harabasz"],
            "smallest_cluster": lo, "largest_cluster": hi,
            "runtime_seconds": round(time.perf_counter() - start, 3),
        }
        search_rows.append(row)
        print(f"  k={k:2d}  Silhouette={row['silhouette']:.4f}  "
              f"DB={row['davies_bouldin']:.4f}  最小簇={lo}", flush=True)
    write_csv(output_dir / "kmeans_k_search.csv", list(search_rows[0]), search_rows)
    best = max(search_rows, key=lambda r: float(r["silhouette"]))
    chosen_k = args.clusters if args.clusters is not None else int(best["k"])
    print(f"自动最佳 k={best['k']}；本次使用 K-Means 聚类数={chosen_k}", flush=True)
    if chosen_k == args.k_min or chosen_k == args.k_max:
        print("提示：最佳 k 位于扫描边界，可结合聚类规模与主题解释继续检查。", flush=True)

    # 2) Run K-Means on original features and on the three fixed 2D embeddings.
    print("\n[2/3] K-Means：原始特征 + 3 种二维降维结果", flush=True)
    labels_by_method = {}
    metrics_rows = []
    reference_labels = None
    for space, x in datasets.items():
        start = time.perf_counter()
        labels = KMeans(n_clusters=chosen_k, n_init=10, random_state=args.seed).fit_predict(x)
        elapsed = time.perf_counter() - start
        if space == "specter2":
            reference_labels = labels
        key = f"kmeans_{space}"
        labels_by_method[key] = labels
        own = cluster_metrics(x, labels)
        high = cluster_metrics(x_high, labels)
        lo, hi = cluster_sizes(labels)
        metrics_rows.append({
            "algorithm": "KMeans", "space": space, "n_clusters": own["n_clusters"],
            "n_noise": 0, "n_scored": own["n_scored"],
            "silhouette_own_space": own["silhouette"],
            "silhouette_specter2": high["silhouette"],
            "davies_bouldin_specter2": high["davies_bouldin"],
            "calinski_harabasz_specter2": high["calinski_harabasz"],
            "ari_vs_kmeans_specter2": round(float(adjusted_rand_score(reference_labels, labels)), 6),
            "smallest_cluster": lo, "largest_cluster": hi,
            "runtime_seconds": round(elapsed, 3),
        })
        print(f"  {space:8s} Silhouette(768D)={high['silhouette']:.4f}  "
              f"ARI={metrics_rows[-1]['ari_vs_kmeans_specter2']:.4f}", flush=True)

    # 2b) NEW: TF-IDF + K-Means using exactly the SAME cluster count.
    # Cluster labels and ARI can be compared directly, but silhouettes measured
    # in different feature spaces must NOT be compared as absolute quality scores.
    print("\n[对照实验] TF-IDF + K-Means（使用相同的聚类数）", flush=True)
    start = time.perf_counter()
    labels_tfidf = KMeans(n_clusters=chosen_k, n_init=10, random_state=args.seed).fit_predict(x_tfidf)
    elapsed = time.perf_counter() - start
    labels_by_method["kmeans_tfidf"] = labels_tfidf
    own = cluster_metrics(x_tfidf, labels_tfidf)
    high = cluster_metrics(x_high, labels_tfidf)
    lo, hi = cluster_sizes(labels_tfidf)
    metrics_rows.append({
        "algorithm": "KMeans", "space": "tfidf", "n_clusters": own["n_clusters"],
        "n_noise": 0, "n_scored": own["n_scored"],
        "silhouette_own_space": own["silhouette"],
        "silhouette_specter2": high["silhouette"],
        "davies_bouldin_specter2": high["davies_bouldin"],
        "calinski_harabasz_specter2": high["calinski_harabasz"],
        "ari_vs_kmeans_specter2": round(float(adjusted_rand_score(reference_labels, labels_tfidf)), 6),
        "smallest_cluster": lo, "largest_cluster": hi,
        "runtime_seconds": round(elapsed, 3),
    })
    print(f"  tfidf     Silhouette(TF-IDF)={own['silhouette']:.4f}  "
          f"Silhouette(SPECTER2)={high['silhouette']:.4f}  "
          f"ARI={metrics_rows[-1]['ari_vs_kmeans_specter2']:.4f}", flush=True)

    # 3) Density-based clustering for comparison (built into scikit-learn >=1.3).
    if not args.no_hdbscan:
        print("\n[3/3] HDBSCAN：原始特征 + 3 种二维降维结果", flush=True)
        for space, x in datasets.items():
            start = time.perf_counter()
            labels = HDBSCAN(
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                metric="euclidean",
            ).fit_predict(x)
            elapsed = time.perf_counter() - start
            labels_by_method[f"hdbscan_{space}"] = labels
            own = cluster_metrics(x, labels)
            high = cluster_metrics(x_high, labels)
            lo, hi = cluster_sizes(labels)
            metrics_rows.append({
                "algorithm": "HDBSCAN", "space": space,
                "n_clusters": own["n_clusters"], "n_noise": own["n_noise"],
                "n_scored": own["n_scored"],
                "silhouette_own_space": own["silhouette"],
                "silhouette_specter2": high["silhouette"],
                "davies_bouldin_specter2": high["davies_bouldin"],
                "calinski_harabasz_specter2": high["calinski_harabasz"],
                "ari_vs_kmeans_specter2": round(float(adjusted_rand_score(reference_labels, labels)), 6),
                "smallest_cluster": lo, "largest_cluster": hi,
                "runtime_seconds": round(elapsed, 3),
            })
            print(f"  {space:8s} 簇数={own['n_clusters']}  "
                  f"噪声点={own['n_noise']}  Silhouette(768D)={high['silhouette']}", flush=True)
    else:
        print("\n[3/3] 已跳过 HDBSCAN", flush=True)

    # Optional metadata: never edit original CSV files.
    metadata_path = features_dir / "papers_metadata_reviewed.csv"
    if not metadata_path.is_file():
        metadata_path = features_dir / "papers_metadata.csv"
    metadata = {}
    if metadata_path.is_file():
        metadata = {str(r.get("pmid", "")).strip(): r for r in read_csv(metadata_path)}

    label_fields = ["row_index_0based", "pmid", "title", "year", "institutions"] + list(labels_by_method)
    label_rows = []
    for i, pmid in enumerate(pmids):
        meta = metadata.get(pmid, {})
        row = {
            "row_index_0based": i, "pmid": pmid,
            "title": meta.get("title", ""), "year": meta.get("year", ""),
            "institutions": meta.get("institutions", ""),
        }
        row.update({name: int(labels[i]) for name, labels in labels_by_method.items()})
        label_rows.append(row)
    write_csv(output_dir / "cluster_labels.csv", label_fields, label_rows)

    # Additional common-reference column: score every grouping in TF-IDF space.
    # Some HDBSCAN results may have fewer than two clusters (blank metric).
    for row in metrics_rows:
        key = row["algorithm"].lower() + "_" + row["space"]
        row["silhouette_tfidf"] = cluster_metrics(x_tfidf, labels_by_method[key])["silhouette"]
    write_csv(output_dir / "clustering_metrics.csv", list(metrics_rows[0]), metrics_rows)

    # Top TF-IDF terms are *clues* to help name themes, not verified cluster names.
    tfidf_path = features_dir / "tfidf_top_terms.csv"
    terms_by_pmid = {}
    if tfidf_path.is_file():
        for row in read_csv(tfidf_path):
            pmid = str(row.get("pmid", "")).strip()
            terms_by_pmid[pmid] = [t.strip().lower() for t in row.get("top_terms", "").split(";") if t.strip()]
    summary_rows = []
    for label in sorted(set(int(x) for x in reference_labels)):
        group_pmids = [pmids[i] for i, x in enumerate(reference_labels) if int(x) == label]
        frequencies = Counter(term for pmid in group_pmids for term in set(terms_by_pmid.get(pmid, [])))
        summary_rows.append({
            "cluster_id": label,
            "paper_count": len(group_pmids),
            "top_terms_by_document_frequency": "; ".join(t for t, _ in frequencies.most_common(10)),
            "sample_pmids": "; ".join(group_pmids[:5]),
        })
    write_csv(output_dir / "cluster_summary.csv", list(summary_rows[0]), summary_rows)

    # TF-IDF cluster terms: mean TF-IDF weights over all papers in the cluster.
    # This is more representative than counting appearances in each paper's Top 15.
    vocab_path = features_dir / "tfidf_vocabulary.csv"
    vocabulary = None
    if vocab_path.is_file():
        vocab_rows = read_csv(vocab_path)
        vocabulary = [None] * x_tfidf.shape[1]
        for row in vocab_rows:
            idx = int(row["index"])
            if not 0 <= idx < len(vocabulary) or vocabulary[idx] is not None:
                raise ValueError("tfidf_vocabulary.csv 存在无效或重复 index")
            vocabulary[idx] = row["term"]
        if any(v is None for v in vocabulary):
            raise ValueError("tfidf_vocabulary.csv 与 TF-IDF 列索引不一致")
    else:
        print("提示：缺少 tfidf_vocabulary.csv，仅保存 TF-IDF 聚类统计，不输出平均权重关键词。", flush=True)

    summary_tfidf = []
    for label in sorted(set(int(x) for x in labels_tfidf)):
        indices = np.flatnonzero(labels_tfidf == label)
        representative_terms = ""
        if vocabulary is not None:
            means = np.asarray(x_tfidf[indices].mean(axis=0)).ravel()
            ranks = np.argsort(means)[::-1][:10]
            representative_terms = "; ".join(vocabulary[i] for i in ranks if means[i] > 0)
        summary_tfidf.append({
            "cluster_id": label, "paper_count": len(indices),
            "top_terms_by_mean_tfidf": representative_terms,
            "sample_pmids": "; ".join(pmids[i] for i in indices[:5]),
        })
    write_csv(output_dir / "cluster_summary_tfidf.csv", list(summary_tfidf[0]), summary_tfidf)

    settings = {
        "number_of_papers": len(pmids),
        "specter2_dimensions": raw.shape[1],
        "tfidf_dimensions": x_tfidf.shape[1],
        "tfidf_normalization": "L2; rows aligned with analysis_pmids.csv",
        "preprocessing_for_specter2": "L2 normalization (matching reduction script)",
        "kmeans_clusters": chosen_k,
        "kmeans_selection": "maximum 768D silhouette over scan (unless --clusters specified)",
        "kmeans_k_min": args.k_min, "kmeans_k_max": args.k_max,
        "kmeans_n_init": 10, "random_seed": args.seed,
        "hdbscan_enabled": not args.no_hdbscan,
        "hdbscan_min_cluster_size": args.min_cluster_size,
        "hdbscan_min_samples": args.min_samples,
        "spaces": list(datasets) + ["tfidf (K-Means only)"],
        "metadata_source": str(metadata_path) if metadata else None,
        "notes": [
            "The number of K-Means clusters is NOT UMAP/LocalMAP's n_neighbors.",
            "silhouette_own_space is not directly comparable across 2D and 768D spaces.",
            "silhouette_specter2 evaluates all labels in a COMMON original embedding space.",
            "HDBSCAN metrics exclude noise points and may be blank for <2 clusters.",
            "ARI compares agreement with 768D K-Means; it is not ground-truth accuracy.",
            "silhouette_tfidf evaluates all groupings in the SAME TF-IDF space.",
            "DO NOT compare silhouette_own_space between 768D SPECTER2 and TF-IDF as absolute scores.",
            "TF-IDF and SPECTER2 use different text inputs (full text vs title/abstract).",
        ],
    }
    (output_dir / "clustering_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n完成！结果目录：{output_dir}", flush=True)
    print("主要查看：clustering_metrics.csv、cluster_summary_tfidf.csv、cluster_labels.csv", flush=True)


if __name__ == "__main__":
    main()
