#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SPECTER2 -> PCA / UMAP / LocalMAP (2D) + common quality metrics.

Run from the project directory:
    python3 dimensionality_reduction.py

Reads: reference/features/paper_embeddings.npy
       reference/features/analysis_pmids.csv
Writes: results/reduction/{pca,umap,localmap}_2d.csv
        results/reduction/reduction_metrics.csv
        results/reduction/reduction_settings.json
"""

import argparse
import csv
import inspect
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def read_pmids(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "pmid" not in reader.fieldnames:
            raise ValueError("analysis_pmids.csv 必须包含 pmid 列")
        rows = list(reader)
    pmids = [str(row["pmid"]).strip() for row in rows]
    if not pmids or any(not p for p in pmids) or len(set(pmids)) != len(pmids):
        raise ValueError("PMID 列为空、存在空值或重复值")
    if "row_index_0based" in reader.fieldnames:
        for i, row in enumerate(rows):
            if str(row["row_index_0based"]).strip() != str(i):
                raise ValueError(f"PMID 行顺序异常：第 {i} 行")
    return pmids


def get_neighbors(distance_matrix, k):
    d = distance_matrix.copy()
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def evaluate(x, y, original_neighbors, original_pairwise, k):
    """Four metrics shared by PCA, UMAP and LocalMAP."""
    trust = float(trustworthiness(x, y, n_neighbors=k, metric="euclidean"))
    neighbors_2d = get_neighbors(pairwise_distances(y), k)
    overlap = float(np.mean([
        len(set(a) & set(b)) / k
        for a, b in zip(original_neighbors, neighbors_2d)
    ]))

    distances_2d = pdist(y, metric="euclidean")
    spearman = float(spearmanr(original_pairwise, distances_2d).statistic)

    # Scale-invariant normalized stress: optimally rescale 2D distances.
    # stress = sqrt(sum((D_high - scale*D_2d)^2) / sum(D_high^2))
    denominator = float(np.dot(distances_2d, distances_2d))
    original_norm = float(np.dot(original_pairwise, original_pairwise))
    if denominator <= 0 or original_norm <= 0:
        raise ValueError("距离全为零，无法计算 Normalized Stress")
    scale = float(np.dot(original_pairwise, distances_2d)) / denominator
    residual = original_pairwise - scale * distances_2d
    stress = float(np.sqrt(np.dot(residual, residual) / original_norm))

    if not np.isfinite([trust, overlap, spearman, stress]).all():
        raise ValueError("评价指标包含 NaN 或 Inf")
    return trust, overlap, spearman, stress


def save_coordinates(path, pmids, y):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index_0based", "pmid", "x", "y"])
        for i, (pmid, point) in enumerate(zip(pmids, y)):
            writer.writerow([i, pmid, float(point[0]), float(point[1])])


def fit_localmap(x, k, seed):
    import pacmap
    if not hasattr(pacmap, "LocalMAP"):
        raise RuntimeError("pacmap 未提供 LocalMAP 类，请检查包版本")
    options = dict(
        n_components=2,
        n_neighbors=k,
        MN_ratio=0.5,
        FP_ratio=2.0,
        low_dist_thres=10.0,
    )
    if "random_state" in inspect.signature(pacmap.LocalMAP).parameters:
        options["random_state"] = seed
    model = pacmap.LocalMAP(**options)
    return model.fit_transform(x, init="pca")


def main():
    parser = argparse.ArgumentParser(description="PCA / UMAP / LocalMAP 二维降维")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project = args.project.expanduser().resolve()
    features_dir = project / "reference" / "features"
    output_dir = project / "results" / "reduction"
    pmids = read_pmids(features_dir / "analysis_pmids.csv")
    x_raw = np.load(features_dir / "paper_embeddings.npy", allow_pickle=False)

    if x_raw.ndim != 2 or x_raw.shape[0] != len(pmids):
        raise ValueError(f"特征矩阵 {x_raw.shape} 与 PMID 数量 {len(pmids)} 不匹配")
    if not np.isfinite(x_raw).all() or np.any(np.linalg.norm(x_raw, axis=1) == 0):
        raise ValueError("SPECTER2 特征包含 NaN、Inf 或全零向量")
    n, original_dim = x_raw.shape
    k = args.neighbors
    if k < 2 or k >= n // 2:
        raise ValueError("--neighbors 必须 >= 2 且小于文献数量的一半")

    # Same preprocessing/distance definition for all three algorithms.
    x = normalize(x_raw.astype(np.float32), norm="l2", axis=1)
    np.random.seed(args.seed)
    import umap
    import pacmap
    if not hasattr(umap, "UMAP") or not hasattr(pacmap, "LocalMAP"):
        raise RuntimeError("请检查 umap-learn 和 pacmap（需包含 LocalMAP）的安装")

    neighbors_high = get_neighbors(pairwise_distances(x), k)
    distances_high = pdist(x, metric="euclidean")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"读取 {n} 篇文献，SPECTER2: {original_dim} 维 -> 2 维，k={k}", flush=True)

    results = []

    def run_method(name, fit_function):
        print(f"\n正在运行 {name}...", flush=True)
        start = time.perf_counter()
        y = np.asarray(fit_function(), dtype=np.float64)
        duration = time.perf_counter() - start  # Only fit-transform time.
        if y.shape != (n, 2) or not np.isfinite(y).all():
            raise ValueError(f"{name} 输出形状错误或包含 NaN/Inf: {y.shape}")
        trust, overlap, spearman, stress = evaluate(
            x, y, neighbors_high, distances_high, k
        )
        save_coordinates(output_dir / f"{name.lower()}_2d.csv", pmids, y)
        results.append({
            "method": name,
            "trustworthiness_k": round(trust, 6),
            "knn_overlap_k": round(overlap, 6),
            "global_spearman": round(spearman, 6),
            "normalized_stress": round(stress, 6),
            "runtime_seconds": round(duration, 3),
        })
        print(f"{name}: Trust={trust:.4f}, kNN={overlap:.4f}, "
              f"Spearman={spearman:.4f}, Stress={stress:.4f}, "
              f"Time={duration:.3f}s", flush=True)

    run_method("PCA", lambda: PCA(n_components=2, svd_solver="full").fit_transform(x))
    run_method("UMAP", lambda: umap.UMAP(
        n_components=2, n_neighbors=k, min_dist=0.1,
        metric="euclidean", random_state=args.seed,
    ).fit_transform(x))
    run_method("LocalMAP", lambda: fit_localmap(x, k, args.seed))

    with (output_dir / "reduction_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fieldnames = ["method", "trustworthiness_k", "knn_overlap_k",
                      "global_spearman", "normalized_stress", "runtime_seconds"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    settings = {
        "input": str(features_dir / "paper_embeddings.npy"),
        "pmid_order": str(features_dir / "analysis_pmids.csv"),
        "number_of_papers": n,
        "original_dimensions": original_dim,
        "target_dimensions": 2,
        "preprocessing": "L2 normalization",
        "distance": "Euclidean",
        "neighbors_k": k,
        "random_seed": args.seed,
        "PCA": {"svd_solver": "full"},
        "UMAP": {"min_dist": 0.1, "n_neighbors": k},
        "LocalMAP": {"n_neighbors": k, "MN_ratio": 0.5,
                     "FP_ratio": 2.0, "low_dist_thres": 10.0, "init": "pca"},
        "metric_direction": {
            "trustworthiness_k": "higher_is_better",
            "knn_overlap_k": "higher_is_better",
            "global_spearman": "higher_is_better",
            "normalized_stress": "lower_is_better",
            "runtime_seconds": "lower_is_better",
        },
    }
    (output_dir / "reduction_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n降维完成，结果已保存至：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
