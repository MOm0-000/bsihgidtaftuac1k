#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正式实验阶段②：读取 d_reduce 的三份降维向量 -> Leiden社区聚类。

只负责聚类，不再运行PCA、UMAP或LocalMAP；直接运行，无命令行参数。
输出：/home/zxx/bigdata/exp1/cluster/
  sweep_all_metrics.csv、all_cluster_labels.csv、全部簇_主题分布.csv、聚类配置.json
  三份降维矩阵仅存于 d_reduce/embeddings/；此目录不复制、不链接矩阵。

只有三组已经选定的配置，不再进行54组参数Sweep。
与 6visualize_UMAP.py 共享标准字段接口；可视化单独读取 d_reduce/embeddings/。
依赖：numpy pandas scipy scikit-learn leidenalg（安装leidenalg一般也会安装igraph）
参考主题CSV是可选的：缺失时照常计算聚类、Silhouette和簇成员。
"""
from __future__ import annotations

import hashlib
import json
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances

# ====================== 固定路径和正式聚类配置 ======================
ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / 'features_final' / 'papers_index.csv'
FEATURE_PATH = ROOT / 'features_final' / 'specter2_proximity.npy'
REDUCE_DIR = ROOT / 'd_reduce'
EMBED_DIR = REDUCE_DIR / 'embeddings'
OUTPUT_DIR = ROOT / 'cluster'

SEED = 42
LEIDEN_ITERATIONS = -1
# 这些是此前探索后选定的配置；改变它们会改变报告中的聚类结果。
METHODS = {
    'PCA':      {'spec_id': 'PCA_20d',                'dim': 20, 'graph_mode': 'mutual', 'graph_k': 10, 'gamma': 1.5},
    'UMAP':     {'spec_id': 'UMAP_10d_n20_md0p00',   'dim': 10, 'graph_mode': 'union',  'graph_k': 5,  'gamma': 1.5},
    'LocalMAP': {'spec_id': 'LocalMAP_10d_n10',      'dim': 10, 'graph_mode': 'union',  'graph_k': 10, 'gamma': 1.0},
}
GRAPH_DISTANCE = 'euclidean'
REFERENCE_PATHS = (
    ROOT / 'cluster' / '参考主题标签_本次使用.csv',
    ROOT / 'reference_topics_v1.csv',
    Path(__file__).resolve().parent / 'reference_topics_v1.csv',
)
# ==================================================================


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return 'not installed'


def need_leiden():
    try:
        import igraph as ig
        import leidenalg as la
    except ImportError as exc:
        raise ImportError('缺少Leiden依赖，请在当前Python环境运行：python -m pip install leidenalg') from exc
    return ig, la


def read_csv(path: Path, **kwargs):
    return pd.read_csv(path, encoding='utf-8-sig', **kwargs)


def load_inputs():
    files = [INDEX_PATH, FEATURE_PATH, REDUCE_DIR / '降维配置.json'] + [
        EMBED_DIR / f'{c["spec_id"]}.npy' for c in METHODS.values()]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError('请先成功运行 4d_reduce.py，缺少以下文件：\n' + '\n'.join(missing))

    index = read_csv(INDEX_PATH, dtype={'pmid': str}, keep_default_na=False)
    if not {'row_index', 'pmid', 'title'}.issubset(index.columns):
        raise ValueError('papers_index.csv缺少row_index/pmid/title')
    n = len(index)
    if n != 180 or index.pmid.duplicated().any() or not np.array_equal(
            pd.to_numeric(index.row_index, errors='raise').to_numpy(dtype=int), np.arange(n)):
        raise ValueError('索引与180篇的行顺序不一致')

    manifest = json.loads((REDUCE_DIR / '降维配置.json').read_text(encoding='utf-8'))
    if manifest.get('index_sha256') != file_sha256(INDEX_PATH):
        raise ValueError('降维时使用的论文索引已改变！请检查/重新运行d_reduce.py')
    if manifest.get('feature_sha256') != file_sha256(FEATURE_PATH):
        raise ValueError('降维时使用的SPECTER2特征已改变！请检查/重新运行d_reduce.py')
    x = np.load(FEATURE_PATH, allow_pickle=False)
    if x.shape != (n, 768) or not np.isfinite(x).all():
        raise ValueError(f'特征矩阵不是180×768有效数据：{x.shape}')
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1)
    if np.any(norm < 1e-12):
        raise ValueError('SPECTER2存在零向量')
    x /= norm[:, None]
    original_d = cosine_distances(x)
    original_d = np.maximum((original_d + original_d.T) / 2, 0)
    np.fill_diagonal(original_d, 0)

    embeddings = {}
    hash_info = manifest.get('embedding_sha256', {})
    for method, cfg in METHODS.items():
        spec = cfg['spec_id']
        path = EMBED_DIR / f'{spec}.npy'
        if hash_info.get(spec) != file_sha256(path):
            raise ValueError(f'{path}与降维配置记录不一致，拒绝混用新旧结果')
        z = np.load(path, allow_pickle=False)
        if z.shape != (n, cfg['dim']) or not np.isfinite(z).all():
            raise ValueError(f'降维坐标异常：{path}, {z.shape}')
        embeddings[method] = np.asarray(z, dtype=np.float64)
    return index, embeddings, original_d, manifest


def optional_reference(index: pd.DataFrame):
    path = next((p for p in REFERENCE_PATHS if p.is_file()), None)
    if path is None:
        print('未发现探索性参考主题标签：将跳过论文对F1，不影响正式聚类。')
        return None, None
    reference = read_csv(path, dtype={'pmid': str}, keep_default_na=False)
    if not {'pmid', 'candidate_topic_code'}.issubset(reference.columns):
        raise ValueError(f'参考标签文件缺少字段：{path}')
    if len(reference) != len(index) or reference.pmid.duplicated().any():
        raise ValueError('参考标签须恰好覆盖180个不重复PMID')
    ref = reference.set_index('pmid').reindex(index.pmid.astype(str))
    if ref.candidate_topic_code.isna().any():
        raise ValueError('参考标签PMID与本轮论文不匹配')
    ref = ref.reset_index(drop=True)
    ref.insert(0, 'pmid', index.pmid.astype(str).to_numpy())
    core_codes = {f'C{i}' for i in range(1, 6)} | {f'I{i}' for i in range(1, 9)}
    ref['主评价130篇'] = ref.candidate_topic_code.isin(core_codes).astype(int)
    if int(ref['主评价130篇'].sum()) != 130:
        raise ValueError('当前参考版本的核心主题不再是130篇，请核对标签版本')
    if {'abstract_available', 'review_note'}.issubset(ref.columns):
        good = (pd.to_numeric(ref.abstract_available, errors='coerce').eq(1) &
                ref.review_note.astype(str).str.strip().eq(''))
        ref['严格子集123篇'] = (ref['主评价130篇'].eq(1) & good).astype(int)
    else:
        ref['严格子集123篇'] = 0
    print(f'探索性主题标签：{path}（主评价130篇，仅作为参考）')
    return ref, str(path)


def knn_graph(z, k, mode, ig):
    """沿用此前Leiden Sweep的图定义：欧氏kNN + 局部尺度高斯边权。"""
    n = len(z)
    d = cdist(z, z, metric=GRAPH_DISTANCE)
    d = np.maximum((d + d.T) / 2, 0)
    np.fill_diagonal(d, np.inf)
    neighbor = np.argsort(d, axis=1, kind='stable')[:, :k]
    chosen = np.zeros((n, n), dtype=bool)
    chosen[np.arange(n)[:, None], neighbor] = True
    mask = chosen | chosen.T if mode == 'union' else chosen & chosen.T
    ii, jj = np.where(np.triu(mask, k=1))
    if not len(ii):
        raise ValueError(f'{mode}图没有边，k={k}')

    sigma = d[np.arange(n), neighbor[:, -1]]
    positive = d[np.isfinite(d) & (d > 0)]
    min_positive = float(positive.min()) if positive.size else 1.0
    sigma = np.maximum(sigma, min_positive * 1e-6)
    weights = np.exp(-np.square(d[ii, jj]) / np.maximum(sigma[ii] * sigma[jj], 1e-20))
    weights = np.maximum(weights, 1e-12)
    g = ig.Graph(n=n, edges=list(zip(ii.tolist(), jj.tolist())), directed=False)
    g.es['weight'] = weights.tolist()
    degrees = np.asarray(g.degree(), dtype=int)
    adjacency = csr_matrix((np.ones(len(ii) * 2), (np.r_[ii, jj], np.r_[jj, ii])), shape=(n, n))
    n_components, comp_idx = connected_components(adjacency, directed=False)
    _, comp_sizes = np.unique(comp_idx, return_counts=True)
    graph_info = {
        'graph_edges': int(len(ii)), 'graph_isolates': int((degrees == 0).sum()),
        'graph_components': int(n_components), 'graph_largest_component': int(comp_sizes.max()),
        'graph_mean_degree': float(degrees.mean()),
    }
    return g, graph_info


def leiden_labels(graph, gamma, la):
    result = la.find_partition(
        graph, la.RBConfigurationVertexPartition, weights='weight',
        resolution_parameter=float(gamma), n_iterations=LEIDEN_ITERATIONS, seed=SEED,
    )
    raw = np.asarray(result.membership, dtype=np.int32)
    # 社区编号依首次出现的论文行号固定，便于比较及复核；不改变分组。
    renumber = {}
    labels = np.empty_like(raw)
    for i, cid in enumerate(raw):
        if int(cid) not in renumber:
            renumber[int(cid)] = len(renumber)
        labels[i] = renumber[int(cid)]
    return labels, float(result.quality())


def pair_score(truth, labels):
    a, b = np.triu_indices(len(labels), 1)
    actual = truth[a] == truth[b]
    pred = (labels[a] >= 0) & (labels[a] == labels[b])
    tp, fp, fn = int(np.sum(actual & pred)), int(np.sum(~actual & pred)), int(np.sum(actual & ~pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {'论文对精确率': p, '论文对召回率': r, '论文对F1': f1,
            'ARI': float(adjusted_rand_score(truth, labels)),
            '成功同类对数': tp, '误合并对数': fp, '漏配对数': fn}


def evaluate(method, cfg, z, labels, quality, graph_info, original_dist, reference):
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    n_noise = int(np.sum(labels < 0))
    n_assigned = len(labels) - n_noise
    sizes = ';'.join(str(v) for v in sorted(counts, reverse=True))
    sil_low = sil_orig = np.nan
    if 2 <= len(ids) < n_assigned:
        selected = labels >= 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sil_low = float(silhouette_score(z[selected], labels[selected], metric='euclidean'))
            sil_orig = float(silhouette_score(
                original_dist[np.ix_(selected, selected)], labels[selected], metric='precomputed'))
    spec = cfg['spec_id']
    gm, k, gamma = cfg['graph_mode'], cfg['graph_k'], cfg['gamma']
    row_id = f'{spec}__Leiden_00plot_{gm}_k{k}_g{str(gamma).replace(".", "p")}'
    record = {
        # 与 visualize_UMAP_v5.py 兼容的字段：不冒用HDBSCAN参数。
        'row_id': row_id, 'spec_id': spec, 'method': method, 'dim': cfg['dim'],
        'neighbors': {'PCA': np.nan, 'UMAP': 20, 'LocalMAP': 10}[method],
        'min_dist': 0.0 if method == 'UMAP' else np.nan,
        'seed': SEED, 'n_clusters': int(len(ids)), 'n_noise': n_noise,
        'coverage': float(n_assigned / len(labels)),
        'largest_cluster': int(counts.max()) if len(counts) else 0,
        'largest_fraction_of_assigned': float(counts.max() / n_assigned) if n_assigned else 0,
        'cluster_sizes': sizes,
        'silhouette_reduced': sil_low, 'silhouette_original': sil_orig,
        'clustering_algorithm': 'Leiden', 'preferred_for_visualization': 1,
        'graph_mode': gm, 'graph_k': k, 'resolution_gamma': gamma,
        'graph_distance': GRAPH_DISTANCE, 'edge_weight_formula': 'local_scaling_gaussian',
        'leiden_objective': 'RBConfigurationVertexPartition',
        'leiden_quality_not_comparable_across_graphs': quality,
        '单篇社区数': int((counts == 1).sum()), '两篇社区数': int((counts == 2).sum()),
        '三篇社区数': int((counts == 3).sum()),
        **graph_info,
    }
    if reference is not None:
        codes = reference.candidate_topic_code.to_numpy()
        for prefix, mask_name in (('核心130', '主评价130篇'), ('严格123', '严格子集123篇')):
            mask = reference[mask_name].to_numpy(dtype=bool)
            if mask.sum() >= 3:
                record.update({f'{prefix}_{key}': val for key, val in
                               pair_score(codes[mask], labels[mask]).items()})
    return record


def cluster_members(index, labels, method, row_id, reference):
    result = []
    for cid in sorted(set(int(c) for c in labels if c >= 0)):
        pos = np.flatnonzero(labels == cid)
        row = {'row_id': row_id, 'method': method, '簇编号': cid,
               '簇论文数': len(pos),
               '完整PMID': ';'.join(index.iloc[pos].pmid.astype(str)),
               '完整论文题名': ' | '.join(f'{index.iloc[i].pmid} {index.iloc[i].title}' for i in pos)}
        if reference is not None:
            is_core = reference['主评价130篇'].to_numpy(dtype=bool)
            core_pos = pos[is_core[pos]]
            distribution = reference.iloc[core_pos].candidate_topic_code.value_counts()
            row['核心参考论文数'] = len(core_pos)
            row['未参与核心参考论文数'] = len(pos) - len(core_pos)
            row['核心主题组成_探索性'] = ';'.join(f'{code}:{size}' for code, size in distribution.items())
        result.append(row)
    return result


def main():
    ig, la = need_leiden()
    index, embeddings, original_dist, manifest = load_inputs()
    reference, reference_source = optional_reference(index)
    metric_rows, member_rows = [], []
    labels_table = {'row_index': index.row_index.astype(int).to_numpy(),
                    'pmid': index.pmid.astype(str).to_numpy()}
    for method, cfg in METHODS.items():
        z = embeddings[method]
        graph, graph_info = knn_graph(z, cfg['graph_k'], cfg['graph_mode'], ig)
        labels, quality = leiden_labels(graph, cfg['gamma'], la)
        scores = evaluate(method, cfg, z, labels, quality, graph_info, original_dist, reference)
        metric_rows.append(scores)
        labels_table[scores['row_id']] = labels
        member_rows.extend(cluster_members(index, labels, method, scores['row_id'], reference))
        f1 = scores.get('核心130_论文对F1')
        detail = f' | 参考F1={f1:.3f}' if f1 is not None else ''
        print(f'{method:8s} k={cfg["graph_k"]:2d} {cfg["graph_mode"]:6s} '
              f'gamma={cfg["gamma"]:.1f} | {scores["n_clusters"]}簇'
              f' | 原空间Silhouette={scores["silhouette_original"]:.4f}{detail}', flush=True)

    metrics = pd.DataFrame(metric_rows)
    assignments = pd.DataFrame(labels_table)
    if len(metrics) != 3 or assignments.shape != (180, 5):
        raise RuntimeError('正式实验应该输出3组Leiden结果与180篇论文的对应标签')
    if metrics.row_id.duplicated().any() or assignments.columns[2:].tolist() != metrics.row_id.tolist():
        raise RuntimeError('指标表与标签表的row_id不匹配')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / 'sweep_all_metrics.csv', index=False, encoding='utf-8-sig')
    assignments.to_csv(OUTPUT_DIR / 'all_cluster_labels.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(member_rows).to_csv(OUTPUT_DIR / '全部簇_主题分布.csv', index=False, encoding='utf-8-sig')
    config = {
        'stage': 'clustering_only', 'algorithm': 'Leiden',
        'n_papers': len(index), 'seed': SEED,
        'input_reduction_dir': str(REDUCE_DIR),
        'input_feature': str(FEATURE_PATH),
        'input_index': str(INDEX_PATH),
        'input_reduction_manifest_index_sha256': manifest['index_sha256'],
        'input_reduction_manifest_feature_sha256': manifest['feature_sha256'],
        'input_embedding_sha256': manifest['embedding_sha256'],
        'chosen_configurations': METHODS, 'graph_distance': GRAPH_DISTANCE,
        'edge_weight_formula': 'exp(-d(i,j)^2 / (sigma_i*sigma_j))',
        'leiden_objective': 'RBConfigurationVertexPartition',
        'leiden_iterations': LEIDEN_ITERATIONS,
        'reference_topics_optional': reference_source,
        'notes': [
            'Only three selected configurations; not a hyperparameter sweep.',
            'The 2D UMAP maps in 6visualize_UMAP.py are visualization-only, not clustering input.',
            'Distinct Leiden graph parameters for each reduction method mean differences are not solely attributable to reducers.',
            'Leiden does not mark noise by default; singleton clusters are counted as valid communities.',
            'Reference topic labels are exploratory, not independent ground truth.',
            'Original-space silhouette uses SPECTER2 cosine distances.',
            'Reduced embeddings are read directly from d_reduce/embeddings. No copies or symlinks are created under cluster/.',
        ],
        'software': {p: package_version(p) for p in
                     ('numpy', 'pandas', 'scipy', 'scikit-learn', 'leidenalg', 'igraph')},
    }
    (OUTPUT_DIR / '聚类配置.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n完成。输出：{OUTPUT_DIR}\n'
          '可视化读取cluster/下的指标与标签，以及d_reduce/embeddings/下的降维矩阵。', flush=True)


if __name__ == '__main__':
    main()
