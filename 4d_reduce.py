#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正式实验阶段①：SPECTER2 -> PCA-20 / UMAP-10 / LocalMAP-10。

只负责降维，绝不聚类、也不绘图；直接运行，无命令行参数。
结果：/home/zxx/bigdata/exp1/d_reduce/embeddings/*.npy
依赖：numpy pandas scipy scikit-learn umap-learn pacmap

沿用前期已选配置。注意：LocalMAP 的内部随机性可能依赖 pacmap 版本，
重新计算的坐标不保证与过去 Sweep 的缓存逐元素相同。后续 Leiden 必须读取本次保存的坐标。
"""
from __future__ import annotations

import hashlib
import json
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# ======================== 固定路径和正式实验参数 ========================
ROOT = Path('/home/zxx/bigdata/exp1')
FEATURE_PATH = ROOT / 'features_final' / 'specter2_proximity.npy'
INDEX_PATH = ROOT / 'features_final' / 'papers_index.csv'
OUTPUT_DIR = ROOT / 'd_reduce'
EMBED_DIR = OUTPUT_DIR / 'embeddings'

SEED = 42
SPECS = {
    'PCA': {'spec_id': 'PCA_20d', 'dim': 20},
    'UMAP': {'spec_id': 'UMAP_10d_n20_md0p00', 'dim': 10, 'n_neighbors': 20, 'min_dist': 0.0},
    'LocalMAP': {'spec_id': 'LocalMAP_10d_n10', 'dim': 10, 'n_neighbors': 10},
}
KNN_K = (5, 10)
# =======================================================================


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return 'not installed'


def load_inputs():
    for path, label in ((FEATURE_PATH, 'SPECTER2特征'), (INDEX_PATH, '论文索引')):
        if not path.is_file():
            raise FileNotFoundError(f'缺少{label}：{path}')
    index = pd.read_csv(INDEX_PATH, encoding='utf-8-sig', dtype={'pmid': str}, keep_default_na=False)
    for col in ('row_index', 'pmid', 'title'):
        if col not in index.columns:
            raise ValueError(f'论文索引缺少{col}列')
    n = len(index)
    if n != 180 or index['pmid'].duplicated().any():
        raise ValueError('本次固定180篇论文，索引行数或PMID重复异常')
    order = pd.to_numeric(index['row_index'], errors='raise').to_numpy(dtype=int)
    if not np.array_equal(order, np.arange(n)):
        raise ValueError('row_index必须按0..179排序，不允许凭标题猜测向量行序')
    x = np.load(FEATURE_PATH, allow_pickle=False)
    if x.shape != (n, 768) or not np.isfinite(x).all():
        raise ValueError(f'SPECTER2特征应为(180,768)且数值有限，实际{x.shape}')
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms < 1e-10):
        raise ValueError('存在零向量，无法归一化')
    x = x / norms[:, None]
    return x, index


def get_reducers():
    try:
        import umap
        from pacmap import LocalMAP
    except (ImportError, AttributeError) as exc:
        raise ImportError('请在当前Python环境安装 umap-learn、pacmap（需提供LocalMAP）') from exc
    return umap, LocalMAP


def fit_one(x, method, umap_module, LocalMAP):
    """精确沿用过去SPECTER2 Sweep的主要参数，且每种方法拟合一次。"""
    np.random.seed(SEED)
    cfg = SPECS[method]
    explained = np.nan
    if method == 'PCA':
        model = PCA(n_components=cfg['dim'], svd_solver='full')
        z = model.fit_transform(x)
        explained = float(model.explained_variance_ratio_.sum())
    elif method == 'UMAP':
        model = umap_module.UMAP(
            n_components=cfg['dim'], n_neighbors=cfg['n_neighbors'],
            min_dist=cfg['min_dist'], metric='cosine',
            random_state=SEED, n_jobs=1,
        )
        z = model.fit_transform(x)
    elif method == 'LocalMAP':
        model = LocalMAP(
            n_components=cfg['dim'], n_neighbors=cfg['n_neighbors'],
            MN_ratio=0.5, FP_ratio=2.0, apply_pca=False, verbose=False,
        )
        z = model.fit_transform(x.copy(), init='pca')
    else:
        raise ValueError(method)
    z = np.asarray(z, dtype=np.float32)
    if z.shape != (len(x), cfg['dim']) or not np.isfinite(z).all():
        raise ValueError(f'{method}降维结果异常：{z.shape}')
    return z, explained


def nearest_index(d, k):
    a = np.array(d, copy=True)
    np.fill_diagonal(a, np.inf)
    return np.argsort(a, axis=1, kind='stable')[:, :k]


def evaluate_reduction(x, z, original_knn):
    d = euclidean_distances(z)
    scores = {}
    for k in KNN_K:
        neighbor = nearest_index(d, k)
        overlap = [len(set(original_knn[k][i]).intersection(neighbor[i])) / k
                   for i in range(len(x))]
        scores[f'knn_overlap_{k}'] = float(np.mean(overlap))
        scores[f'trustworthiness_{k}'] = float(
            trustworthiness(x, z, n_neighbors=k, metric='cosine'))
    return scores


def main():
    x, index = load_inputs()
    umap_module, LocalMAP = get_reducers()
    original_d = cosine_distances(x)
    original_knn = {k: nearest_index(original_d, k) for k in KNN_K}
    result = []
    outputs = {}
    print(f'SPECTER2：{x.shape[0]}篇论文 × {x.shape[1]}维；开始三种正式降维。', flush=True)
    for method, cfg in SPECS.items():
        start = time.perf_counter()
        z, explained = fit_one(x, method, umap_module, LocalMAP)
        sec = time.perf_counter() - start
        record = {'method': method, 'spec_id': cfg['spec_id'], 'dim': cfg['dim'],
                  'seed': SEED, 'reduction_seconds': round(sec, 3),
                  'explained_variance_ratio': explained,
                  **evaluate_reduction(x, z, original_knn)}
        result.append(record)
        outputs[method] = z
        print(f'  {method:8s} {z.shape[1]:2d}维 | '
              f'Trustworthiness@10={record["trustworthiness_10"]:.4f} | '
              f'{sec:.1f}s', flush=True)

    # 全部拟合并验证后才将本次结果写入正式目录。
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for method, z in outputs.items():
        path = EMBED_DIR / f'{SPECS[method]["spec_id"]}.npy'
        np.save(path, z)
        file_hashes[SPECS[method]['spec_id']] = sha256_file(path)
    pd.DataFrame(result).to_csv(OUTPUT_DIR / '降维指标.csv', index=False, encoding='utf-8-sig')
    info = {
        'stage': 'dimensionality_reduction_only', 'n_papers': len(index),
        'original_dim': x.shape[1], 'seed': SEED,
        'input_feature': str(FEATURE_PATH), 'input_index': str(INDEX_PATH),
        'feature_sha256': sha256_file(FEATURE_PATH),
        'index_sha256': sha256_file(INDEX_PATH),
        'specs': SPECS, 'embedding_sha256': file_hashes,
        'method_notes': {
            'PCA': 'full SVD, no feature re-normalization after reduction',
            'UMAP': 'cosine input metric; random_state=42; n_jobs=1',
            'LocalMAP': 'MN_ratio=0.5, FP_ratio=2.0, apply_pca=False, init=pca; internal RNG may vary by version',
        },
        'software': {p: package_version(p) for p in
                     ('numpy', 'pandas', 'scikit-learn', 'umap-learn', 'pacmap')},
    }
    (OUTPUT_DIR / '降维配置.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n完成。输出：{OUTPUT_DIR}；聚类脚本将直接读取这里的3份npy。', flush=True)


if __name__ == '__main__':
    main()
