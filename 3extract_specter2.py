#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正式实验：仅提取180篇论文的 SPECTER2 Proximity 语义特征。

直接运行：python 3extract_specter2.py（无命令行参数）
输入：/home/zxx/bigdata/exp1/cleaned_180_checked/papers_180.jsonl
输出：/home/zxx/bigdata/exp1/features_final/specter2_proximity.npy（180×768）
      同目录 papers_index.csv、corpus_info.json、specter2_config.json、specter2_input_report.csv

沿用之前的 SPECTER2 base + proximity adapter、标题[SEP]摘要、512 token
截断、CLS 向量L2归一化、逐篇缓存；不提取BGE、TF-IDF或LSA。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'cleaned_180_checked' / 'papers_180.jsonl'
OUTPUT = ROOT / 'features_final'
BASE_MODEL = ROOT / 'models' / 'specter2' / 'base'
PROXIMITY_MODEL = ROOT / 'models' / 'specter2' / 'adapters' / 'proximity'
EXCLUDED = {'14450716', '40011409', '8457977'}
MAX_TOKENS = 512
# 使用固定模型；实际换权重时请手动升级版本，避免误用上一版本的缓存。
MODEL_VERSION = 'specter2_base_proximity_v1'
FORCE_RECOMPUTE = False  # True：忽略逐篇缓存，重新推理全部180篇


def read_corpus():
    if not INPUT.is_file():
        raise FileNotFoundError(f'找不到论文语料：{INPUT}')
    papers = [json.loads(s) for s in INPUT.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    pmids = [str(p['pmid']).strip() for p in papers]
    if (len(papers) != 180 or len(set(pmids)) != 180 or
            set(pmids) & EXCLUDED or [p.get('row_index') for p in papers] != list(range(180))):
        raise ValueError('语料应为180篇、PMID唯一、row_index连续为0..179，且不含3篇排除论文')
    for p in papers:
        if not str(p.get('title') or '').strip():
            raise ValueError(f'{p["pmid"]}：标题缺失')
    return papers, hashlib.sha256(INPUT.read_bytes()).hexdigest()


def ensure_paper_index(papers, source_sha):
    """每次按当前语料生成论文索引，允许正常重跑。"""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    idx, info = OUTPUT / 'papers_index.csv', OUTPUT / 'corpus_info.json'
    with idx.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['row_index', 'pmid', 'title', 'has_abstract', 'quality_status'])
        writer.writeheader()
        writer.writerows({
            'row_index': i, 'pmid': str(p['pmid']), 'title': p['title'],
            'has_abstract': int(bool(str(p.get('abstract') or '').strip())),
            'quality_status': p.get('quality_status', ''),
        } for i, p in enumerate(papers))
    info.write_text(json.dumps({
        'input': str(INPUT.relative_to(ROOT)), 'source_sha256': source_sha,
        'n_papers': len(papers),
        'missing_abstract': sum(not str(p.get('abstract') or '').strip() for p in papers),
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def has_weights(root):
    if any(p.stat().st_size > 1_000_000 for p in
           list(root.glob('*.safetensors')) + list(root.glob('pytorch_model*.bin')) if p.is_file()):
        return True
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        path = root / name
        if path.is_file():
            try:
                entries = set(json.loads(path.read_text(encoding='utf-8'))['weight_map'].values())
                if entries and all((root / p).is_file() for p in entries):
                    return True
            except (KeyError, ValueError, OSError):
                pass
    return False


def model_ready(root, kind):
    if kind == 'base':
        return ((root / 'config.json').is_file() and has_weights(root) and
                ((root / 'tokenizer.json').is_file() or (root / 'vocab.txt').is_file()))
    return ((root / 'adapter_config.json').is_file() and any(
        (root / filename).is_file() for filename in ('pytorch_adapter.bin', 'adapter_model.safetensors')))


def find_model(kind):
    preferred = BASE_MODEL if kind == 'base' else PROXIMITY_MODEL
    repo_name = 'models--allenai--' + ('specter2_base' if kind == 'base' else 'specter2')
    hub_cache = Path(os.environ.get('HF_HUB_CACHE') or
                     Path(os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))) / 'hub')
    for root in (preferred, ROOT / 'models' / repo_name, hub_cache / repo_name):
        for path in [root, *sorted((root / 'snapshots').glob('*'), reverse=True)]:
            if model_ready(path, kind):
                return path.resolve()
    return None


def ensure_models():
    base, adapter = find_model('base'), find_model('adapter')
    if base and adapter:
        return base, adapter
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError('缺少SPECTER2模型且无法下载；请安装huggingface_hub') from exc
    if not base:
        print('正在下载SPECTER2基础模型……', flush=True)
        BASE_MODEL.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id='allenai/specter2_base', local_dir=str(BASE_MODEL))
    if not adapter:
        print('正在下载SPECTER2 proximity适配器……', flush=True)
        PROXIMITY_MODEL.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id='allenai/specter2', local_dir=str(PROXIMITY_MODEL))
    base, adapter = find_model('base'), find_model('adapter')
    if not base or not adapter:
        raise RuntimeError('SPECTER2模型文件不完整')
    return base, adapter


def normalize_vector(vector):
    import numpy as np
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if v.ndim != 1 or not np.isfinite(v).all() or norm < 1e-9:
        raise RuntimeError('模型输出的语义向量无效')
    return v / norm


def read_cache(pmid, text, cfg_hash):
    import numpy as np
    path = OUTPUT / 'cache_embeddings' / f'{pmid}_{cfg_hash[:12]}.npz'
    input_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if path.is_file() and not FORCE_RECOMPUTE:
        try:
            with np.load(path, allow_pickle=False) as data:
                if data['input_hash'].item() == input_hash and data['config_hash'].item() == cfg_hash:
                    return normalize_vector(data['vector']), path, input_hash, int(data['total_tokens']), int(data['used_chunks'])
        except (ValueError, OSError, KeyError, EOFError):
            pass
    return None, path, input_hash, 0, 0


def atomic_numpy_save(path, matrix):
    import numpy as np
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npy', delete=False) as f:
        tmp = Path(f.name)
    try:
        np.save(tmp, matrix)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_cache(path, vector, input_hash, cfg_hash, n_tokens):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
        tmp = Path(f.name)
    try:
        np.savez_compressed(tmp, vector=vector, input_hash=np.array(input_hash),
                            config_hash=np.array(cfg_hash), total_tokens=np.array(n_tokens), used_chunks=np.array(1))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def existing_features_complete(papers, source_sha):
    """已有的正式特征如果完整且对应当前语料，就直接复用而不加载模型。

    无 model_version 的旧版配置视为固定 SPECTER2 v1，只用于兼容现有成果。
    以后换了模型权重必须手动升级 MODEL_VERSION，才会重新计算。
    """
    import numpy as np
    try:
        names = ('papers_index.csv', 'corpus_info.json', 'specter2_proximity.npy',
                 'specter2_config.json', 'specter2_input_report.csv')
        if not all((OUTPUT / name).is_file() for name in names):
            return False
        info = json.loads((OUTPUT / 'corpus_info.json').read_text(encoding='utf-8'))
        cfg = json.loads((OUTPUT / 'specter2_config.json').read_text(encoding='utf-8'))
        version = cfg.get('model_version', 'specter2_base_proximity_v1' if cfg.get('model_stamp') else None)
        if (info.get('source_sha256') != source_sha or cfg.get('source_sha256') != source_sha or
                cfg.get('method') != 'specter2' or cfg.get('specter_max_tokens') != MAX_TOKENS or
                version != MODEL_VERSION or info.get('n_papers') != 180):
            return False
        with (OUTPUT / 'papers_index.csv').open(encoding='utf-8-sig', newline='') as f:
            index = list(csv.DictReader(f))
        with (OUTPUT / 'specter2_input_report.csv').open(encoding='utf-8-sig', newline='') as f:
            report = list(csv.DictReader(f))
        ids = [str(p['pmid']) for p in papers]
        if (len(index) != 180 or len(report) != 180 or
                [str(r['pmid']) for r in index] != ids or
                [str(r['pmid']) for r in report] != ids or
                [int(r['row_index']) for r in index] != list(range(180)) or
                [int(r['row_index']) for r in report] != list(range(180))):
            return False
        matrix = np.load(OUTPUT / 'specter2_proximity.npy', allow_pickle=False)
        return (matrix.shape == (180, 768) and np.isfinite(matrix).all() and
                np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def main():
    import numpy as np
    papers, source_sha = read_corpus()
    if not FORCE_RECOMPUTE and existing_features_complete(papers, source_sha):
        print('3：已有与当前180篇语料一致的完整特征结果，跳过模型推理。')
        return
    base, adapter = ensure_models()
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    model = AutoAdapterModel.from_pretrained(str(base), local_files_only=True)
    model.load_adapter(str(adapter), load_as='proximity', set_active=True)
    model = model.to(device).eval()

    # 按标题/摘要和模型设置缓存；缓存格式不变，原有缓存首次可能需要重建。
    settings = {
        'method': 'specter2', 'model_version': MODEL_VERSION,
        'source_sha256': source_sha, 'specter_max_tokens': MAX_TOKENS,
        'missing_abstract': 'title only; never infer from body',
        'vector_pooling': 'CLS normalized',
    }
    # 缓存签名不包含整份语料的 SHA：正文修改不应使所有标题/摘要向量失效。
    cache_settings = {k: v for k, v in settings.items() if k != 'source_sha256'}
    cfg_hash = hashlib.sha256(json.dumps(cache_settings, sort_keys=True).encode()).hexdigest()
    vectors, report = [], []
    print(f'SPECTER2 Proximity：{len(papers)}篇论文；设备：{device}', flush=True)
    sep = tokenizer.sep_token or '[SEP]'
    for i, p in enumerate(papers, start=1):
        abstract = str(p.get('abstract') or '').strip()
        text = str(p['title']).strip() + (sep + abstract if abstract else '')
        vector, cache_path, input_hash, n_tokens, _ = read_cache(str(p['pmid']), text, cfg_hash)
        status = 'cached' if vector is not None else 'computed'
        if vector is None:
            n_tokens = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
            tokenized = tokenizer(text, truncation=True, max_length=MAX_TOKENS,
                                  return_tensors='pt', return_token_type_ids=False)
            with torch.inference_mode():
                hidden = model(**{k: v.to(device) for k, v in tokenized.items()}).last_hidden_state[:, 0, :]
            vector = normalize_vector(hidden.float().cpu().numpy()[0])
            write_cache(cache_path, vector, input_hash, cfg_hash, n_tokens)
        vectors.append(vector)
        report.append({
            'row_index': i - 1, 'pmid': str(p['pmid']), 'has_abstract': int(bool(abstract)),
            'status': status, 'input_sha256': input_hash, 'input_tokens': n_tokens,
            'chunks_used': 1, 'was_truncated_or_sampled': int(n_tokens > MAX_TOKENS),
        })
        if i % 10 == 0 or i == len(papers):
            print(f'  {i}/{len(papers)}', flush=True)
    matrix = np.stack(vectors).astype('float32')
    if (matrix.shape != (180, 768) or not np.isfinite(matrix).all() or
            not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4)):
        raise RuntimeError(f'SPECTER2输出矩阵无效：{matrix.shape}')
    ensure_paper_index(papers, source_sha)
    atomic_numpy_save(OUTPUT / 'specter2_proximity.npy', matrix)
    settings['shape'] = list(matrix.shape)
    (OUTPUT / 'specter2_config.json').write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (OUTPUT / 'specter2_input_report.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print('完成：', OUTPUT / 'specter2_proximity.npy')


if __name__ == '__main__':
    main()
