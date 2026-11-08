#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正式实验阶段③：读取d_reduce中的降维矩阵 + cluster中的Leiden标签，独立绘图。

输入：/home/zxx/bigdata/exp1/d_reduce/embeddings/*.npy
     /home/zxx/bigdata/exp1/cluster/sweep_all_metrics.csv
     /home/zxx/bigdata/exp1/cluster/all_cluster_labels.csv
输出：/home/zxx/bigdata/exp1/visualizations_UMAP/（所有图片直接保存，不再嵌套实验目录）

仅使用UMAP将已保存的10/20维结果二次投影到2维用于展示，不重新降维或聚类。
保留半透明完整包络、主题术语、时间趋势、热力图和研究机构图。
"""
from __future__ import annotations

import math
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from scipy.spatial import Delaunay, QhullError, ConvexHull, distance
from scipy.sparse.csgraph import minimum_spanning_tree
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from umap import UMAP

# ===== 用户提供的真实路径：直接运行，不需要命令行参数 =====
ROOT = Path(__file__).resolve().parent
CLUSTER_DIR = ROOT / 'cluster'
REDUCE_DIR = ROOT / 'd_reduce'
EMBED_DIR = REDUCE_DIR / 'embeddings'
FEATURE_NAME = 'SPECTER2'
OUTPUT_ROOT = ROOT / 'visualizations_UMAP'
INDEX_FILE = ROOT / 'features_final/papers_index.csv'
METADATA_FILE = ROOT / 'extracted' / 'papers_metadata_reviewed.csv'

# 从固定的正式实验目录读取指标/标签，三种降维坐标只从d_reduce读取。
# 通过每组结果的method/spec_id/dim/row_id字段识别可展示的算法和配置。
# 仅用于显示的少量簇不影响聚类本身，也不限制可发现的簇总数。
DISPLAY_NAMES = {
    'tfidf_A': 'TF-IDF A（正文）',
    'tfidf_B': 'TF-IDF B（标题与摘要）',
    'tfidf_C': 'TF-IDF C（加权组合）',
    'lsa_50': 'LSA-50',
    'bge_core': 'BGE-M3 Core',
    'bge_full': 'BGE-M3 Full',
    'specter2': 'SPECTER2',
}
# 以上只是可选的中文显示别名，未列出的新特征同样会被自动发现、绘制。


# 只用每组UMAP结果描述主题和时间；另外两种方法仅参加图1的对照。
TOP_GROUPS = 8
RECENT_START = 2022
LAST_YEAR = 2026
DPI = 220

# 只控制散点图的视觉包络，不改变任何聚类标签或降维坐标。
# Alpha Shape 为凹包：沿局部密集的同簇点形成不规则闭合区域。
HULL_MIN_POINTS = 4                 # 少于4个不同位置的点不画面状包络
HULL_RADIUS_FACTOR = 2.4            # 最大三角形外接圆半径，相对簇内最近邻中位距离
HULL_FILL_ALPHA = 0.12              # 半透明填充，避免掩盖散点
HULL_LINE_ALPHA = 0.72              # 细边线
HULL_LOOSE = False                  # False=严格：圈入任意其他簇/噪声点就不画；True=宽松：允许圈入，仍要求完整包围本簇
# v4: 在显示像素中计算边界，每簇最多一个完整区域；不再拆成局部包络。
# 所有簇的可视化名称统一使用“簇0”“簇1”等编号，不自动判断研究主题。
ENVELOPE_MIN_PAPERS = 4            # 至少4个点才形成一个完整包络
ENVELOPE_PADDING_PX = 5.8          # 圆角包络外扩宽度（基于画面像素，非数据单位）
ENVELOPE_MIN_PIXELS = 2.5          # 临近其他簇时仍尝试较窄的轮廓

# 中文机构名来自指定的英文原始机构名称（仅明示匹配的机构进入排名，不臆造未识别名称）。
# 加州大学系统包含多校区，明确作为系统而不是一所大学统计。
INSTITUTION_PATTERNS = {
    '加州大学系统（多校区）': [r'\bUniversity of California\b', r'\bUCLA\b', r'\bUC Berkeley\b', r'\bUCSF\b'],
    '鲁汶大学': [r'\bKU Leuven\b', r'\bKatholieke Universiteit Leuven\b'],
    '佐治亚大学': [r'\bUniversity of Georgia\b'],
    '多伦多大学': [r'\bUniversity of Toronto\b'],
    '科隆大学': [r'\bUniversity of Cologne\b'],
    '弗吉尼亚大学': [r'\bUniversity of Virginia\b'],
    '阿尔伯塔大学': [r'\bUniversity of Alberta\b'],
    '华盛顿大学': [r'\bUniversity of Washington\b'],
    '剑桥大学': [r'\bUniversity of Cambridge\b'],
    '牛津大学': [r'\bUniversity of Oxford\b'],
    '佛罗里达大学': [r'\bUniversity of Florida\b'],
    '苏黎世大学': [r'\bUniversity of Zurich\b'],
    '匹兹堡大学': [r'\bUniversity of Pittsburgh\b'],
    '哥本哈根大学': [r'\bUniversity of Copenhagen\b'],
    '安特卫普大学': [r'\bUniversity of Antwerp\b'],
    '筑波大学': [r'\bUniversity of Tsukuba\b'],
    '比勒陀利亚大学': [r'\bUniversity of Pretoria\b'],
    '密歇根大学': [r'\bUniversity of Michigan\b'],
    '马里兰大学': [r'\bUniversity of Maryland\b'],
    '得克萨斯大学系统': [r'\bUniversity of Texas\b'],
    '威斯康星大学': [r'\bUniversity of Wisconsin\b'],
    '利兹大学': [r'\bUniversity of Leeds\b'],
    '哈佛大学': [r'\bHarvard University\b'],
    '西北大学': [r'\bNorthwestern University\b'],
    '普渡大学': [r'\bPurdue University\b'],
    '首尔大学': [r'\bSeoul National University\b'],
    '根特大学': [r'\bGhent University\b'],
    '康奈尔大学': [r'\bCornell University\b'],
    '哥伦比亚大学': [r'\bColumbia University\b'],
    '北卡罗来纳州立大学': [r'\bNorth Carolina State University\b'],
    '埼玉医科大学': [r'\bSaitama Medical University\b'],
    '湖南师范大学': [r'\bHunan Normal University\b'],
    '高丽大学': [r'\bKorea University\b'],
    '波士顿大学': [r'\bBoston University\b'],
    '约翰斯·霍普金斯大学': [r'\bJohns Hopkins University\b'],
    '宾夕法尼亚大学': [r'\bUniversity of Pennsylvania\b'],
    '莫纳什大学': [r'\bMonash University\b'],
    '乌普萨拉大学': [r'\bUppsala University\b'],
    '里尔大学': [r'\bUniversit[eé] de Lille\b', r'\bUniversity of Lille\b'],
    '维也纳大学': [r'(?<!Medical )\bUniversity of Vienna\b'],
    '维也纳医科大学': [r'\bMedical University (?:of )?Vienna\b'],
    '中国科学院': [r'\bChinese Academy of Sciences\b'],
    '美国国立卫生研究院': [r'\bNational Institutes of Health\b'],
    '马克斯·普朗克学会': [r'\bMax[ -]Planck\b'],
    '霍华德·休斯医学研究所': [r'\bHoward Hughes Medical Institute\b'],
    '日本理化学研究所': [r'\bRIKEN\b'],
    '日本国立环境研究所': [r'\bNational Institute for Environmental Studies\b'],
    '欧洲食品安全局': [r'\bEuropean Food Safety Authority\b', r'\bEFSA\b'],
    '伦敦大学学院': [r'\bUniversity College London\b', r'\bUCL\b'],
    '麦吉尔大学': [r'\bMcGill University\b'],
    '马希隆大学': [r'\bMahidol University\b'],
    '开普敦大学': [r'\bUniversity of Cape Town\b'],
    '南京医科大学': [r'\bNanjing Medical University\b'],
    '伊利诺伊大学': [r'\bUniversity of Illinois\b'],
    '中国医科大学': [r'\bChina Medical University\b'],
    '伊法卡拉卫生研究所': [r'\bIfakara Health Institute\b'],
    '芝加哥大学': [r'\bUniversity of Chicago\b'],
    '巴西利卡塔大学': [r'\bUniversity of Basilicata\b'],
    '亚当·密茨凯维奇大学': [r'\bAdam Mickiewicz University\b'],
    '南京大学': [r'\bNanjing University\b'],
    '乌尔米耶大学': [r'\bUrmia University\b'],
    '杜兰大学': [r'\bTulane University\b'],
    '帕维亚大学': [r'\bUniversity of Pavia\b'],
    '乔治·梅森大学': [r'\bGeorge Mason University\b'],
    '洛桑大学': [r'\bUniversity of Lausanne\b'],
    '奥卢大学': [r'\bUniversity of Oulu\b'],
    '开罗大学': [r'\bCairo University\b'],
    '图尔库大学': [r'\bUniversity of Turku\b'],
    '南达科他大学': [r'\bUniversity of South Dakota\b'],
    '南加州大学': [r'\bUniversity of Southern California\b'],
    '那不勒斯费德里科二世大学': [r'\bUniversity of Naples Federico II\b'],
    '罗兹大学': [r'\bUniversity of Lodz\b'],
    '贝尼苏韦夫大学': [r'\bBeni-Suef University\b'],
    '东北大学（日本）': [r'\bTohoku University\b'],
    '密歇根州立大学': [r'\bMichigan State University\b'],
    '耶鲁大学': [r'\bYale University\b'],
    '岩手医科大学': [r'\bIwate Medical University\b'],
    '南开大学': [r'\bNankai University\b'],
    '厦门大学': [r'\bXiamen University\b'],
    '宾夕法尼亚州立大学': [r'\bPenn State University\b'],
    '巴黎萨克雷大学': [r'\bUniversit[eé] Paris-Saclay\b', r'\bParis-Saclay University\b'],
}

# 用标题/关键词中明确出现的术语提取中文主题线索。小样本、混合簇不强行命名。
# 这里的匹配不是人工主题标签，也不会改变聚类标签或论文归属。
TOPIC_RULES = [
    ('哮喘与临床管理', [r'\basthma\b', r'bronchial', r'\bwheez', r'\bpediatric asthma\b']),
    ('过敏原与免疫反应', [r'\ballergen', r'\ballergic\b', r'\ballergy\b', r'sensitiz', r'hypersensitiv', r'immunoglobulin e']),
    ('住房、环境与健康', [r'\bhousing\b', r'\bindoor\b', r'residential', r'\bhome\b', r'household', r'public health', r'\bair quality\b']),
    ('昆虫杀虫剂抗性', [r'insecticide', r'pyrethroid', r'resistan', r'knockdown', r'\bp450\b', r'acetylcholinesterase']),
    ('昆虫感受与神经机制', [r'neuronal', r'\bneural\b', r'\bneuron', r'olfact', r'gustat', r'antenna', r'ion channel', r'\bsodium channel']),
    ('昆虫激素与生殖', [r'juvenile hormone', r'ecdys', r'reproduct', r'\bovary\b', r'\bovaries\b', r'fertility', r'hormonal']),
    ('昆虫微生物与共生', [r'microbi', r'symbio', r'endosymbio', r'gut bacteria', r'\bbacillus\b', r'\bbacterial\b']),
    ('土壤生态与生物多样性', [r'\bsoil\b', r'\blitter\b', r'\bforest\b', r'biodivers', r'arthropod abundance', r'decompos']),
    ('昆虫行为与生态', [r'\binsect', r'\bdrosophila\b', r'\bmosquito', r'\bcockroach', r'\bmoth\b', r'\bbee\b', r'\bant\b', r'\bflies\b']),
    ('分子与蛋白质机制', [r'\bprotein', r'\bgenetic', r'\bgenom', r'\bmutation', r'\breceptor', r'\benzyme', r'\bgenes?\b']),
]
# 用于附在主题图旁的中文关键词，均需先在所属文献出现才会显示。
KEYWORD_RULES = [
    ('哮喘', r'\basthma\b'), ('儿童', r'\bchild|\bpediatric|\binfant'),
    ('治疗', r'therap|treat|management|control'), ('过敏原', r'allergen|allerg'),
    ('致敏', r'sensitiz|hypersensit'), ('免疫', r'immun|inflamm'),
    ('住房', r'hous|residen'), ('环境', r'\benvironment|\bindoor|\bair pollution'),
    ('杀虫剂', r'insecticid|pyrethroid'), ('抗性', r'resistan|knockdown'),
    ('蚊虫', r'\bmosquito|\baedes\b'), ('昆虫', r'\binsect|\bdrosophila|\bmoth'),
    ('神经', r'neur|\bbrain\b'), ('离子通道', r'\bchannel'),
    ('激素', r'hormon|ecdyst|juvenile'), ('生殖', r'reproduct|fertil|ovari'),
    ('微生物', r'microbi|bacteri'), ('共生', r'symbios'),
    ('土壤', r'\bsoil\b'), ('生态', r'ecolog|biodivers|\bforest\b'),
    ('基因', r'\bgenom|\bgenetic|\bgenes?\b'), ('蛋白质', r'\bprotein'),
]

# 一致、清爽的颜色：图2的八个群组颜色与图3对应，且在图1的UMAP面板复用。
TOP_COLORS = ['#2786A5', '#E58A47', '#678E68', '#9370A5',
              '#CD6A78', '#53A6A2', '#B99A43', '#6D83C0']
GRAY = '#8E939C'
INK = '#283743'
MUTED = '#61717D'


def set_chinese_style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ['Noto Sans CJK SC', 'Source Han Sans SC', 'Microsoft YaHei',
                  'WenQuanYi Micro Hei', 'SimHei', 'PingFang SC', 'Noto Sans CJK JP']
    choice = next((s for s in candidates if s in installed), None)
    if choice is None:
        raise RuntimeError('系统未找到中文字体。请安装 Noto Sans CJK SC（Linux可安装fonts-noto-cjk）后再绘图。')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': [choice, 'DejaVu Sans'],
        'axes.unicode_minus': False, 'figure.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#BAC5CD', 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'savefig.facecolor': 'white', 'font.size': 10,
    })
    print(f'中文字体：{choice}')


def assert_file(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError(f'缺少{label}：{path}（请核对已有Sweep输出）')


def load_common():
    assert_file(INDEX_FILE, '论文索引')
    assert_file(METADATA_FILE, '核对后的论文元数据')
    idx = pd.read_csv(INDEX_FILE, encoding='utf-8-sig', dtype={'pmid': str}, keep_default_na=False)
    md = pd.read_csv(METADATA_FILE, encoding='utf-8-sig', dtype={'pmid': str}, keep_default_na=False)
    need_idx = {'pmid', 'row_index', 'title'}
    need_md = {'pmid', 'year', 'institutions', 'title'}
    if not need_idx.issubset(idx.columns) or not need_md.issubset(md.columns):
        raise ValueError('论文索引或元数据缺少必要列，请检查源CSV字段')
    if idx.pmid.duplicated().any() or md.pmid.duplicated().any():
        raise ValueError('源CSV存在重复PMID；不能按行号猜测匹配')
    if not np.array_equal(idx.row_index.astype(int).to_numpy(), np.arange(len(idx))):
        raise ValueError('论文索引row_index不连续，停止绘图')
    full = idx.merge(md, on='pmid', how='left', validate='one_to_one', indicator=True, suffixes=('', '_metadata'))
    missing = full._merge.ne('both')
    print(f'论文索引{len(idx)}条，元数据{len(md)}条，按PMID成功关联{len(full)-missing.sum()}条。')
    if missing.any():
        raise ValueError(f'缺少这几篇的元数据：{full.loc[missing,"pmid"].tolist()}')
    full['year'] = pd.to_numeric(full.year, errors='coerce')
    full.loc[~full.year.between(1900, LAST_YEAR), 'year'] = np.nan
    full.drop(columns='_merge', inplace=True)
    return full


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def verify_reduction_source():
    """防止降维文件重算后，拿旧Leiden标签配新坐标造成错图。"""
    cluster_cfg = CLUSTER_DIR / '聚类配置.json'
    reduction_cfg = REDUCE_DIR / '降维配置.json'
    assert_file(cluster_cfg, '聚类配置')
    assert_file(reduction_cfg, '降维配置')
    c = json.loads(cluster_cfg.read_text(encoding='utf-8'))
    r = json.loads(reduction_cfg.read_text(encoding='utf-8'))
    if c.get('input_reduction_manifest_index_sha256') != r.get('index_sha256') or \
            c.get('input_reduction_manifest_feature_sha256') != r.get('feature_sha256'):
        raise ValueError('聚类结果与降维记录的输入版本不一致；请重新运行5cluster_leiden.py。')
    hash_in_cluster = c.get('input_embedding_sha256', {})
    hash_in_reduce = r.get('embedding_sha256', {})
    if not hash_in_cluster or hash_in_cluster != hash_in_reduce:
        raise ValueError('聚类配置与降维矩阵版本不一致；请重新运行5cluster_leiden.py。')
    for spec, expected_hash in hash_in_cluster.items():
        path = EMBED_DIR / (spec + '.npy')
        assert_file(path, '降维矩阵')
        if file_sha256(path) != expected_hash:
            raise ValueError(f'降维矩阵已改变：{path}；请重新运行5cluster_leiden.py。')


def _maybe_float(v, fallback=0.0):
    try:
        x = float(v)
        return x if np.isfinite(x) else fallback
    except (ValueError, TypeError):
        return fallback


def _candidate_priority(row):
    """固定且透明的展示配置偏好，不声称自动选出的就是主题最优聚类。

    优先：20维、leaf、小簇阈值3/5、min_samples 2；UMAP优先不经SVD的输入，
    同类配置再优先较高的邻域保真度。若数据不含目标参数，则自然退化到实际可用行。
    """
    method = str(row['method'])
    spec = str(row['spec_id']).lower()
    sparse = ('sparse' in spec or str(row.get('input_mode', '')).lower() in ('sparse', 'svd100'))
    target_mcs = 5 if method == 'LocalMAP' and not sparse else 3
    target_n = 5 if method == 'LocalMAP' else (20 if sparse else 10)
    target_md = 0.15 if sparse else 0.0
    dim = _maybe_float(row.get('dim'), 20)
    cost = (
        abs(dim - 20) / 10
        + abs(_maybe_float(row.get('min_cluster_size'), target_mcs) - target_mcs) * 0.30
        + abs(_maybe_float(row.get('min_samples'), 2) - 2) * 0.25
        + (0 if str(row.get('selection', '')).lower() == 'leaf' else 0.4)
        + abs(_maybe_float(row.get('neighbors'), target_n) - target_n) * 0.035
    )
    if method == 'UMAP':
        cost += (0.0 if ('svd' not in spec and 'pre' not in spec) else 1.5)
        cost += abs(_maybe_float(row.get('min_dist'), target_md) - target_md) * 1.5
    # 稳定地选择用户已经计算好的表示，避免使用“最好看”的低维Silhouette来选行。
    return (round(cost, 6), -_maybe_float(row.get('trustworthiness_10'), 0), str(row['row_id']))


def _method_order(method):
    priority = {'PCA': 0, 'SVD': 0, 'UMAP': 1, 'LocalMAP': 2}
    return priority.get(method, 3), method


def load_sweep(parent: Path, meta: pd.DataFrame):
    """只读取已有的真实Sweep记录；每种method自动选一条可用配置。"""
    feature = FEATURE_NAME
    m = pd.read_csv(parent / 'sweep_all_metrics.csv', encoding='utf-8-sig')
    l = pd.read_csv(parent / 'all_cluster_labels.csv', encoding='utf-8-sig', dtype={'pmid': str})
    needed_m = {'row_id','spec_id','method','dim','n_clusters','n_noise'}
    needed_l = {'row_index','pmid'}
    if not needed_m.issubset(m.columns) or not needed_l.issubset(l.columns) or m.empty:
        raise ValueError(f'{feature}结果格式不符合Sweep约定，缺少关键列或指标表为空')
    if m.row_id.astype(str).duplicated().any():
        raise ValueError(f'{feature}指标表含重复row_id')
    if len(l) != len(meta) or l.pmid.astype(str).tolist() != meta.pmid.astype(str).tolist() or \
       not np.array_equal(pd.to_numeric(l.row_index, errors='coerce').to_numpy(), np.arange(len(meta))):
        raise ValueError(f'{feature}聚类标签与论文索引的PMID、行顺序不一致，不能画图')
    candidates = {}
    warnings_list = []
    for _, row in m.iterrows():
        method = str(row['method']).strip()
        rowid = str(row['row_id'])
        if not method or rowid not in l.columns:
            continue
        path = EMBED_DIR / f'{row["spec_id"]}.npy'
        if not path.is_file():
            continue
        candidates.setdefault(method, []).append((row, path))
    runs, choices = {}, []
    for method in sorted(candidates, key=_method_order):
        group = sorted(candidates[method], key=lambda item: (
            -_maybe_float(item[0].get('preferred_for_visualization', 0), 0),
            _candidate_priority(item[0])))
        for row, path in group:
            try:
                z = np.load(path, allow_pickle=False)
                y = pd.to_numeric(l[str(row['row_id'])], errors='raise').to_numpy(dtype=int)
                if (z.ndim != 2 or z.shape != (len(meta), int(row['dim'])) or
                        z.shape[1] < 2 or not np.isfinite(z).all()):
                    raise ValueError(f'降维矩阵形状/数值异常：{path.name}')
                if len(set(y[y >= 0])) != int(row['n_clusters']) or int((y < 0).sum()) != int(row['n_noise']):
                    raise ValueError('标签数量和指标表不一致')
                runs[method] = (row, y, z)
                choices.append({'method': method, 'row_id': str(row['row_id']),
                                'spec_id': str(row['spec_id']), 'dim': int(row['dim']),
                                'n_clusters': int(row['n_clusters']), 'n_noise': int(row['n_noise'])})
                break
            except (OSError, ValueError, KeyError, TypeError) as e:
                warnings_list.append(f'{method}/{row["row_id"]}: {e}')
    if not runs:
        raise ValueError(f'{feature}未发现可用于绘图的完整配置；请检查embeddings和标签列')
    return runs, choices, warnings_list


def choose_topic_method(runs):
    """主题/时间图优先用UMAP，有缺失时回退到实际已有的降维方法。"""
    for name in ('UMAP', 'LocalMAP', 'PCA', 'SVD'):
        if name in runs:
            return name
    return next(iter(runs))

def get_major_clusters(y: np.ndarray):
    sizes = pd.Series(y[y >= 0]).value_counts()
    return [(int(cid), int(n)) for cid, n in sizes.head(TOP_GROUPS).items()]


def infer_topics(meta: pd.DataFrame, y: np.ndarray):
    text = (meta['title'].fillna('').astype(str) + ' | ' +
            meta['keywords'].fillna('').astype(str) if 'keywords' in meta.columns
            else meta['title'].fillna('').astype(str)).str.lower().tolist()
    items = []
    for order, (cid, n) in enumerate(get_major_clusters(y)):
        paper_texts = [t for t, lab in zip(text, y) if lab == cid]
        # 可视化只使用客观簇编号，不自动判定或展示中文主题名称。
        title = f'簇{cid}'
        hits = Counter()
        for term, pattern in KEYWORD_RULES:
            cnt = sum(bool(re.search(pattern, t, re.I)) for t in paper_texts)
            if cnt >= 2:
                hits[term] = cnt
        words = [word for word, _ in hits.most_common(3)]
        new_n = int((meta.loc[y == cid, 'year'].between(RECENT_START, LAST_YEAR)).sum())
        items.append({'cluster_id': cid, 'cluster_n': n, 'topic': title,
                      'keywords': '、'.join(words) if words else '共同术语不足',
                      'recent_n': new_n, 'color': TOP_COLORS[order]})
    return items


def save(fig, file: Path):
    file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'已生成：{file}')


def _concave_regions(points: np.ndarray, all_xy: np.ndarray, labels: np.ndarray,
                     cluster_id: int, radius_factor: float = HULL_RADIUS_FACTOR) -> list[np.ndarray]:
    """用 Delaunay 的 alpha complex 提取不规则边界；不依赖额外的 alphashape/shapely 包。

    只连接相对紧密的点；相距很远的片段不会被强行连成一个区域。
    严格模式下若区域包含其他类别或噪声点则跳过；宽松模式允许重叠。
    注意：这些是二维展示包络，不是20维HDBSCAN的真实决策边界。
    """
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(pts) < HULL_MIN_POINTS or not np.isfinite(pts).all():
        return []
    try:
        tri = Delaunay(pts)
    except (QhullError, ValueError):
        return []  # 点近似共线或重叠，不能安全地创建二维多边形
    d = distance.cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    typical_nn = float(np.median(d.min(axis=1)))
    if typical_nn <= 1e-12 or not np.isfinite(typical_nn):
        return []
    max_radius = radius_factor * typical_nn
    kept = []
    for face in tri.simplices:
        a, b, c = pts[face]
        ab, bc, ca = np.linalg.norm(a-b), np.linalg.norm(b-c), np.linalg.norm(c-a)
        area2 = abs(float((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])))  # 两倍面积
        if area2 <= 1e-12 * max(1.0, ab*bc, bc*ca, ca*ab):
            continue
        radius = (ab*bc*ca)/(2*area2)
        if radius <= max_radius:
            kept.append(face)
    if not kept:
        return []

    # 被保留三角形的边：只出现一次的边是 alpha 复形的外轮廓。
    edge_counts = Counter()
    for face in kept:
        for i, j in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_counts[tuple(sorted((int(i), int(j))))] += 1
    boundary = [edge for edge, count in edge_counts.items() if count == 1]
    if not boundary:
        return []
    graph = {}
    for a, b in boundary:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    # 边界交叉、接触或不封闭时跳过不可靠的连通块，绝不补线跨越空白。
    remaining = set(boundary)
    regions = []
    while remaining:
        a, b = min(remaining)
        poly_idx = [a]
        prev, curr = a, b
        valid = True
        for _ in range(len(boundary)+1):
            poly_idx.append(curr)
            remaining.discard(tuple(sorted((prev, curr))))
            if curr == a:
                break
            neigh = graph.get(curr, [])
            if len(neigh) != 2:
                valid = False
                break
            nxt = neigh[0] if neigh[1] == prev else neigh[1] if neigh[0] == prev else None
            if nxt is None or (nxt in poly_idx and nxt != a):
                valid = False
                break
            prev, curr = curr, nxt
        else:
            valid = False
        if not valid or poly_idx[-1] != a or len(set(poly_idx[:-1])) < HULL_MIN_POINTS:
            continue
        polygon = pts[poly_idx[:-1]]
        if not np.isfinite(polygon).all():
            continue
        # 只有严格模式才检查区域是否圈入其他簇/噪声论文。
        if not HULL_LOOSE:
            other = np.asarray(all_xy)[np.asarray(labels) != cluster_id]
            if len(other) and MplPath(polygon).contains_points(other, radius=1e-10).any():
                continue
        regions.append(polygon)
    return regions



def _polygon_covered(poly: np.ndarray, pts: np.ndarray) -> int:
    """计算多边形内本簇论文数；仅用于绘图选择，不作为聚类评价。"""
    if len(poly) < 3:
        return 0
    return int(MplPath(poly).contains_points(pts, radius=1e-5).sum())


def _polygon_valid(poly: np.ndarray, foreign: np.ndarray) -> bool:
    """严格模式排除异类点；宽松模式只检查多边形本身是否合法。"""
    if len(poly) < 3 or not np.isfinite(poly).all():
        return False
    if not HULL_LOOSE and len(foreign) and MplPath(poly).contains_points(foreign, radius=1e-6).any():
        return False
    return True


def _rounded_hull(pts: np.ndarray, pad: float) -> np.ndarray:
    """在全部同簇点外侧生成一个圆角凸包；共线或细长点群也可绘制。"""
    theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    cloud = (pts[:, None, :] + pad * circle[None, :, :]).reshape(-1, 2)
    try:
        vertices = ConvexHull(cloud).vertices
    except (QhullError, ValueError):
        return np.empty((0, 2), dtype=float)
    return cloud[vertices]


def _polygon_covers_all(poly: np.ndarray, pts: np.ndarray) -> bool:
    """单个包络必须包含整个簇，包括贴在轮廓上的点。"""
    if len(poly) < 3:
        return False
    return bool(MplPath(poly).contains_points(pts, radius=0.8).all())


def _cluster_regions(points: np.ndarray, all_xy: np.ndarray, labels: np.ndarray,
                     cluster_id: int) -> tuple[list[np.ndarray], str]:
    """每簇至多一个完整的包络，绝不把同色论文拆成多个填色岛屿。

    坐标在显示像素空间，统一对三个子图使用同一套屏幕尺度的绘制规则。
    同簇论文形成离散岛屿时不强行跨越空白；严格模式还排除圈入异类点的包络。
    宽松模式只放宽异类点检查，仍必须用一个完整区域包住本簇全部论文。
    不允许选择性地只包围某簇的一部分论文，更不移动或删除任何散点。
    """
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) < ENVELOPE_MIN_PAPERS or not np.isfinite(pts).all():
        return [], 'none'
    foreign = np.asarray(all_xy, dtype=float)[np.asarray(labels) != cluster_id]

    # 先判定整簇在画面上是否连续：一个孤立片段不能靠长线跨越空白接入主簇。
    d = distance.cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    near = np.min(d, axis=1)
    finite_nn = near[np.isfinite(near)]
    if not len(finite_nn):
        return [], 'none'
    typical = float(np.median(finite_nn))
    if typical <= 1e-8:  # 重复点太多时无可靠空间尺度
        return [], 'none'
    tree_d = d.copy()
    np.fill_diagonal(tree_d, 0)
    tree = minimum_spanning_tree(tree_d).tocoo()
    if len(tree.data) != len(pts) - 1:
        return [], 'none'
    # 不强行围住远处孤立点；限值同时考虑像素尺度和该簇自身点间距。
    largest_bridge = float(np.max(tree.data))
    bridge_limit = max(32.0, 4.6*typical, 2.0*float(np.percentile(finite_nn, 75)))
    if largest_bridge > bridge_limit:
        return [], 'none'

    # 优先尝试沿完整点群外缘形成的单个凹包；不允许选取多个分离子包络。
    # 调整 alpha 尺度是为了让沿曲线分布的点形成完整闭合边界，而不是改变聚类。
    for factor in (HULL_RADIUS_FACTOR, 3.4, 4.8, 7.0, 10.0):
        shapes = _concave_regions(pts, all_xy, labels, cluster_id, radius_factor=factor)
        for poly in shapes:
            if _polygon_covers_all(poly, pts) and _polygon_valid(poly, foreign):
                return [poly], 'alpha'

    # Alpha Shape无法包围全簇时，尝试窄圆角凸包，适用于细长、弧形、共线点群。
    # 圆角包络同样遵守 HULL_LOOSE 开关；不画只包一半的分段包络。
    for padding in (ENVELOPE_PADDING_PX, 4.0, ENVELOPE_MIN_PIXELS):
        outline = _rounded_hull(pts, padding)
        if (_polygon_covers_all(outline, pts) and
                _polygon_valid(outline, foreign)):
            return [outline], 'rounded'
    return [], 'none'


def _place_cluster_labels(fig, ax, xy: np.ndarray, y: np.ndarray, rank: list[tuple[int, int]]):
    """为最多5个重点簇寻找不重叠的文字位置，必要时带短引导线。"""
    fig.canvas.draw()
    existing = []
    offsets = [(0, 9), (19, 10), (-19, 10), (0, -16), (19, -15),
               (-19, -15), (30, 0), (-30, 0), (0, 22), (0, -27)]
    renderer = fig.canvas.get_renderer()
    axes_bb = ax.get_window_extent(renderer=renderer)
    for cid, _ in rank[:5]:
        pts = xy[y == cid]
        center = np.median(pts, axis=0)
        chosen = None
        for dx, dy in offsets:
            ann = ax.annotate(f'簇{cid}', xy=center, xytext=(dx, dy),
                              textcoords='offset points', fontsize=8.0,
                              ha='center', va='center', weight='bold', color=INK,
                              bbox={'boxstyle': 'round,pad=.16', 'fc': 'white',
                                    'ec': 'none', 'alpha': .88},
                              arrowprops=({'arrowstyle': '-', 'color': '#77838D',
                                           'lw': .65, 'alpha': .7} if abs(dx)+abs(dy) >= 19 else None),
                              zorder=9)
            fig.canvas.draw()
            bb = ann.get_window_extent(renderer=renderer).expanded(1.06, 1.18)
            inside = (bb.xmin >= axes_bb.xmin and bb.xmax <= axes_bb.xmax and
                      bb.ymin >= axes_bb.ymin and bb.ymax <= axes_bb.ymax)
            overlap = any(bb.overlaps(old) for old in existing)
            if inside and not overlap:
                chosen = bb
                break
            ann.remove()
        if chosen is None:
            # 不用重叠标签遮住簇；该簇仍用颜色标识。
            continue
        existing.append(chosen)


def plot_scatter(feature: str, runs: dict, topics: list[dict], folder: Path, topic_method: str):
    n_methods = len(runs)
    fig, axs = plt.subplots(1, n_methods, figsize=(max(7, 5.5*n_methods), 6),
                            squeeze=False, gridspec_kw={'wspace': .17})
    axes = axs[0]
    # 先固定最终画布布局，随后按屏幕像素计算包络，避免坐标转换受布局变化影响。
    fig.subplots_adjust(top=.86, bottom=.17, left=.07, right=.985)
    main_color = {row['cluster_id']: row['color'] for row in topics}
    name = DISPLAY_NAMES.get(feature, feature.replace('_',' '))
    boundary_counts = {}
    for ax, (method, (row, y, z)) in zip(axes, runs.items()):
        xy = z if z.shape[1] == 2 else UMAP(n_components=2, n_neighbors=15,
                                         min_dist=0.1, metric='euclidean',
                                         random_state=42).fit_transform(z)
        unique = sorted(set(y[y >= 0]))
        rank = get_major_clusters(y)
        keycolors = main_color if method == topic_method else {c: TOP_COLORS[i] for i, (c, _) in enumerate(rank)}
        # 几何包络以当前子图的实际显示像素计算，不被数值坐标尺度误导。
        # 只根据180篇点的位置预先固定坐标范围，包络不会反向修改坐标轴。
        xspan, yspan = np.ptp(xy, axis=0)
        ax.set_xlim(float(xy[:,0].min() - .05*xspan), float(xy[:,0].max() + .05*xspan))
        ax.set_ylim(float(xy[:,1].min() - .06*yspan), float(xy[:,1].max() + .06*yspan))
        ax.set_autoscale_on(False)
        fig.canvas.draw()
        shown_xy = ax.transData.transform(xy)
        shown_to_data = ax.transData.inverted()
        cmap = plt.get_cmap('hsv')
        drawn = 0
        drawn_alpha = 0
        drawn_rounded = 0
        outlined_clusters = 0
        for i, cid in enumerate(unique):
            where = y == cid
            if cid in keycolors:
                col, siz, alpha, edge = keycolors[cid], 34, .96, 'white'
            else:
                col, siz, alpha, edge = cmap((i*.61803398875) % 1), 23, .80, 'white'

            # 绘制局部紧密区域的半透明 Alpha Shape 凹包，优先于散点绘制。
            # 散乱的簇、小簇、二维重叠簇不强行围成面状区域。
            regions, region_kind = _cluster_regions(shown_xy[where], shown_xy, y, int(cid))
            if regions:
                outlined_clusters += 1
            if region_kind == 'rounded':
                drawn_rounded += len(regions)
            else:
                drawn_alpha += len(regions)
            for poly_screen in regions:
                poly = shown_to_data.transform(poly_screen)
                # 将同色填充和描边放在散点下层；不连接不同簇，不改变论文坐标。
                ax.add_patch(Polygon(poly, closed=True, facecolor=col,
                                     edgecolor='none', linewidth=0,
                                     alpha=HULL_FILL_ALPHA, zorder=1.6))
                ax.plot(*np.vstack([poly, poly[0]]).T, color=col, linewidth=.9,
                        alpha=HULL_LINE_ALPHA, zorder=1.8)
                drawn += 1
            ax.scatter(xy[where, 0], xy[where, 1], color=col, s=siz, alpha=alpha,
                       edgecolors=edge, linewidths=.35, zorder=3)
        boundary_counts[method] = (outlined_clusters, len(unique), drawn, drawn_alpha, drawn_rounded)
        mask = y < 0
        if mask.any():
            ax.scatter(xy[mask, 0], xy[mask, 1], marker='x', s=31, color='#9A9EA5',
                       alpha=.7, linewidths=.85, zorder=4)
        method_display = '截断SVD' if method == 'SVD' else method
        ax.set_title(f'{method_display} · {int(row["dim"])}维 · {len(unique)}簇／{int(mask.sum())}篇噪声',
                     fontsize=11.5, fontweight='bold', pad=11)
        ax.set_xlabel('二维展示坐标一')
        if ax is axes[0]:
            ax.set_ylabel('二维展示坐标二')
        ax.grid(alpha=.09)
        ax.tick_params(labelsize=8)
        ax.set_facecolor('#FAFCFD')
        # 在坐标轴范围确定后避让注释，避免直接把多个编号压在同一个点团上。
        _place_cluster_labels(fig, ax, xy, y, rank)
    fig.suptitle(f'{name}｜降维方法的论文聚类对照', fontsize=17, weight='bold', y=.992)
    overlap_note = ('宽松包络允许与其他簇点重叠；' if HULL_LOOSE
                    else '严格包络不包含其他簇点；')
    fig.text(.5, .018,
             '每点代表1篇论文；同簇同色，灰×为噪声。每簇最多一个完整半透明包络；'
             + overlap_note + '分散的簇可能不画包络。边界不代表真实主题边界；二维投影采用UMAP。',
             ha='center', fontsize=8.7, color=MUTED)
    fig.subplots_adjust(top=.86, bottom=.17, left=.07, right=.985)
    save(fig, folder/'01_三种降维聚类图.png')
    print(f'  包络模式：{"宽松（允许圈入其他簇点）" if HULL_LOOSE else "严格（不允许圈入其他簇点）"}')
    print('  散点图完整包络数量：' + ', '.join(
        f'{k}={covered}/{ncls}簇（包络区域{total}，凹包{alpha}，圆角包络{rounded}）'
        for k, (covered, ncls, total, alpha, rounded) in boundary_counts.items()))

def plot_topics(feature: str, topics: list[dict], folder: Path, topic_method: str):
    name = DISPLAY_NAMES.get(feature, feature)
    vals = [p['cluster_n'] for p in topics][::-1]
    colors = [p['color'] for p in topics][::-1]
    labels = [p['topic'] for p in topics][::-1]
    kws = [p['keywords'] for p in topics][::-1]
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    ys = np.arange(len(vals))
    ax.barh(ys, vals, height=.64, color=colors, alpha=.88, zorder=2)
    ax.set_yticks(ys, labels, fontsize=11)
    ax.set_xlim(0, max(vals)+14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('该簇包含的论文数（篇）')
    ax.grid(axis='x', color='#D9E1E7', alpha=.55, zorder=0)
    for i, (v, kw) in enumerate(zip(vals, kws)):
        ax.text(v+.18, i, str(v), va='center', fontsize=10, weight='bold', color=INK)
        ax.text(max(vals)+2.1, i, f'关键词：{kw}', fontsize=9.4, va='center', color=MUTED)
    ax.set_title(f'{name}｜主要研究群组及关键词线索', loc='left', fontsize=16, weight='bold', pad=19)
    total = sum(vals)
    fig.text(.5, .02,
             f'基于{topic_method}聚类展示论文数最多的{len(topics)}个簇，共{total}篇。'
             '左侧仅使用簇编号；右侧术语基于论文标题与关键词统计，仅供探索，不代表人工确认的主题名称。',
             ha='center', fontsize=9, color=MUTED)
    fig.subplots_adjust(top=.85, bottom=.14, left=.275, right=.98)
    save(fig, folder/'02_主题术语图.png')


def decade_label(year: float):
    if not np.isfinite(year):
        return None
    y = int(year)
    if y >= 2020:
        return '2020—2026'
    return f'{y // 10 * 10}年代'


def plot_time(feature: str, meta: pd.DataFrame, y: np.ndarray, topics: list[dict], folder: Path):
    name = DISPLAY_NAMES.get(feature, feature)
    known_years = meta.year.dropna().astype(int)
    if known_years.empty:
        raise ValueError('年份全部为空，无法生成历史图')
    years = list(range((known_years.min()//10)*10, 2020, 10))
    periods = [f'{i}年代' for i in years] + ['2020—2026']
    overall = Counter(decade_label(x) for x in known_years)
    series = np.array([overall.get(p, 0) for p in periods], dtype=int)
    mat = np.zeros((len(topics), len(periods)), dtype=int)
    for i, p in enumerate(topics):
        matched_years = meta.loc[y == p['cluster_id'], 'year'].dropna().astype(int)
        bins = Counter(decade_label(v) for v in matched_years)
        mat[i] = [bins.get(pp, 0) for pp in periods]
    fig = plt.figure(figsize=(13.4, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.7], hspace=.34)
    ax = fig.add_subplot(gs[0])
    xx = np.arange(len(periods))
    ax.plot(xx, series, color='#247E9A', marker='o', ms=7, lw=2.3, zorder=3)
    ax.fill_between(xx, 0, series, alpha=.13, color='#247E9A', zorder=2)
    for i, number in enumerate(series):
        if number:
            ax.annotate(str(number), (i,number), xytext=(0,8), textcoords='offset points',
                        ha='center', fontsize=9, color=INK)
    ax.set_xticks(xx, periods, rotation=25, ha='right')
    ax.set_ylabel('论文数量（篇）')
    ax.set_title('① 样本文献的年代分布', loc='left', fontsize=12, weight='bold')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', color='#D6DEE5', alpha=.58)
    ax.set_xlim(-.3,len(periods)-.7)

    ax2 = fig.add_subplot(gs[1])
    cmap = LinearSegmentedColormap.from_list('topic_time', ['#F6F9FB','#C4DFE2','#3992A5','#1C617D'])
    im = ax2.imshow(mat, cmap=cmap, vmin=0, vmax=max(1,int(mat.max())), aspect='auto')
    ax2.set_xticks(np.arange(len(periods)), periods, rotation=25, ha='right')
    labels = [p['topic'] for p in topics]
    ax2.set_yticks(np.arange(len(topics)), labels, fontsize=9.8)
    for i, p in enumerate(topics):
        ax2.get_yticklabels()[i].set_color(p['color'])
    ax2.tick_params(axis='both', length=0)
    for i in range(len(topics)):
        for j in range(len(periods)):
            if mat[i,j]:
                ax2.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=9,
                         color='white' if mat[i,j] > .58*max(1,mat.max()) else INK)
    cb = fig.colorbar(im, ax=ax2, fraction=.02, pad=.02)
    cb.set_label('该簇该时期的论文数（篇）')
    cb.locator = MaxNLocator(integer=True)
    cb.update_ticks()
    ax2.set_title('② 主要研究群组的年代分布（与术语图使用相同的簇编号）',
                  loc='left', fontsize=12, weight='bold', pad=12)
    fig.suptitle(f'{name}｜研究主题的历史分布与近期线索', fontsize=17, weight='bold', y=.987)
    fig.text(.5, .025,
             '2020—2026时间段尚不完整、与其他十年区间长度不同；样本并非逐年代表性抽样，'
             '不能将论文数量变化直接解释为整个领域研究热度变化。',
             ha='center', fontsize=9.2, color=MUTED)
    fig.subplots_adjust(left=.31, right=.955, top=.91, bottom=.11)
    save(fig, folder/'03_主题时间图.png')


def plot_institutions(meta: pd.DataFrame):
    pattern_map = {k: [re.compile(p, re.I) for p in pats] for k,pats in INSTITUTION_PATTERNS.items()}
    counts = Counter()
    matched_papers = 0
    with_info = 0
    for raw in meta.institutions.fillna('').astype(str):
        if not raw.strip():
            continue
        with_info += 1
        found = {name for name, patterns in pattern_map.items() if any(p.search(raw) for p in patterns)}
        if found:
            matched_papers += 1
            counts.update(found)  # 同一篇出现多次的单位，只计一次
    top = counts.most_common(10)
    if not top:
        raise RuntimeError('未匹配到机构名称；请检查元数据机构字段，而不是填造机构名称')
    names, vals = zip(*top[::-1])
    fig, ax = plt.subplots(figsize=(11.7, 6.5))
    ys = np.arange(len(vals))
    bars = ax.barh(ys, vals, color='#438997', height=.67)
    bars[-1].set_color('#DB9557')
    ax.set_yticks(ys, names, fontsize=11)
    ax.set_xlabel('涉及的样本论文数（篇；一篇论文中同一机构只计一次）')
    ax.set_xlim(0, max(vals)+1.2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='x', color='#D9E1E7', alpha=.55)
    ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        ax.text(v+.12, i, str(v), va='center', fontsize=10, weight='bold', color=INK)
    ax.set_title(f'{len(meta)}篇样本文献中较常出现的研究机构', loc='left', fontsize=16, weight='bold', pad=22)
    fig.text(.5, .021,
             f'{len(meta)}篇样本中{with_info}篇有机构记录；按明确别名可识别{matched_papers}篇。'
             '机构排名不代表学术影响力；加州大学按系统合计。',
             ha='center', fontsize=9.2, color=MUTED)
    fig.subplots_adjust(left=.30, top=.85, right=.97, bottom=.16)
    save(fig, OUTPUT_ROOT/'04_研究机构图.png')
    return counts, matched_papers, with_info


def main():
    set_chinese_style()
    meta = load_common()
    assert_file(CLUSTER_DIR / 'sweep_all_metrics.csv', 'Leiden聚类指标')
    assert_file(CLUSTER_DIR / 'all_cluster_labels.csv', 'Leiden聚类标签')
    verify_reduction_source()
    # 读取和验证通过后再创建输出目录，防止无效输入生成空文件夹。
    runs, choices, load_warnings = load_sweep(CLUSTER_DIR, meta)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    topic_method = choose_topic_method(runs)
    y = runs[topic_method][1]
    topics = infer_topics(meta, y)
    plot_scatter(FEATURE_NAME, runs, topics, OUTPUT_ROOT, topic_method)
    if topics:
        plot_topics(FEATURE_NAME, topics, OUTPUT_ROOT, topic_method)
        plot_time(FEATURE_NAME, meta, y, topics, OUTPUT_ROOT)
        pd.DataFrame([{k: val for k, val in t.items() if k != 'color'} for t in topics]).to_csv(
            OUTPUT_ROOT / '簇关键词统计.csv', index=False, encoding='utf-8-sig')
    else:
        print('没有非噪声簇，跳过术语图与时间图。')
    pd.DataFrame(choices).to_csv(OUTPUT_ROOT / '选用配置.csv', index=False, encoding='utf-8-sig')
    for warning in load_warnings:
        print('  跳过无效配置：' + warning)
    selected = ', '.join(f'{c["method"]}:{c["row_id"]}' for c in choices)
    print('可视化采用：' + selected)
    try:
        _, matched, recorded = plot_institutions(meta)
        institution_note = f'机构信息：{recorded}篇有记录，明确匹配{matched}篇。'
    except (RuntimeError, ValueError) as exc:
        print('机构图未生成：' + str(exc))
        institution_note = f'机构图未生成：{exc}'
    descriptions = [
        '正式实验可视化：从d_reduce/embeddings读取三种降维向量，从cluster/读取Leiden指标和标签。',
        '所有输出直接保存到visualizations_UMAP目录，不使用额外子文件夹。',
        'Leiden聚类使用10/20维降维结果；二维UMAP只用于视觉展示，不参与聚类。',
        '二维投影参数：n_neighbors=15, min_dist=0.1, metric=euclidean, random_state=42。',
        '格式塔原则：同簇同色（相似性）；二维接近性；闭合包络与共同区域；背景弱化。',
        f'包络：HULL_LOOSE={HULL_LOOSE}，每簇至多一个完整包络；包络不是真实学术主题边界。',
        '聚类散点图、簇术语图、主题时间折线/热力图、研究机构图使用同一批180篇论文。',
        '主题与时间图使用UMAP对应聚类的前8个较大簇，仅显示簇编号，不自动命名主题。',
        '跨方法的簇编号不表示相同主题；不同面板的二维坐标均为独立UMAP投影。',
        '时间图来自非代表性样本，2020—2026段未完整，不能推论整个学科的绝对热度。',
        '研究机构依据元数据中明确匹配名称统计，不代表机构学术影响力。',
        '本轮可视化选用：' + selected,
        institution_note,
    ]
    (OUTPUT_ROOT / '可视化说明_格式塔原则.txt').write_text(
        '\n'.join(descriptions) + '\n', encoding='utf-8')
    print(f'完成。全部图片直接保存到：{OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
