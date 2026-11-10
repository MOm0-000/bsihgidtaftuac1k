#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正式实验阶段③：读取d_reduce中的降维矩阵 + cluster中的Leiden标签，独立绘图。

输入：/home/zxx/bigdata/exp1/d_reduce/embeddings/*.npy
     /home/zxx/bigdata/exp1/cluster/sweep_all_metrics.csv
     /home/zxx/bigdata/exp1/cluster/all_cluster_labels.csv
输出：/home/zxx/bigdata/exp1/visualizations_UMAP/（所有图片直接保存，不再嵌套实验目录）

仅使用UMAP将已保存的10/20维结果二次投影到2维用于展示，不重新降维或聚类。
保留半透明完整包络；展示全部Leiden簇、关键词与可核实的人工候选主题；
输出主题年代计数/占比、近期论文分布和研究机构识别核查。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Patch
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
# 可视化展示全部簇，不改变聚类本身。
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


# 本轮根据“具体簇论文题名的主题一致性”人工审阅后选UMAP用于主题分析。
# 这不是按Silhouette自动选出的数学最优方法；低维Silhouette最高的是LocalMAP。
# 如未来希望改用LocalMAP，可把下面的值改成'LocalMAP'，但需重新人工审核主题名称。
TOPIC_METHOD = 'UMAP'
TOP_GROUPS = None  # None=展示全部Leiden簇，避免按规模筛掉小而主题集中的簇
RECENT_START = 2022
LAST_YEAR = 2026
DPI = 220

# 以下主题候选名称是基于本次公开的簇成员论文题名进行的人工归纳，绝非算法直接产生的标签。
# 仅当聚类标签CSV与已审阅的GitHub版本 *字节完全相同* 时才在图片中使用这些名称。
# 这样未来重跑5cluster_leiden.py或更改标签后，不会错误地把旧名称套用到新簇。
REVIEWED_LABELS_GIT_BLOB_SHA1 = '5a06940ed3d600e851911cda8e226d8595989819'
REVIEWED_UMAP_TOPICS = {
    0: '环境暴露与儿童呼吸健康（少量跨主题）',
    1: '昆虫发育、激素与生殖（少量跨主题）',
    2: '昆虫神经、感受与运动（部分跨主题）',
    3: '感觉与行为等跨主题混合簇',
    4: '生态、分子与生物医学混合簇',
    5: '离子通道与毒性机制（混合）',
    6: '微生物与昆虫共生（部分跨主题）',
    7: '遗传、演化与分类等混合簇',
    8: '儿童哮喘管理与生活质量',
    9: '过敏原、致敏与过敏性疾病',
    10: '植物源杀虫剂与农药相关研究',
    11: '免疫、微生物组与屏障机制（混合）',
    12: '媒介昆虫控制与抗性',
    13: '哮喘、气道屏障与炎症机制',
}
REVIEWED_MIXED_CLUSTERS = {3, 4, 5, 7}
REVIEWED_FINE_TOPIC_CLUSTERS = {8, 9, 11, 12, 13}

# 只控制散点图的视觉包络，不改变任何聚类标签或降维坐标。
# Alpha Shape 为凹包：沿局部密集的同簇点形成不规则闭合区域。
HULL_MIN_POINTS = 4                 # 少于4个不同位置的点不画面状包络
HULL_RADIUS_FACTOR = 2.4            # 最大三角形外接圆半径，相对簇内最近邻中位距离
HULL_FILL_ALPHA = 0.12              # 半透明填充，避免掩盖散点
HULL_LINE_ALPHA = 0.72              # 细边线
HULL_LOOSE = True                  # False=严格：圈入其他簇点就不画；True=宽松：允许圈入，仍要求完整包围本簇
# v4: 在显示像素中计算边界，每簇最多一个完整区域；不再拆成局部包络。
# 所有簇的可视化名称统一使用“簇0”“簇1”等编号，不自动判断研究主题。
ENVELOPE_MIN_PAPERS = 4            # 至少4个点才形成一个完整包络
ENVELOPE_PADDING_PX = 5.8          # 圆角包络外扩宽度（基于画面像素，非数据单位）
ENVELOPE_MIN_PIXELS = 2.5          # 临近其他簇时仍尝试较窄的轮廓

# 机构图只展示至少出现在3篇样本论文中的机构，避免在大量并列2篇机构中任意截取Top 10。
# 完整排名另存CSV供核查；改阈值只影响图中展示，不影响机构识别和完整统计。
MIN_INSTITUTION_PAPERS_TO_PLOT = 3

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

# ===== 基于“机构匹配核查.csv”逐条复核后增加的机构名称 / 多语种别名 =====
# 不从城市、电子邮箱或学科部门推断大学；只按机构字段明示名称匹配。
# 同一篇涉及多个机构：分别计入各机构，同一机构在一篇内最多计1次。
# 学院/医学院属于明确大学的，统一归入该大学；科研院所、医院和企业可单独计数。
# 原有机构名的确定别名（不改变已有规范化名称）：
INSTITUTION_PATTERNS['鲁汶大学'].extend([r'\bCatholic University of Leuven\b'])
INSTITUTION_PATTERNS['加州大学系统（多校区）'].extend([r'\bUCSD\b'])
INSTITUTION_PATTERNS['洛桑大学'].extend([r'\bUniversit[eé] de Lausanne\b'])
INSTITUTION_PATTERNS['哈佛大学'].extend([r'\bHarvard Medical School\b', r'\bHarvard Business School\b'])
INSTITUTION_PATTERNS['耶鲁大学'].extend([r'\bYale School of Public Health\b'])
INSTITUTION_PATTERNS['约翰斯·霍普金斯大学'].extend([r'\bJohns Hopkins Bloomberg School of Public Health\b'])

# 新机构：英文、德文、法文、葡文、西文的实体名称均以CSV明示字符串为依据。
# 这不是通用命名实体识别器，不会把无法确认的上级单位推测出来。
INSTITUTION_PATTERNS.update({
    '北布朗克斯中心医院': [r'\bNorth Central Bronx Hospital\b'],
    '东门诺学院': [r'\bEastern Mennonite College\b'],
    '高申学院': [r'\bGoshen College\b'],
    '法政大学': [r'\bHosei University\b'],
    '顺天堂大学': [r'\bJuntendo University\b'],
    '富士通': [r'\bFUJITSU Limited\b'],
    '圣托马斯医院（伦敦）': [r"\bSt[ .]?Thomas['’]? Hospital\b"],
    'Astra Charnwood（原文单位）': [r'\bAstra Charnwood\b'],
    '美国毒物与疾病登记署': [r'\bAgency for Toxic Substances and Disease Registry\b'],
    '鹿特丹伊拉斯姆斯大学医学中心': [r'\bErasmus University Medical Center\b'],
    '荷兰国家公共卫生与环境研究所': [r'\bNational Institute of Public Health and the Environment\b'],
    '马凯雷雷大学医学院': [r'\bMakerere Medical School\b'],
    '麦克马斯特大学': [r'\bMcMaster University\b'],
    '美国国家健康住房中心': [r'\bNational Center for Healthy Housing\b'],
    '堪萨斯州立大学': [r'\bKansas State University\b'],
    '韦恩州立大学': [r'\bWayne State University\b'],
    '女王大学（加拿大）': [r"\bQueen['’]s University\b"],
    '洛克菲勒大学': [r'\bRockefeller University\b'],
    '孟席斯健康研究学院': [r'\bMenzies School of Health Research\b'],
    '查尔斯·达尔文大学': [r'\bCharles Darwin University\b'],
    '本-古里安大学': [r'\bBen-Gurion University\b'],
    '穆欣比利健康与相关科学大学': [r'\bMuhimbili University of Health and Allied Sciences\b'],
    '保罗·埃尔利希研究所': [r'\bPaul[-– ]Ehrlich[-– ]Institut\b'],
    '美国国家癌症研究所': [r'\bNational Cancer Institute\b'],
    '乔斯大学': [r'\bUniversity of Jos\b'],
    '克瑞顿大学': [r'\bCreighton University Medical Center\b', r'\bCreighton University\b'],
    '慕尼黑大学': [r'\bLudwig[-– ]Maximilians[-– ]Universit[aä]t\b'],
    '阿伯里斯特威斯大学': [r'\bAberystwyth University\b'],
    '江西中医药大学': [r'\bJiangxi University of Traditional Chinese Medicine\b'],
    '马萨诸塞州总医院': [r'\bMassachusetts General Hospital\b'],
    '马尔堡大学': [r'\bPhilipps[-– ]Universit[aä]t Marburg\b'],
    '巴尼亚卢卡临床中心': [r'\bClinical Center of Banja Luka\b'],
    '巴尼亚卢卡卫生中心': [r'\bHealth Center, Banja Luka\b'],
    '美国疾控中心': [r'\bCenters for Disease Control and Prevention\b'],
    '博伊西州立大学': [r'\bBoise State,? University\b'],
    '斯旺西大学': [r'\bSwansea University\b'],
    '萨伦托大学': [r'\bUniversity of Salento\b'],
    'UNIRG大学中心': [r'\bUniversity Center UNIRG\b'],
    '戈亚斯联邦大学': [r'\bFederal University of Goias\b'],
    '美国农业部农业研究局': [r'\bUSDA Agricultural Research Service\b',
                      r'\bAgricultural Research Services?, United States Department of Agriculture\b'],
    '约翰布朗大学': [r'\bJohn Brown University\b'],
    '诺华': [r'\bNovartis Pharma(?:ceuticals?|ceutical Corporation)?\b'],
    '索菲亚医科大学': [r'\bMedical University Sofia\b'],
    '蒂宾根大学': [r'\b(?:Eberhard Karls )?Universit[aä]t T[uü]bingen\b'],
    '弗劳恩霍夫协会': [r'\bFraunhofer[-– ]Institut\b'],
    '不来梅大学': [r'\bUniversit[aä]t Bremen\b'],
    '阿里德农业大学（拉瓦尔品第）': [r'\bPir Mehr Ali Shah Arid Agriculture University\b'],
    '阳光海岸大学': [r'\bUniversity of the Sunshine Coast\b'],
    '利物浦大学': [r'\bUniversity of Liverpool\b'],
    '香港大学': [r'\bUniversity of Hong Kong\b'],
    '爱尔康实验室': [r'\bAlcon Laboratories\b'],
    '皇家墨尔本理工大学': [r'\bRMIT University\b'],
    '埃及国家研究中心': [r'\bNational Research Centre\b'],
    '基尔大学': [r'\bUniversity of Kiel\b'],
    '长春中医药大学': [r'\bChangchun University of Traditional Chinese Medicine\b'],
    '卫材': [r'\bEisai Inc\b'],
    '海尔大学（沙特）': [r'\bUniversity of Hail\b'],
    '蓬塔格罗萨州立大学': [r'\bUniversidade Estadual de Ponta Grossa\b'],
    '圣卡洛斯联邦大学': [r'\b(?:UFSCar|Universidade Federal de S[aã]o Carlos)\b'],
    '圣保罗联邦大学': [r'\bUniversidade Federal de S[aã]o Paulo\b'],
    '圣保罗大学': [r'\bUniversidade de S[aã]o Paulo\b'],
    '生物多样性研究所（巴西）': [r'\bIPBio\s*[-– ]\s*Instituto de Pesquisas da Biodiversidade\b'],
    '广州医科大学': [r'\bGuangzhou Medical University\b'],
    '里约热内卢联邦大学': [r'\bUniversidade Federal do Rio de Janeiro\b'],
    '圣埃斯皮里图州联邦大学': [r'\bUniversidade Federal do Esp[ií]rito Santo\b'],
    '华南农业大学': [r'\bSouth China Agricultural University\b'],
    '中国农业科学院': [r'\bChinese Academy of Agricultural Sciences\b'],
    '广西农业科学院': [r'\bGuangxi Academy of Agricultural Sciences\b'],
    '波兹南生命科学大学': [r'\bPozna[nń] University of Life Sciences\b'],
    '上海交通大学': [r'\bShanghai Jiaotong University\b', r'\bShanghai Jiao Tong University\b'],
    '西北农林科技大学': [r'\bNorthwest A\s*&\s*F University\b'],
    '卡马拉杰学院': [r'\bKamaraj College\b'],
    'V.H.N.S.N.学院': [r'\bV\.H\.N\.S\.N\.\s*College\b'],
    '萨塔姆王子大学': [r'\bPrince Sattam Bin Abdulaziz University\b'],
    '北新墨西哥学院': [r'\bNorthern New Mexico College\b'],
    '亚利桑那大学': [r'\bUniversity of Arizona\b'],
    '海梅一世大学': [r'\bUniversitat Jaume I\b'],
    '萨拉曼卡宗座大学': [r'\bUniversidad Pontificia de Salamanca\b'],
    '西班牙卡洛斯三世卫生研究所': [r'\bInstituto Carlos III\b'],
    '巴西利亚大学': [r'\bUniversidade de Bras[ií]lia\b'],
    '奥斯瓦尔多·克鲁兹基金会': [r'\bFiocruz\b', r'\bFund[aã]c[aã]o Oswaldo Cruz\b'],
    '上海科技大学': [r'\bShanghaiTech University\b'],
    '柏林工业大学': [r'\bTechnische Universit[aä]t Berlin\b'],
    '西班牙国家研究委员会': [r'\b(?:Spanish National Research Council|CSIC)\b'],
    '土耳其卫生科学大学': [r'\bHealth Sciences University\b'],
    '穆拉锡特克曼大学': [r'\bMu[gğ]la S[ıi]tk[ıi] Ko[cç]man University\b'],
    '诺丁汉大学': [r'\bUniversity of Nottingham\b'],
    '弗罗茨瓦夫环境与生命科学大学': [r'\bWroc[lł]aw University of Environmental and Life Sciences\b'],
    'SEGi大学': [r'\bSEGi University\b'],
    '中佛罗里达大学': [r'\bUniversity of Central Florida\b'],
    '浙江大学': [r'\bZhejiang University School of Medicine\b', r'\bZhejiang University\b'],
    '杭州湾医院': [r'\bHangzhou Bay Hospital\b'],
    '布里斯托大学': [r'\bUniversity of Bristol\b'],
    '同济大学': [r'\bTongji University\b'],
    '博洛尼亚大学': [r'\bUniversity of Bologna\b'],
    '拉普拉塔国立大学': [r'\bUniversidad Nacional de La Plata\b', r'\bUNLP\b'],
    '阿根廷国家科学技术研究委员会': [r'\bCONICET\b'],
    '厄瓜多尔国际大学': [r'\bUniversidad Internacional del Ecuador\b'],
    '伦敦圣乔治大学': [r'\bSt George[’\']?s University of London\b'],
    '新不伦瑞克大学': [r'\bUniversity of New Brunswick\b'],
    '四川农业大学': [r'\bSichuan Agricultural University\b'],
    '赫拉德茨-克拉洛韦大学': [r'\bUniversity of Hradec Kr[aá]lov[eé]\b'],
    '拉特罗布大学': [r'\bLa Trobe University\b'],
    '巴伐利亚森林国家公园研究部门': [r'\bBavarian Forest National Park Nature Conservation and Research\b'],
    '西悉尼大学': [r'\bWestern Sydney University\b'],
    '埃克塞特大学': [r'\bUniversity of Exeter\b'],
    '伊利诺伊州立大学': [r'\bIllinois State University\b'],
    '滨州医学院': [r'\bBinzhou Medical University\b'],
    '庞培法布拉大学': [r'\bUniversitat Pompeu Fabra\b'],
    '巴塞罗那基因组调控中心': [r'\bCentre for Genomic Regulation\b'],
    '巴塞罗那科学技术研究所': [r'\bBarcelona Institute of Science and Technology\b'],
    '加泰罗尼亚高等研究院': [r'\bInstituci[oó] Catalana de Recerca i Estudis Avan[cç]ats\b'],
    '贝尔格莱德大学': [r'\bUniversity of Belgrade\b'],
    '史密森热带研究所': [r'\bSmithsonian Tropical Research Institute\b'],
    '巴拿马大学': [r'\bUniversity of Panama\b'],
    '浙江省疾病预防控制中心': [r'\bZhejiang Provincial Center for Disease Control and Prevention\b'],
    '浙江省爱国卫生发展中心': [r'\bZhejiang Patriotic Health Development Center\b'],
    '杭州医学院': [r'\bHangzhou Medical College\b'],
    '伦敦国王学院': [r"\bKing['’]?s College London\b"],
    '伯明翰大学': [r'\bUniversity of Birmingham\b'],
    '墨西哥研究与高级研究中心': [r'\bCenter for Research and Advanced Studies\b'],
    '格雷罗自治大学': [r'\bUniversidad Aut[oó]noma de Guerrero\b'],
    '苏丹穆莱苏莱曼大学': [r'\bSultan Moulay Slimane University\b'],
    '摩洛哥巴斯德研究所': [r'\bInstitut Pasteur du Maroc\b'],
    '法拉比哈萨克国立大学': [r'\bAl[-– ]Farabi Kazakh National University\b'],
    '新南威尔士大学': [r'\bUNSW Medicine\b', r'\bUniversity of New South Wales\b'],
    '中国医学科学院': [r'\bChinese Academy of Medical Sciences\b'],
    '北京协和医院': [r'\bPeking Union Medical College Hospital\b'],
    '扬州大学': [r'\bYangzhou University\b'],
})
# ===== 增补规则结束 =====

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
    # control 多见于 pest/vector control，不能翻成临床‘治疗’。改成区分语义的两个术语。
    ('治疗', r'therap|treat|clinical management|asthma management|disease management'),
    ('防控', r'\b(?:vector|mosquito|pest|insect|insecticide) control\b|\bcontrol of (?:vectors|mosquitoes|pests|insects)\b|\bbiocontrol\b'),
    ('过敏原', r'allergen|allerg'),
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

# 各簇按规模降序固定颜色；图1选用方法、图2、图3及近期图使用同一套颜色。
# 14簇亦有独立颜色，不再将规模较小的簇全部改成灰色。
TOP_COLORS = ['#2786A5', '#E58A47', '#678E68', '#9370A5',
              '#CD6A78', '#53A6A2', '#B99A43', '#6D83C0',
              '#9F7A4D', '#387EB0', '#A65D9B', '#5A8A4C',
              '#C16D4B', '#7A6CA8']
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
    """同种方法多份配置时使用明确展示偏好和固定ID排序，不沿用HDBSCAN旧参数。"""
    return (-_maybe_float(row.get('preferred_for_visualization', 0), 0),
            str(row.get('spec_id', '')), str(row.get('row_id', '')))


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
        group = sorted(candidates[method], key=lambda item: _candidate_priority(item[0]))
        for row, path in group:
            try:
                z = np.load(path, allow_pickle=False)
                y = pd.to_numeric(l[str(row['row_id'])], errors='raise').to_numpy(dtype=int)
                if (z.ndim != 2 or z.shape != (len(meta), int(row['dim'])) or
                        z.shape[1] < 2 or not np.isfinite(z).all()):
                    raise ValueError(f'降维矩阵形状/数值异常：{path.name}')
                if np.any(y < 0):
                    raise ValueError('当前脚本只绘制Leiden社区；发现负标签，请核对标签来源')
                if len(set(y)) != int(row['n_clusters']) or int(row.get('n_noise', 0)) != 0:
                    raise ValueError('Leiden社区数量与指标表不一致')
                runs[method] = (row, y, z)
                choices.append({'method': method, 'row_id': str(row['row_id']),
                                'spec_id': str(row['spec_id']), 'dim': int(row['dim']),
                                'n_clusters': int(row['n_clusters'])})
                break
            except (OSError, ValueError, KeyError, TypeError) as e:
                warnings_list.append(f'{method}/{row["row_id"]}: {e}')
    if not runs:
        raise ValueError(f'{feature}未发现可用于绘图的完整配置；请检查embeddings和标签列')
    return runs, choices, warnings_list


def choose_topic_method(runs):
    """主题分析方法由审阅后显式指定，不再静默优先选UMAP。"""
    if TOPIC_METHOD not in runs:
        raise ValueError(f'已选主题方法{TOPIC_METHOD}没有对应的有效聚类结果：{list(runs)}')
    return TOPIC_METHOD


def reviewed_labels_match() -> bool:
    """通过Git Blob SHA-1核对人工主题名称所对应的整份聚类标签文件。"""
    path = CLUSTER_DIR / 'all_cluster_labels.csv'
    raw = path.read_bytes()
    blob_hash = hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()
    return TOPIC_METHOD == 'UMAP' and blob_hash == REVIEWED_LABELS_GIT_BLOB_SHA1


def get_major_clusters(y: np.ndarray, limit: int | None = None):
    """簇规模排序；默认返回全部簇。同规模时按簇编号升序，确保结果稳定。"""
    counts = Counter(int(i) for i in y)
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return ranked if limit is None else ranked[:limit]


def infer_topics(meta: pd.DataFrame, y: np.ndarray, manual_labels_ok: bool):
    # 此处只做关键词的确定性匹配；不是ML训练的主题分类器。
    if 'keywords' in meta.columns:
        text = (meta['title'].fillna('').astype(str) + ' | ' +
                meta['keywords'].fillna('').astype(str)).str.lower().tolist()
    else:
        text = meta['title'].fillna('').astype(str).str.lower().tolist()
    items = []
    for order, (cid, n) in enumerate(get_major_clusters(y, limit=TOP_GROUPS)):
        paper_texts = [t for t, lab in zip(text, y) if int(lab) == cid]
        hits = Counter()
        for term, pattern in KEYWORD_RULES:
            cnt = sum(bool(re.search(pattern, t, re.I)) for t in paper_texts)
            if cnt >= 2:
                hits[term] = cnt
        words = [word for word, _ in hits.most_common(3)]
        new_n = int(meta.loc[y == cid, 'year'].between(RECENT_START, LAST_YEAR).sum())
        reviewed_name = REVIEWED_UMAP_TOPICS.get(cid, '') if manual_labels_ok else ''
        items.append({
            'cluster_id': cid, 'cluster_n': n, 'topic': f'簇{cid}',
            'manual_topic_candidate': reviewed_name,
            'review_status': ('混合簇（人工核对）' if cid in REVIEWED_MIXED_CLUSTERS else
                              '人工题名归纳') if manual_labels_ok else '未匹配人工复核版本',
            'keywords': '、'.join(words) if words else '共同术语不足',
            'recent_n': new_n,
            'recent_fraction': new_n / n if n else 0.0,
            'color': TOP_COLORS[order % len(TOP_COLORS)],
        })
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
    严格模式下若区域包含其他社区节点则跳过；宽松模式允许重叠。
    注意：这些只是二维展示包络，不代表Leiden的实际社区边界。
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
        # 只有严格模式才检查区域是否圈入其他簇论文。
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
    # 图1划出三个互不重叠的垂直区域：主图、UMAP专属图例、两行解释文字。
    # 先固定主图坐标轴，保证后续按显示像素计算的聚类包络不会因调布局而位移。
    fig, axs = plt.subplots(1, n_methods, figsize=(max(8, 6*n_methods), 8.2),
                            squeeze=False, gridspec_kw={'wspace': .17})
    axes = axs[0]
    fig.subplots_adjust(top=.875, bottom=.345, left=.07, right=.985)
    # 图例轴和说明文字刻意位于主图横轴标题的下方，不再用整张图的 figure.legend
    # 覆盖上方主图，也不在绘制包络后再次修改主图坐标轴位置。
    legend_ax = fig.add_axes([.06, .105, .88, .18])
    legend_ax.set_axis_off()
    main_color = {row['cluster_id']: row['color'] for row in topics}
    name = DISPLAY_NAMES.get(feature, feature.replace('_',' '))
    boundary_counts = {}
    for ax, (method, (row, y, z)) in zip(axes, runs.items()):
        xy = z if z.shape[1] == 2 else UMAP(n_components=2, n_neighbors=15,
                                         min_dist=0.1, metric='euclidean',
                                         random_state=42).fit_transform(z)
        unique = sorted(set(y[y >= 0]))
        rank = get_major_clusters(y)
        keycolors = main_color if method == topic_method else {c: TOP_COLORS[i % len(TOP_COLORS)] for i, (c, _) in enumerate(rank)}
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
        method_display = '截断SVD' if method == 'SVD' else method
        ax.set_title(f'{method_display} · {int(row["dim"])}维 · {len(unique)}个Leiden社区',
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
    handles = [Patch(facecolor=t['color'], edgecolor='none', label=t['topic']) for t in topics]
    if handles:
        legend_ax.legend(handles=handles, loc='center', ncol=min(7, len(handles)),
                         frameon=False, fontsize=8.8,
                         title=f'颜色图例仅对应{topic_method}面板（其他面板的同色不代表同主题）',
                         title_fontsize=9.5, handlelength=1.05, columnspacing=1.2,
                         borderaxespad=0.0)
    note = fig.text(.5, .047,
             '每点代表1篇论文；同簇同色。每簇最多一个完整半透明包络；'
             + overlap_note + '分散簇可能不画包络。\n'
             '包络不是主题边界；三个面板独立进行二维UMAP投影，二维坐标不能跨面板直接比较。',
             ha='center', va='center', fontsize=8.4, color=MUTED, linespacing=1.45)
    # 运行时核验文字层不会压到相邻区域；不改变任何坐标和聚类包络。
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_obj = legend_ax.get_legend()
    if legend_obj is not None:
        legend_box = legend_obj.get_window_extent(renderer=renderer)
        axis_label_boxes = [a.xaxis.label.get_window_extent(renderer=renderer) for a in axes]
        if any(legend_box.overlaps(box) for box in axis_label_boxes):
            raise RuntimeError('图1的图例与横轴标题发生重叠，请调整 plot_scatter 中的布局区域。')
        if legend_box.overlaps(note.get_window_extent(renderer=renderer)):
            raise RuntimeError('图1的图例与脚注发生重叠，请调整 plot_scatter 中的布局区域。')
    save(fig, folder/'01_三种降维聚类图.png')
    print(f'  包络模式：{"宽松（允许圈入其他簇点）" if HULL_LOOSE else "严格（不允许圈入其他簇点）"}')
    print('  散点图完整包络数量：' + ', '.join(
        f'{k}={covered}/{ncls}簇（包络区域{total}，凹包{alpha}，圆角包络{rounded}）'
        for k, (covered, ncls, total, alpha, rounded) in boundary_counts.items()))

def plot_topics(feature: str, topics: list[dict], folder: Path, topic_method: str):
    """完整簇分布：左侧为计数，右侧为词频和带有来源限制的人工候选名称。"""
    name = DISPLAY_NAMES.get(feature, feature)
    ordered = topics[::-1]
    height = max(9.0, 2.7 + len(topics) * .61)
    fig, (ax, notes_ax) = plt.subplots(
        1, 2, figsize=(17.5, height), sharey=True,
        gridspec_kw={'width_ratios': [1.05, 1.65], 'wspace': .05})
    yy = np.arange(len(ordered))
    vals = [p['cluster_n'] for p in ordered]
    cols = [p['color'] for p in ordered]
    ax.barh(yy, vals, height=.68, color=cols, alpha=.90, zorder=2)
    ax.set_yticks(yy, [p['topic'] for p in ordered], fontsize=10.7)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, max(vals) + 4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('论文数（篇）')
    ax.grid(axis='x', color='#D9E1E7', alpha=.58, zorder=0)
    ax.set_axisbelow(True)
    for i, v in enumerate(vals):
        ax.text(v+.22, i, str(v), va='center', fontsize=10, weight='bold', color=INK)

    notes_ax.set_xlim(0, 1)
    notes_ax.set_xticks([])
    notes_ax.tick_params(axis='y', left=False, labelleft=False)
    for spine in notes_ax.spines.values():
        spine.set_visible(False)
    for i, p in enumerate(ordered):
        candidate = p['manual_topic_candidate'] or '待人工复核主题（标签文件版本未通过校验）'
        if p['review_status'].startswith('混合'):
            candidate = '〔混合〕' + candidate
        notes_ax.text(.01, i+.14, candidate, va='center', color=INK, fontsize=10.4, weight='medium')
        notes_ax.text(.01, i-.19, f'匹配词：{p["keywords"]}  |  近年{p["recent_n"]}篇',
                      va='center', color=MUTED, fontsize=9.2)
    ax.set_title('全部Leiden社区的文献数量', loc='left', fontsize=14, weight='bold', pad=14)
    notes_ax.set_title('人工题名归纳的候选主题 / 程序统计的术语',
                       loc='left', fontsize=14, weight='bold', pad=14)
    fig.suptitle(f'{name}｜{topic_method}全部{len(topics)}簇的主题线索',
                 fontsize=17, weight='bold', y=.985)
    fig.text(.5, .026,
             f'覆盖{sum(vals)}篇；词频基于标题/关键词正则匹配，人工候选名称不等于算法真值。'
             '标“〔混合〕”者为人工复核发现的混合簇；小簇也保留。',
             ha='center', color=MUTED, fontsize=9)
    fig.subplots_adjust(top=.88, bottom=.095, left=.09, right=.985)
    save(fig, folder / '02_主题术语图.png')


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
    fig = plt.figure(figsize=(13.7, max(9.2, 5.5 + .35*len(topics))))
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
    ax2.set_title('② 全部研究簇的年代计数（与主题图使用相同的簇编号）',
                  loc='left', fontsize=12, weight='bold', pad=12)
    fig.suptitle(f'{name}｜研究主题的历史分布与近期线索', fontsize=17, weight='bold', y=.987)
    fig.text(.5, .025,
             '2020—2026时间段尚不完整、与其他十年区间长度不同；样本并非逐年代表性抽样，'
             '不能将论文数量变化直接解释为整个领域研究热度变化。',
             ha='center', fontsize=9.2, color=MUTED)
    fig.subplots_adjust(left=.21, right=.955, top=.91, bottom=.11)
    save(fig, folder/'03_主题时间图.png')

    # 同一簇不同年代占比：避免将大簇绝对篇数直接误解为更活跃。
    # 分母为该簇有有效年份记录的论文数量，而非全部180篇。
    den = mat.sum(axis=1, keepdims=True)
    rates = np.divide(mat, den, out=np.zeros_like(mat, dtype=float), where=den > 0) * 100.0
    fig2, ax3 = plt.subplots(figsize=(13.7, max(6.4, 2.6 + .39 * len(topics))))
    im2 = ax3.imshow(rates, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax3.set_xticks(np.arange(len(periods)), periods, rotation=25, ha='right')
    ax3.set_yticks(np.arange(len(topics)), [p['topic'] for p in topics])
    for i, p in enumerate(topics):
        ax3.get_yticklabels()[i].set_color(p['color'])
    ax3.tick_params(axis='both', length=0)
    for i in range(len(topics)):
        for j in range(len(periods)):
            if mat[i, j]:
                ax3.text(j, i, f'{rates[i, j]:.0f}%', ha='center', va='center',
                         fontsize=8.5, color='white' if rates[i, j] >= 53 else INK)
    fig2.colorbar(im2, ax=ax3, fraction=.024, pad=.025, label='占该簇有年份论文的比例（%）')
    ax3.set_title(f'{name}｜各研究簇内部的年代构成（行归一化）',
                  loc='left', fontsize=15.5, weight='bold', pad=17)
    fig2.text(.5, .025, '分母为各簇有有效年份的论文数；2020—2026区间未满十年。'
              '本图是样本年代构成，不等于全领域研究增长率。', ha='center', color=MUTED, fontsize=9)
    fig2.subplots_adjust(left=.21, right=.935, top=.88, bottom=.13)
    save(fig2, folder/'03b_主题年代占比.png')


def plot_recent(feature: str, meta: pd.DataFrame, topics: list[dict], folder: Path, topic_method: str):
    """近期研究线索，展示绝对篇数与簇内比例；不宣称可自动识别学科前沿。"""
    ordered = topics[::-1]
    total_recent = int(meta.year.between(RECENT_START, LAST_YEAR).sum())
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(13.8, max(7.5, 2.9 + .49*len(topics))),
                                 sharey=True, gridspec_kw={'width_ratios': [1, 1], 'wspace': .12})
    pos = np.arange(len(ordered))
    counts = np.array([p['recent_n'] for p in ordered])
    rates = np.array([p['recent_fraction'] * 100 for p in ordered])
    cols = [p['color'] for p in ordered]
    ax.barh(pos, counts, height=.64, color=cols, alpha=.91)
    axp.barh(pos, rates, height=.64, color=cols, alpha=.91)
    ax.set_yticks(pos, [p['topic'] for p in ordered])
    ax.set_xlim(0, max(3, counts.max()+2))
    axp.set_xlim(0, max(15, min(100, rates.max()+15)))
    ax.set_xlabel('近年论文数（篇）')
    axp.set_xlabel('占该簇论文的比例（%）')
    ax.set_title('绝对数量', loc='left', fontsize=12.5, weight='bold')
    axp.set_title('簇内比例', loc='left', fontsize=12.5, weight='bold')
    for i,(v,r) in enumerate(zip(counts, rates)):
        ax.text(v+.12, i, str(v), va='center', fontsize=9)
        axp.text(r+1, i, f'{r:.0f}%', va='center', fontsize=9)
    for axis in (ax,axp):
        axis.grid(axis='x', alpha=.18)
        axis.set_axisbelow(True)
    fig.suptitle(f'{feature}｜{topic_method}聚类中{RECENT_START}—{LAST_YEAR}年的文献分布',
                 fontsize=16, weight='bold', y=.985)
    fig.text(.5, .025, f'样本中该时段共{total_recent}篇；2026年尚未结束。'
             '仅用于观察近期研究线索，不能用发表年份单独证明“前沿性”。',
             fontsize=9, color=MUTED, ha='center')
    fig.subplots_adjust(top=.87, bottom=.12, left=.12, right=.965)
    save(fig, folder/'05_近期论文主题分布.png')


def plot_institutions(meta: pd.DataFrame):
    pattern_map = {k: [re.compile(p, re.I) for p in pats] for k,pats in INSTITUTION_PATTERNS.items()}
    counts = Counter()
    matched_papers = 0
    with_info = 0
    audits = []
    for paper in meta.itertuples(index=False):
        raw = str(getattr(paper, 'institutions', '') or '')
        valid = bool(raw.strip())
        found = sorted({name for name, patterns in pattern_map.items()
                        if valid and any(p.search(raw) for p in patterns)})
        with_info += int(valid)
        matched_papers += int(bool(found))
        counts.update(found)  # 同一篇同一机构只计一次
        audits.append({'pmid': str(getattr(paper, 'pmid', '')),
                       '是否有机构记录': int(valid),
                       '是否匹配到规则': int(bool(found)),
                       '匹配到的机构': '；'.join(found),
                       '原始机构字段_供人工核查': raw})
    pd.DataFrame(audits).to_csv(OUTPUT_ROOT/'机构匹配核查.csv',
                                index=False, encoding='utf-8-sig')
    # 完整排序先按论文数降序，同频机构按名称排序，不使用原始出现顺序决胜。
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    if not ranked:
        raise RuntimeError('未匹配到机构名称；请检查元数据机构字段，而不是填造机构名称')
    pd.DataFrame([
        {'机构（规范名称）': org, '涉及样本论文数': int(n),
         '占全部样本文献比例': float(n / len(meta)),
         '占有机构记录文献比例': float(n / with_info) if with_info else 0.0}
        for org, n in ranked
    ]).to_csv(OUTPUT_ROOT/'机构频次_完整统计.csv', index=False, encoding='utf-8-sig')
    top = [pair for pair in ranked if pair[1] >= MIN_INSTITUTION_PAPERS_TO_PLOT]
    threshold_used = MIN_INSTITUTION_PAPERS_TO_PLOT
    if not top:  # 防止未来使用另一批数据时无图
        top = ranked[:min(10, len(ranked))]
        threshold_used = top[-1][1]
    names, vals = zip(*top[::-1])
    fig, ax = plt.subplots(figsize=(11.7, max(5.5, 2.7 + .47 * len(top))))
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
    ax.set_title(f'{len(meta)}篇样本文献中出现不少于{threshold_used}篇的机构', loc='left', fontsize=16, weight='bold', pad=22)
    fig.text(.5, .021,
             f'{len(meta)}篇样本中{with_info}篇有机构记录；规则匹配{matched_papers}篇'
             f'（占有记录论文{matched_papers / with_info:.1%}）。'
             f'共识别{len(counts)}个规范机构；图中展示{len(top)}个，完整统计见CSV。'
             '同篇多机构分别计数；加州大学按系统合计；频次不代表学术影响力。',
             ha='center', fontsize=9.2, color=MUTED)
    fig.subplots_adjust(left=.30, top=.85, right=.97, bottom=.16)
    save(fig, OUTPUT_ROOT/'04_研究机构图.png')
    return counts, matched_papers, with_info



def plot_scatter_details(feature: str, runs: dict, folder: Path):
    """每种方案的独立高清聚类图。独立图例不把不同方法的簇号混为同一主题。"""
    for method, (row, y, z) in runs.items():
        xy = z if z.shape[1] == 2 else UMAP(n_components=2, n_neighbors=15,
                 min_dist=0.1, metric='euclidean', random_state=42).fit_transform(z)
        rank = get_major_clusters(y)
        cmap_local = {cid: TOP_COLORS[i % len(TOP_COLORS)] for i, (cid, _) in enumerate(rank)}
        fig, ax = plt.subplots(figsize=(12.8, 9.1))
        fig.subplots_adjust(top=.89, bottom=.13, left=.075, right=.72)
        xspan, yspan = np.ptp(xy, axis=0)
        ax.set_xlim(float(xy[:, 0].min() - .07 * xspan), float(xy[:, 0].max() + .07 * xspan))
        ax.set_ylim(float(xy[:, 1].min() - .07 * yspan), float(xy[:, 1].max() + .07 * yspan))
        ax.set_autoscale_on(False)
        fig.canvas.draw()
        shown_xy = ax.transData.transform(xy)
        shown_to_data = ax.transData.inverted()
        patches = []
        for cid, n in rank:
            mask = y == cid
            col = cmap_local[cid]
            shapes, _ = _cluster_regions(shown_xy[mask], shown_xy, y, int(cid))
            for outline_screen in shapes:
                outline = shown_to_data.transform(outline_screen)
                ax.add_patch(Polygon(outline, facecolor=col, edgecolor='none',
                                     alpha=HULL_FILL_ALPHA, zorder=1.5))
                ax.plot(*np.vstack([outline, outline[0]]).T,
                        color=col, lw=1.0, alpha=HULL_LINE_ALPHA, zorder=1.7)
            ax.scatter(xy[mask, 0], xy[mask, 1], s=61, c=col,
                       edgecolors='white', linewidths=.5, alpha=.94, zorder=3)
            patches.append(Patch(facecolor=col, edgecolor='none', label=f'簇{cid}  （{n}篇）'))
        _place_cluster_labels(fig, ax, xy, y, rank)
        ax.grid(alpha=.08)
        ax.set_facecolor('#FAFCFD')
        ax.set_xlabel('二维展示坐标一', fontsize=12)
        ax.set_ylabel('二维展示坐标二', fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_title(f'{method}：{int(row["dim"])}维空间的Leiden聚类 / 二维投影',
                     fontsize=16, weight='bold', pad=15)
        fig.legend(handles=patches, loc='upper left', bbox_to_anchor=(.75, .84),
                   ncol=1, frameon=False, fontsize=11, title='当前方法的社区', title_fontsize=12)
        fig.text(.075, .035,
                 '散点为文献，颜色表示当前方法的社区；半透明包络仅作视觉引导。'
                 '投影不参与聚类，且不同方法的簇编号不能直接对应。',
                 ha='left', fontsize=10, color=MUTED)
        save(fig, folder / f'01a_{method}_高清聚类详图.png')


def plot_institution_topic_matrix(meta: pd.DataFrame, y: np.ndarray,
                                  topics: list[dict], folder: Path):
    """论文所属机构 × Leiden簇的真实交叉计数，允许一篇论文对应多个机构。"""
    pattern_map = {org: [re.compile(p, re.I) for p in pats]
                   for org, pats in INSTITUTION_PATTERNS.items()}
    org_clusters = Counter()
    org_totals = Counter()
    records = []
    for paper, cluster in zip(meta.itertuples(index=False), y):
        raw = str(getattr(paper, 'institutions', '') or '')
        orgs = sorted({org for org, patterns in pattern_map.items()
                       if raw.strip() and any(pt.search(raw) for pt in patterns)})
        for org in orgs:
            org_clusters[(org, int(cluster))] += 1
            org_totals[org] += 1
            records.append({'pmid': str(getattr(paper, 'pmid', '')),
                            '年份': getattr(paper, 'year', np.nan),
                            '机构': org, 'UMAP簇': int(cluster)})
    if not org_totals:
        raise RuntimeError('机构—主题关联没有任何明示匹配机构，无法制作图表')
    pd.DataFrame(records).to_csv(folder / '机构主题关联_逐篇记录.csv',
                                index=False, encoding='utf-8-sig')
    order_org = sorted(org_totals, key=lambda org: (-org_totals[org], org))
    order_cluster = [int(t['cluster_id']) for t in topics]
    long_records = []
    for org in order_org:
        for c in order_cluster:
            n = org_clusters.get((org, c), 0)
            long_records.append({'机构': org, 'UMAP簇': c,
                  '簇候选主题': next((t['manual_topic_candidate'] for t in topics
                                if t['cluster_id'] == c), ''),
                  '涉及论文数': n, '该机构出现篇数': org_totals[org],
                  '机构内部主题占比': n / org_totals[org]})
    pd.DataFrame(long_records).to_csv(folder / '机构主题关联_全部交叉统计.csv',
                                      index=False, encoding='utf-8-sig')
    show_orgs = [org for org in order_org
                 if org_totals[org] >= MIN_INSTITUTION_PAPERS_TO_PLOT]
    if not show_orgs:
        show_orgs = order_org[:min(10, len(order_org))]
    matrix = np.array([[org_clusters.get((org, c), 0) for c in order_cluster]
                       for org in show_orgs], dtype=int)
    # 横向与图2保持一致的簇规模次序；一个机构跨多个社区并不等于存在合作关系。
    fig, ax = plt.subplots(figsize=(17.0, max(5.6, 2.8 + .52 * len(show_orgs))))
    cmap = LinearSegmentedColormap.from_list('org_topics',
                                             ['#F6F9FB', '#D9E9EC', '#79B6C1', '#2A738A'])
    im = ax.imshow(matrix, cmap=cmap, vmin=0,
                   vmax=max(1, int(matrix.max())), aspect='auto')
    ax.set_xticks(np.arange(len(order_cluster)), [f'簇{c}' for c in order_cluster], fontsize=10)
    ax.set_yticks(np.arange(len(show_orgs)),
                  [f'{org}（{org_totals[org]}篇）' for org in show_orgs], fontsize=10.5)
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if v:
                ax.text(j, i, str(v), ha='center', va='center',
                        color='white' if v > .60 * max(1, matrix.max()) else INK,
                        fontsize=10, weight='bold')
    bar = fig.colorbar(im, ax=ax, fraction=.027, pad=.018)
    bar.set_label('机构与社区共同出现的论文数（篇）')
    bar.locator = MaxNLocator(integer=True)
    bar.update_ticks()
    ax.set_title('机构—研究主题关联：高频机构在UMAP各社区中的出现次数',
                 loc='left', fontsize=16, weight='bold', pad=19)
    ax.set_xlabel('UMAP社区（按论文数降序；主题解释参见图2）', fontsize=11)
    fig.text(.5, .022,
             '仅统计元数据明确出现的机构；同篇多机构分别计数，同篇同机构只计一次。'
             '数字衡量本样本内参与位置，不是合作网络或机构影响力排名。',
             ha='center', color=MUTED, fontsize=9.5)
    fig.subplots_adjust(left=.245, right=.94, top=.85, bottom=.16)
    save(fig, folder / '06_机构主题关联图.png')
    print(f'  机构—主题交叉表：{len(order_org)}个机构，绘图展示{len(show_orgs)}个机构。')


def plot_historical_period_comparison(meta: pd.DataFrame, y: np.ndarray,
                                      topics: list[dict], folder: Path):
    """每个阶段内部的主题构成（列归一化），与现有簇内年代占比（行归一化）互补。"""
    valid = meta['year'].notna().to_numpy()
    years = meta['year'].to_numpy(dtype=float)
    period_labels = ['2010年以前', '2010—2019年', '2020—2026年']
    masks = [valid & (years < 2010), valid & (years >= 2010) & (years < 2020),
             valid & (years >= 2020) & (years <= LAST_YEAR)]
    denom = np.array([int(m.sum()) for m in masks], dtype=int)
    cids = [int(t['cluster_id']) for t in topics]
    mat = np.array([[int(np.sum(m & (y == c))) for m in masks]
                    for c in cids], dtype=int)
    if not (denom > 0).any():
        raise RuntimeError('缺少有效年份，无法绘制分阶段对照')
    proportions = np.divide(mat, denom[None, :],
                            out=np.zeros_like(mat, dtype=float), where=denom[None, :] > 0)*100
    rows = []
    for i, c in enumerate(cids):
        for j, period in enumerate(period_labels):
            rows.append({'UMAP簇': c, '候选主题': topics[i]['manual_topic_candidate'],
                         '时期': period, '该簇该期论文数': int(mat[i, j]),
                         '该时期全部论文数': int(denom[j]),
                         '占该时期论文比例': float(proportions[i, j]/100)})
    pd.DataFrame(rows).to_csv(folder / '历史阶段_各时期主题构成.csv',
                              index=False, encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(11.4, max(7.6, 2.5 + .48 * len(topics))))
    cmap = LinearSegmentedColormap.from_list('stage_topics',
                                             ['#F6F9FB', '#CDE3E7', '#77B3BE', '#286F85'])
    im = ax.imshow(proportions, aspect='auto', cmap=cmap, vmin=0,
                   vmax=max(20, float(proportions.max())))
    ax.set_xticks(np.arange(3), [f'{p}\n（{n}篇）' for p, n in zip(period_labels, denom)], fontsize=11)
    ax.set_yticks(np.arange(len(cids)), [f'簇{c}（{topics[i]["cluster_n"]}篇）'
                                          for i,c in enumerate(cids)], fontsize=10.5)
    ax.tick_params(length=0)
    for i in range(len(cids)):
        for j in range(3):
            if mat[i, j]:
                ax.text(j, i, f'{mat[i,j]}篇 / {proportions[i,j]:.0f}%',
                        ha='center', va='center', fontsize=9.3,
                        color='white' if proportions[i,j] > .65 * max(20, proportions.max()) else INK)
    bar = fig.colorbar(im, ax=ax, fraction=.033, pad=.03)
    bar.set_label('占该时期样本文献比例（%）')
    ax.set_title('不同历史阶段的研究主题构成（每列按该期总量归一）',
                 fontsize=15, loc='left', pad=18, weight='bold')
    fig.text(.5, .022,
             '与既有“簇内年代构成”（图5）分母不同；2020—2026年不足十年，教师清单非代表性抽样，'
             '不能据此推断整个领域的兴衰。',
             ha='center', color=MUTED, fontsize=9.3)
    fig.subplots_adjust(left=.17, right=.91, top=.91, bottom=.13)
    save(fig, folder / '07_历史阶段主题构成图.png')

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
    manual_labels_ok = reviewed_labels_match()
    print('人工主题候选名称：' + ('标签版本吻合，启用' if manual_labels_ok else '标签版本不同，禁用以避免错标'))
    topics = infer_topics(meta, y, manual_labels_ok)
    plot_scatter(FEATURE_NAME, runs, topics, OUTPUT_ROOT, topic_method)
    plot_scatter_details(FEATURE_NAME, runs, OUTPUT_ROOT)
    if topics:
        plot_topics(FEATURE_NAME, topics, OUTPUT_ROOT, topic_method)
        plot_time(FEATURE_NAME, meta, y, topics, OUTPUT_ROOT)
        plot_recent(FEATURE_NAME, meta, topics, OUTPUT_ROOT, topic_method)
        plot_historical_period_comparison(meta, y, topics, OUTPUT_ROOT)
        pd.DataFrame([{k: val for k, val in t.items() if k != 'color'} for t in topics]).to_csv(
            OUTPUT_ROOT / '簇关键词统计.csv', index=False, encoding='utf-8-sig')
    else:
        print('没有可用Leiden社区，跳过术语图与时间图。')
    pd.DataFrame(choices).to_csv(OUTPUT_ROOT / '选用配置.csv', index=False, encoding='utf-8-sig')
    for warning in load_warnings:
        print('  跳过无效配置：' + warning)
    selected = ', '.join(f'{c["method"]}:{c["row_id"]}' for c in choices)
    print('可视化采用：' + selected)
    try:
        _, matched, recorded = plot_institutions(meta)
        plot_institution_topic_matrix(meta, y, topics, OUTPUT_ROOT)
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
        f'主题与时间图使用{topic_method}全部{len(topics)}个Leiden簇，避免遗漏小规模研究方向。',
        '若聚类标签与已审阅GitHub版本完全一致，可显示逐簇人工归纳的候选主题；混合簇有明确提示。',
        f'人工主题名称版本核验成功：{manual_labels_ok}。',
        '研究热点/前沿不是由单一Silhouette或自动词频决定；主题名称需结合成员论文人工复核。',
        '跨方法的簇编号不表示相同主题；不同面板的二维坐标均为独立UMAP投影。',
        '时间图来自非代表性样本，2020—2026段未完整，不能推论整个学科的绝对热度。',
        '机构依据原始字段明示名称与多语种别名匹配；同篇多机构分别计数，不明母机构不猜测；未匹配见机构匹配核查.csv；不代表学术影响力。',
        '新增图06及交叉统计CSV：机构×UMAP社区的出现次数，用于观察样本中的机构主题参与，不能推断机构主导某方向。',
        '新增图07及阶段构成CSV：每个时期内部各社区比例（列归一化）；与原图03b的簇内年代比例（行归一化）互补。',
        '新增01a_各方法_高清聚类详图，每幅仅有该方法自己的图例，便于替换报告中字号偏小的缩略图。',
        '已将关键词control从“治疗”规则移除；仅在明确昆虫/媒介防控语境输出“防控”词，重新运行后须同步替换旧图2和报告主题统计表。',
        f'近期主题图：{RECENT_START}—{LAST_YEAR}年论文按簇统计（数量和簇内比例），不宣称自动判断前沿性。',
        '本轮可视化选用：' + selected,
        institution_note,
    ]
    (OUTPUT_ROOT / '可视化说明_格式塔原则.txt').write_text(
        '\n'.join(descriptions) + '\n', encoding='utf-8')
    print(f'完成。全部图片直接保存到：{OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
