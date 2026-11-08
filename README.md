# 基于 SPECTER2 的文献语义分析与主题聚类

> **SPECTER2 · PCA / UMAP / LocalMAP · Leiden**  
> 从 PubMed 文献整理、正文清洗和人工复核，到论文向量表示、降维、社区发现与可视化的可复用分析流程。

本仓库包含一套面向 **183 篇候选文献** 的研究数据处理与聚类分析代码。经过质量筛选，保留 **180 篇论文** 进入正式实验；使用 **SPECTER2 Proximity** 生成每篇论文的 768 维语义向量，再分别采用 PCA、UMAP 和 LocalMAP 降维，构建近邻图并进行 Leiden 社区聚类，最后生成对比和探索性可视化。

**说明：** 目前正式实验的语义特征使用的是论文的**标题和摘要**（缺少摘要时仅使用标题），而非整篇正文。正文提取、清洗与复核是独立的数据整理和质量控制环节，不能理解为将全文直接输入了 SPECTER2。

## 目录

- [分析流程](#分析流程)
- [仓库结构](#仓库结构)
- [环境配置](#环境配置)
- [快速开始：使用已有结果](#快速开始使用已有结果)
- [完整流程：从原始文献开始](#完整流程从原始文献开始)
- [实验配置与现有结果](#实验配置与现有结果)
- [数据与复现注意事项](#数据与复现注意事项)

## 分析流程

```text
PubMed PMID 列表 / 本地 PDF、HTML
                  │
                  ▼
     1.1 文献正文提取与元数据整理
                  │
                  ▼
     1.2 机构信息及元数据复核
                  │
                  ▼
     2.1 正文清洗与文献筛选
           183 篇 → 180 篇
                  │
                  ▼
     2.2 问题论文定向复核与修复
                  │
                  ▼
     3   SPECTER2 Proximity 特征提取
               180 × 768
                  │
                  ▼
     4   PCA / UMAP / LocalMAP 降维
         20 维 / 10 维 / 10 维
                  │
                  ▼
     5   近邻图构建 + Leiden 聚类
                  │
                  ▼
     6   2D 投影、聚类对比与统计可视化
```

各阶段按数字前缀组织，输入与输出存放在仓库根目录的相应文件夹内。**降维、聚类和绘图是相互独立的步骤**：聚类使用保存的高维降维结果；二维可视化投影仅用于展示，不作为 Leiden 聚类的输入。

## 仓库结构

```text
.
├── 1.1prepare_corpus.py          # 提取 HTML/PDF 正文，生成元数据与提取报告
├── 1.2update_institutions.py     # 根据人工核查信息补充机构等元数据
├── 2.1clean_corpus.py           # 清洗正文，筛选生成 180 篇语料
├── 2.2review_problem_papers.py  # 针对已核实问题进行定向修复
├── 3extract_specter2.py         # 生成 180 × 768 的 SPECTER2 特征
├── 4d_reduce.py                 # PCA / UMAP / LocalMAP 降维
├── 5cluster_leiden.py           # 构建近邻图、运行 Leiden 聚类
├── 6visualize_UMAP.py           # 对比图、主题/时间/机构等可视化
├── requirements.txt             # Python 依赖（不含 PyTorch）
│
├── 下载code/                     # PubMed 下载与缺失文献报告工具
├── reference/                   # 下载报告；原始 PDF/HTML 不纳入 Git
│   ├── PDF/                      # 需要自行准备，仓库中不包含
│   └── FullText/                 # 需要自行准备，仓库中不包含
├── extracted/                   # 原始提取语料、元数据、审核记录
├── cleaned_180/                 # 清洗后的 180 篇正文与质量报告
├── cleaned_180_checked/         # 进一步复核后的正式语料
├── features_final/              # SPECTER2 特征、论文索引及缓存
├── d_reduce/                    # 降维后的嵌入及质量指标
│   └── embeddings/              # PCA / UMAP / LocalMAP 的 .npy 文件
├── cluster/                     # 聚类标签、指标、配置及主题分布
└── visualizations_UMAP/         # 绘图时生成；当前仓库未包含此目录
```

关键文件：

| 路径 | 内容 |
| --- | --- |
| `extracted/papers_metadata_reviewed.csv` | 复核后的标题、摘要、作者、机构等元数据 |
| `cleaned_180_checked/papers_180.jsonl` | 供正式实验使用的 180 篇语料 |
| `features_final/papers_index.csv` | 论文索引，使用 `row_index` 保证向量和论文一一对应 |
| `features_final/specter2_proximity.npy` | 180 × 768 的论文向量 |
| `d_reduce/embeddings/*.npy` | 三种方法的降维结果 |
| `d_reduce/降维指标.csv` | 近邻重合率、Trustworthiness 等降维指标 |
| `cluster/all_cluster_labels.csv` | 三组配置下的聚类标签 |
| `cluster/sweep_all_metrics.csv` | 聚类数量、Silhouette 等聚类指标 |
| `cluster/全部簇_主题分布.csv` | 簇与参考主题的分布统计 |

## 环境配置

项目的 `requirements.txt` 按研究环境中的 **Python 3.12** 软件版本固定；建议使用 Python 3.12 以便复现。

```bash
git clone https://github.com/MOm0-000/bsihgidtaftuac1k.git
cd bsihgidtaftuac1k

python -m venv .venv
source .venv/bin/activate         # Linux / macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

主要依赖包括 `numpy`、`pandas`、`scikit-learn`、`umap-learn`、`pacmap`、`leidenalg`、`igraph`、`matplotlib`、`transformers`、`adapters`、`PyMuPDF` 等。

**PyTorch 需单独安装。** `requirements.txt` 不包含 `torch`，以避免覆盖研究机器上的 CUDA 版本。仅在需要**重新计算 SPECTER2 特征**时，按自己的 CPU/CUDA 环境安装兼容的 PyTorch；如果机器已经配置好 PyTorch，不要为运行本仓库而盲目重装。

## 快速开始：使用已有结果

仓库已经提交了正式实验所需的 `features_final/`、`d_reduce/` 与 `cluster/` 的主要数据文件，**不需要先下载原始 PDF/HTML，也无需先重新推理 SPECTER2**，即可检查结果或运行下游分析。

安装所需 Python 依赖后，在仓库根目录依次运行：

```bash
# 重新生成三组降维坐标及其评估指标
python 4d_reduce.py

# 基于这三组坐标进行 Leiden 聚类
python 5cluster_leiden.py

# 生成可视化图片
python 6visualize_UMAP.py
```

图片将写入新生成的 `visualizations_UMAP/` 目录。

> **只想查看已有结果？** 可以直接打开 `d_reduce/降维指标.csv`、`cluster/sweep_all_metrics.csv` 和 `cluster/all_cluster_labels.csv`，不必重新运行脚本。注意：重新计算 LocalMAP 时，结果可能随 `pacmap` 版本或内部随机性而变化；若需要严格使用仓库保存的结果，请勿覆盖对应 `.npy` 文件。

## 完整流程：从原始文献开始

完整复现前，需要准备研究使用的 PMID 列表和原始论文文件。仓库的 `.gitignore` 排除了 `reference/PDF/`、`reference/FullText/` 以及 `models/`，因此**仅克隆本仓库无法从头执行全部步骤**。

### 0. 准备文献

`下载code/` 目录提供 PubMed 文献下载及缺失文件检查脚本；例如：

```bash
# 先测试少量文献
python "下载code/pubmed_downloader.py" --excel "下载code/zzx.xlsx" --limit 5

# 根据清单尝试下载可公开获取的文献
python "下载code/pubmed_downloader.py" --excel "下载code/zzx.xlsx"

# 对本地缺失的 PDF / HTML 生成检查报告
python "下载code/generate_pubmed_missing_report.py"
```

文献按 PMID 命名，例如 `reference/PDF/2257383.pdf` 和 `reference/FullText/2257383.html`。下载工具不保证每篇论文都能自动获取，缺失文件可能需要人工核实并在合法授权范围内补齐。

### 1. 正文提取与元数据补充

```bash
python 1.1prepare_corpus.py
python 1.2update_institutions.py
```

- `1.1prepare_corpus.py` 从本地 HTML/PDF 提取正文，并整理 `extracted/papers_metadata.csv` 及提取报告；首次运行可尝试通过 PubMed 补充元数据。
- `1.2update_institutions.py` 在**不修改原始元数据 CSV** 的前提下，输出 `extracted/papers_metadata_reviewed.csv`。
- 正文提取结果位于 `extracted/corpus/`。

如需强制重建提取正文，可执行 `python 1.1prepare_corpus.py --force`。执行前请备份已有人工作业，避免误覆盖下游产物。

### 2. 正文清洗与定向复核

```bash
python 2.1clean_corpus.py
python 2.2review_problem_papers.py
```

- 第一步根据清洗规则和已核实的筛选决定，将 183 篇候选文献整理为 180 篇；结果和 `cleaning_report.csv`、`needs_review.csv`、`excluded.csv` 写入 `cleaned_180/`。
- 第二步仅对特定论文的已知问题执行有证据约束的修复，结果写入 `cleaned_180_checked/`。该步骤对若干论文**依赖对应的原始 PDF**，缺失时无法完整重跑。
- 两步均支持 `--force` 重新生成输出；不会直接改写上游输入。

本研究排除的 3 篇论文及其具体原因记录在 `cleaned_180/cleaning_config.json` 中。`needs_review.csv` 中的记录仍应由研究者结合原文人工核查。

### 3. 生成 SPECTER2 特征

```bash
python 3extract_specter2.py
```

脚本读取 `cleaned_180_checked/papers_180.jsonl`，使用 **SPECTER2 base + proximity adapter**，以标题、摘要为输入，截断到最多 512 token，提取 CLS 表示并做 L2 归一化，输出 `features_final/specter2_proximity.npy`（形状为 `180 × 768`）。缺少摘要的论文只使用标题，不自动以正文替代摘要。

脚本支持检查并复用已存在的完整特征，以及基于论文输入和配置的逐篇缓存。需要重新推理时，可在脚本中修改 `FORCE_RECOMPUTE`。模型权重不随仓库发布：脚本会查找本地模型/缓存，缺少时尝试从 Hugging Face 获取 `allenai/specter2_base` 和 `allenai/specter2`（proximity adapter）。网络、镜像地址及离线模式设置以脚本为准。

### 4–6. 降维、聚类与可视化

```bash
python 4d_reduce.py
python 5cluster_leiden.py
python 6visualize_UMAP.py
```

这三个脚本不需要命令行参数，使用根目录下固定的输入输出路径。为保证实验一致性，`5cluster_leiden.py` 会检查上游论文索引、特征以及降维结果的一致性；不要手动打乱 `row_index` 对应的向量行序。

## 实验配置与现有结果

当前提交的是**三组已经选定的正式实验配置**，而不是重新搜索所有超参数的网格实验。

| 降维方法 | 维度 | 降维核心设置 | Leiden 图设置 | 簇数* | 降维空间 Silhouette* |
| --- | ---: | --- | --- | ---: | ---: |
| PCA | 20 | Full SVD | Mutual kNN，`k=10`，`γ=1.5` | 11 | 0.0933 |
| UMAP | 10 | `n_neighbors=20`，`min_dist=0`，cosine | Union kNN，`k=5`，`γ=1.5` | 14 | 0.3057 |
| LocalMAP | 10 | `n_neighbors=10` | Union kNN，`k=10`，`γ=1.0` | 10 | 0.5030 |

\* 簇数和 Silhouette 读取自仓库现有的 `cluster/sweep_all_metrics.csv`，为当前保存的数据/配置对应结果；并不代表不同配置间已控制所有变量。三组方法同时使用了**不同的图构建参数**，因此聚类差异不能完全归因于降维算法本身。Leiden 社区编号也**不是人工验证的主题标签**。

此外，`d_reduce/降维指标.csv` 提供近邻重合率与 Trustworthiness，`cluster/全部簇_主题分布.csv` 提供参考主题的探索性分布。这些指标适合分析与比较，但不应当作独立的论文分类准确率。

## 数据与复现注意事项

1. **原始数据未完整公开于此仓库。** `reference/PDF/`、`reference/FullText/` 和 `models/` 在 `.gitignore` 中；仓库虽然保留了中间语料、特征和正式实验结果，但不能仅凭现有文件完整重建所有上游步骤。
2. **研究规模固定。** 多处脚本校验 180 篇论文、唯一 PMID、连续 `row_index` 以及 `180 × 768` 特征矩阵；如果替换语料，不能只改输入文件，还需要调整脚本中的验证逻辑、配置及缓存。
3. **质量状态不等于完全人工验真。** 自动清理的 `ok` 表示通过规则，不代表全文、图表、表格和 OCR 均经人工逐项核对；`review` 项及修复记录需要结合原文解读。
4. **结果依赖版本与随机性。** 正式降维设置使用 `SEED=42`，但 LocalMAP 的内部随机性可能受 `pacmap` 版本影响。推荐参照 `requirements.txt` 及各阶段的配置 JSON 对照环境、参数和 SHA256。
5. **可视化与聚类分离。** `6visualize_UMAP.py` 使用 UMAP 将已保存的降维坐标再次投影到二维，仅用于作图；二维位置不参与第 5 步的聚类计算。部分图片中的中文文字可能需要系统安装合适的 CJK 字体。
6. **授权与引用。** 请遵守原始论文、下载平台以及所用预训练模型的版权和使用条款。本仓库当前未提供独立的 `LICENSE` 文件，不能据此默认全部内容采用某种开源许可证。

---

**Repository:** [`MOm0-000/bsihgidtaftuac1k`](https://github.com/MOm0-000/bsihgidtaftuac1k)  
**主要技术：** Python · PubMed · SPECTER2 · PCA · UMAP · LocalMAP · Leiden · Matplotlib
