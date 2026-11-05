文献解析 + 特征提取（Ubuntu）
============================
把以下三个文件放到 /home/zxx/bigdata/exp1/：
  - extract_paper_features.py
  - requirements_features.txt
  - zzx.xlsx （你本来已有）
文章存放在：
  - /home/zxx/bigdata/exp1/reference/PDF/<PMID>.pdf
  - /home/zxx/bigdata/exp1/reference/FullText/<PMID>.html

建议在原有 bigdata 虚拟环境中操作：
  cd /home/zxx/bigdata/exp1
  python3 -m pip install -r requirements_features.txt

先不加载模型，验证 HTML/PDF 解析及 TF-IDF：
  python3 extract_paper_features.py --parse-only

正式生成 SPECTER2 向量：
  python3 extract_paper_features.py

首次正式运行需要联网从 Hugging Face 下载 SPECTER2 模型及 adapter。
PubMed 元数据也是联网获取；不需要邮箱/账号，失败时保留本地提取内容。
如果你只想从本地提取、不请求 PubMed：
  python3 extract_paper_features.py --no-pubmed --parse-only

结果在 /home/zxx/bigdata/exp1/reference/features/
  papers_metadata.csv      文献元数据、来源、质量标记（全体183篇）
  papers_text.jsonl        全文/摘要文字及来源（全体183篇）
  extraction_report.csv    提取是否成功/原因
  analysis_pmids.csv       特征矩阵行号 ↔ PMID 严格对应（仅可用篇目）
  tfidf_matrix.npz        稀疏 TF-IDF 特征矩阵
  tfidf_vocabulary.csv    TF-IDF 每列特征对应的关键词
  tfidf_top_terms.csv     每篇关键词及权重
  paper_embeddings.npy    SPECTER2 向量（仅正式运行成功时生成）
  embedding_model_info.json  嵌入模型信息
  pubmed_cache.json       PubMed 结构化元数据缓存（避免重复查询）
  summary.txt             总篇数、可用篇数和向量维度

注意：
* 任何特征矩阵的第 i 行都对应 analysis_pmids.csv 中的第 i 行。
* 无法确认的机构、作者、年份会留空，绝不会假造。
* 扫描 PDF 如果不含文本层，程序不会 OCR，需人工处理。
* SPECTER2 使用标题+摘要（无摘要使用正文短片段并标记），不把全文直接塞入模型。
* TF-IDF 使用清洗的全文，HTML 不可用时提取 PDF 文本。
* 多学科样本聚类可能首先按学科分类，不代表单一领域科研热点。
