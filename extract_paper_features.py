#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取本地 PMID 清单及 HTML/PDF，提取文献元数据、全文 TF-IDF 和 SPECTER2 向量。

修复：识别 PDF-only HTML 信息页并回退 PDF；去除 LaTeX 排版杂项；保留统计表。

默认路径：/home/zxx/bigdata/exp1
不下载论文，不要求邮箱/API Key；可按需使用 PubMed EFetch 补充元数据。
"""
import os
# 通过镜像站下载模型；如终端已指定 HF_ENDPOINT，则优先使用终端配置。
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '60')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')

import hashlib
import argparse
import csv
import json
import re
import sys
import time
import zipfile
from collections import Counter
from unicodedata import east_asian_width
from pathlib import Path
from xml.etree import ElementTree as ET

BASE = Path('/home/zxx/bigdata/exp1')
XLSX_NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
MODEL_NAME = 'allenai/specter2_base'
ADAPTER_NAME = 'allenai/specter2_classification'


def tidy(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip()


def xmltext(node):
    return tidy(' '.join(node.itertext())) if node is not None else ''


def csv_write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_excel_pmids(path):
    """用 Python 标准库解析 Excel 的 A 列序号/B 列 PMID，无需 openpyxl。"""
    if not path.is_file():
        raise FileNotFoundError(f'找不到 Excel：{path}')
    output = []
    with zipfile.ZipFile(path) as z:
        ss = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall(f'{XLSX_NS}si'):
                ss.append(''.join(t.text or '' for t in si.iter(f'{XLSX_NS}t')))
        candidates = sorted((x for x in z.namelist()
                             if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', x)),
                            key=lambda x: int(re.search(r'sheet(\d+)\.xml', x).group(1)))
        if not candidates:
            raise ValueError('Excel 中找不到工作表')
        root = ET.fromstring(z.read(candidates[0]))
        for row in root.findall(f'.//{XLSX_NS}sheetData/{XLSX_NS}row'):
            values = {}
            for cell in row.findall(f'{XLSX_NS}c'):
                letter = re.match(r'[A-Z]+', cell.get('r', '') or '')
                if not letter or letter.group(0) not in ('A', 'B'):
                    continue
                value = cell.findtext(f'{XLSX_NS}v')
                if cell.get('t') == 's' and value is not None:
                    value = ss[int(value)]
                if cell.get('t') == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.iter(f'{XLSX_NS}t'))
                values[letter.group(0)] = tidy(value)
            pmid = values.get('B', '').removesuffix('.0')
            seq = values.get('A', '').removesuffix('.0')
            if re.fullmatch(r'\d{5,10}', pmid):
                output.append({'sequence': seq, 'pmid': pmid})
    unique, seen = [], set()
    for item in output:
        if item['pmid'] not in seen:
            unique.append(item)
            seen.add(item['pmid'])
    if not unique:
        raise ValueError('Excel 的 B 列没有识别出 PMID，请检查表格')
    return unique


def choose_path(directory, pmid, suffix):
    if not directory.exists():
        return None
    direct = directory / f'{pmid}{suffix}'
    if direct.is_file() and direct.stat().st_size > 0:
        return direct
    return next((p for p in directory.glob(f'{pmid}.*')
                 if p.suffix.lower() == suffix and p.is_file() and p.stat().st_size > 0), None)


def html_meta(soup, key):
    for tag in soup.find_all('meta'):
        name = tidy(tag.get('name') or tag.get('property')).lower()
        if name == key.lower():
            content = tidy(tag.get('content'))
            if content:
                return content
    return ''


def html_metas(soup, key):
    values = []
    for tag in soup.find_all('meta'):
        if tidy(tag.get('name') or tag.get('property')).lower() == key.lower():
            value = tidy(tag.get('content'))
            if value and value not in values:
                values.append(value)
    return values



# HTML 有时只有 PMC 信息页，真正正文仅在 PDF 中；不能单凭文本长度判定全文。
PDF_ONLY_NOTICE = re.compile(
    r'(?:the\s+)?full\s+text\s+of\s+(?:this|the)\s+article\s+is\s+available\s+as\s+(?:a\s+)?pdf'
    r'|full\s+text\s+(?:is\s+)?available\s+(?:only\s+)?(?:as|in)\s+(?:a\s+)?pdf',
    flags=re.I,
)
# 删除 PMC HTML 里为数学公式附带的 LaTeX 排版预处理代码。
# 常见形态：\documentclass...\begin{document}$$公式$$\end{document}可读公式
# 仅删除短的内嵌 LaTeX 片段，不影响整篇文献，也尽量保留旁边已渲染的公式。
LATEX_LAYOUT_DOCUMENT = re.compile(
    r'\\documentclass[\s\S]{0,4500}?\\end\{document\}', flags=re.I
)
LATEX_LAYOUT_COMMAND = re.compile(
    r'\\(?:documentclass|usepackage|setlength|setcounter|pagestyle|thispagestyle|'
    r'DeclareFontFamily|DeclareFontShape|DeclareSymbolFont|DeclareMathAlphabet|'
    r'geometry|newcommand|renewcommand|begin\{document\}|end\{document\})'
    r'(?:\[[^\]\n]{0,250}\])?(?:\{[^{}\n]{0,250}\}){0,3}',
    flags=re.I,
)


def clean_scientific_text(text, source='HTML'):
    """保守清洗版面噪声；不删除有效学术数字、章节、引用的研究实体。"""
    text = str(text or '').replace('\xa0', ' ')
    text = LATEX_LAYOUT_DOCUMENT.sub(' ', text)
    text = LATEX_LAYOUT_COMMAND.sub(' ', text)
    # PDF 的自动换行可能把单词拆成两段（transmis-\nsible）。
    # 仅连接小写字母之间的行末断词，不改变正常行内连字符。
    if source == 'PDF':
        # 尽量不破坏非临床(non-clinical)等有意义的连字符。
        def join_broken_word(match):
            left, right = match.group(1), match.group(2)
            return left + right if (len(left) >= 4 and left.lower() not in
                                    {'anti', 'post', 'para', 'meta', 'intra', 'inter'}) else left + '-' + right
        text = re.sub(r'\b([a-zA-Z]+)-\s*\n\s*([a-z]+)\b', join_broken_word, text)
    # 单独一行的排版垃圾，保留正常图注、公式和数字型科研术语。
    lines = []
    for line in text.splitlines():
        line = tidy(line)
        if not line:
            continue
        if re.fullmatch(r'(?:documentclass|usepackage|setlength|begin\{document\}|end\{document\})',
                        line, flags=re.I):
            continue
        lines.append(line)
    return '\n'.join(lines)


def validate_html_fulltext(content, abstract='', page_text='', para_count=0):
    """启发式判定；对明确 PDF-only 的页面宁可回退，不伪称它有 HTML 正文。"""
    normalized = tidy(content)
    if PDF_ONLY_NOTICE.search(normalized) and len(normalized) < 3000:
        return False, 'HTML 仅提示全文在 PDF 中（PDF-only 信息页）'
    # 有些 PDF-only 提示在容器的其他节点；短页面且无足量文章段落才触发。
    if PDF_ONLY_NOTICE.search(page_text) and len(normalized) < 1200 and para_count < 4:
        return False, 'HTML 页面提示 PDF 全文，且缺少足够正文段落'
    if len(normalized) < 600:
        return False, f'HTML 正文过短（{len(normalized)} 字符）'
    if abstract and len(normalized) <= len(abstract) + 250:
        return False, 'HTML 内容长度接近摘要，可能仅有摘要'
    # 短文章可能合法存在，不按统一 1000 字符阈值直接拒绝。
    if len(normalized) < 1200 and para_count < 2 and re.search(
        r'(?i)\b(?:PMC Copyright notice|Author information|Article notes|Find articles by)\b', normalized
    ):
        return False, 'HTML 主要为文章元信息，缺少独立正文段落'
    return True, ''

def html_parse(path):
    from bs4 import BeautifulSoup
    # 浏览器保存的 HTML 可能带非 UTF-8 编码，BeautifulSoup 会识别
    soup = BeautifulSoup(path.read_bytes(), 'lxml')
    title = (html_meta(soup, 'citation_title') or html_meta(soup, 'dc.title'))
    if not title:
        h = soup.select_one('h1.content-title, h1.article-title, h1')
        title = tidy(h.get_text(' ', strip=True)) if h else ''
    if re.search(r'^(pubmed|pubmed central|ncbi|article|full text)', title, re.I):
        title = ''
    if not title and soup.title:
        raw_title = tidy(soup.title.get_text(' ', strip=True))
        title = re.split(r'\s+[|–-]\s+(?:PubMed|PMC|NCBI)', raw_title, flags=re.I)[0]
        if title.lower() in {'pubmed', 'pubmed central', 'ncbi'}:
            title = ''
    pubdate = (html_meta(soup, 'citation_date')
               or html_meta(soup, 'citation_publication_date')
               or html_meta(soup, 'citation_online_date')
               or html_meta(soup, 'dc.date'))
    year = re.search(r'(19\d{2}|20\d{2})', pubdate)
    doi = html_meta(soup, 'citation_doi')
    pmcid = html_meta(soup, 'citation_pmcid')
    authors = html_metas(soup, 'citation_author')
    institutions = html_metas(soup, 'citation_author_institution')
    keywords = html_metas(soup, 'citation_keywords')
    for tag in soup.select('.affiliation, .affiliations, [id^="aff-"], [id^="affiliation-"]')[:30]:
        val = tidy(tag.get_text(' ', strip=True))
        if len(val) > 12 and val not in institutions:
            institutions.append(val)
    abstract = html_meta(soup, 'citation_abstract')
    if not abstract:
        for element in soup.select('section.abstract, div.abstract, #abstract, #Abs1, .abstract-content, [id^="abstract-"]'):
            val = tidy(element.get_text(' ', strip=True))
            val = re.sub(r'^Abstract\s*[:.]?\s*', '', val, flags=re.I)
            if len(val) >= 90:
                abstract = val[:12000]
                break
    # 有些论文的摘要只有标题加若干兄弟段落
    if not abstract:
        for h in soup.find_all(re.compile(r'^h[1-6]$')):
            if tidy(h.get_text(' ', strip=True)).lower().strip(':') == 'abstract':
                parent = h.parent
                val = tidy(parent.get_text(' ', strip=True)) if parent else ''
                if 90 <= len(val) <= 14000:
                    abstract = re.sub(r'^Abstract\s*', '', val, flags=re.I)
                    break
    # 优先选择文章内容容器；过滤导航栏、参考文献、脚本等
    container = (soup.select_one('article') or
                 soup.select_one('#main-content .article') or
                 soup.select_one('.article-content') or
                 soup.select_one('#article-level-0-body') or
                 soup.select_one('main') or soup.body)
    content = ''
    if container:
        copied = BeautifulSoup(str(container), 'lxml')
        for tag in copied.select('script, style, nav, header, footer, aside, form, '
                                 '.ref-list, .references, .reference-list, '
                                 '#ref-list, #references, .article-nav, .supplementary-material, '
                                 '.article-tools, .fig, .table-wrap, figure, table'):
            tag.decompose()
        for section in list(copied.find_all(['section', 'div'])):
            ident = tidy(section.get('id', '')).lower()
            if ident in {'ref-list', 'references', 'ack', 'acknowledgments'} or ident.startswith('ref-list-'):
                section.decompose()
        # 为减少脚注、按钮污染，按段落读取；保留章节标题
        chunks = []
        for tag in copied.find_all(['p', 'h2', 'h3', 'h4', 'li']):
            if tag.find_parent(['p', 'li']):
                continue
            val = tidy(tag.get_text(' ', strip=True))
            if (tag.name in ('h2', 'h3', 'h4') and len(val) >= 3) or len(val) >= 35:
                chunks.append(val)
        content = '\n'.join(chunks)
        para_count = sum(1 for tag in copied.find_all('p')
                         if len(tidy(tag.get_text(' ', strip=True))) >= 35)
        # 如果没有标准 p 标签，退回容器纯文本，但不接受明显的导航页
        if len(content) < 300:
            content = tidy(copied.get_text(' ', strip=True))
    else:
        para_count = 0
    content = re.split(r'\n(?:References|Bibliography|Literature cited)\s*\n', content, maxsplit=1, flags=re.I)[0]
    content = clean_scientific_text(content, source='HTML')
    is_fulltext, html_warning = validate_html_fulltext(
        content, abstract=abstract,
        page_text=tidy(soup.get_text(' ', strip=True))[:30000],
        para_count=para_count,
    )
    return {
        'title': title, 'abstract': abstract, 'year': year.group(1) if year else '',
        'authors': authors, 'institutions': institutions,
        'doi': doi, 'pmcid': pmcid, 'keywords': keywords,
        'body': content if is_fulltext else '',
        'html_body_chars': len(content), 'html_fulltext_candidate': is_fulltext,
        'html_validation_note': html_warning,
    }


def pdf_parse(path):
    import fitz
    text_parts = []
    try:
        with fitz.open(path) as doc:
            if doc.is_encrypted:
                return '', 'PDF 加密，无法提取正文'
            for page in doc:
                text_parts.append(page.get_text('text', sort=True))
    except Exception as exc:
        return '', f'PDF 解析失败: {type(exc).__name__}: {exc}'
    body = clean_scientific_text('\n'.join(text_parts), source='PDF')
    if len(body) < 400:
        return body, 'PDF 文字太少，可能是扫描版（需要 OCR）'
    return body, ''


def pubmed_extract_xml(blob):
    root = ET.fromstring(blob)
    data = {}
    for article in root.findall('.//PubmedArticle'):
        pmid = tidy(article.findtext('./MedlineCitation/PMID'))
        if not pmid:
            continue
        a = article.find('./MedlineCitation/Article')
        if a is None:
            continue
        title = xmltext(a.find('ArticleTitle'))
        abstract = ' '.join(xmltext(n) for n in a.findall('./Abstract/AbstractText') if xmltext(n))
        date = (a.findtext('./Journal/JournalIssue/PubDate/Year') or
                a.findtext('./ArticleDate/Year') or
                a.findtext('./Journal/JournalIssue/PubDate/MedlineDate') or '')
        y = re.search(r'\b(19\d{2}|20\d{2})\b', date)
        authors, institutions = [], []
        for author in a.findall('./AuthorList/Author'):
            name = tidy(' '.join(filter(None, [author.findtext('ForeName'), author.findtext('LastName')])))
            if not name:
                name = tidy(author.findtext('CollectiveName'))
            if name and name not in authors:
                authors.append(name)
            for node in author.findall('./AffiliationInfo/Affiliation'):
                aff = xmltext(node)
                if aff and aff not in institutions:
                    institutions.append(aff)
        keyword_list = [xmltext(k) for k in article.findall('./MedlineCitation/KeywordList/Keyword')]
        ids = {t.get('IdType', ''): xmltext(t) for t in article.findall('./PubmedData/ArticleIdList/ArticleId')}
        data[pmid] = {'title': title, 'abstract': tidy(abstract), 'year': y.group(1) if y else '',
                      'authors': authors, 'institutions': institutions,
                      'doi': ids.get('doi', ''), 'pmcid': ids.get('pmc', ''),
                      'keywords': [s for s in keyword_list if s]}
    return data


def fetch_pubmed(pmids, cache_file):
    """NCBI EFetch 批量获取文献元数据，失败时保留本地解析结果。"""
    import requests
    cache = {}
    if cache_file.is_file():
        try:
            cache = json.loads(cache_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            print('  警告：PubMed 缓存不可读，重新请求缺失项', flush=True)
    pending = [p for p in pmids if p not in cache]
    if not pending:
        print(f'  PubMed 元数据已缓存 {len(cache)} 篇，跳过联网', flush=True)
        return cache
    session = requests.Session()
    session.headers.update({'User-Agent': 'AcademicLiteratureFeatureExtractor/1.0'})
    for idx in range(0, len(pending), 40):
        ids = pending[idx:idx + 40]
        try:
            response = session.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
                                   params={'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml'},
                                   timeout=(10, 35))
            response.raise_for_status()
            batch = pubmed_extract_xml(response.content)
            if not batch:
                raise ValueError('返回的 XML 未包含 PubmedArticle')
            cache.update(batch)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'  PubMed 批量查询 {min(idx + 40, len(pending))}/{len(pending)}：获取 {len(batch)} 篇', flush=True)
        except Exception as exc:
            print(f'  警告：本批 PMID 元数据补充失败：{type(exc).__name__}: {exc}', flush=True)
        time.sleep(0.4)  # 遵守 NCBI 无 API Key 时的基础速率限制
    return cache


def merge_record(base, offline, pubmed):
    row = {'sequence': base['sequence'], 'pmid': base['pmid']}
    for field in ('title', 'abstract', 'year', 'doi', 'pmcid'):
        row[field] = pubmed.get(field) or offline.get(field) or ''
        row[field + '_source'] = ('PubMed' if pubmed.get(field) else
                                  'HTML' if offline.get(field) else 'missing')
    for field in ('authors', 'institutions', 'keywords'):
        arr = pubmed.get(field) or offline.get(field) or []
        row[field] = list(dict.fromkeys(s for s in arr if s))
        row[field + '_source'] = 'PubMed' if pubmed.get(field) else ('HTML' if offline.get(field) else 'missing')
    return row


def build_tfidf(records, outdir):
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    from sklearn.preprocessing import normalize
    from scipy.sparse import save_npz
    import numpy as np
    # Citation artifacts cause large, spurious TF-IDF weights. Preserve source text.
    texts = []
    for rec in records:
        text = re.sub(r'\bet\s+al\.?(?=\W|$)', ' ', rec['tfidf_text'], flags=re.I)
        # Preserve IL-4 / IL-13 identities instead of collapsing both to 'il'.
        text = re.sub(r'\bIL[-\u2010\u2011\u2012\u2013\u2014](\d{1,3})\b',
                      lambda m: 'IL' + m.group(1), text, flags=re.I)
        texts.append(text)
    # 文本已预清洗；额外排除图注/排版常见噪声词，保留字母+数字组合术语。
    noise_terms = {'figure', 'figures', 'fig', 'table', 'tables', 'copyright',
                   'usepackage', 'setlength', 'documentclass', 'supplementary',
                   'supplemental', 'download', 'publisher'}
    academic_noise = {'et', 'al', 'using', 'different', 'analysis', 'study', 'studies',
                      'method', 'methods', 'result', 'results', 'paper', 'papers',
                      'article', 'articles'}
    # 'al' is also the chemical element Al; this choice favors citation cleanup.
    stops = sorted(set(ENGLISH_STOP_WORDS) | noise_terms | academic_noise)
    vec = TfidfVectorizer(lowercase=True, stop_words=stops,
                          token_pattern=r'(?u)\b[^\W\d_][\w]{1,}\b',
                          strip_accents='unicode', ngram_range=(1, 2),
                          min_df=2 if len(records) >= 5 else 1,
                          max_df=0.92 if len(records) >= 5 else 1.0,
                          max_features=12000, sublinear_tf=True)
    try:
        x = vec.fit_transform(texts)
    except ValueError as exc:
        print(f'  TF-IDF 的双词特征过滤过严，调整为 min_df=1：{exc}', flush=True)
        vec.set_params(min_df=1, max_df=1.0)
        x = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    # Drop artificial repeated bigrams, e.g. 'il il', while keeping 'il'.
    keep = np.asarray([not (len(parts := term.split()) == 2 and parts[0] == parts[1])
                       for term in terms], dtype=bool)
    if not keep.all():
        x = normalize(x[:, keep].tocsr(), norm='l2', axis=1, copy=False)
        terms = terms[keep]
    save_npz(outdir / 'tfidf_matrix.npz', x)
    df = (x > 0).sum(axis=0).A1
    csv_write(outdir / 'tfidf_vocabulary.csv',
              ({'index': i, 'term': term, 'document_frequency': int(df[i])}
               for i, term in enumerate(terms)), ['index', 'term', 'document_frequency'])
    keyword_rows = []
    for i, record in enumerate(records):
        ranked = x.getrow(i)
        order = ranked.data.argsort()[::-1][:15]
        best = [(str(terms[ranked.indices[k]]), round(float(ranked.data[k]), 6)) for k in order]
        keyword_rows.append({'pmid': record['pmid'], 'top_terms': '; '.join(t for t, _ in best),
                             'term_weights': '; '.join(f'{t}:{v}' for t, v in best),
                             'text_source': record['tfidf_source']})
    csv_write(outdir / 'tfidf_top_terms.csv', keyword_rows,
              ['pmid', 'top_terms', 'term_weights', 'text_source'])
    return x.shape


def build_specter2(records, outdir, batch_size=8, device_name='auto'):
    """以 Classification Adapter 生成语义向量，保持原有的 PMID 行序与文件名。"""
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else device_name
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('选择了 CUDA，但 PyTorch 未检测到 CUDA。请用 --device cpu')

    print(f'加载 SPECTER2 Classification Adapter，设备：{device}', flush=True)
    print(f'Hugging Face 地址：{os.environ.get("HF_ENDPOINT", "默认源")}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)
    adapter_name = model.load_adapter(
        ADAPTER_NAME, source='hf', load_as='classification', set_active=True
    )
    if 'classification' not in str(model.active_adapters).lower():
        raise RuntimeError(f'Classification Adapter 未激活：{model.active_adapters}')
    print(f'已激活 Adapter：{model.active_adapters}', flush=True)
    model = model.to(device)
    model.eval()

    sep = tokenizer.sep_token or '[SEP]'
    model_inputs = [r['title'] + sep + r['model_context'] for r in records]
    input_sha256 = hashlib.sha256('\n\0\n'.join(model_inputs).encode('utf-8')).hexdigest()
    vectors = []
    for start in range(0, len(model_inputs), batch_size):
        inputs = model_inputs[start:start + batch_size]
        tokens = tokenizer(inputs, padding=True, truncation=True, max_length=512,
                           return_tensors='pt', return_token_type_ids=False)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            output = model(**tokens)
            cls = output.last_hidden_state[:, 0, :].float().cpu().numpy()
        vectors.append(cls)
        print(f'  向量生成进度：{min(start + batch_size, len(records))}/{len(records)}', flush=True)

    arr = np.concatenate(vectors, axis=0).astype(np.float32)
    if arr.shape != (len(records), 768) or not np.isfinite(arr).all():
        raise ValueError(f'Classification 向量形状或数值异常：{arr.shape}')
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('检测到全零 Classification 向量')
    arr = (arr / norms).astype(np.float32)

    np.save(outdir / 'paper_embeddings.npy', arr)
    info = {
        'model': MODEL_NAME,
        'adapter': ADAPTER_NAME,
        'adapter_loaded_as': str(adapter_name),
        'active_adapters': str(model.active_adapters),
        'input': 'title [SEP] abstract (or flagged body fallback)',
        'input_sha256': input_sha256,
        'max_length_tokens': 512,
        'pooling': 'CLS',
        'L2_normalized': True,
        'rows': int(arr.shape[0]),
        'dimensions': int(arr.shape[1]),
        'device': device,
        'pmid_order': 'analysis_pmids.csv',
        'abstract_count': sum(r['model_input_source'] == 'abstract' for r in records),
        'body_fallback_count': sum(r['model_input_source'] == 'body_fallback_not_abstract'
                                   for r in records),
    }
    (outdir / 'embedding_model_info.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return arr.shape



def _visual_width(value):
    """计算中文终端列宽；不需要额外的格式化依赖。"""
    return sum(2 if east_asian_width(ch) in ('F', 'W') else 1 for ch in str(value))


def _pad_visual(value, width):
    s = str(value)
    return s + ' ' * max(1, width - _visual_width(s))


def make_quality_summary(rows, valid_rows, outdir, tfidf_shape=None, embeddings_shape=None):
    """仅使用本次解析数据，输出可审计的完整率表和可疑文献列表。"""
    total = len(rows)
    metrics = []

    def add(section, label, count, denominator=total, note=''):
        pct = f'{count / denominator * 100:.1f}%' if denominator else '—'
        metrics.append({'分类': section, '指标': label, '数量': count,
                        '比例': pct, '说明': note})

    def nonempty(field):
        return sum(bool(row.get(field)) for row in rows)

    def source_count(source):
        return sum(row.get('fulltext_source') == source for row in rows)

    add('基本情况', 'Excel 文献总数', total, note='读取 Excel 后按 PMID 去重')
    add('基本情况', '特征可用', len(valid_rows), note='仅满足脚本的最低可用条件')
    add('基本情况', '跳过', total - len(valid_rows))
    add('基本情况', '输出 PMID 重复数', total - len(set(row['pmid'] for row in rows)),
        note='仅检查解析输出；Excel 输入读取阶段已去重')
    for field, label in [('title', '标题非空'), ('abstract', '摘要非空'),
                         ('year', '发表年份非空'), ('authors', '作者非空'),
                         ('institutions', '机构非空'), ('keywords', '关键词非空'),
                         ('doi', 'DOI 非空'), ('pmcid', 'PMCID 非空')]:
        add('元数据', label, nonempty(field), note='字段非空不等于内容经人工核对')
    add('文本来源', '本地 HTML 文件存在', sum(bool(r['html_exists']) for r in rows))
    add('文本来源', '本地 PDF 文件存在', sum(bool(r['pdf_exists']) for r in rows))
    add('文本来源', '以 HTML 提取正文', source_count('HTML'))
    add('文本来源', '以 PDF 提取正文', source_count('PDF'))
    add('文本来源', '未提取到正文', source_count(''))
    add('正文初筛', 'HTML 判定非全文',
        sum(bool(r.get('html_validation_note')) for r in rows),
        note='此项记录 HTML 拒绝原因；可自动回退 PDF')
    add('正文初筛', 'HTML 无效且已回退 PDF',
        sum(bool(r.get('html_validation_note')) and r.get('fulltext_source') == 'PDF' for r in rows))
    add('模型输入', '使用正式摘要', sum(r.get('model_input_source') == 'abstract' for r in rows))
    add('模型输入', '正文开头替代摘要',
        sum(r.get('model_input_source') == 'body_fallback_not_abstract' for r in rows),
        note='不是正式摘要，应人工复核')
    add('正文初筛', '正文少于 1000 字符',
        sum(0 < int(r.get('fulltext_chars') or 0) < 1000 for r in rows),
        note='仅作为可疑短文本提醒，短文章可能正常')
    add('正文初筛', '正文超过 200000 字符',
        sum(int(r.get('fulltext_chars') or 0) > 200000 for r in rows),
        note='可能是会议合集或长综述，不一定错误')

    # 可疑条目的具体内容需要结合原始 PDF/HTML 人工核对。
    review_rows = []
    for row in rows:
        flags = []
        n = int(row.get('fulltext_chars') or 0)
        if row.get('html_validation_note'):
            flags.append('HTML 可疑或无效：' + row['html_validation_note'])
        if not row.get('fulltext_source'):
            flags.append('未提取到正文')
        elif n < 1000:
            flags.append('正文少于1000字符')
        elif n > 200000:
            flags.append('正文超过200000字符')
        if row.get('model_input_source') == 'body_fallback_not_abstract':
            flags.append('模型输入使用正文代替摘要')
        if not row.get('institutions'):
            flags.append('机构缺失')
        if not row.get('title'):
            flags.append('标题缺失')
        if flags:
            review_rows.append({'pmid': row['pmid'], 'title': row['title'],
                                'fulltext_source': row.get('fulltext_source', ''),
                                'fulltext_chars': n, 'review_flags': '；'.join(flags)})

    csv_write(outdir / 'data_quality_summary.csv', metrics,
              ['分类', '指标', '数量', '比例', '说明'])
    csv_write(outdir / 'quality_review_candidates.csv', review_rows,
              ['pmid', 'title', 'fulltext_source', 'fulltext_chars', 'review_flags'])
    # 保留终端可读中文对齐格式，并可直接作为报告草稿附件。
    w1, w2 = 27, 13
    lines = ['=' * 70, '数据完整性检查（本次实际解析结果）', '=' * 70,
             f'{_pad_visual("检查指标", w1)} {_pad_visual("数量", w2)} 比例', '-' * 70]
    previous = None
    for entry in metrics:
        if previous is not None and previous != entry['分类']:
            lines.append('-' * 70)
        lines.append(f'{_pad_visual(entry["指标"], w1)} '
                     f'{_pad_visual(entry["数量"], w2)} {entry["比例"]}')
        previous = entry['分类']
    lines.extend(['-' * 70,
                  f'TF-IDF 特征维度: {tfidf_shape if tfidf_shape is not None else "未生成"}',
                  f'SPECTER2 向量维度: {embeddings_shape if embeddings_shape is not None else "未生成（或生成失败）"}',
                  f'建议人工复核的文献: {len(review_rows)} 篇（可由多种原因同时触发）',
                  '注意：数量和完整率只反映字段非空/脚本判定，不代表全文已核实完整。',
                  '建议核查：quality_review_candidates.csv',
                  '=' * 70])
    table = '\n'.join(lines)
    (outdir / 'data_quality_summary.txt').write_text(table + '\n', encoding='utf-8')
    return table


def main():
    parser = argparse.ArgumentParser(description='183 篇文献：HTML/PDF 解析 + PubMed 元数据 + TF-IDF + SPECTER2')
    parser.add_argument('--excel', type=Path, default=BASE / 'zzx.xlsx')
    parser.add_argument('--reference', type=Path, default=BASE / 'reference')
    parser.add_argument('--out', type=Path, default=BASE / 'reference' / 'features')
    parser.add_argument('--no-pubmed', action='store_true', help='完全不请求 PubMed 元数据，仅使用本地 HTML/PDF')
    parser.add_argument('--parse-only', action='store_true', help='只解析并生成 TF-IDF，不下载/运行 SPECTER2')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size 必须为正整数')
    args.out.mkdir(parents=True, exist_ok=True)
    originals = read_excel_pmids(args.excel)
    print(f'Excel 要求的 PMID：{len(originals)} 篇', flush=True)
    print(f'数据目录：{args.reference}\n输出目录：{args.out}', flush=True)
    pmids = [x['pmid'] for x in originals]
    pubmed = {} if args.no_pubmed else fetch_pubmed(pmids, args.out / 'pubmed_cache.json')
    all_rows, valid_rows, texts = [], [], []
    for i, item in enumerate(originals, 1):
        pmid = item['pmid']
        html_path = choose_path(args.reference / 'FullText', pmid, '.html')
        pdf_path = choose_path(args.reference / 'PDF', pmid, '.pdf')
        offline, notes = {}, []
        if html_path:
            try:
                offline = html_parse(html_path)
                if not offline.get('html_fulltext_candidate'):
                    notes.append('HTML 未识别到可信全文：' +
                                 (offline.get('html_validation_note') or '正文完整性不足') +
                                 '；尝试 PDF')
            except Exception as exc:
                notes.append(f'HTML 解析失败: {type(exc).__name__}: {exc}')
        else:
            notes.append('HTML 不存在')
        body = offline.get('body', '')
        text_source = 'HTML' if body else ''
        if not body and pdf_path and offline.get('html_validation_note'):
            print(f'  ↳ PMID {pmid}: {offline["html_validation_note"]}，回退 PDF', flush=True)
        if not body and pdf_path:
            pdf_text, warning = pdf_parse(pdf_path)
            if warning:
                notes.append(warning)
            if len(pdf_text) >= 400:
                body = pdf_text
                text_source = 'PDF'
        if not body and not pdf_path:
            notes.append('PDF 不存在或无法提取正文')
        elif not body and pdf_path:
            notes.append('已尝试 PDF，但未提取到有效正文')
        r = merge_record(item, offline, pubmed.get(pmid, {}))
        r.update({'html_exists': bool(html_path), 'pdf_exists': bool(pdf_path),
                  'fulltext_source': text_source, 'fulltext_chars': len(body),
                  'html_body_chars': offline.get('html_body_chars', 0),
                  'html_validation_note': offline.get('html_validation_note', '')})
        # 记录来源；不能把正文片段冒充成原始摘要
        context = r['abstract'] if len(r['abstract']) >= 100 else ''
        model_src = 'abstract' if context else ''
        if not context and len(body) >= 400:
            context, model_src = body[:1500], 'body_fallback_not_abstract'
        r['model_input_source'] = model_src
        tfidf_text = body if len(body) >= 400 else (r['title'] + '\n' + r['abstract'])
        r['tfidf_source'] = text_source if len(body) >= 400 else 'title_abstract_fallback'
        # 如果提取到文本，但没有标题，不能伪造标题或生成论文级 SPECTER2 嵌入
        reason = ''
        if not r['title']:
            reason = '没有可靠的标题'
        elif len(context) < 100:
            reason = '没有摘要且无法从正文提取足够内容'
        elif len(tfidf_text.strip()) < 100:
            reason = '没有可供 TF-IDF 使用的文本'
        r['usable_for_features'] = not bool(reason)
        r['excluded_reason'] = reason
        r['notes'] = '；'.join(notes)
        all_rows.append(r)
        texts.append({'pmid': pmid, 'title': r['title'], 'abstract': r['abstract'],
                      'full_text': body, 'full_text_source': text_source,
                      'full_text_length': len(body), 'warning': r['notes']})
        if not reason:
            valid_rows.append(dict(r, model_context=context, tfidf_text=tfidf_text))
        print(f'[{i:03d}/{len(originals)}] PMID {pmid}: '
              f'HTML={"有" if html_path else "无"} PDF={"有" if pdf_path else "无"} '
              f'正文={len(body)}字({text_source or "无"}) '
              f'标题={"有" if r["title"] else "无"} '
              f'摘要={"有" if r["abstract"] else "无"} '
              f'特征={"可用" if not reason else "跳过"}', flush=True)
    cols = ['sequence', 'pmid', 'title', 'abstract', 'year', 'authors', 'institutions',
            'keywords', 'doi', 'pmcid', 'title_source', 'abstract_source', 'year_source',
            'authors_source', 'institutions_source', 'keywords_source',
            'html_exists', 'pdf_exists', 'fulltext_source', 'fulltext_chars',
            'html_body_chars', 'html_validation_note',
            'model_input_source', 'tfidf_source', 'usable_for_features',
            'excluded_reason', 'notes']
    csv_rows = []
    for r in all_rows:
        s = dict(r)
        for f in ('authors', 'institutions', 'keywords'):
            s[f] = ' | '.join(r[f])
        csv_rows.append(s)
    csv_write(args.out / 'papers_metadata.csv', csv_rows, cols)
    with (args.out / 'papers_text.jsonl').open('w', encoding='utf-8') as f:
        for item in texts:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    csv_write(args.out / 'extraction_report.csv',
              ({k: r[k] for k in ('pmid', 'html_exists', 'pdf_exists', 'fulltext_source',
                                  'fulltext_chars', 'html_validation_note',
                                  'model_input_source', 'usable_for_features',
                                  'excluded_reason', 'notes')} for r in all_rows),
              ['pmid', 'html_exists', 'pdf_exists', 'fulltext_source', 'fulltext_chars',
               'html_validation_note', 'model_input_source',
               'usable_for_features', 'excluded_reason', 'notes'])
    csv_write(args.out / 'analysis_pmids.csv',
              ({'row_index_0based': i, 'pmid': r['pmid'], 'sequence': r['sequence'],
                'input_source': r['model_input_source'], 'tfidf_source': r['tfidf_source']}
               for i, r in enumerate(valid_rows)),
              ['row_index_0based', 'pmid', 'sequence', 'input_source', 'tfidf_source'])
    counts = Counter(r['fulltext_source'] or '无' for r in all_rows)
    summary = [f'Excel 总篇数: {len(originals)}',
               f'特征可用篇数: {len(valid_rows)}',
               f'跳过篇数: {len(originals) - len(valid_rows)}',
               f'HTML 正文来源: {counts["HTML"]}',
               f'PDF 正文来源: {counts["PDF"]}',
               f'缺少可解析正文: {counts["无"]}',
               f'使用 PubMed 缓存/联网元数据: {"否" if args.no_pubmed else "是（请求失败会回退本地）"}',
               '说明: 发表年份/作者/机构按有证据的元数据填写；空值代表无法确认。',
               '说明: SPECTER2 主要使用标题+摘要；无摘要时可使用正文开头作为显式标注的替代输入。',
               '说明: analysis_pmids.csv 是所有特征矩阵的行顺序，不能按文件名另行排序。',
               '说明: HTML 若仅提供 PDF 全文提示，将回退 PDF；正文已过滤已知 LaTeX 排版噪声。']
    shape = None
    emb_shape = None
    embedding_failed = False
    if valid_rows:
        shape = build_tfidf(valid_rows, args.out)
        summary.append(f'TF-IDF 稀疏矩阵维度: {shape}')
        if not args.parse_only:
            # 仅在新向量成功生成后覆盖旧向量；失败时保留原备份并以非零状态退出。
            try:
                emb_shape = build_specter2(valid_rows, args.out, args.batch_size, args.device)
                summary.append(f'SPECTER2 Classification 语义向量维度: {emb_shape}')
            except Exception as exc:
                embedding_failed = True
                summary.append(f'SPECTER2 Classification 失败: {type(exc).__name__}: {exc}')
                print(f'\n错误：SPECTER2 Classification 未生成，原因：{type(exc).__name__}: {exc}', flush=True)
                print('元数据和 TF-IDF 已保存；现存向量（如有）属于以前的运行结果，不代表本次成功。', flush=True)
        else:
            summary.append('SPECTER2: --parse-only 已跳过（尚未生成 paper_embeddings.npy）')
    else:
        summary.append('没有文献能用于特征提取，请检查 HTML/PDF 文件内容和是否成功获取 PubMed 元数据。')
    summary_file = args.out / 'summary.txt'
    summary_file.write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print('\n======== 本次结果 ========\n' + '\n'.join(summary), flush=True)
    quality_table = make_quality_summary(all_rows, valid_rows, args.out,
                                         tfidf_shape=shape, embeddings_shape=emb_shape)
    with summary_file.open('a', encoding='utf-8') as f:
        f.write('\n' + quality_table + '\n')
    print('\n' + quality_table, flush=True)
    print(f'\n文件已保存：{args.out}', flush=True)
    if valid_rows and not args.parse_only and (embedding_failed or not (args.out / 'paper_embeddings.npy').is_file()):
        sys.exit(2)


if __name__ == '__main__':
    main()
