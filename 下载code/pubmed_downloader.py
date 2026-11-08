#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PubMed 文献批量下载器（Ubuntu 修复版 V4，无需邮箱，本地优先）

从 zzx.xlsx 第一张工作表的 B 列读取 PMID，下载公开且允许自动访问的
PDF 和完整正文 HTML，按 PMID 命名并生成失败名单。

默认保存目录：/home/zxx/bigdata/exp1/reference

首次测试：
    python3 pubmed_downloader_ubuntu_v4.py --limit 5
全部处理：
    python3 pubmed_downloader_ubuntu_v4.py
指定 Excel：
    python3 pubmed_downloader_ubuntu_v4.py --excel /path/zzx.xlsx

注：无需邮箱/账号。HTML 优先由 PMC 官方 JATS XML/TXT 全文转换；不把摘要冒充全文。
    如需从已有 PDF 提取文字并生成 HTML，可另加 --pdf-to-html（并非原版网页）。
"""
from __future__ import annotations

import argparse
import csv
import html
import logging
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import quote, urlparse
from xml.etree import ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "reference"
S3 = 'https://pmc-oa-opendata.s3.amazonaws.com'
PUBMED_EFETCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
PMC_IDCONV = 'https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/'
OPENALEX = 'https://api.openalex.org/works/'
TOOL = 'PubmedUbuntuBatchDownloaderV4'
BIOC_FULLTEXT = 'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/'

REPORT_FIELDS = [
    '序号', 'PMID', '标题', 'PMCID', 'DOI',
    'PDF状态', 'HTML状态', 'PDF来源', 'HTML来源',
    '是否需要手动处理', '缺少文件', '失败原因',
    'PubMed链接', 'PMC链接', 'DOI链接', '开放获取链接'
]
FAIL_FIELDS = [
    '序号', 'PMID', '标题', '缺少文件', '失败原因',
    'PubMed链接', 'PMC链接', 'DOI链接', '开放获取链接'
]


def read_excel_pmids(path: Path) -> list[tuple[str, str]]:
    """只依赖 Python 标准库，不需要用户安装 openpyxl/pandas。

    读取 xlsx 的首张工作表：A 列=序号，B 列=PMID；自动去掉标题行和重复 PMID。
    """
    main_ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel_ns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    ns = {'m': main_ns}
    with zipfile.ZipFile(path) as z:
        book = ET.fromstring(z.read('xl/workbook.xml'))
        sheets = book.find(f'{{{main_ns}}}sheets')
        if sheets is None or len(sheets) == 0:
            raise ValueError('Excel 没有工作表')
        rid = sheets[0].attrib.get(f'{{{rel_ns}}}id')
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        target = next((e.attrib.get('Target') for e in rels
                       if e.attrib.get('Id') == rid), None)
        if not target:
            raise ValueError('无法定位 Excel 的首张工作表')
        if target.startswith('/'):
            member = target.lstrip('/')
        elif target.startswith('xl/'):
            member = target
        else:
            member = 'xl/' + target.lstrip('./')
        # 处理 Excel 中普通字符、共享字符串和行内字符串三种常见情况。
        strings: list[str] = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared_root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in shared_root.findall('m:si', ns):
                strings.append(''.join(t.text or '' for t in si.findall('.//m:t', ns)))
        root = ET.fromstring(z.read(member))
        result, seen = [], set()
        for row in root.findall('.//m:sheetData/m:row', ns):
            values = {}
            for c in row.findall('m:c', ns):
                match = re.match(r'[A-Z]+', c.get('r', ''))
                if not match or match.group() not in ('A', 'B'):
                    continue
                v = c.find('m:v', ns)
                raw = (v.text or '') if v is not None else ''
                if c.get('t') == 's' and raw:
                    raw = strings[int(raw)]
                elif c.get('t') == 'inlineStr':
                    raw = ''.join(t.text or '' for t in c.findall('.//m:t', ns))
                values[match.group()] = str(raw).strip()
            pmid = re.sub(r'\.0$', '', values.get('B', ''))
            if re.fullmatch(r'[0-9]{5,10}', pmid) and pmid not in seen:
                seen.add(pmid)
                result.append((values.get('A', ''), pmid))
    if not result:
        raise ValueError('没有找到 PMID：请检查第一张工作表 B 列是否为 PubMed 编号')
    return result


class WebClient:
    def __init__(self, interval: float):
        self.interval = max(interval, 0.4)
        self.last_request = 0.0
        self.session = requests.Session()
        retry = Retry(total=2, connect=2, read=2, backoff_factor=0.8,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=frozenset({'GET'}),
                      respect_retry_after_header=True)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        self.session.mount('https://', adapter)
        self.session.headers.update({
            'User-Agent': f'{TOOL}/4.0 (Ubuntu; public-OA-research)',
            'Accept': '*/*',
        })

    def get(self, url: str, **kwargs):
        pause = self.interval - (time.monotonic() - self.last_request)
        if pause > 0:
            time.sleep(pause)
        self.last_request = time.monotonic()
        return self.session.get(url, timeout=(10, 45), allow_redirects=True, **kwargs)

    def blob(self, url: str, max_mb: int = 60, **kwargs) -> bytes:
        with self.get(url, stream=True, **kwargs) as r:
            r.raise_for_status()
            size = bytearray()
            for chunk in r.iter_content(128 * 1024):
                if chunk:
                    size.extend(chunk)
                    if len(size) > max_mb * 1024 * 1024:
                        raise ValueError(f'文件超过 {max_mb} MB，已跳过')
            return bytes(size)

    def json(self, url: str, **kwargs) -> dict:
        result = self.blob(url, 5, **kwargs)
        import json
        return json.loads(result)


def plain(node: ET.Element | None) -> str:
    return ' '.join(''.join(node.itertext()).split()) if node is not None else ''


def tag(node: ET.Element) -> str:
    return node.tag.rsplit('}', 1)[-1]


def child(node: ET.Element | None, name: str) -> ET.Element | None:
    if node is None:
        return None
    return next((e for e in node if tag(e) == name), None)


def pubmed_metadata(client: WebClient, pmids: list[str]) -> dict[str, dict]:
    """分批取标题/DOI/PMCID；批量失败后仍生成可用的失败名单。"""
    data = {p: {'title': '', 'pmcid': '', 'doi': ''} for p in pmids}
    for i in range(0, len(pmids), 60):
        chunk = pmids[i:i + 60]
        try:
            xml = client.blob(PUBMED_EFETCH, 12, params={
                'db': 'pubmed', 'id': ','.join(chunk), 'retmode': 'xml',
                'tool': TOOL
            })
            root = ET.fromstring(xml)
            for article in root.findall('.//PubmedArticle'):
                pmid = (article.findtext('./MedlineCitation/PMID') or '').strip()
                if pmid not in data:
                    continue
                a = data[pmid]
                a['title'] = plain(article.find('./MedlineCitation/Article/ArticleTitle'))
                for ident in article.findall('./PubmedData/ArticleIdList/ArticleId'):
                    kind, value = ident.get('IdType', '').lower(), (ident.text or '').strip()
                    if kind == 'pmc' and re.fullmatch(r'PMC\d+', value):
                        a['pmcid'] = value
                    if kind == 'doi':
                        a['doi'] = value
                if not a['doi']:
                    for loc in article.findall('./MedlineCitation/Article/ELocationID'):
                        if loc.get('EIdType') == 'doi' and loc.text:
                            a['doi'] = loc.text.strip()
                            break
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            logging.warning('PubMed 元数据第 %s 批获取失败：%s', i // 60 + 1, exc)

    # 新版 PMC ID converter（原 /pmc/utils/idconv/ 已迁移）。
    for i in range(0, len(pmids), 100):
        chunk = pmids[i:i + 100]
        try:
            result = client.json(PMC_IDCONV, params={
                'ids': ','.join(chunk), 'idtype': 'pmid', 'format': 'json',
                'tool': TOOL
            })
            for item in result.get('records', []):
                pmid = str(item.get('pmid') or item.get('requested-id') or '')
                if pmid not in data:
                    continue
                if re.fullmatch(r'PMC\d+', str(item.get('pmcid', ''))):
                    data[pmid]['pmcid'] = str(item['pmcid'])
                if item.get('doi') and not data[pmid]['doi']:
                    data[pmid]['doi'] = str(item['doi'])
        except (requests.RequestException, ValueError) as exc:
            logging.warning('PMC ID 关联查询第 %s 批失败：%s', i // 100 + 1, exc)
    return data


def to_https(url: str) -> str:
    """新 PMC AWS 元数据含 s3:// URL，并可能带 ?md5=...。"""
    if not isinstance(url, str):
        return ''
    if url.startswith('s3://pmc-oa-opendata/'):
        return S3 + '/' + url[len('s3://pmc-oa-opendata/'):]
    if url.startswith(S3 + '/'):
        return url
    return ''  # 只接受官方 S3 来源，避免意外转跳到未知域名


def pmc_dataset_metadata(client: WebClient, pmcid: str) -> dict:
    """按 PMCID 列举所有可用版本；再获取各版本的官方 JSON 元数据。"""
    if not re.fullmatch(r'PMC\d+', pmcid):
        return {}
    xml = client.blob(S3 + '/', 2, params={
        'list-type': '2', 'prefix': pmcid + '.', 'delimiter': '/'
    })
    root = ET.fromstring(xml)
    versions = []
    for item in root.iter():
        if tag(item) == 'CommonPrefixes':
            prefix = child(item, 'Prefix')
            name = (prefix.text or '').rstrip('/') if prefix is not None else ''
            if re.fullmatch(re.escape(pmcid) + r'\.\d+', name):
                versions.append(name)
    candidates = []
    for name in versions:
        try:
            info = client.json(f'{S3}/metadata/{name}.json')
            if not isinstance(info, dict):
                continue
            if not any(info.get(key) for key in ('pdf_url', 'xml_url', 'text_url')):
                continue
            final = str(info.get('is_manuscript', '')).lower() not in ('yes', 'true', '1')
            candidates.append((int(final), int(bool(info.get('xml_url'))),
                               int(bool(info.get('pdf_url'))), info))
        except (requests.RequestException, ValueError) as exc:
            logging.info('%s 元数据获取失败：%s', name, exc)
    return max(candidates, key=lambda x: x[:3])[3] if candidates else {}


def valid_pdf(path: Path) -> bool:
    try:
        with path.open('rb') as f:
            return path.stat().st_size >= 700 and f.read(5) == b'%PDF-'
    except OSError:
        return False


def valid_html(path: Path) -> bool:
    try:
        if path.stat().st_size < 400:
            return False
        with path.open('r', encoding='utf-8') as f:
            start = f.read(3000).lower()
        return '<html' in start and '<body' in start
    except (OSError, UnicodeError):
        return False


def save_pdf(client: WebClient, url: str, target: Path) -> None:
    temp = target.with_name(target.name + '.part')
    try:
        temp.write_bytes(client.blob(url, max_mb=100))
        if not valid_pdf(temp):
            raise ValueError('返回内容不是有效 PDF，可能是登录页面或需手动点击 PDF 按钮')
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def inline(node: ET.Element) -> str:
    parts = [html.escape(node.text or '')]
    for sub in node:
        kind = tag(sub)
        inner = inline(sub)
        if kind in ('italic', 'emphasis'):
            inner = f'<em>{inner}</em>'
        elif kind in ('bold', 'strong'):
            inner = f'<strong>{inner}</strong>'
        elif kind in ('sup', 'sub'):
            inner = f'<{kind}>{inner}</{kind}>'
        parts.extend((inner, html.escape(sub.tail or '')))
    return ''.join(parts)


def render(node: ET.Element, level: int = 2) -> str:
    kind = tag(node)
    if kind == 'sec':
        title = child(node, 'title')
        size = min(level, 6)
        header = f'<h{size}>{inline(title)}</h{size}>' if title is not None else ''
        return '<section>' + header + ''.join(render(c, level + 1) for c in node if c is not title) + '</section>'
    if kind == 'p':
        return '<p>' + inline(node) + '</p>'
    if kind == 'list':
        ordered = node.get('list-type', '').startswith(('order', 'alpha', 'roman'))
        which = 'ol' if ordered else 'ul'
        return f'<{which}>' + ''.join(render(c, level) for c in node) + f'</{which}>'
    if kind == 'list-item':
        return '<li>' + ''.join(render(c, level) for c in node) + '</li>'
    if kind == 'fig':
        return '<figure><figcaption>' + html.escape(plain(node)) + '</figcaption>' \
            '<p>[图片请参阅原始 PDF 或 PMC 网页]</p></figure>'
    if kind == 'table-wrap':
        return '<section><h3>' + html.escape(plain(child(node, 'caption'))) + '</h3>' \
            + ''.join(render(c, level) for c in node if tag(c) == 'table') + '</section>'
    if kind in ('table', 'thead', 'tbody', 'tfoot', 'tr'):
        return f'<{kind}>' + ''.join(render(c, level) for c in node) + f'</{kind}>'
    if kind in ('td', 'th'):
        return f'<{kind}>' + inline(node) + f'</{kind}>'
    if kind == 'ref-list':
        return '<section><h2>References</h2><ol>' + ''.join(
            '<li>' + html.escape(plain(c)) + '</li>' for c in node if tag(c) == 'ref'
        ) + '</ol></section>'
    if kind in ('ack', 'app', 'app-group', 'boxed-text', 'supplementary-material'):
        return '<section>' + ''.join(render(c, level + 1) for c in node) + '</section>'
    if kind == 'title':
        return '<h3>' + inline(node) + '</h3>'
    if len(node):
        return ''.join(render(c, level) for c in node)
    return '<p>' + html.escape(plain(node)) + '</p>' if plain(node) else ''


STYLES = ('body{font:16px/1.65 system-ui,Arial,sans-serif;max-width:1000px;'
          'margin:30px auto;padding:0 22px;color:#222}h1{line-height:1.3}'
          'table{border-collapse:collapse;display:block;overflow-x:auto}'
          'td,th{border:1px solid #aaa;padding:5px}pre{white-space:pre-wrap}'
          'section{margin:18px 0}a{color:#1361a8}')


def html_document(pmid: str, title: str, source: str, content: str) -> str:
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<title>{html.escape(title or pmid)}</title><style>{STYLES}</style></head>'
            f'<body><h1>{html.escape(title or "PMID " + pmid)}</h1>'
            f'<p>PMID: {pmid} | 来源：{html.escape(source)} '
            f'| <a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/">PubMed</a></p>'
            f'<main>{content}</main></body></html>\n')


def from_xml(data: bytes, pmid: str, fallback_title: str) -> str:
    root = ET.fromstring(data)
    if tag(root) != 'article':
        root = next((e for e in root.iter() if tag(e) == 'article'), None)
        if root is None:
            raise ValueError('XML 缺少 article 节点')
    body = child(root, 'body')
    if body is None or len(plain(body)) < 200:
        raise ValueError('XML 没有完整正文 body（不会把摘要保存为全文）')
    front = child(root, 'front')
    meta = child(front, 'article-meta')
    title_group = child(meta, 'title-group')
    full_title = plain(child(title_group, 'article-title')) or fallback_title
    abstracts = [x for x in meta if tag(x) == 'abstract'] if meta is not None else []
    abstract_part = ''.join('<section><h2>Abstract</h2>' + ''.join(render(x) for x in ab) + '</section>'
                            for ab in abstracts)
    back = child(root, 'back')
    content = abstract_part + ''.join(render(x) for x in body)
    if back is not None:
        content += ''.join(render(x) for x in back)
    return html_document(pmid, full_title, 'PMC 官方 JATS XML 转 HTML', content)


def from_text(data: bytes, pmid: str, title: str) -> str:
    text = data.decode('utf-8-sig', errors='replace').strip()
    if len(text) < 800 or text.lstrip().lower().startswith(('<html', '<!doctype')):
        raise ValueError('官方 TXT 内容过短或不是全文')
    return html_document(pmid, title, 'PMC 官方全文 TXT 转 HTML',
                         '<pre>' + html.escape(text) + '</pre>')



def from_bioc_xml(data: bytes, pmid: str, title: str) -> str:
    """NCBI 官方 BioC PMC OA 全文 -> HTML。明确拒绝仅含摘要的响应。"""
    root = ET.fromstring(data)
    docs = [e for e in root.iter() if tag(e) == 'document']
    if not docs:
        raise ValueError('BioC 未返回可用 document（可能未开放全文）')

    parts = []
    body_lengths = 0
    body_paragraphs = 0
    last_section = ''
    for passage in docs[0].iter():
        if tag(passage) != 'passage':
            continue
        infons = {
            str(e.get('key', '')).lower(): (e.text or '').strip()
            for e in passage if tag(e) == 'infon'
        }
        section = (infons.get('section_type') or infons.get('type') or '').upper()
        text_node = child(passage, 'text')
        paragraph = plain(text_node)
        if not paragraph:
            continue
        if section in {'TITLE', 'ABSTRACT', 'ABSTRACT_TEXT', 'FRONT'} or \
                section.startswith('ABSTRACT'):
            continue
        if section != last_section and section and not section.startswith('REF'):
            parts.append(f'<h2>{html.escape(section.title())}</h2>')
            last_section = section
        parts.append(f'<p>{html.escape(paragraph)}</p>')
        if not (section.startswith('REF') or section in {'ACK', 'ACKNOWLEDGEMENTS'}):
            body_paragraphs += 1
            body_lengths += len(paragraph)
    if body_paragraphs < 3 or body_lengths < 800:
        raise ValueError('BioC 不包含足够的正文段落，不把摘要充当 Full Text')
    return html_document(pmid, title, 'NCBI PMC BioC 全文转 HTML', '\n'.join(parts))

def from_pdf(path: Path, pmid: str, title: str) -> str:
    """仅在显式使用 --pdf-to-html 时调用。"""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError('PDF 转 HTML 缺少 PyMuPDF：pip install -r requirements.txt') from exc
    with fitz.open(path) as pdf:
        pages = [p.get_text('text') for p in pdf]
    if len(''.join(pages).strip()) < 800:
        raise ValueError('PDF 正文字数太少，可能是扫描件或仅摘要')
    contents = ''.join(f'<h2>Page {i}</h2><pre>{html.escape(t)}</pre>'
                       for i, t in enumerate(pages, 1) if t.strip())
    return html_document(pmid, title, 'PDF 提取文字转 HTML（非原始网页版式）', contents)


def save_html(target: Path, contents: str) -> None:
    temp = target.with_name(target.name + '.part')
    try:
        temp.write_text(contents, encoding='utf-8')
        if not valid_html(temp):
            raise ValueError('保存的文件未通过 HTML 格式检查')
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def openalex_pdf_links(client: WebClient, doi: str) -> tuple[list[str], str]:
    """用无需邮箱的 OpenAlex 公共查询获取合法开放获取 PDF 链接。

    OpenAlex 返回的 PDF 链接由各机构/出版社托管，有些链接仍会拦截脚本访问；
    此时仅记录失败，不绕过登录、付费墙、验证码。
    """
    if not doi:
        return [], ''
    work_url = OPENALEX + 'https://doi.org/' + quote(doi, safe='/')
    response = client.json(work_url)
    locations = [response.get('best_oa_location')] + (response.get('locations') or [])
    candidates, landing = [], ''
    for loc in locations:
        if not isinstance(loc, dict) or not loc.get('is_oa'):
            continue
        landing = landing or str(loc.get('landing_page_url') or '')
        pdf_url = str(loc.get('pdf_url') or '')
        parsed = urlparse(pdf_url)
        domain = (parsed.hostname or '').lower()
        # PMC 官方数据集已单独处理，不把期刊 HTML 当 PDF。
        if parsed.scheme == 'https' and domain and not domain.endswith('ncbi.nlm.nih.gov') \
                and pdf_url not in candidates:
            candidates.append(pdf_url)
    return candidates[:4], landing


def article_row(client: WebClient, order: str, pmid: str, meta: dict,
                out: Path, pdf_to_html: bool) -> dict[str, str]:
    pdf_path, html_path = out / 'PDF' / (pmid + '.pdf'), out / 'FullText' / (pmid + '.html')
    pdf_ok, html_ok = valid_pdf(pdf_path), valid_html(html_path)
    pdf_src = '已存在' if pdf_ok else ''
    html_src = '已存在' if html_ok else ''
    pmcid = str(meta.get('pmcid', ''))
    doi = str(meta.get('doi', ''))
    title = str(meta.get('title', ''))
    problems, landing = [], ''

    if pmcid and not (pdf_ok and html_ok):
        try:
            info = pmc_dataset_metadata(client, pmcid)
            if info:
                if not title:
                    title = str(info.get('title') or '')
                if not doi:
                    doi = str(info.get('doi') or '')
                if not pdf_ok and (pdf_url := to_https(info.get('pdf_url', ''))):
                    try:
                        save_pdf(client, pdf_url, pdf_path)
                        pdf_ok, pdf_src = True, 'PMC 官方 PDF'
                    except (requests.RequestException, OSError, ValueError) as e:
                        problems.append(f'PMC PDF: {e}')
                if not html_ok:
                    for key, convert, desc in (
                        ('xml_url', from_xml, 'PMC XML 全文转 HTML'),
                        ('text_url', from_text, 'PMC TXT 全文转 HTML')
                    ):
                        url = to_https(info.get(key, ''))
                        if not url:
                            continue
                        try:
                            body = client.blob(url, max_mb=30)
                            save_html(html_path, convert(body, pmid, title))
                            html_ok, html_src = True, desc
                            break
                        except (requests.RequestException, ET.ParseError, OSError, ValueError) as e:
                            problems.append(f'{key}: {e}')
            else:
                problems.append('PMC 没有向批量下载接口开放该文章版本')
        except (requests.RequestException, ET.ParseError, ValueError) as e:
            problems.append(f'PMC 数据集访问失败: {e}')

    # 官方 BioC 接口是第二个全文来源，部分文章即使 S3 查询暂时失败仍可获取。
    # BioC-PubMed 返回的可能是摘要；这里专门调用 pmcoa 全文接口并检查正文。
    if not html_ok:
        try:
            body = client.blob(BIOC_FULLTEXT + pmid + '/unicode', max_mb=30)
            save_html(html_path, from_bioc_xml(body, pmid, title))
            html_ok, html_src = True, 'NCBI PMC BioC 全文转 HTML'
        except (requests.RequestException, ET.ParseError, OSError, ValueError) as e:
            problems.append(f'PMC BioC 全文：{e}')

    if not pdf_ok and doi:
        try:
            pdf_urls, landing = openalex_pdf_links(client, doi)
            for url in pdf_urls:
                try:
                    save_pdf(client, url, pdf_path)
                    pdf_ok, pdf_src = True, 'OpenAlex 收录的开放 PDF'
                    landing = landing or url
                    break
                except (requests.RequestException, OSError, ValueError) as e:
                    problems.append(f'开放 PDF 链接: {e}')
        except (requests.RequestException, ValueError) as e:
            problems.append(f'OpenAlex 查询失败: {e}')

    if pdf_ok and not html_ok and pdf_to_html:
        try:
            save_html(html_path, from_pdf(pdf_path, pmid, title))
            html_ok, html_src = True, 'PDF 提取文字生成 HTML（非原网页）'
        except (OSError, ValueError, RuntimeError) as e:
            problems.append(f'PDF 转 HTML: {e}')

    missing = '+'.join(x for x, status in [('PDF', pdf_ok), ('HTML', html_ok)] if not status)
    if not pdf_ok:
        problems.append('PDF 可能需进入期刊网页手动点击下载按钮')
    if not html_ok:
        problems.append('未获得完整正文 HTML（未用 PubMed 摘要充数）')
    if not meta.get('title') and not pmcid and not doi:
        problems.append('元数据不完整；请检查 PubMed API 网络连通性')
    return {
        '序号': order, 'PMID': pmid, '标题': title, 'PMCID': pmcid, 'DOI': doi,
        'PDF状态': '成功' if pdf_ok else '失败',
        'HTML状态': '成功' if html_ok else '失败',
        'PDF来源': pdf_src, 'HTML来源': html_src,
        '是否需要手动处理': '是' if missing else '否', '缺少文件': missing,
        '失败原因': '；'.join(problems),
        'PubMed链接': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
        'PMC链接': f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/' if pmcid else '',
        'DOI链接': 'https://doi.org/' + quote(doi, safe='/') if doi else '',
        '开放获取链接': landing,
    }


def safe_csv(value: object) -> str:
    text = str(value or '')
    return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) else text


def write_csv(target: Path, fields: list[str], records: list[dict]) -> None:
    with target.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for row in records:
            writer.writerow([safe_csv(row.get(name, '')) for name in fields])


def save_reports(out: Path, records: list[dict], total: int) -> list[str]:
    fails = [x for x in records if x['是否需要手动处理'] == '是']
    write_csv(out / '下载报告.csv', REPORT_FIELDS, records)
    write_csv(out / '下载失败名单.csv', FAIL_FIELDS, fails)
    good_pdf = sum(x['PDF状态'] == '成功' for x in records)
    good_html = sum(x['HTML状态'] == '成功' for x in records)
    both = len(records) - len(fails)
    messages = [
        f'Excel 要求总数：{total} 篇',
        f'本次已检查：{len(records)} 篇',
        f'尚未检查：{total - len(records)} 篇（--limit 测试或中断时会出现）',
        f'PDF 当前有效文件：{good_pdf} 篇（包含此前已下载或手动放入的文件）',
        f'HTML 当前有效文件：{good_html} 篇（包含此前已保存或手动放入的文件）',
        f'本次运行自动新下载 PDF：{sum(x["PDF来源"] not in ("", "已存在") for x in records)} 篇',
        f'本次运行自动新保存 HTML：{sum(x["HTML来源"] not in ("", "已存在") for x in records)} 篇',
        f'PDF+HTML 两种都齐全：{both} 篇',
        f'已检查但缺少 PDF：{len(records) - good_pdf} 篇',
        f'已检查但缺少 HTML：{len(records) - good_html} 篇',
        f'需要手动补文件：{len(fails)} 篇（同一篇缺两个文件只计一次）',
        f'保存目录：{out}',
    ]
    (out / '下载统计.txt').write_text('\n'.join(messages) + '\n', encoding='utf-8')
    lines = ['需要人工补充的文献', f'本次检查 {len(records)} 篇，需要处理 {len(fails)} 篇',
             f'尚未检查 {total - len(records)} 篇',
             '把手工下载的文件命名为 <PMID>.pdf 或 <PMID>.html 放到相应目录后可重新运行。', '']
    for index, item in enumerate(fails, 1):
        lines += [f'{index}. 序号 {item["序号"]} | PMID {item["PMID"]} | 缺少 {item["缺少文件"]}',
                  f'   题名：{item["标题"]}',
                  f'   PubMed：{item["PubMed链接"]}']
        for label, key in [('PMC', 'PMC链接'), ('出版社 DOI', 'DOI链接'), ('开放获取', '开放获取链接')]:
            if item.get(key):
                lines.append(f'   {label}：{item[key]}')
        if item['失败原因']:
            lines.append(f'   原因：{item["失败原因"]}')
        lines.append('')
    (out / '下载失败名单.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return messages


def previous_reports(out: Path) -> dict[str, dict[str, str]]:
    """读取上次记录，跳过已齐全文献时保留标题、DOI、PMCID 等信息；只读本地文件。"""
    report = out / '下载报告.csv'
    if not report.is_file():
        return {}
    try:
        with report.open('r', encoding='utf-8-sig', newline='') as f:
            return {row['PMID']: row for row in csv.DictReader(f) if row.get('PMID')}
    except (OSError, csv.Error, UnicodeError, KeyError) as exc:
        logging.warning('读取上次报告失败，将只按本地文件状态跳过：%s', exc)
        return {}


def already_complete_row(order: str, pmid: str, old: dict[str, str]) -> dict[str, str]:
    """PDF 与 HTML 都已有效：构造报告行，不调用任何网络 API。"""
    doi = old.get('DOI', '')
    pmcid = old.get('PMCID', '')
    return {
        '序号': order, 'PMID': pmid, '标题': old.get('标题', ''),
        'PMCID': pmcid, 'DOI': doi,
        'PDF状态': '成功', 'HTML状态': '成功',
        'PDF来源': '已存在', 'HTML来源': '已存在',
        '是否需要手动处理': '否', '缺少文件': '', '失败原因': '',
        'PubMed链接': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
        'PMC链接': f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/' if pmcid else '',
        'DOI链接': 'https://doi.org/' + quote(doi, safe='/') if doi else '',
        '开放获取链接': old.get('开放获取链接', ''),
    }


def excel_path(arg: str | None, out: Path) -> Path:
    if arg:
        path = Path(arg).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Excel 不存在：{path}')
        return path
    for path in (Path(__file__).resolve().parent / 'zzx.xlsx',
                 Path.cwd() / 'zzx.xlsx', out.parent / 'zzx.xlsx'):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError('找不到 zzx.xlsx；请把它放在脚本同目录，或使用 --excel 完整路径')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Ubuntu：按照 Excel PMID 下载 PDF 和全文 HTML')
    parser.add_argument('--excel', help='zzx.xlsx 的完整路径；默认在脚本目录及当前目录查找')
    parser.add_argument('--output', type=Path, default=DEFAULT_DIR,
                        help='文件保存根目录；默认为 /home/zxx/bigdata/exp1/reference')
    parser.add_argument('--limit', type=int, default=0, help='只处理前 N 篇，0 为全部')
    parser.add_argument('--pdf-to-html', action='store_true',
                        help='如无 PMC HTML，允许从 PDF 提取纯文字生成 HTML（不等于原始网页）')
    parser.add_argument('--delay', type=float, default=0.6, help='两次 HTTP 请求至少间隔多少秒')
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error('--limit 不能为负数')

    out = args.output.expanduser().resolve()
    try:
        excel = excel_path(args.excel, out)
        items = read_excel_pmids(excel)
        for directory in (out, out / 'PDF', out / 'FullText'):
            directory.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, KeyError, ET.ParseError, zipfile.BadZipFile) as e:
        parser.error(str(e))

    logging.basicConfig(filename=out / '运行日志.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8')
    total = len(items)
    chosen = items[:args.limit] if args.limit else items
    print(f'Excel：{excel}\n总共要求：{total} 篇；本次检查：{len(chosen)} 篇')
    print(f'存储目录：{out}\n', flush=True)
    # 核心优化：先只用本地文件系统检查完成情况，再决定哪些 PMID 需要联网。
    # 双格式都存在的文献：不查询 PubMed / PMC / OpenAlex，也不等待 --delay。
    complete = {
        pmid for _, pmid in chosen
        if valid_pdf(out / 'PDF' / f'{pmid}.pdf')
        and valid_html(out / 'FullText' / f'{pmid}.html')
    }
    pending_pmids = [pmid for _, pmid in chosen if pmid not in complete]
    print(f'本地 PDF+HTML 已齐全：{len(complete)} 篇（直接跳过联网）')
    print(f'仍缺文件、需要尝试联网：{len(pending_pmids)} 篇\n', flush=True)

    old_rows = previous_reports(out)
    client = WebClient(args.delay)
    # 只有缺文件的 PMID 才会进入批量元数据查询。
    metadata = pubmed_metadata(client, pending_pmids) if pending_pmids else {}
    rows = []
    try:
        for i, (order, pmid) in enumerate(chosen, 1):
            if pmid in complete:
                row = already_complete_row(order, pmid, old_rows.get(pmid, {}))
                logging.info('PMID %s PDF+HTML 均已存在，跳过全部网络请求', pmid)
            else:
                try:
                    row = article_row(client, order, pmid, metadata.get(pmid, {}), out, args.pdf_to_html)
                except Exception as e:  # 不让单篇论文出错中断整个批次
                    logging.exception('PMID %s 未知错误', pmid)
                    pdf_ok = valid_pdf(out / 'PDF' / f'{pmid}.pdf')
                    html_ok = valid_html(out / 'FullText' / f'{pmid}.html')
                    missing = '+'.join(x for x, state in [('PDF', pdf_ok), ('HTML', html_ok)] if not state)
                    row = {
                        '序号': order, 'PMID': pmid, '标题': metadata.get(pmid, {}).get('title', ''),
                        'PMCID': metadata.get(pmid, {}).get('pmcid', ''),
                        'DOI': metadata.get(pmid, {}).get('doi', ''),
                        'PDF状态': '成功' if pdf_ok else '失败', 'HTML状态': '成功' if html_ok else '失败',
                        'PDF来源': '已存在' if pdf_ok else '', 'HTML来源': '已存在' if html_ok else '',
                        '是否需要手动处理': '是' if missing else '否', '缺少文件': missing,
                        '失败原因': f'{type(e).__name__}: {e}',
                        'PubMed链接': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                        'PMC链接': '', 'DOI链接': '', '开放获取链接': ''
                    }
            rows.append(row)
            label = '本地齐全，跳过联网' if pmid in complete else '已检查'
            print(f'[{i:>3}/{len(chosen)}] PMID {pmid} | {label} | PDF {row["PDF状态"]} | '
                  f'HTML {row["HTML状态"]} | 缺少 {row["缺少文件"] or "无"}', flush=True)
            save_reports(out, rows, total)  # 每篇都写报告，中断也不会丢记录
    except KeyboardInterrupt:
        print('\n用户中断：已有文件及本次进度已保存。', flush=True)
        return 130
    summary = save_reports(out, rows, total)
    print('\n' + '\n'.join(summary))
    print('\n失败名单：' + str(out / '下载失败名单.txt'))
    print('CSV 报告：' + str(out / '下载报告.csv'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
