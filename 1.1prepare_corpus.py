#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文文本提取（不含TF-IDF/Embedding）。Python 3.10+。

固定输入：reference/FullText/*.html、reference/PDF/*.pdf
默认运行：python 1.1prepare_corpus.py
重跑正文：python 1.1prepare_corpus.py --force

首次运行自动生成 extracted/papers_metadata.csv（优先尝试PubMed，失败回退本地文件）。
此后保留该CSV中的人工修改；--force只重建extracted/corpus/。
不在提取阶段排除论文，后续清理阶段自行筛选。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
REF = ROOT / "reference"
OUT = ROOT / "extracted"
META = OUT / "papers_metadata.csv"
CORPUS = OUT / "corpus"
VERSION = "7.0-conservative-frontmatter-and-references"
MIN_BODY = 900
# 保留旧版全部实质性论文元数据字段，不再生成精简的10列CSV。
META_FIELDS = ("title", "abstract", "year", "authors", "institutions", "keywords", "doi", "pmcid")
META_COLS = ["paper_id", "pmid", *META_FIELDS,
             "title_source", "abstract_source", "year_source", "authors_source",
             "institutions_source", "keywords_source", "doi_source", "pmcid_source",
             "metadata_flags"]
LIST_FIELDS = {"authors", "institutions", "keywords"}
REPORT_COLS = ["paper_id", "status", "source", "body_chars", "has_abstract",
               "warnings", "cleanup_actions", "attempts", "html_file", "pdf_file"]
END_HEADINGS = {"references", "selected references", "bibliography", "literature cited", "works cited"}
ABSTRACT_HEADINGS = {"abstract", "summary"}


def tidy(value):
    s = unicodedata.normalize("NFKC", str(value or ""))
    s = s.replace("\ufeff", "").replace("\u00ad", "").replace("\x00", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def normalized(s):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(s or "")).lower())


def header(s):
    return re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", tidy(s).lstrip("# ").lower().rstrip(" :.;"))


def fake_abstract(s):
    s = tidy(s)
    return "图片请参阅原始" in s or bool(re.match(r"(?i)^images?\s+p?\[", s) and len(s) < 900)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def doc_id(path):
    """PMID.html / PMID_extra.html / PMID.pdf -> PMID。"""
    stem = path.stem.strip()
    match = re.match(r"^(\d{5,10})(?:[._-]|$)", stem)
    return match.group(1) if match else stem


def discover():
    groups = {}
    for folder, kind, extensions in ((REF / "FullText", "html", {".html", ".htm", ".shtml"}),
                                     (REF / "PDF", "pdf", {".pdf"})):
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到原始论文目录：{folder}")
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in extensions:
                pid = doc_id(p)
                if not re.fullmatch(r"[\w.\-]+", pid):
                    raise ValueError(f"原始文件名无法安全识别为ID：{p.name}")
                groups.setdefault(pid, {"html": [], "pdf": []})[kind].append(p)
    if not groups:
        raise ValueError("reference/FullText/ 和 reference/PDF/ 中均没有发现论文")
    duplicate = [(pid, kind) for pid, paths in groups.items()
                 for kind, files in paths.items() if len(files) != 1 and files]
    if duplicate:
        raise ValueError(f"同一论文有多个同类型文件，请先检查：{duplicate[:10]}")
    return dict(sorted(groups.items(), key=lambda v: (not v[0].isdigit(), int(v[0]) if v[0].isdigit() else v[0])))


def html_meta(soup, key, many=False):
    result = []
    for tag in soup.find_all('meta'):
        k = tidy(tag.get('name') or tag.get('property')).lower()
        if k == key.lower():
            val = tidy(tag.get('content'))
            if val and val not in result:
                result.append(val)
    return result if many else (result[0] if result else '')


def xmltext(node):
    return tidy(' '.join(node.itertext())) if node is not None else ''


def pubmed_xml_parse(blob):
    root = ET.fromstring(blob)
    out = {}
    for article in root.findall('.//PubmedArticle'):
        pmid = tidy(article.findtext('./MedlineCitation/PMID'))
        a = article.find('./MedlineCitation/Article')
        if not pmid or a is None:
            continue
        title = xmltext(a.find('ArticleTitle'))
        abstract = tidy(' '.join(xmltext(x) for x in a.findall('./Abstract/AbstractText')))
        date = (a.findtext('./Journal/JournalIssue/PubDate/Year')
                or a.findtext('./ArticleDate/Year')
                or a.findtext('./Journal/JournalIssue/PubDate/MedlineDate') or '')
        match = re.search(r'\b(19\d{2}|20\d{2})\b', date)
        authors, institutions = [], []
        for author in a.findall('./AuthorList/Author'):
            name = tidy(' '.join(filter(None, [author.findtext('ForeName'), author.findtext('LastName')])))
            name = name or tidy(author.findtext('CollectiveName'))
            if name and name not in authors:
                authors.append(name)
            for el in author.findall('./AffiliationInfo/Affiliation'):
                aff = xmltext(el)
                if aff and aff not in institutions:
                    institutions.append(aff)
        keywords = [xmltext(x) for x in article.findall('./MedlineCitation/KeywordList/Keyword')]
        ids = {x.get('IdType', ''): xmltext(x) for x in article.findall('./PubmedData/ArticleIdList/ArticleId')}
        out[pmid] = {'title': title, 'abstract': abstract, 'year': match.group(1) if match else '',
                     'authors': authors, 'institutions': institutions,
                     'doi': ids.get('doi', ''), 'pmcid': ids.get('pmc', ''),
                     'keywords': [k for k in keywords if k]}
    return out


def old_html_metadata(path):
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(path.read_bytes(),'lxml')
    title=(html_meta(soup,'citation_title') or html_meta(soup,'dc.title'))
    if not title:
        tag=soup.select_one('h1')
        if tag:title=tidy(tag.get_text(' ',strip=True))
    if not title and soup.title:title=tidy(soup.title.get_text(' ',strip=True))
    ab=html_meta(soup,'citation_abstract')
    if not ab:
        el=soup.select_one('.abstract-content, section.abstract, div.abstract, #abstract, #Abs1')
        if el:
            text=tidy(el.get_text(' ',strip=True))
            if len(text)>80: ab=re.sub(r'^Abstract\s*[:.]?\s*','',text,flags=re.I)
    pubdate=(html_meta(soup,'citation_date') or html_meta(soup,'citation_publication_date') or
             html_meta(soup,'citation_online_date') or html_meta(soup,'dc.date'))
    year=re.search(r'\b(19\d\d|20\d\d)\b',pubdate)
    return {'title':title,'abstract':ab,'year':year.group(1) if year else '',
            'authors':html_meta(soup,'citation_author',many=True),
            'institutions':html_meta(soup,'citation_author_institution',many=True),
            'keywords':html_meta(soup,'citation_keywords',many=True),
            'doi':html_meta(soup,'citation_doi'),'pmcid':html_meta(soup,'citation_pmcid')}



def html_metadata_focused(path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_bytes(), "lxml")

    def meta(*keys):
        wanted = {k.lower() for k in keys}
        for tag in soup.find_all("meta"):
            if tidy(tag.get("name") or tag.get("property")).lower() in wanted:
                value = tidy(tag.get("content"))
                if value:
                    return value
        return ""

    title = meta("citation_title", "dc.title", "dc:title")
    if not title:
        h1 = soup.select_one("article h1, main h1, h1")
        page = tidy(soup.title.get_text(" ", strip=True)) if soup.title else ""
        options = (tidy(h1.get_text(" ", strip=True)) if h1 else "", page)
        title = next((s for s in options if 10 <= len(s) <= 450 and
                      header(s) not in {"abstract", "summary", "full text", "pubmed", "pmc"}), "")
    title = re.sub(r"\s*[-–|]\s*(?:PMC|PubMed|NCBI)\s*$", "", title, flags=re.I).rstrip(" *†‡")
    if header(title) in {"abstract", "summary", "full text"}:
        title = ""
    abstract = meta("citation_abstract")
    if not abstract:
        # 有些PMC页面先展示Significance / Simple Summary，再展示真正的Abstract；
        # 不能无条件取第一个section.abstract。
        candidates = []
        for h in soup.find_all(["h2", "h3"]):
            if header(h.get_text(" ", strip=True)) != "abstract":
                continue
            section = h.parent
            if section.name not in {"section", "div"}:
                continue
            text = tidy(section.get_text(" ", strip=True))
            text = re.sub(r"^abstract\s*[:.]?\s*", "", text, flags=re.I)
            if not 90 <= len(text) <= 8000:
                continue
            # 容器若以Significance/Simple Summary开头，常是文章导读而非摘要。
            misleading = bool(re.match(r"(?i)^(significance|simple summary)\b", text))
            candidates.append((misleading, text))
        if candidates:
            abstract = min(candidates, key=lambda pair: pair[0])[1]
        else:
            section = soup.select_one(".abstract-content, #abstract, #Abs1, section.abstract, div.abstract")
            if section is not None:
                abstract = re.sub(r"^(?:abstract|summary)\s*[:.]?\s*", "",
                                  tidy(section.get_text(" ", strip=True)), flags=re.I)
    if not 90 <= len(abstract) <= 8000 or fake_abstract(abstract):
        abstract = ""
    date = meta("citation_date", "citation_publication_date", "citation_online_date", "dc.date")
    year = re.search(r"\b(?:19|20)\d{2}\b", date)
    return {"title": title, "abstract": abstract, "year": year.group() if year else "",
            "doi": meta("citation_doi"), "pmcid": meta("citation_pmcid"),
            "title_source": "HTML" if title else "missing",
            "abstract_source": "HTML" if abstract else "missing"}


def html_metadata(path):
    """旧版完整元数据字段 + 已修复的PMC摘要/标题识别。"""
    old = old_html_metadata(path)
    focused = html_metadata_focused(path)
    # 新版只替换成功识别的值，不能用空字符串擦掉旧版已经识别的字段。
    for field in ("title", "abstract", "year", "doi", "pmcid"):
        if focused.get(field):
            old[field] = focused[field]
    old["title_source"] = "HTML" if old.get("title") else "missing"
    old["abstract_source"] = "HTML" if old.get("abstract") else "missing"
    if fake_abstract(old.get("abstract")):
        old["abstract"] = ""
        old["abstract_source"] = "missing"
    return old


def pdf_metadata(path):
    """无HTML时或HTML字段不完整时，从PDF提取低置信度的元数据。"""
    import pymupdf
    with pymupdf.open(path) as pdf:
        title = tidy((pdf.metadata or {}).get("title"))
        if not 12 <= len(title) <= 300 or re.match(r"(?i)^(untitled|microsoft word|scan|document|article|pdf)", title):
            title = ""
        abstract = ""
        if pdf:
            first = tidy(pdf[0].get_text("text", sort=True))[:7000]
            match = re.search(r"(?im)^\s*(?:abstract|summary)\s*[:.]?\s*$", first)
            if match:
                next_text = first[match.end():match.end() + 4000]
                end = re.search(r"(?im)^\s*(?:introduction|keywords?|background|methods?)\s*[:.]?\s*$", next_text)
                abstract = tidy(next_text[:end.start()] if end else next_text)
                if not 90 <= len(abstract) <= 3500 or fake_abstract(abstract):
                    abstract = ""
    return {"title": title, "abstract": abstract,
            "title_source": "PDF_unverified" if title else "missing",
            "abstract_source": "PDF_unverified" if abstract else "missing"}


def pubmed_metadata(ids):
    """按旧版 XML 解析器获取完整八类字段，网络故障时回退本地。"""
    valid = [v for v in ids if re.fullmatch(r"\d{5,10}", v)]
    result = {}
    for pos in range(0, len(valid), 40):
        batch = valid[pos:pos + 40]
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urlencode(
            {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"})
        try:
            with urlopen(Request(url, headers={"User-Agent": "PaperCorpusExtractor/5.0 (academic research)"}),
                         timeout=8) as reply:
                records = pubmed_xml_parse(reply.read())  # 直接沿用旧代码
                result.update(records)
        except Exception as exc:
            print(f"  PubMed不可用（{type(exc).__name__}），回退本地HTML/PDF；缺失项标记复核。")
            break
        print(f"  PubMed：本批{len(batch)}篇，返回{sum(p in result for p in batch)}篇", flush=True)
    return result


def metadata_for(pid, paths, pubmed, existing=None):
    """填空不覆盖：旧CSV的人工修订永远优先，新增字段使用PubMed优先。"""
    row = dict.fromkeys(META_COLS, "")
    if existing:
        row.update(existing)
    row["paper_id"] = row.get("paper_id") or pid
    row["pmid"] = row.get("pmid") or (pid if pid.isdigit() else "")
    flags = []
    local = {}
    if paths["html"]:
        try:
            local = html_metadata(paths["html"][0])
        except Exception as exc:
            flags.append(f"html_metadata_error:{type(exc).__name__}")
    if paths["pdf"] and (not local.get("title") or not local.get("abstract")):
        try:
            pdf = pdf_metadata(paths["pdf"][0])
            for field in ("title", "abstract"):
                if not local.get(field) and pdf.get(field):
                    local[field] = pdf[field]
                    local[field + "_source"] = pdf[field + "_source"]
        except Exception as exc:
            flags.append(f"pdf_metadata_error:{type(exc).__name__}")
    for field in META_FIELDS:
        if tidy(row.get(field)):
            continue  # 绝不覆盖已有/人工修订的单元格
        for source, data in (("PubMed", pubmed), ("HTML", local)):
            value = data.get(field)
            if isinstance(value, list):
                value = " | ".join(dict.fromkeys(tidy(v) for v in value if tidy(v)))
            value = tidy(value)
            if field == "abstract" and fake_abstract(value):
                value = ""
            if value:
                row[field] = value
                explicit_source = data.get(field + "_source")
                row[field + "_source"] = explicit_source or source
                break
    for field in ("title", "abstract"):
        if not tidy(row.get(field)):
            flags.append("missing_" + field)
    if not tidy(row.get("year")):
        flags.append("missing_year")
    row["metadata_flags"] = ";".join(dict.fromkeys(filter(None, [tidy(row.get("metadata_flags")), *flags])))
    return row


def load_metadata():
    if not META.exists():
        # 对已有项目的一次性兼容：迁移旧文件名，但不修改其中任何单元格。
        legacy = OUT / "papers_metadata_final.csv"
        if legacy.exists():
            legacy.rename(META)
            print("将 papers_metadata_final.csv 更名为 papers_metadata.csv（内容未修改）。")
    elif (OUT / "papers_metadata_final.csv").exists():
        raise ValueError("extracted/ 同时存在两个元数据CSV，请人工确认哪一个是定稿文件")
    if not META.exists():
        return {}, META_COLS
    with META.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or META_COLS
        rows = list(reader)
    data = {}
    for row in rows:
        pid = tidy(row.get("paper_id") or row.get("pmid"))
        if not pid or pid in data:
            raise ValueError(f"元数据CSV有空ID或重复ID：{pid}")
        data[pid] = row
    return data, columns


def save_metadata(rows, columns):
    temp = META.with_suffix(".csv.tmp")
    write_csv(temp, rows, list(dict.fromkeys(columns + META_COLS)))
    temp.replace(META)


def strip_html_frontmatter(text):
    """只剔除HTML首部有明确作者信息栏标记的出版信息；保留Highlights/Key Messages。"""
    parts = re.split(r"\n\s*\n", tidy(text))
    first = parts[:25]
    labels = {"author information", "article notes", "copyright and license information",
              "author details", "author affiliations"}
    markers = [i for i, part in enumerate(first) if header(part) in labels]
    if not markers or sum(len(x) + 2 for x in first[:markers[0]]) > 1400:
        return text, []
    # 页首可能先有研究亮点/摘要，不能见到 Author information 就全部截断。
    prefix = first[:markers[0]]
    if any(len(x) > 230 or header(x) in {"highlights", "key messages", "abstract", "summary",
                                              "significance", "simple summary"} for x in prefix):
        return text, ["possible_leading_metadata_review"]

    publication = re.compile(
        r"(?i)^(?:author information|article notes|copyright and license information|"
        r"author details|author affiliations|received\b|accepted\b|published\b|"
        r"issue date\b|collection date\b|all rights reserved\b|copyright\b|"
        r"©|e-?mail\b|address(?:es)? correspondence\b|correspond(?:ing|ence)\b|"
        r"contributions?\s*:|these authors\b|the authors declare\b|"
        r"financial support\s*:|present address\s*:|co-first authors\b|"
        r"one contribution to a special feature\b|electronic supplementary material\b|"
        r"published by\b|authors[’']? addresses\s*:|\*?klinik und poliklinik\b)"
    )
    # 作者、标题短行只有在有上述明确出版信息标记时才允许删去。
    end = markers[0] + 1
    for i in range(end, min(len(parts), 45)):
        val = parts[i].strip()
        if header(val) in labels or publication.search(val) or ("@" in val and len(val) < 1000):
            end = i + 1
            continue
        break
    removed = sum(len(x) + 2 for x in parts[:end])
    if removed > 3300 or removed > len(text) * .30 or len(text) - removed < MIN_BODY:
        return text, ["possible_leading_metadata_review"]
    return tidy("\n\n".join(parts[end:])), ["removed_html_frontmatter"]


def normalize_pdf_markup(text):
    """仅删除Markdown/HTML排版标记，保留上标、下标、表格及链接的可见文字。"""
    import html
    text = html.unescape(text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"</?(?:sup|sub|i|b|em|strong|span)[^>]*>", "", text, flags=re.I)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*\|?\s*:?-{3,}(?:\s*\|\s*:?-{3,})*\s*\|?\s*$", "", text)
    text = re.sub(r"(?m)^\s*_[0-9]{1,4}_\s*$", "", text)
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = text.replace("**", "").replace("__", "")
    # 在整行单独的Markdown强调也保留标签文字本身。
    text = re.sub(r"(?m)^(\s*)_([^_\n]{3,100})_(\s*)$", r"\1\2\3", text)
    return tidy(text)


def trim_reference_tail(text):
    """有独立/行尾标题且尾部具有引用证据时删除；无标题则要求连续编号+年份。"""
    heading = re.compile(
        r"(?i)^\s*(?:#{1,6}\s*)?[_*`]*\s*"
        r"(?:references|selected references|bibliography|literature cited|works cited)"
        r"\s*[_*`]*\s*:?\s*$"
    )
    citation_no = re.compile(r"(?m)^\s*(?:[-*]\s*)?(?:\[\d{1,3}\]|\d{1,3}[.)]?)[.\s]+[A-Za-z]")
    years = re.compile(r"\b(?:18|19|20)\d{2}\b")

    def evidence(tail, require_numbered=False):
        head = tail[:6500]
        count = len(citation_no.findall(head))
        year_count = len(years.findall(head))
        if require_numbered:
            return count >= 6 and year_count >= 4
        return (count >= 3 and year_count >= 2) or (year_count >= 7 and len(tail) >= 450)

    offset, match_pos = 0, []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if heading.fullmatch(stripped):
            match_pos.append(offset)
        # 某些老PDF的REFERENCES紧接最后一个正文句子，仍要求大写且在行尾。
        elif m := re.search(r"\bREFERENCES\s*$", stripped):
            if m.group().strip() == "REFERENCES" and len(stripped[:m.start()].strip()) >= 50:
                match_pos.append(offset + line.find("REFERENCES"))
        offset += len(line)
    for pos in reversed(match_pos):
        if pos < max(1500, len(text) * .48):
            continue  # Bibliography可能本身就是论文正文主题。
        tail = text[pos:]
        # 科学意见/附录文档的中途参考文献后面可能还有实质正文。
        later = [header(line) for line in tail.splitlines()[1:]]
        if any(h in {"introduction", "background", "methods", "results", "discussion",
                     "conclusions", "purpose of appendix", "appendix", "annex"}
               or re.match(r"^(?:appendix|annex)\s+[a-z0-9]", h) for h in later):
            return text, ["ambiguous_reference_heading_kept"]
        if evidence(tail):
            return tidy(text[:pos]), ["removed_references"]
        return text, ["ambiguous_reference_heading_kept"]

    # 仅在没有明显参考文献标题时识别连续编号的尾部书目。
    # 需要 1..6 连续编号、多个年份，而且起始处是典型引文而非章标题。
    if not match_pos:
        entries = list(re.finditer(r"(?m)^\s*(?:[-*]\s*)?1[.)]\s+[A-Z][^\n]{20,}", text))
        for m in entries:
            if m.start() < len(text) * .55:
                continue
            tail = text[m.start():]
            nums = {int(x) for x in re.findall(r"(?m)^\s*(?:[-*]\s*)?(\d{1,2})[.)]\s+[A-Z]", tail[:9000])}
            if set(range(1, 7)).issubset(nums) and evidence(tail, require_numbered=True):
                return tidy(text[:m.start()]), ["removed_numbered_references"]
    return text, []


def trim_body(text, title, abstract, metadata=None):
    """只清理可信的前置信息和已验证的尾部参考文献；不对正文做全局替换。"""
    text = tidy(text)
    flags = []
    if not text:
        return "", flags
    lines = text.splitlines()

    def full_title(line):
        # 只处理独立标题行，避免删掉正文中自然出现的标题。
        val = re.sub(r"\s*[-–|]\s*(?:PMC|PubMed|NCBI)\s*$", "", line, flags=re.I)
        return len(normalized(title)) >= 14 and normalized(val) == normalized(title)

    for i in range(min(25, len(lines))):
        if full_title(lines[i]):
            lines[i] = ""
            flags.append("removed_title")

    # 仅处理紧邻文首、单独成行且与CSV相符的字段；正文中的相同词句必须保留。
    meta = metadata or {}
    doi = normalized(meta.get("doi", ""))
    pmcid = normalized(meta.get("pmcid", ""))
    affiliations = [normalized(x) for x in re.split(r"\s*\|\s*", tidy(meta.get("institutions", ""))) if len(normalized(x)) > 18]
    for i in range(min(35, len(lines))):
        line = lines[i].strip()
        if not line or len(line) > 350:
            continue
        norm = normalized(line)
        if ((doi and norm in {doi, "doi" + doi, "httpsdoiorg" + doi}) or
            (pmcid and norm in {pmcid, "pmcid" + pmcid}) or
            (norm in affiliations) or
            re.match(r"(?i)^(?:keywords?|key words|article info|citation|published|received|accepted|copyright|correspondence|corresponding author|author information)\s*[:：]", line)):
            lines[i] = ""
            flags.append("removed_leading_metadata")
    text = tidy("\n".join(lines))
    lines = text.splitlines()

    # HTML/PDF摘要往往有明确的 Abstract -> Introduction 边界。
    # 只在摘要块位于文首且切除比例合理时使用边界；不按孤立的 Methods/Results 标题截断。
    abstract_i = next((i for i, v in enumerate(lines[:120]) if header(v) in ABSTRACT_HEADINGS), None)
    intro_i = next((i for i, v in enumerate(lines[:180]) if header(v) == "introduction"), None)
    if abstract_i is not None and intro_i is not None and abstract_i < intro_i:
        prefix = "\n".join(lines[:intro_i])
        if len(prefix) < 12000 and len(prefix) < max(1500, len(text) * .45):
            # 保留摘要块之前可能已有的正文（例如历史评论序言），只切明确摘要段。
            lines = lines[:abstract_i] + lines[intro_i:]
            text = tidy("\n".join(lines))
            flags.append("removed_abstract_front")

    # 精确匹配完整摘要，允许HTML把结构化摘要分成Background/Methods/Results小节。
    # 匹配时忽略独立的小节标题，但必须完整匹配CSV摘要正文才自动删除。
    a = normalized(abstract)
    if 150 <= len(a) <= 10000 and text:
        abstract_subheads = {"abstract", "summary", "background", "objective", "objectives",
                             "method", "methods", "results", "result", "conclusion", "conclusions",
                             "purpose", "aim", "aims", "design", "setting", "settings", "findings",
                             "interpretation", "importance", "patients", "participants",
                             "intervention", "interventions", "main outcome measures", "significance"}
        norm_chars, offsets = [], []
        pos = 0
        for line in text.splitlines(keepends=True):
            # 这里仅为匹配摘要忽略标题；并不会真的从正文删掉这些小节标题。
            if header(line) not in abstract_subheads:
                for j, ch in enumerate(line):
                    for n in unicodedata.normalize("NFKD", ch).lower():
                        if "a" <= n <= "z" or "0" <= n <= "9":
                            norm_chars.append(n)
                            offsets.append(pos + j)
            pos += len(line)
            if pos > 12000:
                break
        at = "".join(norm_chars).find(a)
        if at >= 0:
            start, end = offsets[at], offsets[at + len(a) - 1] + 1
            # 确认完整摘要位于文首并且后面还有足够的真实正文。
            if start < 1200 and end < len(text) * .55 and len(text) - end >= MIN_BODY:
                while end < len(text) and text[end] in ".:; \t\n":
                    end += 1
                prefix = text[:start]
                # 移除摘要前紧邻的独立标题（Abstract / Background等），保留其他前置信息。
                prefix_lines = prefix.splitlines()
                if prefix_lines and header(prefix_lines[-1]) in abstract_subheads:
                    prefix = "\n".join(prefix_lines[:-1])
                text = tidy(prefix + "\n" + text[end:])
                flags.append("removed_exact_leading_abstract")
            else:
                flags.append("possible_abstract_duplicate")

    # 参考文献边界以标题/条目/年份共同验证；无法确认时不删除。
    text, reference_actions = trim_reference_tail(text)
    flags.extend(reference_actions)
    if abstract and len(a) >= 120 and a[:90] in normalized(text[:8000]):
        flags.append("possible_abstract_duplicate")
    if title and len(normalized(title)) >= 18 and normalized(title)[:60] in normalized(text[:1800]):
        flags.append("possible_title_duplicate")
    return text, list(dict.fromkeys(flags))

# 从旧版移植的章节结构识别常量。
HEAD_RE = re.compile(r'^\s*(#{1,6})\s+(.+?)\s*#*\s*$')
REF_HEADING = re.compile(r'^(references|bibliography|literature cited|works cited)$', re.I)
NONRESEARCH_HEAD = re.compile(r'^(acknowledg(e)?ments?|author contributions?|funding( information)?|competing interests?|conflicts? of interest|data availability( statement)?|supplementary (materials?|information)|supporting information|publisher.s note|ethics (approval|statement)|author information)$', re.I)
ABSTRACT_HEAD = re.compile(r'^(abstract|summary)$', re.I)
BODY_HEAD = re.compile(r'^(introduction|background|materials? and methods?|methods?|results?|discussion|conclusions?|results? and discussion)$', re.I)
SCIENTIFIC_HEADER = re.compile(r'\b(19\d\d|20\d\d)\b|\bdoi\b|\bet al\.?\b', re.I)

def norm_match(s):
    return normalized(s)


def plain_text(s):
    s = unicodedata.normalize('NFKC', str(s or '')).replace('\ufeff', '').replace('\u00ad', '')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n[ \t]*\n[ \t]*\n+', '\n\n', s)
    return s.strip()


def tag_name(tag):
    return str(tag).split('}')[-1].lower()


def children(node, name):
    return [x for x in node if tag_name(x.tag) == name]


def first_child(node, name):
    return next((x for x in node if tag_name(x.tag) == name), None)


def elem_text(node):
    return tidy(' '.join(node.itertext())) if node is not None else ''


def mksec(heading, text, level=2, kind=None):
    return {'heading': tidy(heading), 'type': kind or classify(heading),
            'text': plain_text(text), 'level': int(level)}


def classify(heading, parent='other'):
    s = re.sub(r'^\s*\d+(?:\.\d+)*\s*[-.)]?\s*', '', tidy(heading)).lower().rstrip(':')
    if REF_HEADING.fullmatch(s): return 'references'
    if ABSTRACT_HEAD.fullmatch(s): return 'abstract'
    if NONRESEARCH_HEAD.fullmatch(s): return 'nonresearch'
    if s in ('introduction','literature review'): return 'introduction'
    if s == 'background': return 'background'
    if 'method' in s or 'experimental procedure' in s: return 'methods'
    if 'result' in s: return 'results'
    if 'discussion' in s: return 'discussion'
    if 'conclu' in s: return 'conclusion'
    return parent


def render_body(sections):
    parts=[]
    for s in sections:
        t=plain_text(s.get('text',''))
        if not t: continue
        h=tidy(s.get('heading',''))
        # Keep readable titles; don't emit exact 'References' as article content
        # because the existing BGE-M3 split_sections stops at that marker.
        if classify(h)=='references' and s.get('type')!='references':
            h='Bibliographic discussion'
        parts.append((h+'\n' if h else '')+t)
    return '\n\n'.join(parts).strip()


def likely_reference_section(text):
    """Conservative filter: only a *heading* plus citation-like entries.
    An isolated References column name in a table must never truncate body.
    """
    lines=[tidy(s) for s in plain_text(text).splitlines() if tidy(s)][:35]
    if len(lines)<3: return False
    numbered=sum(bool(re.match(r'^(?:\[\d+\]|\d+[.)]\s+)\s*[A-Z]', s)) for s in lines)
    citation=sum(bool(SCIENTIFIC_HEADER.search(s)) and len(s)>25 for s in lines)
    return numbered>=3 or citation>=4


def prune_sections(sections, abstract='', source=''):
    """Never globally chop on a bare References string. Only trim sections.
    A suspicious boundary is kept and flagged instead of deleting article body.
    """
    out=[]; notes=[]
    full=render_body(sections)
    raw_count=len(full)
    skip_nested_abstract_level=None
    # Determine whether deleting the Abstract heading and its descendants
    # would discard an implausibly large fraction of the article.
    for i,s in enumerate(sections):
        kind=s.get('type','other')
        head=s.get('heading','')
        lvl=int(s.get('level',2))
        if skip_nested_abstract_level is not None:
            if lvl>skip_nested_abstract_level:
                notes.append('removed_abstract_subsection')
                continue
            skip_nested_abstract_level=None
        if kind=='abstract':
            end=i+1
            while end<len(sections) and int(sections[end].get('level',2))>lvl:
                end+=1
            candidate_size=sum(len(t.get('text','')) for t in sections[i:end])
            # Use AND: a broad Abstract ancestor must NEVER wipe all body text.
            if candidate_size<12000 and candidate_size<raw_count*.40:
                notes.append('removed_abstract_section')
                skip_nested_abstract_level=lvl
                continue
            notes.append('ambiguous_abstract_kept')
            s=dict(s);s['type']='other'
        if kind=='nonresearch':
            notes.append('removed_nonresearch_section');continue
        if kind=='references':
            # A bibliography may itself be the scholarly object (e.g. PMID
            # 14450716), or a table column. Prefer retaining doubtful content.
            citation_like=likely_reference_section(s.get('text',''))
            is_bibliography=bool(re.match(r'^\s*bibliography\s*$',head,re.I))
            is_last=i==len(sections)-1
            if citation_like and is_last and not (source=='PDF' and is_bibliography):
                notes.append('removed_verified_references');continue
            else:
                notes.append('ambiguous_reference_heading_kept')
                s=dict(s);s['type']='other'
        out.append(s)
    # Some HTML extractors include abstract as plain text without a separate
    # heading. Only remove an exact leading copy, NEVER search globally.
    if abstract and out:
        a=norm_match(abstract)
        first=out[0]
        t=plain_text(first.get('text',''))
        lead=norm_match(t[:min(9000,len(abstract)*3)])
        if len(a)>150 and a[:min(100,len(a))] in lead[:max(len(a)*2,400)]:
            # Can't robustly map normalized spans back to original punctuation.
            # Flag for review rather than risk deleting Methods / Results.
            notes.append('possible_inline_abstract_duplicate')
    if raw_count>4000 and len(render_body(out)) < raw_count*.25:
        notes.append('large_prune_ratio_review')
    return out,notes


def sections_from_markdown(markdown):
    """Use document parser's heading markers, preserving all unclassified text."""
    sections=[]; h=''; level=0; lines=[]; parent_kind='other'; parents=[]
    def flush():
        if lines:
            body=plain_text('\n'.join(lines))
            if body: sections.append(mksec(h,body,level or 2,parent_kind))
        lines.clear()
    for line in plain_text(markdown).splitlines():
        m=HEAD_RE.match(line)
        if m:
            flush();level=len(m.group(1));h=m.group(2)
            while parents and parents[-1][0]>=level: parents.pop()
            inherited=parents[-1][1] if parents else 'other'
            parent_kind=classify(h,inherited)
            parents.append((level,parent_kind))
        else:
            lines.append(line)
    flush()
    return sections


def sections_from_trafilatura_xml(blob):
    """Convert Trafilatura's structural XML into ordered headed sections."""
    root=ET.fromstring(blob)
    main=next((e for e in root.iter() if tag_name(e.tag)=='main'),None)
    root=main if main is not None else root
    seq=[]
    def walk(el):
        tag=tag_name(el.tag)
        if tag in {'head','heading','h1','h2','h3','h4','h5','h6'}:
            t=elem_text(el)
            if t:
                lvl=int(tag[-1]) if re.fullmatch('h[1-6]',tag) else 2
                rend=str(el.attrib.get('rend',''))
                m=re.search(r'h([1-6])',rend)
                if m: lvl=int(m.group(1))
                seq.append((True,lvl,t))
            return
        if tag in {'p','item','li','quote'}:
            t=elem_text(el)
            if t: seq.append((False,0,t))
            return
        for ch in el: walk(ch)
    walk(root)
    if not seq: return []
    blocks=[]
    for hd,l,t in seq:
        blocks.append(('#'*l+' ' if hd else '')+t)
    return sections_from_markdown('\n\n'.join(blocks))


def parse_jats(blob):
    """JATS is authoritative about <front>/<body>/<back>, unlike CSS rules."""
    root=ET.fromstring(blob)
    article=next((x for x in root.iter() if tag_name(x.tag)=='article'),root)
    front=first_child(article,'front'); body=first_child(article,'body')
    if body is None: raise ValueError('JATS XML has no <body>')
    abstract=''
    if front is not None:
        ab=next((e for e in front.iter() if tag_name(e.tag)=='abstract'),None)
        abstract=elem_text(ab)
    sections=[]
    def append_non_section(blocks,parent_heading,kind,level):
        ps=[]
        for node in blocks:
            nm=tag_name(node.tag)
            if nm in {'sec','ref-list','ack','fn-group','supplementary-material','floats-group','table-wrap'}:
                continue
            if nm in {'p','list','disp-quote','def-list','statement','boxed-text'}:
                value=elem_text(node)
                if value: ps.append(value)
            elif nm=='fig':
                cap=next((e for e in node.iter() if tag_name(e.tag)=='caption'),None)
                if cap is not None: ps.append(elem_text(cap))
        if ps: sections.append(mksec(parent_heading,'\n'.join(ps),level,kind))
    def visit(items,heading='',kind='other',level=2):
        # Preserve original ordering within a section; don't duplicate nested secs.
        buf=[]
        def flush():
            if buf: append_non_section(buf,heading,kind,level);buf.clear()
        for n in items:
            t=tag_name(n.tag)
            if t=='sec':
                flush()
                title=elem_text(first_child(n,'title'))
                subkind=classify(title,kind)
                visit([x for x in n if tag_name(x.tag)!='title'],title,subkind,level+1)
            else: buf.append(n)
        flush()
    visit(list(body))
    if not sections: raise ValueError('JATS body contains no extractable paragraphs')
    kept,notes=prune_sections(sections,abstract,'JATS')
    return {'sections':kept,'raw_text':render_body(sections),'abstract':abstract,
            'notes':notes,'parser':'jats'}


def parse_bioc(blob):
    data=json.loads(blob)
    docs=data.get('documents') or data.get('document') or []
    if isinstance(docs,dict): docs=[docs]
    if not docs: raise ValueError('BioC has no documents')
    passages=docs[0].get('passages',[])
    sections=[]; abstract=[]; heading='';kind='other'
    for p in passages:
        inf=p.get('infons') or {}
        typ=tidy(inf.get('type') or inf.get('Type') or '').lower()
        sec=tidy(inf.get('section_type') or inf.get('section') or '')
        txt=plain_text(p.get('text',''))
        if not txt: continue
        label=(sec or typ).upper().replace('-', '_')
        if label in ('ABSTRACT','ABSTR') or 'ABSTRACT' in label:
            abstract.append(txt);continue
        excluded={'REF','REFERENCES','ACK','ACK_FUND','FUND','SUPPL','SUPPLEMENTARY',
                  'TABLE','FIG','FIGURE','CAPTION','AFFILIATION','AUTHORS','TITLE',
                  'AUTHOR','LICENSE','COPYRIGHT','FRONT'}
        if label in excluded or any(label.startswith(z+'_') for z in excluded):
            continue
        if typ in ('title','article-title','article_title'):continue
        if typ in ('front','keyword','journal'):continue
        if typ in ('section','section_title','heading'):
            heading=txt;kind=classify(txt);continue
        bioctypes={'INTRO':'introduction','INTRODUCTION':'introduction',
                   'METHOD':'methods','METHODS':'methods','METH':'methods',
                   'RESULT':'results','RESULTS':'results',
                   'DISCUSS':'discussion','DISCUSSION':'discussion',
                   'CONCL':'conclusion','CONCLUSION':'conclusion',
                   'BACKGROUND':'background','BODY':'other'}
        known=bioctypes.get(label,classify(sec) if sec else 'other')
        k=known if known!='other' else kind
        if k in ('references','nonresearch','abstract'):continue
        sections.append(mksec(heading if heading else sec,txt,2,k))
    if not sections: raise ValueError('BioC returned no body passages')
    kept,notes=prune_sections(sections,' '.join(abstract),'BioC')
    return {'sections':kept,'raw_text':render_body(sections),
            'abstract':' '.join(abstract),'notes':notes,'parser':'bioc'}


class PMCFulltext:
    """Cache successes, prefer Europe PMC JATS, then NCBI BioC; never cache 404.
    Connection failures trigger a session-wide circuit breaker to avoid 183 timeouts.
    """
    def __init__(self, cache_dir, enabled=True):
        self.enabled=enabled;self.cache=Path(cache_dir);self.cache.mkdir(parents=True,exist_ok=True)
        self.failures=0;self.network_disabled=False
        self.sess=None

    def get(self,pmid,pmcid):
        if not self.enabled:return None,[]
        keys=[('jats',self.cache/f'{pmid}.jats.xml'),('bioc',self.cache/f'{pmid}.bioc.json')]
        for parser,path in keys:
            if path.is_file() and path.stat().st_size>300:
                try:
                    b=path.read_bytes()
                    obj=parse_jats(b) if parser=='jats' else parse_bioc(b)
                    if len(render_body(obj['sections']))>=400:
                        return obj,['PMC_local_cache']
                except Exception as e:
                    print(f'  缓存不可解析 {path.name}: {e}',flush=True)
        if self.network_disabled:return None,['PMC_network_disabled_after_errors']
        if self.sess is None:
            import requests
            self.sess=requests.Session()
            self.sess.headers.update({'User-Agent':'PaperFulltextResearchExtractor/4.0 (academic use)',
                                      'Accept':'application/xml, application/json, */*'})
        urls=[]
        if pmcid:
            pid=pmcid.upper().replace(' ','')
            if not pid.startswith('PMC'):pid='PMC'+pid
            urls.append(('jats',f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pid}/fullTextXML'))
            urls.append(('bioc',f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pid}/unicode'))
        else:
            urls.append(('bioc',f'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmid}/unicode'))
        notes=[]
        for parser,url in urls:
            try:
                res=self.sess.get(url,timeout=(5,15))
                if res.status_code in (400,403,404,410):continue
                res.raise_for_status()
                blob=res.content
                obj=parse_jats(blob) if parser=='jats' else parse_bioc(blob)
                if len(render_body(obj['sections'])) < 400:continue
                dst=self.cache/f'{pmid}.{ "jats.xml" if parser=="jats" else "bioc.json"}'
                tmp=dst.with_suffix(dst.suffix+'.tmp')
                tmp.write_bytes(blob);os.replace(tmp,dst)
                self.failures=0
                time.sleep(.35)
                return obj,['PMC_network_success']
            except Exception as exc:
                notes.append(f'{parser}_network_or_parse_error:{type(exc).__name__}')
                # Only connectivity issues, not parse failures, disable future requests.
                if 'requests' in type(exc).__module__:
                    self.failures+=1
                    if self.failures>=2:
                        self.network_disabled=True
                        notes.append('PMC_network_disabled_after_errors')
                        break
            time.sleep(.35)
        return None,notes


def parse_html_trafilatura(path,abstract=''):
    import trafilatura
    html=path.read_bytes()
    # No custom DOM deletion: past regressions deleted 20 full articles.
    xml=trafilatura.extract(html,output_format='xml',favor_recall=True,
                            include_comments=False,include_tables=False)
    sections=[];notes=[]
    if xml:
        try:sections=sections_from_trafilatura_xml(xml)
        except Exception as e:notes.append(f'trafilatura_xml_parse:{type(e).__name__}')
    if not sections:
        md=trafilatura.extract(html,output_format='markdown',favor_recall=True,
                               include_comments=False,include_tables=False)
        if md:sections=sections_from_markdown(md)
    if not sections:raise ValueError('Trafilatura extracted no article body')
    raw_html_text=render_body(sections)
    sections,more=prune_sections(sections,abstract,'HTML');notes+=more
    body=render_body(sections)
    if len(body)<400:raise ValueError(f'Trafilatura content too short ({len(body)} characters)')
    return {'sections':sections,'raw_text':raw_html_text,
            'notes':notes,'parser':'trafilatura'}


def jats_body(path):
    doc = parse_jats(path.read_bytes())  # 复用旧版JATS解析，front/body/back结构分离
    return render_body(doc["sections"]), doc["notes"]


def bioc_body(path):
    doc = parse_bioc(path.read_bytes())  # 复用旧版BioC解析
    return render_body(doc["sections"]), doc["notes"]


def html_body(path, title, abstract):
    from bs4 import BeautifulSoup, Tag
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    main = soup.select_one("article, main, #main-content") or soup.body or soup
    root = BeautifulSoup(str(main), "lxml")
    flags = []
    for node in list(root.select("script,style,nav,aside,footer,form,noscript,svg")):
        node.decompose()
    total = len(root.get_text(" ", strip=True))
    for node in list(root.select("section,div")):
        if node.parent is None:
            continue
        ident = " ".join(node.get("class", [])) + " " + str(node.get("id") or "")
        h = node.find(["h1", "h2", "h3", "h4"], recursive=False)
        name = header(h.get_text(" ", strip=True)) if h else ""
        remove_abstract = bool(re.search(r"(?:^|[\s_-])(abstract|abs\d*)(?:$|[\s_-])", ident, re.I)) or name in ABSTRACT_HEADINGS
        remove_refs = bool(re.search(r"(?:^|[\s_-])(references|ref-list|bibliography)(?:$|[\s_-])", ident, re.I)) or name in END_HEADINGS
        if remove_abstract:
            if len(node.get_text(" ", strip=True)) < max(9000, total * .5):
                node.decompose()
            else:
                flags.append("uncertain_abstract_boundary")
        elif remove_refs:
            node.decompose()
        elif name in {"funding", "acknowledgments", "acknowledgements"}:
            if len(node.get_text(" ", strip=True)) < 9000:
                node.decompose()
    for el in list(root.find_all(["h1", "h2", "h3"])):
        if title and el.parent and normalized(el.get_text(" ", strip=True)) == normalized(title):
            el.decompose()
    blocks = {"p", "li", "blockquote", "pre", "h2", "h3", "h4"}
    parts = []
    for el in root.descendants:
        if not isinstance(el, Tag) or el.name not in blocks:
            continue
        if any(parent is not root and parent.name in blocks for parent in el.parents):
            continue
        value = tidy(el.get_text(" ", strip=True))
        if len(value) > 8:
            parts.append(value)
    text = tidy("\n\n".join(parts)) or tidy(root.get_text("\n", strip=True))
    # PMC等页面可能只有摘要、PDF入口及Selected References。
    headings = [header(h.get_text(" ", strip=True)) for h in soup.find_all(["h2", "h3"])]
    substantive = any(re.match(r"^(?:introduction|background|materials|methods|results|discussion|conclusions?)\b", h)
                      for h in headings)
    ref_only = any(h in END_HEADINGS for h in headings) and not substantive
    pdf_link = bool(soup.select_one('a[href$=".pdf"], a[href*=".pdf?"]'))
    notice = bool(re.search(r"(?i)(full text available as pdf|selected references|these references are in pubmed)",
                            soup.get_text(" ", strip=True)[:4500]))
    if (ref_only and pdf_link) or (notice and not substantive and len(text) < 7000):
        flags.append("html_pdf_preview")
    if not substantive and len(text) > 15000 and len(soup.find_all("p")) <= 6:
        flags.append("html_unstructured_review")
    return text, flags


def pdf_body(path, title, abstract):
    import pymupdf
    flags = []
    try:
        import pymupdf4llm
        text = pymupdf4llm.to_markdown(str(path), use_ocr=False)
        if len(text) < MIN_BODY:
            raise ValueError("pymupdf4llm文字过少")
    except (ImportError, ValueError, RuntimeError):
        flags.append("pdf_plain_text_reading_order_review")
        with pymupdf.open(path) as doc:
            text = "\n\n".join(page.get_text("text", sort=True) for page in doc)
    # 仅做版式风险提示，绝不把版式检测当作阅读顺序验证。
    with pymupdf.open(path) as doc:
        for page in list(doc)[:3]:
            width = page.rect.width
            boxes = [line["bbox"] for b in page.get_text("dict").get("blocks", []) if b.get("type") == 0
                     for line in b.get("lines", []) if "bbox" in line]
            left = sum(x1 < .49 * width for x0, y0, x1, y1 in boxes)
            right = sum(x0 > .51 * width for x0, y0, x1, y1 in boxes)
            if min(left, right) >= 8:
                flags.append("pdf_multicolumn_review")
                break
    return text, flags


def content_inspection(text, kind):
    """在所有HTML解析器上识别“PDF入口+Selected References”这类假正文。"""
    start = text[:3000]
    start_lines = [header(v) for v in start.splitlines() if v.strip()][:14]
    first = start_lines[:4]
    selected = "selected references" in first
    refs_notice = "these references are in pubmed" in start.lower()
    pdf_notice = bool(re.search(r"(?i)full\s*text.{0,120}available\s*as\s*a?\s*pdf", start))
    flags = []
    if kind.startswith("HTML") and (selected or (refs_notice and "selected references" in start.lower())):
        flags.append("html_reference_only")
    elif kind.startswith("HTML") and pdf_notice and len(text) < 8000:
        flags.append("html_pdf_preview")
    elif selected and len(text) < 3000:
        flags.append("reference_list_only_review")
    if "images" in first[:2] and len(start) >= 300:
        # 很多旧PMC纯图像页面会先导出乱码，不能因为全文足够长就判为通过。
        first_chunk = start[:500]
        if (sum(c.isalnum() for c in first_chunk) / max(1, len(first_chunk)) < .45 or
                re.search(r"(?:~{4,}|\.{7,}|<\|)", first_chunk)):
            flags.append("possible_ocr_noise_review")
    return flags


# 有意义的修复记录与真正的质量异常必须分开，避免清理成功也被标记review。
CLEAN_ACTIONS = {
    "removed_title", "removed_leading_metadata", "removed_abstract_front",
    "removed_exact_leading_abstract", "removed_references",
    "removed_abstract_section", "removed_abstract_subsection",
    "removed_nonresearch_section", "removed_verified_references",
    "removed_html_frontmatter", "removed_numbered_references",
    "PMC_network_success",
}
SEVERE_HTML = {"html_reference_only", "html_pdf_preview", "reference_list_only_review"}


def evaluate(text, flags):
    n = len(text)
    visible = sum(not c.isspace() for c in text)
    alnum = sum(c.isalnum() for c in text) / max(1, visible)
    bad = sum(c in "\ufffd\u25a1" for c in text) / max(1, n)
    all_flags = list(dict.fromkeys(flags))
    actions = [f for f in all_flags if f in CLEAN_ACTIONS]
    warnings = [f for f in all_flags if f not in CLEAN_ACTIONS]
    if n < MIN_BODY:
        warnings.append("body_too_short_or_scanned_pdf")
    elif n < 2500:
        warnings.append("short_body_review")
    if alnum < .45 or bad > .01:
        warnings.append("ocr_or_encoding_review")
    if n > 260000:
        warnings.append("very_long_body_review")
    if any(f in SEVERE_HTML for f in warnings):
        warnings.append("non_article_content_blocked")
    blocked = n < MIN_BODY or alnum < .30 or bad > .03 or "non_article_content_blocked" in warnings
    return ("blocked" if blocked else "review" if warnings else "ok"), list(dict.fromkeys(warnings)), actions

def choose_body(pid, files, metadata, pmc=None):
    title, abstract = tidy(metadata.get("title")), tidy(metadata.get("abstract"))
    candidates, attempts = [], []

    def try_parse(kind, path, parser):
        try:
            text, flags = parser(path)
            # 所有提取器统一按CSV移除可验证的前置标题、摘要等，避免JATS/HTML/PDF清理标准不同。
            if kind.startswith("PDF"):
                text = normalize_pdf_markup(text)
            text, edits = trim_body(text, title, abstract, metadata)
            if kind.startswith("HTML"):
                text, front_actions = strip_html_frontmatter(text)
                edits += front_actions
            if "removed_references" in edits or "removed_numbered_references" in edits:
                flags = [f for f in flags if f != "ambiguous_reference_heading_kept"]
            flags = flags + edits + content_inspection(text, kind)
            if kind.startswith("PDF") and re.search(r"(?i)<(?:sup|sub|br)\b|^\s*#{1,6}\s", text, re.M):
                flags.append("pdf_markup_residue_review")
            status, warnings, actions = evaluate(text, flags)
            candidates.append({"source": kind, "body": text, "status": status,
                               "warnings": warnings, "actions": actions})
            attempts.append(f"{kind}:{status}:{len(text)}")
        except Exception as exc:
            attempts.append(f"{kind}:ERROR:{type(exc).__name__}")

    cache = REF / "pmc_source_cache_v5"
    if re.fullmatch(r"\d{5,10}", pid):
        jats = cache / f"{pid}.jats.xml"
        bioc = cache / f"{pid}.bioc.json"
        if jats.is_file():
            try_parse("JATS", jats, jats_body)
        if bioc.is_file() and not any(c["source"] == "JATS" and c["status"] != "blocked" for c in candidates):
            try_parse("BioC", bioc, bioc_body)
    if files["html"]:
        try:
            import trafilatura  # 可选依赖；缺失时仍可用BeautifulSoup与PDF
            def trafilatura_body(path):
                doc = parse_html_trafilatura(path, abstract)
                return render_body(doc["sections"]), doc["notes"]
            try_parse("HTML_TRAFILATURA", files["html"][0], trafilatura_body)
        except ImportError:
            pass
        try_parse("HTML", files["html"][0], lambda p: html_body(p, title, abstract))

    def is_usable(c):
        return c["status"] != "blocked" and len(c["body"]) >= 2500 and not any(
            w in c["warnings"] for w in ("html_unstructured_review", "html_reference_only", "html_pdf_preview"))

    if files["pdf"] and not any(is_usable(c) for c in candidates):
        try_parse("PDF", files["pdf"][0], lambda p: pdf_body(p, title, abstract))

    if pmc and pid.isdigit() and not any(is_usable(c) for c in candidates):
        try:
            d, reasons = pmc.get(pid, tidy(metadata.get("pmcid")))
            if d:
                src = "PMC_" + d["parser"].upper()
                text, edits = trim_body(render_body(d["sections"]), title, abstract, metadata)
                status, warnings, actions = evaluate(text, d.get("notes", []) + reasons + edits)
                candidates.append({"source": src, "body": text, "status": status,
                                   "warnings": warnings, "actions": actions})
                attempts.append(f"{src}:{status}:{len(text)}")
        except Exception as exc:
            attempts.append(f"PMC:ERROR:{type(exc).__name__}")

    # 优先完整、通过检查的正文，而不是固定选择Trafilatura。
    # JATS/BioC结构清晰时优先；HTML预览页/纯参考文献永远不能击败可读PDF。
    order = {
        'JATS': 0,
        'BioC': 1,
        'PMC_JATS': 2,
        'PMC_BIOC': 3,
        'HTML_TRAFILATURA': 4,
        'HTML': 5,
        'PDF': 7,
    }
    good = [c for c in candidates if c["status"] != "blocked"]
    if good:
        # 质量合格优先。长度作为覆盖性补充，不让短HTML的固定顺序压过完整PDF。
        def rank(c):
            return (0 if c["status"] == "ok" else 1,
                    len(c["warnings"]), order.get(c["source"], 99),
                    -min(len(c["body"]), 80000))
        winner = min(good, key=rank)
    else:
        # 没有合格候选就如实标记blocked；避免把Selected References误当正文。
        fallback = [c for c in candidates if not any(w in SEVERE_HTML for w in c["warnings"])]
        winner = max(fallback or candidates,
                     key=lambda c: len(c["body"]), default={"source": "NONE", "body": "",
                     "status": "blocked", "warnings": ["extraction_failed"], "actions": []})
    warnings = list(winner["warnings"])
    status = winner["status"]
    if not title:
        warnings.append("missing_title_review")
    if (metadata.get("title_source", "").endswith("unverified") or
            metadata.get("abstract_source", "").endswith("unverified")):
        warnings.append("pdf_metadata_unverified")
    if status == "ok" and warnings:
        status = "review"
    return (winner["body"], winner["source"], status, list(dict.fromkeys(warnings)),
            winner["actions"], attempts)

def corpus_complete(folder: Path) -> bool:
    """检验本阶段保存的 CSV/JSONL/正文文件是否一致，而不只检查目录存在。"""
    try:
        required = ('papers.jsonl', 'extraction_report.csv', 'needs_review.csv', 'config.json')
        if not META.is_file() or not folder.is_dir() or not all((folder / p).is_file() for p in required):
            return False
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        if config.get('version') != VERSION:
            return False
        with META.open(encoding='utf-8-sig', newline='') as f:
            metas = list(csv.DictReader(f))
        with (folder / 'extraction_report.csv').open(encoding='utf-8-sig', newline='') as f:
            reports = list(csv.DictReader(f))
        with (folder / 'needs_review.csv').open(encoding='utf-8-sig', newline='') as f:
            reviews = list(csv.DictReader(f))
        records = [json.loads(s) for s in (folder / 'papers.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        ids = [str(r['paper_id']) for r in records]
        if (not ids or len(ids) != config.get('paper_count') or len(ids) != len(set(ids)) or
                len(metas) != len(ids) or len(reports) != len(ids) or
                set(ids) != {str(m['paper_id']) for m in metas} or
                set(ids) != {str(r['paper_id']) for r in reports} or
                [r.get('row_index') for r in records] != list(range(len(ids)))):
            return False
        if {r['paper_id'] for r in reviews} != {r['paper_id'] for r in reports if r['status'] != 'ok'}:
            return False
        if {p.stem for p in (folder / 'body').glob('*.txt')} != set(ids):
            return False
        for row in records:
            body = str(row['body_text'])
            if (folder / 'body' / (str(row['paper_id']) + '.txt')).read_text(encoding='utf-8') != body + ('\n' if body else ''):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def main():
    cli = argparse.ArgumentParser(description="论文文本提取：首次运行生成一份元数据CSV和独立正文")
    cli.add_argument("--force", action="store_true", help="重建正文和报告，但绝不覆盖人工修订的元数据CSV")
    args = cli.parse_args()
    # 自动复用完整结果；只有不完整或显式 --force 时才重建。
    if CORPUS.exists() and not args.force and corpus_complete(CORPUS):
        print('1.1：已有完整正文与元数据，跳过。')
        return
    groups = discover()
    OUT.mkdir(parents=True, exist_ok=True)
    current, columns = load_metadata()
    missing = [pid for pid in groups if pid not in current]
    # 如果输入CSV来自此前10列版本，补齐旧版的实质性元数据能力。
    # 对旧的27列已核对CSV不触碰：它本身具有所有META_FIELDS。
    upgrading = bool(current) and any(field not in columns for field in META_FIELDS)
    requested = list(groups) if upgrading else missing
    print(f"发现原始论文 {len(groups)} 篇；已有元数据 {len(current)} 篇；新增 {len(missing)} 篇；旧版字段升级：{upgrading}。", flush=True)
    if requested:
        remote = pubmed_metadata(requested)
        for pid in requested:
            current[pid] = metadata_for(pid, groups[pid], remote.get(pid, {}), current.get(pid))
        save_metadata(list(current.values()), columns)
        print(f"元数据CSV已保存：{META}（已有人工修改保留）", flush=True)
    else:
        print(f"直接使用已有元数据CSV：{META}（不覆盖）", flush=True)

    pmc = PMCFulltext(REF / "pmc_source_cache_v5")
    tmp = Path(tempfile.mkdtemp(prefix=".corpus_tmp_", dir=OUT))
    reports = []
    try:
        (tmp / "body").mkdir()
        with (tmp / "papers.jsonl").open("w", encoding="utf-8") as output:
            for i, (pid, paths) in enumerate(groups.items()):
                metadata = current[pid]
                title = tidy(metadata.get("title"))
                abstract = tidy(metadata.get("abstract"))
                if fake_abstract(abstract):
                    raise ValueError(f"元数据存在图片占位假摘要，请先核对 {pid} 的 abstract")
                body, source, status, flags, actions, attempts = choose_body(pid, paths, metadata, pmc)
                (tmp / "body" / f"{pid}.txt").write_text(body + ("\n" if body else ""), encoding="utf-8")
                record = {"row_index": i, "paper_id": pid, "pmid": tidy(metadata.get("pmid")),
                          "title": title, "abstract": abstract, "body_text": body,
                          "has_abstract": bool(abstract), "body_source": source, "quality_status": status,
                          "cleanup_actions": actions}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                reports.append({"paper_id": pid, "status": status, "source": source,
                                "body_chars": len(body), "has_abstract": bool(abstract),
                                "warnings": ";".join(flags), "cleanup_actions": ";".join(actions),
                                "attempts": ";".join(attempts),
                                "html_file": str(paths["html"][0]) if paths["html"] else "",
                                "pdf_file": str(paths["pdf"][0]) if paths["pdf"] else ""})
                print(f"[{i+1}/{len(groups)}] {pid}: {source} {len(body)} chars; {status}", flush=True)
        write_csv(tmp / "extraction_report.csv", reports, REPORT_COLS)
        write_csv(tmp / "needs_review.csv", [r for r in reports if r["status"] != "ok"], REPORT_COLS)
        (tmp / "config.json").write_text(json.dumps({
            "version": VERSION, "paper_count": len(groups), "metadata_csv": "../papers_metadata.csv",
            "status_counts": dict(Counter(row["status"] for row in reports)),
            "note": "本程序只提取文本；质量标签是自动筛查，不等于人工验证。"
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if not corpus_complete(tmp):
            raise RuntimeError('新生成的 extracted/corpus/ 未通过完整性检查，保留原结果')
        if CORPUS.exists():
            backup = OUT / ".corpus_backup"
            if backup.exists():
                raise FileExistsError(f"发现未清理的旧备份：{backup}")
            CORPUS.rename(backup)
            try:
                tmp.rename(CORPUS)
            except Exception:
                backup.rename(CORPUS)
                raise
            shutil.rmtree(backup)
        else:
            tmp.rename(CORPUS)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print(f"\n完成：{len(groups)}篇；质量状态 {dict(Counter(x['status'] for x in reports))}")
    print(f"元数据：{META}\n正文：{CORPUS / 'body'}\n待检查：{CORPUS / 'needs_review.csv'}")


if __name__ == "__main__":
    main()
