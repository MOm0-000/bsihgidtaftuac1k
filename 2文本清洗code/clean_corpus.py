#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理 V7 已提取正文，生成供特征提取使用的 180 篇语料（Python 标准库）。

输入（脚本所在项目目录下）：
  extracted/corpus/papers.jsonl
  extracted/papers_metadata_reviewed.csv
  extracted/corpus/extraction_report.csv  # 可选，补充风险详情

输出：cleaned_180/papers_180.jsonl、body/*.txt、cleaning_report.csv、
      needs_review.csv、excluded.csv、cleaning_config.json。

只清洗 body_text；title/abstract 始终以 reviewed CSV 为准。不修改任何输入。
正常运行：python clean_corpus.py
重新生成：python clean_corpus.py --force
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'extracted/corpus/papers.jsonl'
METADATA = ROOT / 'extracted/papers_metadata_reviewed.csv'
EXTRACT_REPORT = ROOT / 'extracted/corpus/extraction_report.csv'
OUTPUT = ROOT / 'cleaned_180'
VERSION = '1.0'

# 当前研究的人工排除决定：保留在报告中，但不进入180篇正文。
EXCLUDED = {
    '14450716': '书目汇编，不作为单篇研究论文',
    '40011409': '多篇会议摘要合集，不作为单篇研究论文',
    '8457977': '已核实PDF多栏阅读顺序错乱，无法可靠还原正文',
}

NAV_LINE = re.compile(r'^(?:full text|view full text|view pdf|download pdf|back to top)$', re.I)
HTML_TAG = re.compile(r'</?(?:br|sup|sub|span|em|strong|i|b|small)\b[^>]*>', re.I)
REF_HEAD = re.compile(r'(?im)^\s*(?:references|selected references|bibliography|literature cited)\s*:?\s*$')
YEAR = re.compile(r'\b(?:18|19|20)\d{2}\b')
REF_TAIL = re.compile(r'^\s*[-*]?\s*(\d{1,4})\s*[.)]?\s+(.{35,})$')
LONG_WORD = re.compile(r'\b[A-Za-z]{28,}\b')
PUBLISHING_START = re.compile(
    r'^(?:this article is published and distributed under|'
    r'edited by .{8,} approved |author contributions\s*:|'
    r'go to neurology\.org for full disclosures|corresponding author\.?$|'
    r'received \d{4}\b.{0,120}\baccepted\b|'
    r'accepted \d{4}\b|published under .{0,70}license|'
    r'series information$|from the cover$|'
    r'copyright and license information$|article notes$|author information$)', re.I | re.S)
REPORT_FIELDS = [
    'pmid', 'status', 'reason', 'source', 'extract_status', 'before_chars', 'after_chars',
    'removed_chars', 'actions', 'warnings', 'unrecoverable_chars',
    'long_unspaced_words_at_start', 'source_sha256', 'clean_sha256',
]


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_sources() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    if not INPUT.is_file() or not METADATA.is_file():
        raise FileNotFoundError(f'需要 {INPUT} 和 {METADATA}；请先运行正文提取和元数据复核')
    records = [json.loads(line) for line in INPUT.read_text(encoding='utf-8').splitlines() if line.strip()]
    metadata = load_csv(METADATA)
    if not records or not metadata:
        raise ValueError('正文或元数据为空')
    for group, label in [(records, '正文'), (metadata, '元数据')]:
        keys = [str(r.get('pmid', '')).strip() for r in group]
        if not all(re.fullmatch(r'\d{5,10}', k) for k in keys) or len(keys) != len(set(keys)):
            raise ValueError(f'{label}存在无效或重复 PMID')
    by_meta = {r['pmid'].strip(): r for r in metadata}
    by_body = {r['pmid'].strip(): r for r in records}
    if by_body.keys() != by_meta.keys():
        raise ValueError(f'正文/元数据 PMID 不匹配：正文独有 {sorted(by_body.keys()-by_meta.keys())[:8]}；'
                         f'元数据独有 {sorted(by_meta.keys()-by_body.keys())[:8]}')
    if len(records) == 183 and not set(EXCLUDED).issubset(by_body):
        raise ValueError('183篇中缺少约定排除的3篇，停止以免生成错误的180篇语料')
    report = {}
    if EXTRACT_REPORT.is_file():
        report = {str(r.get('paper_id', '')).strip(): r for r in load_csv(EXTRACT_REPORT)}
    return records, by_meta, report


def tidy_text(text: str) -> tuple[str, list[str]]:
    """轻量清理：不删除URL、表格内容、标题相似句或研究词汇。"""
    actions = []
    old = text
    text = html.unescape(text).replace('\ufeff', '').replace('\x00', '').replace('\u00ad', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = HTML_TAG.sub('', text)
    if text != old:
        actions.append('normalized_control_and_inline_markup')
    lines, removed_nav, converted_tables = [], 0, 0
    for line in text.splitlines():
        s = line.strip()
        if NAV_LINE.fullmatch(s):
            removed_nav += 1
            continue
        # 表格数据有价值，不删除；仅移除明确的Markdown分隔行。
        if s.count('|') >= 3 and s.startswith('|') and s.endswith('|'):
            cells = [c.strip() for c in s[1:-1].split('|')]
            if len([c for c in cells if c]) >= 2:
                line = ' '.join(c for c in cells if c)
                converted_tables += 1
        lines.append(line.rstrip())
    text = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()
    if removed_nav:
        actions.append(f'removed_nav_lines:{removed_nav}')
    if converted_tables:
        actions.append(f'preserved_table_cells_as_text:{converted_tables}')
    return text, actions


def strip_initial_publication_info(text: str) -> tuple[str, int]:
    """只删除开头连续的、有明确格式特征的出版信息段落。"""
    parts = re.split(r'\n\s*\n', text, maxsplit=12)
    removed = 0
    while len(parts) > 1 and removed < 8 and PUBLISHING_START.match(parts[0].strip()):
        # 不吞噬意外很长的正文段落
        if len(parts[0]) > 1100:
            break
        parts.pop(0)
        removed += 1
    if not removed:
        return text, 0
    return '\n\n'.join(parts).strip(), removed


def strip_verified_ocr_prefix(pmid: str, text: str) -> tuple[str, bool]:
    """移植旧后处理对21433778的已核实锚点；不补标题、不重排正文。"""
    if pmid != '21433778':
        return text, False
    m = re.search(r'Medical advancement in any field is never continuous\s*;', text[:3500], re.I)
    if not m:
        return text, False
    prefix = text[:m.start()]
    if ('Images' not in prefix[:80] or
            not (re.search(r'~{12,}', prefix) or re.search(r'\.{12,}', prefix)) or
            len(text[m.start():]) < 25000):
        return text, False
    return text[m.start():].strip(), True


def strip_trailing_numbered_references(text: str, source: str) -> tuple[str, int]:
    """仅删除文末>=3条连续编号且含年份的明显参考文献；绝不从中间截断。"""
    if source != 'PDF' or len(text) < 4000:
        return text, 0
    lines = text.splitlines(keepends=True)
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    numbers = []
    examined = 0
    while i >= 0 and examined < 60 and len(numbers) < 30:
        s = lines[i].strip()
        examined += 1
        if not s:  # PDF参考文献之间常有空行
            i -= 1
            continue
        m = REF_TAIL.match(s)
        if not m or not YEAR.search(m.group(2)):
            break
        numbers.append(int(m.group(1)))
        i -= 1
    numbers.reverse()
    if len(numbers) < 3 or not all(b == a + 1 for a, b in zip(numbers, numbers[1:])):
        return text, 0
    pos = sum(len(s) for s in lines[:i + 1])
    if pos < len(text) * 0.5:
        return text, 0
    return text[:pos].rstrip(), len(numbers)


def make_clean(paper: dict, meta: dict, extract: dict) -> tuple[str, dict]:
    pmid = paper['pmid'].strip()
    body = paper.get('body_text')
    if not isinstance(body, str) or not body.strip():
        raise ValueError(f'{pmid}: 正文为空或不是字符串')
    source = str(paper.get('body_source', ''))
    original = body
    text, actions = tidy_text(original)
    text, trimmed = strip_initial_publication_info(text)
    if trimmed:
        actions.append(f'removed_leading_publication_blocks:{trimmed}')
    text, ocr_fixed = strip_verified_ocr_prefix(pmid, text)
    if ocr_fixed:
        actions.append('removed_verified_21433778_ocr_prefix')
    text, tail_refs = strip_trailing_numbered_references(text, source)
    if tail_refs:
        actions.append(f'removed_trailing_numbered_references:{tail_refs}')
    unknown = text.count('\ufffd')
    if unknown:
        text = text.replace('\ufffd', ' ')
        actions.append(f'marked_unrecoverable_glyphs_as_spaces:{unknown}')
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    warnings = []
    extract_status = str(paper.get('quality_status', '')).lower()
    old_flags = [x.strip() for x in extract.get('warnings', '').split(';') if x.strip()]
    # 保留提取阶段的风险，但成功执行的清理操作不算警告。
    warnings.extend('extract:' + x for x in old_flags)
    if extract_status not in ('ok', 'pass') and not old_flags:
        warnings.append('extract:needs_review')
    if not text:
        warnings.append('empty_after_cleaning')
    if len(text) < 1000:
        warnings.append('short_body_under_1000_chars')
    reduction = (len(original) - len(text)) / max(1, len(original))
    if reduction > 0.3:
        warnings.append('large_reduction_over_30pct_review')
    if REF_HEAD.search(text[-max(3000, len(text)//3):]):
        warnings.append('possible_reference_heading_remaining')
    word_runs = len(LONG_WORD.findall(text[:6000])) if source == 'PDF' else 0
    if word_runs >= 3:
        warnings.append('pdf_words_joined_together_review')
    if unknown:
        warnings.append('unrecoverable_characters_present')
    if 'preserved_table_cells_as_text' in ' '.join(actions) and source == 'PDF':
        warnings.append('pdf_table_structure_converted_review')
    if pmid == '21433778' and not ocr_fixed and text.startswith('Images'):
        warnings.append('21433778_ocr_prefix_not_safely_identified')
    if meta.get('title', '').strip() != str(paper.get('title', '')).strip():
        warnings.append('metadata_title_changed_from_extracted')
    if meta.get('abstract', '').strip() != str(paper.get('abstract', '')).strip():
        warnings.append('metadata_abstract_changed_from_extracted')
    warnings = list(dict.fromkeys(warnings))
    if not text:
        raise ValueError(f'{pmid}: 清理后正文为空，拒绝写入')
    stats = {
        'pmid': pmid, 'status': 'review' if warnings else 'ok', 'reason': '',
        'source': source, 'extract_status': extract_status, 'before_chars': len(original),
        'after_chars': len(text), 'removed_chars': len(original) - len(text),
        'actions': ';'.join(actions), 'warnings': ';'.join(warnings),
        'unrecoverable_chars': unknown, 'long_unspaced_words_at_start': word_runs,
        'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'clean_sha256': hashlib.sha256(text.encode()).hexdigest(),
    }
    return text, stats


def write_csv(path: Path, rows: list[dict]):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='183篇V7正文清理，生成180篇独立语料；原文件不修改')
    parser.add_argument('--force', action='store_true', help='替换本脚本此前生成的 cleaned_180 结果')
    args = parser.parse_args()
    if OUTPUT.exists() and not args.force:
        parser.error(f'{OUTPUT} 已存在。如需重做，请加 --force')
    if OUTPUT.exists() and not (OUTPUT / 'cleaning_config.json').is_file():
        parser.error(f'拒绝覆盖非本脚本生成的目录：{OUTPUT}')
    docs, metadata, reports = load_sources()
    stage = Path(tempfile.mkdtemp(prefix='.cleaned_180_stage_', dir=ROOT))
    try:
        (stage / 'body').mkdir()
        all_rows, kept, excluded = [], [], []
        with (stage / 'papers_180.jsonl').open('w', encoding='utf-8') as out:
            for paper in docs:
                pmid = str(paper['pmid']).strip()
                orig = str(paper.get('body_text', ''))
                if pmid in EXCLUDED:
                    row = dict.fromkeys(REPORT_FIELDS, '')
                    row.update(pmid=pmid, status='excluded', reason=EXCLUDED[pmid],
                               source=paper.get('body_source', ''),
                               extract_status=paper.get('quality_status', ''), before_chars=len(orig),
                               source_sha256=hashlib.sha256(orig.encode()).hexdigest())
                    excluded.append(row)
                    all_rows.append(row)
                    continue
                cleaned, row = make_clean(paper, metadata[pmid], reports.get(pmid, {}))
                (stage / 'body' / f'{pmid}.txt').write_text(cleaned + '\n', encoding='utf-8')
                m = metadata[pmid]
                record = {
                    'row_index': len(kept), 'original_row_index': paper.get('row_index'),
                    'pmid': pmid, 'title': m.get('title', ''), 'abstract': m.get('abstract', ''),
                    'has_abstract': bool(m.get('abstract', '').strip()),
                    'body_text': cleaned, 'body_source': paper.get('body_source', ''),
                    'quality_status': row['status'],
                    'cleaning_actions': row['actions'].split(';') if row['actions'] else [],
                    'cleaning_warnings': row['warnings'].split(';') if row['warnings'] else [],
                }
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                kept.append(row)
                all_rows.append(row)
        if len(docs) == 183 and (len(kept), len(excluded)) != (180, 3):
            raise ValueError(f'期望183→180+3，实际{len(docs)}→{len(kept)}+{len(excluded)}')
        write_csv(stage / 'cleaning_report.csv', all_rows)
        write_csv(stage / 'needs_review.csv', [r for r in kept if r['status'] == 'review'])
        write_csv(stage / 'excluded.csv', excluded)
        config = {
            'version': VERSION, 'input': str(INPUT.relative_to(ROOT)),
            'metadata': str(METADATA.relative_to(ROOT)), 'source_count': len(docs),
            'kept_count': len(kept), 'excluded_count': len(excluded),
            'excluded_pmids': EXCLUDED,
            'status_note': 'ok=通过自动规则，并非人工确认全文正确；review=保留在180篇中等待复核',
        }
        (stage / 'cleaning_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        stage.replace(OUTPUT)
        summary = Counter(r['status'] for r in all_rows)
        print(f'完成：输入 {len(docs)} 篇；保留 {len(kept)} 篇；排除 {len(excluded)} 篇；'
              f'正常 {summary["ok"]} 篇；需复核 {summary["review"]} 篇')
        print('清洗语料：', OUTPUT / 'papers_180.jsonl')
        print('复核报告：', OUTPUT / 'needs_review.csv')
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == '__main__':
    main()
