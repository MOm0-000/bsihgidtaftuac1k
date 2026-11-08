#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""15篇重点论文中已核实问题的定向复核与修复（V1.2，保留180篇）。

输入：cleaned_180/papers_180.jsonl、reference/PDF/<PMID>.pdf
输出：cleaned_180_checked/papers_180.jsonl、body/、repair_report.csv、needs_review.csv
不改动原始提取/元数据；只对固定的已核实PMID进行修复，拒绝未匹配锚点。
用法：python 2.2review_problem_papers.py [--force]
依赖：pip install pymupdf
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'cleaned_180' / 'papers_180.jsonl'
TARGET = ROOT / 'cleaned_180_checked'
PDF_225 = ROOT / 'reference' / 'PDF' / '2257383.pdf'
PDF_121 = ROOT / 'reference' / 'PDF' / '12194887.pdf'
VERSION = '1.2'

# 未确认的PDF缺损和表格数值不自动猜测修复。
MANUAL = {
    '21433778': '已清除已知前置OCR噪声；旧论文仍有OCR误字风险，不自动补写。',
    '16995156': '历史PDF字体及表格格式风险；暂不重写正文。',
    '25144551': '后部大型结果表格未完整结构化；本研究只采用主题正文。',
}



def sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def norm(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def remove_verified_225_footnote(content: str) -> tuple[str, int]:
    """仅移除1990年原PDF首页已核实的整块脚注，不跨区域猜测。"""
    expected = (
        '*Presented as part of a Workshop on Housing and Health: Interrelationship '
        'and Community Impact held by the Committee on Public Health of the New '
        'York Academy of Medicine November 17 and 18, 1989. Address for reprint '
        'requests: Room 4M-08, North Central Bronx Hospital, 3424 Kossuth Avenue, '
        'Bronx, NY 10467'
    )
    if not PDF_225.is_file():
        raise FileNotFoundError(f'缺少原始PDF：{PDF_225}')
    import pymupdf
    with pymupdf.open(PDF_225) as doc:
        first_page = norm(doc[0].get_text('text'))
    # 脚注必须能从原始PDF首页逐字核实，才允许删除。
    if norm(expected).casefold() not in first_page.casefold():
        raise ValueError('2257383：原PDF首页脚注文字未匹配，拒绝自动删除')
    if content.count(expected) != 1:
        raise ValueError('2257383：修复候选中脚注出现次数不是1，拒绝自动删除')
    start = content.index(expected)
    prefix, suffix = content[:start], content[start + len(expected):]
    if (not prefix.rstrip().endswith('war-time care of children has')
            or not suffix.lstrip().startswith('to be more elaborate')):
        raise ValueError('2257383：脚注两侧研究正文锚点不符合已核查样本')
    return norm(prefix.rstrip() + ' ' + suffix.lstrip()), len(expected)


def pdf_linebreak_pairs(path: Path) -> set[tuple[str, str]]:
    """只采集同一PDF页内明确的断行连字符；不跨页或跨栏推测。"""
    import pymupdf
    pairs: set[tuple[str, str]] = set()
    with pymupdf.open(path) as doc:
        if not 3 <= len(doc) <= 12:
            raise ValueError('12194887：PDF页数与已核查版本不一致')
        for page in doc:
            # 需要实际换行证据。已经扁平化的文字不能作为推断依据。
            raw = unicodedata.normalize('NFKC', page.get_text('text', sort=False))
            pairs.update((a.casefold(), b.casefold()) for a, b in re.findall(
                r'(?m)\b([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})\b', raw))
    return pairs


def repair_verified_hyphens(body: str, corpus_words: set[str]) -> tuple[str, int, int]:
    """PDF断行+语料中真实独立词两重证据；不更改未证实复合词。"""
    if not PDF_121.is_file():
        raise FileNotFoundError(f'缺少核查用原始PDF：{PDF_121}')
    pairs = pdf_linebreak_pairs(PDF_121)
    merged = 0
    ambiguous = 0
    # ``undertook``在本文确属断行单词，别处的语料可能恰好未包含它。
    confirmed_extra = {'undertook'}
    pattern = re.compile(r'\b([A-Za-z]{2,})-\s+([a-z]{2,})\b')

    def decide(match: re.Match[str]) -> str:
        nonlocal merged, ambiguous
        a, b = match.group(1), match.group(2)
        joined = (a + b).casefold()
        if (a.casefold(), b.casefold()) not in pairs:
            return match.group(0)
        if joined not in corpus_words and joined not in confirmed_extra:
            ambiguous += 1
            return match.group(0)
        merged += 1
        return a + b

    return pattern.sub(decide, body), merged, ambiguous


def repair_2257383(old: str) -> tuple[str, str]:
    """复用原PDF文字层，避开会合并单词的Markdown/PDF提取路径。"""
    if not PDF_225.is_file():
        raise FileNotFoundError(f'缺少核查用原始PDF：{PDF_225}')
    import pymupdf
    parts = []
    with pymupdf.open(PDF_225) as doc:
        if not 10 <= len(doc) <= 25:
            raise ValueError('2257383：原PDF页数与已审查版本不一致，拒绝自动替换')
        for page in doc:
            lines = page.get_text('text', sort=False).splitlines()
            kept = []
            for line in lines:
                s = line.strip()
                if re.fullmatch(r'50\d|51[0-4]', s):
                    continue
                if re.fullmatch(r'HOUSING STATUS\s*(?:50\d|51[0-4])?', s, re.I):
                    continue
                if re.fullmatch(r'M\. SAN AGUSTIN|Bull\. N\.Y\. Acad\. Med\.', s, re.I):
                    continue
                if re.fullmatch(r'Vol\. 66, No\. 5, September-October 1990', s, re.I):
                    continue
                kept.append(line)
            parts.append('\n'.join(kept))
    content = '\n\n'.join(parts)
    lead = re.search(r'\bDOUBTLESS,\s+we\s+would\s+all\s+agree\b', content, re.I)
    tail = re.search(r'(?m)^\s*ACKNOWLEDGEMENTS\s*$', content)
    references = re.search(r'(?m)^\s*REFERENCES\s*$', content)
    if not lead or not tail or not references or not lead.start() < tail.start() < references.start():
        raise ValueError('2257383：正文起点/致谢/参考文献锚点未全部匹配，拒绝自动替换')
    content = content[lead.start():tail.start()]
    # 只合并明显的跨行断字；不猜测本来拼写错误的单词。
    content = re.sub(r'(?<=[A-Za-z])-\s*\n\s*(?=[a-z])', '', content)
    content = re.sub(r'\n(?!\n)', ' ', content)
    content = norm(content)
    content, footnote_chars = remove_verified_225_footnote(content)
    bad_old = len(re.findall(r'\b[A-Za-z]{28,}\b', old[:6000]))
    bad_new = len(re.findall(r'\b[A-Za-z]{28,}\b', content[:6000]))
    if not 25000 <= len(content) <= 45000 or 'we would all agree' not in content.lower():
        raise ValueError('2257383：PDF候选长度或关键文字异常，拒绝自动替换')
    if bad_old >= 3 and bad_new >= bad_old:
        raise ValueError('2257383：重新提取未改善可量化的单词粘连，拒绝替换')
    return content, (f'PDF直接文字层重提取；前6000字符超长粘连词 {bad_old}→{bad_new}；'
                     f'删除已核实的首页插入脚注{footnote_chars}字符；排除致谢/参考文献')


def repair_12194887(old: str, corpus_words: set[str]) -> tuple[str, str]:
    """仅删除已核对的插入区，不全局删相似摘要。"""
    start_marker = 'This article is part of the monograph Environmental Air Toxics:'
    resume_marker = 'associations between asthma exacerbations and a panel of air contaminants'
    ref_marker = 'REFERENCES AND NOTES 1. Rappaport'
    start = old.find(start_marker)
    stop = old.find(resume_marker, start) if start >= 0 else -1
    ref = old.find(ref_marker, stop) if stop >= 0 else -1
    if min(start, stop, ref) < 0 or not 1500 <= stop - start <= 4500:
        raise ValueError('12194887：摘要插入块或参考文献锚点不符，拒绝自动修改')
    inserted = old[start:stop]
    if not all(k in inserted for k in ('People with asthma who live near or downwind',
                                       'Key words: asthma', 'Addressing Community Concerns',
                                       'Received 25 October 2001')):
        raise ValueError('12194887：待删块不符合已核实的摘要/出版信息结构')
    if not old[:start].rstrip().endswith('temporal'):
        raise ValueError('12194887：插入处的研究正文接缝与原文不符')
    body = old[:start].rstrip() + ' ' + old[stop:]
    ref = body.find(ref_marker)
    if ref < len(body) * 0.6 or len(body) - ref < 2500:
        raise ValueError('12194887：末尾参考文献边界异常')
    body = body[:ref].strip()
    body = re.sub(r'^Environmental Health Perspectives\s*\*\s*VOLUME 110\s*\|\s*SUPPLEMENT 4\s*\|\s*AUGUST 2002\s*561\s*', '', body)
    # PDF第2/3页跨页页眉页脚被拼进正文。锚点必须完整、各出现一次。
    page_furniture = (
        ('Asthma Occurrence * White et al. 562 VOLUME 110 | SUPPLEMENT 4 | '
         'AUGUST 2002 * Environmental Health Perspectives',
         'one lower-bound estimate of ', 'asthma prevalence among school-aged'),
        ('Asthma Occurrence * Community concerns about asthma and air toxics '
         'Environmental Health Perspectives * VOLUME 110 | SUPPLEMENT 4 | '
         'AUGUST 2002 563',
         'scientific knowledge of the problem. ', 'Summary Public health agencies'),
    )
    removed_furniture_chars = 0
    for marker, before, after in page_furniture:
        if body.count(marker) != 1:
            raise ValueError('12194887：跨页页眉页脚数量或内容变化，拒绝自动删除')
        idx = body.index(marker)
        if not body[:idx].endswith(before) or not body[idx + len(marker):].startswith(' ' + after):
            raise ValueError('12194887：页眉页脚前后正文衔接不符合已核查版本')
        body = body[:idx] + body[idx + len(marker):]
        removed_furniture_chars += len(marker)

    body, dehyphenated, unconfirmed = repair_verified_hyphens(body, corpus_words)
    if not 15000 <= len(body) <= 35000:
        raise ValueError('12194887：修复后正文长度不符合预期')
    if 'temporal associations between asthma exacerbations' not in body:
        raise ValueError('12194887：删除插入块后，正文衔接检查未通过')
    return body, (f'删除中途插入的摘要/出版信息{stop-start}字符及文末书目；'
                  f'删除跨页页眉页脚{removed_furniture_chars}字符；'
                  f'按原PDF换行及语料词证据修复断词{dehyphenated}处，'
                  f'保留未确认复合词/断词{unconfirmed}处')


# 下面的规则只用于15篇PDF对照中已核实的样本；锚点不吻合即停止，绝不泛化到其他论文。
PDF_DIR = ROOT / 'reference' / 'PDF'
PDF_LAYOUT_REPAIR = {'6487732', '17246411'}


def verified_pdf(pmid: str, *, pages: tuple[int, int] | None = None):
    """核对PDF版式/页数，拒绝误用同名但不同版本的PDF。"""
    path = PDF_DIR / f'{pmid}.pdf'
    if not path.is_file():
        raise FileNotFoundError(f'{pmid}：缺少原版PDF：{path}')
    import pymupdf
    doc = pymupdf.open(path)
    if pages is not None and not pages[0] <= len(doc) <= pages[1]:
        doc.close()
        raise ValueError(f'{pmid}：原PDF页数与已核查版本不一致')
    return doc


def pdf_layout_repair(pmid: str, old: str) -> tuple[str, str]:
    """仅修复2篇已证实的PDF错栏：按原始文本块顺序，而不是左右栏同一高度拼接。"""
    expected_pages = {'6487732': 8, '17246411': 14}
    expected_first = {
        '6487732': 'A METHOD FOR TESTING AN EXTENDED',
        '17246411': 'Copyright 0 1987 by the Genetics Society of America',
    }
    with verified_pdf(pmid, pages=(expected_pages[pmid], expected_pages[pmid])) as pdf:
        chunks = [page.get_text('text', sort=False) for page in pdf]
    if expected_first[pmid] not in chunks[0][:450]:
        raise ValueError(f'{pmid}：PDF第一页标志不符，拒绝全文替换')
    if pmid == '6487732':
        start = re.search(r'(?m)^INTRODUCTION\s*$', chunks[0])
        end = re.search(r'(?m)^REFERENCES\s*$', chunks[-1])
        if not start or not end:
            raise ValueError('6487732：Introduction/References锚点缺失')
        chunks[0] = chunks[0][start.start():]
        chunks[-1] = chunks[-1][:end.start()]
        # 删除页脚所独有的出版/页码信息；不改动图注或结果表。
        pages = []
        for chunk in chunks:
            chunk = re.sub(r'(?m)^BIOPHYSICAL JOURNAL\s*\nVolume 46\s*\n1984\s*\n', '', chunk)
            chunk = re.sub(r'(?m)^YANA ET AL\.\s*\nSpontaneous Quantal Transmitter Release\s*\n', '', chunk)
            chunk = re.sub(r'(?m)^\s*(?:32[3-9]|330)\s*$', '', chunk)
            pages.append(chunk.strip())
        candidate = '\n\n'.join(pages).strip()
        # 首页版心之外的版权刊号会插入第一句中间，明确据原PDF删除。
        footer = ('BIOPHYS. J. © Biophysical Society\n*\n0006-3495/84/09/323/8\n'
                  'Volume 46\nSeptember 1984\n323-330\n')
        if candidate.count(footer) != 1 or candidate.count('$1.00') != 1:
            raise ValueError('6487732：首页出版页脚与审核样本不符')
        candidate = candidate.replace(footer, '', 1)
        candidate = re.sub(r'(?m)^\$1\.00\s*$', '', candidate, count=1)
        tail = ('Received for publication 15 July 1983 and in final form 16 December\n1983.')
        if not candidate.endswith(tail):
            raise ValueError('6487732：文末出版日期与审核样本不符')
        candidate = candidate[:-len(tail)].rstrip()
        if ('DISCUSSION AND CONCLUSION\nSome phenomena that could be interpreted as non-Poisson'
                not in candidate):
            raise ValueError('6487732：第6页错栏句序未被纠正，拒绝替换')
    else:
        start = re.search(r'(?m)^OMPARISON of genetic structure between spe-\s*$', chunks[0])
        end = re.search(r'(?m)^LITERATURE CITED\s*$', chunks[11])
        if not start or not end:
            raise ValueError('17246411：正文下沉首字母/文献锚点缺失')
        chunks[0] = chunks[0][start.start():]
        # 只修正原PDF中确认的跨行下沉大写C和spe- / cies。
        chunks[0], n = re.subn(
            r'OMPARISON of genetic structure between spe-\s*\nC cies can provide',
            'COMPARISON of genetic structure between species can provide', chunks[0], count=1)
        if n != 1:
            raise ValueError('17246411：首段下沉大写C修复锚点变化')
        chunks = chunks[:12]
        chunks[-1] = chunks[-1][:end.start()]
        pages = []
        for chunk in chunks:
            # 仅移除这份PDF重复的两种页首刊头，不删除材料方法、图表。
            chunk = re.sub(r'(?m)^\s*(?:69[7-9]|70[0-9]|710)\s*\n(?:M\. Choudhary and R\. S\. Singh\s*\n)?', '', chunk)
            chunk = re.sub(r'(?m)^Genetic Structure\s*\n(?:69[7-9]|70[0-9]|710)\s*\n', '', chunk)
            pages.append(chunk.strip())
        candidate = '\n\n'.join(pages).strip()
        # PDF第1页页脚刊号、其他页的短页首题名，均有明确页面来源。
        candidate = re.sub(r'(?m)^Genetics 117: 697-710 \(December, 1987\)\s*$', '', candidate)
        candidate = re.sub(r'(?m)^Genetic Structure\s*$', '', candidate)
        if 'COMPARISON of genetic structure between species can provide' not in candidate:
            raise ValueError('17246411：正文首段句序修复失败')
    if not 0.66 * len(old) < len(candidate) < 1.3 * len(old):
        raise ValueError(f'{pmid}：PDF重提取长度变化过大（{len(old)}→{len(candidate)}），拒绝替换')
    if pmid == '6487732' and not all(x in candidate for x in
        ('INTRODUCTION', 'DISCUSSION AND CONCLUSION', 'Statistical properties')):
        raise ValueError('6487732：重要章节缺失')
    if pmid == '17246411' and not all(x in candidate for x in
        ('MATERIALS AND METHODS', 'RESULTS', 'COMPARISON of genetic structure')):
        raise ValueError('17246411：重要章节缺失')
    old_words = set(x.casefold() for x in re.findall(r'[A-Za-z]{5,}', old))
    new_words = set(x.casefold() for x in re.findall(r'[A-Za-z]{5,}', candidate))
    overlap = len(old_words & new_words) / max(1, len(new_words))
    if overlap < 0.68:
        raise ValueError(f'{pmid}：PDF重新提取与原文词项覆盖过低({overlap:.1%})，拒绝替换')
    return candidate, (f'原版PDF按原始文本块顺序重新提取；取消跨栏同高度拼接；'
                       f'核对关键章节和篇幅，去除起始出版信息/末尾书目；'
                       f'词项交集占新文本词项{overlap:.1%}；图表数值仍未逐项认证')


def repair_8761444(old: str) -> tuple[str, str]:
    """明确位于Introduction之前的摘要/出版信息与独立行的期刊页眉。"""
    with verified_pdf('8761444', pages=(14, 14)) as pdf:
        first = pdf[0].get_text('text', sort=False)
        last_main = pdf[12].get_text('text', sort=False)
    if 'The lipocalin protein family: structure and function' not in first:
        raise ValueError('8761444：原PDF论文标题不匹配')
    if 'INTRODUCTION' not in first or 'The lipocalin protein family' not in last_main:
        raise ValueError('8761444：PDF版面锚点不匹配')
    hits = list(re.finditer(r'(?m)^INTRODUCTION\s*$', old))
    if len(hits) != 1 or not 800 < hits[0].start() < 3500:
        raise ValueError('8761444：正文Introduction边界变化，拒绝删除摘要')
    body = old[hits[0].start():]
    body, n = re.subn(r'(?m)^The lipocalin protein family\s*\n', '', body)
    # PDF交替打印作者/论文短题名+页码，和独立题名行属于同类页眉。
    body, by_author = re.subn(r'(?m)^_\d{1,2}_ D\. R\. Flower\s*$', '', body)
    body, by_title = re.subn(r'(?m)^The lipocalin protein family _\d{1,2}_\s*$', '', body)
    if not 3 <= n <= 9 or by_author != 6 or by_title != 1:
        raise ValueError(f'8761444：页眉数量与已审阅PDF不符({n}/{by_author}/{by_title})')
    if 'Members of the lipocalin protein family are typically small' not in body[:400]:
        raise ValueError('8761444：正文开头异常')
    return body.strip(), f'删除已重复保存在元数据中的前置摘要/刊头{hits[0].start()}字符；去除{n+by_author+by_title}处独立行/页码页眉；表格损坏字符保留风险'


def repair_12456544(old: str) -> tuple[str, str]:
    """只保留PDF首页真正的正文起点至讨论结尾，去掉互相交叉的侧栏/摘要。"""
    with verified_pdf('12456544', pages=(7, 7)) as pdf:
        first = pdf[0].get_text('text', sort=False)
        last = pdf[-1].get_text('text', sort=False)
    if 'The prevalence of allergic disease varies considerably' not in first.replace('\n', ' '):
        # PDF下沉首字母T可能被单独置于不同文本块。
        if 'he prevalence of allergic disease varies considerably' not in first.replace('\n', ' '):
            raise ValueError('12456544：原PDF引言起点不匹配')
    if 'ACKNOWLEDGEMENTS' not in last or 'Authors’ affiliations' not in last:
        raise ValueError('12456544：原PDF文末机构边界不匹配')
    start = old.find('he prevalence of allergic disease varies considerably')
    end = old.find('ACKNOWLEDGEMENTS', start)
    if not 1500 < start < 4500 or not end > 20000 or old.count('ACKNOWLEDGEMENTS') != 1:
        raise ValueError('12456544：前后研究正文边界不满足核查条件')
    body = old[start:end].strip()
    if not body.startswith('he prevalence of allergic disease varies considerably'):
        raise ValueError('12456544：引言断点无法验证')
    body = 'T' + body
    # 原PDF中的孤立下沉大写T被附着到了本页后续Western前面。
    if body.count('prevalence Tin Western countries') == 1:
        body = body.replace('prevalence Tin Western countries', 'prevalence in Western countries')
    if not all(x in body for x in ('METHODS', 'DISCUSSION')):
        raise ValueError('12456544：Methods/Discussion正文被截断')
    # 页面装饰在PDF每页重复，唯一可匹配的刊头/页码独立行直接删除。
    body, nav = re.subn(r'(?m)^(?:www\.archdischild\.com|The PIAMA study|Koopman, Wijga, Smit, et al|48[2-8])\s*$', '', body)
    if not 15 <= nav <= 26:
        raise ValueError(f'12456544：期刊页眉页脚数量异常({nav})')
    return body, (f'保留PDF中真实Introduction起点至讨论结尾；移除被右侧栏打断的前置摘要与出版信息{start}字符，'
                  f'及致谢/作者机构尾部{len(old)-end}字符；删去{nav}处页眉/页码')


def repair_19873223(old: str) -> tuple[str, str]:
    """只在原PDF书目标题能确认时截断，不校正历史OCR字词。"""
    with verified_pdf('19873223', pages=(21, 21)) as pdf:
        page = pdf[18].get_text('text', sort=False)
    if not re.search(r'(?im)^BIBLIOGRAPHY\s*$', page):
        raise ValueError('19873223：PDF文末BIBLIOGRAPHY无法核实')
    match = re.search(r'(?i)\bBIBLIOGRAPHY\s+Adametz,\s*L\.', old)
    if match is None or not 0.8 <= match.start() / len(old) <= 0.94:
        raise ValueError('19873223：正文参考文献边界不在核查范围')
    body = old[:match.start()].rstrip()
    if 'The possible structure of the active groups' not in body[-1000:]:
        raise ValueError('19873223：书目前Summary结尾缺失')
    return body, f'依原PDF第19页确认BIBLIOGRAPHY起点；截断书目{len(old)-len(body)}字符，历史OCR错字保持不变'


def repair_32625401(old: str) -> tuple[str, str]:
    """225页EFSA：只移除8个具有双侧边界的书目小节，保留所有后续附录。"""
    with verified_pdf('32625401', pages=(225, 225)) as pdf:
        # 检查实际有后续附录，不能把References之后全部舍弃。
        tail = pdf[-1].get_text('text', sort=False)
    if 'Uncertainty' not in tail and 'uncertaint' not in tail.casefold():
        raise ValueError('32625401：原PDF最终附录结构无法核实')
    ends = [
        'Appendix B – Time of development',
        'Overview of recovery potential for soil microorganisms.',
        'Appendix F – Summary of (extended) laboratory test systems',
        'Appendix G – ISO standards',
        'Appendix I – Advantages and disadvantages of methods',
        'Appendix J – Evaluation of the ECPA proposal',
        'Potential (semi)field approaches to the risk assessment',
        'Background\nUncertainty was identified',
    ]
    starts = list(re.finditer(r'(?m)^References\s*$', old))
    if len(starts) != 8:
        raise ValueError(f'32625401：预期8个附录参考文献区块，实际{len(starts)}个，拒绝修改')
    ranges = []
    for i, (match, anchor) in enumerate(zip(starts, ends)):
        stop = old.find(anchor, match.end())
        next_start = starts[i + 1].start() if i < 7 else len(old)
        if stop < 0 or not match.end() < stop < next_start:
            raise ValueError(f'32625401：第{i+1}个书目边界未匹配，拒绝修改')
        block = old[match.end():stop]
        # 必须实际包含多条作者-年份书目，才视为可删部分。
        if len(re.findall(r'\b(?:19|20)\d{2}\b', block)) < 2 or len(block) < 100:
            raise ValueError(f'32625401：第{i+1}个书目缺乏文献证据')
        ranges.append((match.start(), stop))
    result = old
    for a, b in reversed(ranges):
        result = result[:a].rstrip() + '\n\n' + result[b:].lstrip()
    if not all(x in result for x in ('Appendix B', 'Appendix D', 'Appendix F',
                                    'Appendix G', 'Appendix H', 'Appendix I', 'Appendix J',
                                    'Background\nUncertainty was identified')):
        raise ValueError('32625401：附录检查失败，拒绝替换')
    return result, f'对照225页PDF结构，保守去除8个由后续附录/小节界定的书目区块，共{sum(b-a for a,b in ranges)}字符；正文及后续附录保留'


def review_complete(folder: Path) -> bool:
    """核对180篇候选语料、逐篇正文以及修复报告的前后 SHA256。"""
    try:
        required = ('papers_180.jsonl', 'repair_config.json', 'repair_report.csv', 'needs_review.csv')
        if not folder.is_dir() or not all((folder / p).is_file() for p in required):
            return False
        config = json.loads((folder / 'repair_config.json').read_text(encoding='utf-8'))
        if config.get('version') != VERSION or config.get('source') != 'cleaned_180/papers_180.jsonl':
            return False
        rows = [json.loads(s) for s in (folder / 'papers_180.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        ids = [str(row['pmid']) for row in rows]
        if len(ids) != 180 or len(ids) != len(set(ids)) or [r.get('row_index') for r in rows] != list(range(180)):
            return False
        with (folder / 'repair_report.csv').open(encoding='utf-8-sig', newline='') as f:
            report = list(csv.DictReader(f))
        with (folder / 'needs_review.csv').open(encoding='utf-8-sig', newline='') as f:
            reviews = list(csv.DictReader(f))
        if len(report) != 180 or {r['pmid'] for r in report} != set(ids):
            return False
        if {r['pmid'] for r in reviews} != {r['pmid'] for r in rows if r.get('quality_status') == 'review'}:
            return False
        if {p.stem for p in (folder / 'body').glob('*.txt')} != set(ids):
            return False
        by_report = {r['pmid']: r for r in report}
        for row in rows:
            pid = str(row['pmid'])
            body = str(row['body_text'])
            if (folder / 'body' / (pid + '.txt')).read_text(encoding='utf-8') != body + '\n':
                return False
            if sha(body) != by_report[pid]['after_sha256']:
                return False
        if SOURCE.is_file():
            source = [json.loads(s) for s in SOURCE.read_text(encoding='utf-8').splitlines() if s.strip()]
            if len(source) != 180 or [str(r['pmid']) for r in source] != ids:
                return False
            for row in source:
                if sha(str(row.get('body_text', ''))) != by_report[str(row['pmid'])]['before_sha256']:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description='15篇PDF人工核查后的定向修复（固定180篇，不改动原始文件）')
    parser.add_argument('--force', action='store_true', help='替换本脚本上次生成的候选目录')
    args = parser.parse_args()
    if not SOURCE.is_file():
        parser.error(f'找不到输入：{SOURCE}，请先运行 2.1clean_corpus.py')
    if TARGET.exists() and not args.force and review_complete(TARGET):
        print('2.2：已有完整人工复核结果，跳过。')
        return
    rows = [json.loads(x) for x in SOURCE.read_text('utf-8').splitlines() if x.strip()]
    ids = [str(x.get('pmid', '')) for x in rows]
    expected_targets = {'2257383', '12194887', '6487732', '17246411',
                        '8761444', '12456544', '19873223', '32625401'}
    if len(rows) != 180 or len(set(ids)) != 180 or not expected_targets <= set(ids):
        parser.error('当前输入并非预期180篇语料，或缺少定向修复目标论文，停止')
    fixes = {'2257383': repair_2257383, '12194887': repair_12194887,
             '6487732': lambda old: pdf_layout_repair('6487732', old),
             '17246411': lambda old: pdf_layout_repair('17246411', old),
             '8761444': repair_8761444,
             '12456544': repair_12456544,
             '19873223': repair_19873223,
             '32625401': repair_32625401}
    # 只用其余论文里独立出现的单词作断词词表，不下载模型/词典。
    corpus_words = set()
    for row in rows:
        if str(row['pmid']) != '12194887':
            corpus_words.update(w.casefold() for w in re.findall(
                r'[A-Za-z]{5,}', str(row.get('body_text', ''))))
    stage = Path(tempfile.mkdtemp(prefix='.review_candidate_', dir=ROOT))
    try:
        (stage / 'body').mkdir()
        logs = []
        for row in rows:
            pmid = str(row['pmid'])
            old = str(row.get('body_text', ''))
            if not old.strip():
                raise ValueError(f'{pmid}：输入正文为空')
            new, note = (repair_12194887(old, corpus_words) if pmid == '12194887'
                         else fixes[pmid](old) if pmid in fixes
                         else (old, MANUAL.get(pmid, '未修改')))
            if pmid in fixes:
                row['body_text'] = new
                row.setdefault('cleaning_actions', []).append('targeted_repair_' + pmid)
                old_warnings = list(row.get('cleaning_warnings') or [])
                resolved = ({'pdf_words_joined_together_review'} if pmid == '2257383'
                            else {'extract:possible_abstract_duplicate'}
                            if pmid in {'12194887', '17246411', '8761444', '12456544'}
                            else set())
                row['cleaning_warnings'] = [x for x in old_warnings if x not in resolved]
                row['cleaning_warnings'].append('targeted_repair_pending_manual_verification')
                row['quality_status'] = 'review'
            (stage / 'body' / f'{pmid}.txt').write_text(new + '\n', encoding='utf-8')
            logs.append({
                'pmid': pmid, 'action': ('repaired_candidate' if pmid in fixes else
                                      'manual_review_only' if pmid in MANUAL else 'unchanged'),
                'before_chars': len(old), 'after_chars': len(new),
                'before_sha256': sha(old), 'after_sha256': sha(new),
                'note': note,
            })
        with (stage / 'papers_180.jsonl').open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        with (stage / 'repair_report.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(logs[0]))
            writer.writeheader(); writer.writerows(logs)
        with (stage / 'needs_review.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['pmid', 'warnings'])
            writer.writeheader()
            for row in rows:
                if row.get('quality_status') == 'review':
                    writer.writerow({'pmid': row['pmid'], 'warnings': '; '.join(row.get('cleaning_warnings', []))})
        (stage / 'repair_config.json').write_text(json.dumps({
            'version': VERSION, 'source': 'cleaned_180/papers_180.jsonl',
            'repaired': list(fixes), 'manual_review_only': MANUAL,
            'note': '候选语料；原始文件未修改，定向修复后的正文仍保留PDF复核状态，图表未逐项人工认证。',
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if not review_complete(stage):
            raise RuntimeError('新生成的 cleaned_180_checked/ 未通过完整性检查，保留旧结果')
        backup = ROOT / '.cleaned_180_checked_backup'
        if TARGET.exists():
            if backup.exists():
                raise FileExistsError(f'旧备份仍在：{backup}，请人工检查后再执行')
            TARGET.rename(backup)
        try:
            stage.rename(TARGET)
        except Exception:
            if backup.exists():
                backup.rename(TARGET)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        counts = Counter(row['action'] for row in logs)
        print('完成：' + '，'.join(f'{k}={v}' for k, v in counts.items()))
        for row in logs:
            if row['action'] != 'unchanged':
                print(f"  {row['pmid']} {row['action']}：{row['before_chars']}→{row['after_chars']}字符")
        print('候选语料：', TARGET / 'papers_180.jsonl')
        print('审查报告：', TARGET / 'repair_report.csv')
        print('原 cleaned_180/ 未修改。')
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == '__main__':
    main()
