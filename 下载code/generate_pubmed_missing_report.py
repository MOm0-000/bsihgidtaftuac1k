#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描本地 PDF / FullText，生成可筛选的 PubMed 缺失文献 HTML 导航报告。

默认使用：
    Excel:  下载code/zzx.xlsx
    目录:   reference/
    HTML:   reference/pubmed_missing_report.html

依赖：Python 3.9+ 标准库；不联网、不下载、不修改 PDF/HTML。
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXCEL = Path(__file__).resolve().parent / 'zzx.xlsx'
DEFAULT_ROOT = PROJECT_ROOT / 'reference'
OUTPUT_NAME = 'pubmed_missing_report.html'

# 以前的 64 篇失败文献的离线元数据，方便在没有网络时显示 PMC、PMC PDF、DOI。
# 对于其它文献，还会尝试读取本地“下载报告.csv”补充信息。
EMBEDDED_METADATA = json.loads(r'''{"21288949":{"title":"Sib mating without inbreeding in the longhorn crazy ant.","pmcid":"PMC3136830","doi":"10.1098/rspb.2010.2562"},"29363738":{"title":"Aeroallergen sensitization predicts acute chest syndrome in children with sickle cell anaemia.","pmcid":"PMC5945280","doi":"10.1111/bjh.15076"},"24926113":{"title":"Individual based and mean-field modeling of direct aggregation.","pmcid":"PMC4047626","doi":"10.1016/j.physd.2012.11.003"},"27215490":{"title":"Claudin-18 deficiency is associated with airway epithelial barrier dysfunction and asthma.","pmcid":"PMC5073041","doi":"10.1016/j.jaci.2016.02.035"},"28880992":{"title":"Innate lymphoid cells at the interface between obesity and asthma.","pmcid":"PMC5721241","doi":"10.1111/imm.12832"},"25297317":{"title":"Finite-element modelling reveals force modulation of jaw adductors in stag beetles.","pmcid":"PMC4223912","doi":"10.1098/rsif.2014.0908"},"34432544":{"title":"Determining the Best Tool Comparable with Global Initiative for Asthma Criteria for Assessing Pediatric Asthma Control.","pmcid":"PMC8664125","doi":"10.1089/ped.2020.1334"},"23486880":{"title":"Estimating cerebral microinfarct burden from autopsy samples.","pmcid":"PMC3662273","doi":"10.1212/WNL.0b013e31828c2f52"},"40011409":{"title":"2025 ACMT Annual Scientific Meeting Abstracts - Vancouver, Canada.","pmcid":"PMC11933582","doi":"10.1007/s13181-025-01061-0"},"17246411":{"title":"A Comprehensive Study of Genic Variation in Natural Populations of Drosophila melanogaster. III. Variations in Genetic Structure and Their Causes between Drosophila melanogaster and Its Sibling Species Drosophila simulans.","pmcid":"PMC1203242","doi":"10.1093/genetics/117.4.697"},"25144551":{"title":"Assessing asthma control and associated risk factors among persons with current asthma - findings from the child and adult Asthma Call-back Survey.","pmcid":"PMC4552346","doi":"10.3109/02770903.2014.956894"},"23038250":{"title":"Glutamate-gated chloride channels.","pmcid":"PMC3504739","doi":"10.1074/jbc.R112.406280"},"22920660":{"title":"Is rural school-aged children's quality of life affected by their responses to asthma?","pmcid":"PMC3428597","doi":"10.1016/j.pedn.2011.06.012"},"19382295":{"title":"Diverging functions of Scr between embryonic and post-embryonic development in a hemimetabolous insect, Oncopeltus fasciatus.","pmcid":"PMC2775506","doi":"10.1016/j.ydbio.2009.01.032"},"32299029":{"title":"Associations between prenatal exposure to DDT and DDE and allergy symptoms and diagnoses in the Venda Health Examination of Mothers, Babies and their Environment (VHEMBE), South Africa.","pmcid":"PMC7336873","doi":"10.1016/j.envres.2020.109366"},"25290173":{"title":"Distinct dendritic cell subsets actively induce Th2 polarization.","pmcid":"PMC4737705","doi":"10.1016/j.coi.2014.09.006"},"18025266":{"title":"Genomics, transcriptomics, and peptidomics of neuropeptides and protein hormones in the red flour beetle Tribolium castaneum.","pmcid":"PMC2134770","doi":"10.1101/gr.6714008"},"8761444":{"title":"The lipocalin protein family: structure and function.","pmcid":"PMC1217580","doi":"10.1042/bj3180001"},"19647861":{"title":"Total IgE levels and asthma prevalence in the US population: results from the National Health and Nutrition Examination Survey 2005-2006.","pmcid":"PMC2758573","doi":"10.1016/j.jaci.2009.06.011"},"14450716":{"title":"A bookshelf on environmental health.","pmcid":"PMC1522886","doi":"10.2105/ajph.52.4.559"},"36081367":{"title":"Geospatial data in pediatric asthma in the United States: a scoping review protocol.","pmcid":"PMC9669090","doi":"10.11124/JBIES-21-00284"},"16995156":{"title":"Afferent impulses from the teeth resulting from a vibratory stimulus.","pmcid":"PMC1393894","doi":"10.1113/jphysiol.1939.sp003801"},"21048818":{"title":"Arboprotozoae.","pmcid":"PMC2928831","doi":"10.1159/000187642"},"17704070":{"title":"Unsteady locomotion: integrating muscle function with whole body dynamics and neuromuscular control.","pmcid":"PMC2651961","doi":"10.1242/jeb.005801"},"7057412":{"title":"Moodiness: a heavenly problem? Observations on the pathology of moodiness.","pmcid":"PMC1437376","doi":"10.1177/014107688207500104"},"21326971":{"title":"Indoor air mycoflora of residential dwellings in Jos metropolis.","pmcid":"PMC2956288","doi":""},"12456544":{"title":"Early respiratory and skin symptoms in relation to ethnic background: the importance of socioeconomic status; the PIAMA study.","pmcid":"PMC1755839","doi":"10.1136/adc.87.6.482"},"31807421":{"title":"Correlation between use of immunosuppressive agents and transplant-acquired allergies in renal transplant recipients.","pmcid":"PMC6842773","doi":"10.21037/tau.2019.09.24"},"2257383":{"title":"Health effects of housing status on children: a house is not a home.","pmcid":"PMC1809768","doi":""},"8457977":{"title":"Compounds active against cell walls of medically important fungi.","pmcid":"PMC358263","doi":"10.1128/CMR.6.1.1"},"25379034":{"title":"Medicinal history of North American Veratrum.","pmcid":"PMC4217314","doi":"10.1007/s11101-013-9328-y"},"29084804":{"title":"Conservation and variation in pair-rule gene expression and function in the intermediate-germ beetle Dermestes maculatus.","pmcid":"PMC5769621","doi":"10.1242/dev.154039"},"32526226":{"title":"Ex vivo characterization of the circulating hemocytes of bed bugs and their responses to bacterial exposure.","pmcid":"PMC9254597","doi":"10.1016/j.jip.2020.107422"},"29118130":{"title":"What have humans done for evolutionary biology? Contributions from genes to populations.","pmcid":"PMC5698635","doi":"10.1098/rspb.2017.1164"},"21123808":{"title":"Carbohydrates contribute to the interactions between cockroach allergen Bla g 2 and a monoclonal antibody.","pmcid":"PMC3099132","doi":"10.4049/jimmunol.1002318"},"21983428":{"title":"Differential state-dependent modification of rat Na(v)1.6 sodium channels expressed in human embryonic kidney (HEK293) cells by the pyrethroid insecticides tefluthrin and deltamethrin.","pmcid":"PMC3226847","doi":"10.1016/j.taap.2011.09.021"},"24161540":{"title":"Advances in allergen-microarray technology for diagnosis and monitoring of allergy: the MeDALL allergen-chip.","pmcid":"PMC4687054","doi":"10.1016/j.ymeth.2013.10.008"},"28689792":{"title":"Functional role of kynurenine and aryl hydrocarbon receptor axis in chronic rhinosinusitis with nasal polyps.","pmcid":"PMC5937692","doi":"10.1016/j.jaci.2017.06.013"},"21216485":{"title":"Gene-environment interactions in human disease: nuisance or opportunity?","pmcid":"PMC3073697","doi":"10.1016/j.tig.2010.12.004"},"9108117":{"title":"Conserved and sexually dimorphic behavioral responses to biogenic amines in decapitated Drosophila.","pmcid":"PMC20580","doi":"10.1073/pnas.94.8.4131"},"33094817":{"title":"Relevance of Electrical Light on Circadian, Neuroendocrine, and Neurobehavioral Regulation in Laboratory Animal Facilities.","pmcid":"PMC7947598","doi":"10.1093/ilar/ilaa010"},"5960261":{"title":"Deposition of labeled beta-alanine in ebony and non-ebony Drosophila melanogaster with notes on other amino acids.","pmcid":"PMC1211057","doi":"10.1093/genetics/53.4.777"},"33661048":{"title":"A low-cost computational approach to analyze spiking activity in cockroach sensory neurons.","pmcid":"PMC8091935","doi":"10.1152/advan.00034.2020"},"21645908":{"title":"Comparison of biomarkers and parent report of tobacco exposure to predict wheeze.","pmcid":"PMC3170672","doi":"10.1016/j.jpeds.2011.04.025"},"35862663":{"title":"Comparative Analysis of Brucepastera parasyntrophica gen. nov., sp. nov. and Teretinema zuelzerae gen. nov., comb. nov. (Treponemataceae) Reveals the Importance of Interspecies Hydrogen Transfer in the Energy Metabolism of Spirochetes.","pmcid":"PMC9317865","doi":"10.1128/aem.00503-22"},"10369463":{"title":"Minor structural changes in nicotinoid insecticides confer differential subtype selectivity for mammalian nicotinic acetylcholine receptors.","pmcid":"PMC1566001","doi":"10.1038/sj.bjp.0702526"},"23658153":{"title":"Widespread sensitivity to looming stimuli and small moving objects in the central complex of an insect brain.","pmcid":"PMC6619629","doi":"10.1523/JNEUROSCI.5390-12.2013"},"19553462":{"title":"Suppression of spreading depression-like events in locusts by inhibition of the NO/cGMP/PKG pathway.","pmcid":"PMC6666049","doi":"10.1523/JNEUROSCI.1652-09.2009"},"16141137":{"title":"Clean drinking water for homes in Africa and other less developed countries.","pmcid":"PMC1199012","doi":"10.1136/bmj.331.7515.468"},"34260722":{"title":"Comparison of Diet Preferences of Laboratory-Reared and Apartment-Collected German Cockroaches.","pmcid":"PMC12102598","doi":"10.1093/jee/toab139"},"22115961":{"title":"Distinct roles of isoforms of the heme-liganded nuclear receptor E75, an insect ortholog of the vertebrate Rev-erb, in mosquito reproduction.","pmcid":"PMC3306807","doi":"10.1016/j.mce.2011.11.006"},"24972604":{"title":"Quantitative investigations of axonal and dendritic arbors: development, structure, function, and pathology.","pmcid":"PMC4723268","doi":"10.1177/1073858414540216"},"29279788":{"title":"Diet, Lung Function, and Asthma Exacerbations in Puerto Rican Children.","pmcid":"PMC5733741","doi":"10.1089/ped.2017.0803"},"31866066":{"title":"Structure of the Cardiac Sodium Channel.","pmcid":"PMC6986426","doi":"10.1016/j.cell.2019.11.041"},"26307762":{"title":"Diverse set of Turing nanopatterns coat corneae across insect lineages.","pmcid":"PMC4553778","doi":"10.1073/pnas.1505748112"},"6487732":{"title":"A method for testing an extended poisson hypothesis of spontaneous quantal transmitter release at neuromuscular junctions.","pmcid":"PMC1434949","doi":"10.1016/S0006-3495%2884%2984028-X"},"28722633":{"title":"Characterization of Sarcoptes scabiei Tropomyosin and Paramyosin: Immunoreactive Allergens in Scabies.","pmcid":"PMC5590589","doi":"10.4269/ajtmh.16-0976"},"38166479":{"title":"Mechanosensory encoding of forces in walking uphill and downhill: force feedback can stabilize leg movements in stick insects.","pmcid":"PMC11286306","doi":"10.1152/jn.00414.2023"},"24407477":{"title":"Polycyclic aromatic hydrocarbon exposure, obesity and childhood asthma in an urban cohort.","pmcid":"PMC3912566","doi":"10.1016/j.envres.2013.12.002"},"22532821":{"title":"Insect stings: clinical features and management.","pmcid":"PMC3334720","doi":"10.3238/arztebl.2012.0238"},"22822349":{"title":"Overview of Reviews The prevention of eczema in infants and children: an overview of Cochrane and non-Cochrane reviews.","pmcid":"PMC3399595","doi":"10.1002/ebch.827"},"34697248":{"title":"Epoxidation of juvenile hormone was a key innovation improving insect reproductive fitness.","pmcid":"PMC8609300","doi":"10.1073/pnas.2109381118"},"33271296":{"title":"Efficacy of type 2-targeted biologics in patients with asthma and bronchiectasis.","pmcid":"PMC8028023","doi":"10.1016/j.anai.2020.11.014"},"12876197":{"title":"Molecular genetics and evolution of pheromone biosynthesis in Lepidoptera.","pmcid":"PMC170892","doi":"10.1073/pnas.1233767100a"}}''')


def read_pmids_xlsx(path: Path) -> list[tuple[str, str]]:
    """仅使用标准库读取 xlsx 中 A=序号、B=PMID（首个工作表）。"""
    if not path.is_file():
        raise FileNotFoundError(f'找不到 Excel 文件：{path}')
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as zf:
        shared = []
        if 'xl/sharedStrings.xml' in zf.namelist():
            tree = ET.fromstring(zf.read('xl/sharedStrings.xml'))
            for si in tree.findall('m:si', ns):
                shared.append(''.join(t.text or '' for t in si.findall('.//m:t', ns)))
        # 本任务的 zzx.xlsx 采用第一个工作表 sheet1.xml。
        sheet_name = 'xl/worksheets/sheet1.xml'
        if sheet_name not in zf.namelist():
            matches = sorted(n for n in zf.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', n))
            if not matches:
                raise ValueError('Excel 内没有可读取的工作表')
            sheet_name = matches[0]
        worksheet = ET.fromstring(zf.read(sheet_name))

    result = []
    seen = set()
    for row in worksheet.findall('.//m:sheetData/m:row', ns):
        data = {}
        for cell in row.findall('m:c', ns):
            ref = cell.get('r', '')
            col = re.match(r'[A-Z]+', ref)
            if not col or col.group() not in {'A', 'B'}:
                continue
            raw = cell.find('m:v', ns)
            inline = cell.find('m:is', ns)
            value = raw.text if raw is not None and raw.text else ''
            if cell.get('t') == 's' and value:
                value = shared[int(value)]
            elif cell.get('t') == 'inlineStr' and inline is not None:
                value = ''.join(t.text or '' for t in inline.findall('.//m:t', ns))
            value = str(value).strip()
            data[col.group()] = value
        pmid = data.get('B', '').split('.')[0].strip()
        order = data.get('A', '').split('.')[0].strip()
        if not pmid.isdigit() or pmid in seen:
            continue
        seen.add(pmid)
        result.append((order, pmid))
    if not result:
        raise ValueError('未在 Excel B 列中找到有效 PMID，请检查 zzx.xlsx 内容')
    return result


def get_local_metadata(root: Path) -> dict[str, dict]:
    """用历史下载报告补充原先 64 篇之外的 PMCID/DOI/标题，不联网。"""
    metas = {k: dict(v) for k, v in EMBEDDED_METADATA.items()}
    report_path = root / '下载报告.csv'
    if not report_path.is_file():
        return metas
    try:
        with report_path.open('r', encoding='utf-8-sig', newline='') as f:
            for record in csv.DictReader(f):
                pmid = (record.get('PMID') or '').strip()
                if not pmid.isdigit():
                    continue
                old = metas.setdefault(pmid, {})
                for k, names in {
                    'title': ('标题', '题名', 'Title'),
                    'pmcid': ('PMCID',),
                    'doi': ('DOI',),
                }.items():
                    for name in names:
                        value = (record.get(name) or '').strip()
                        if value and not old.get(k):
                            old[k] = value
                            break
    except (OSError, UnicodeError, csv.Error) as exc:
        print(f'[提示] 旧下载报告无法读取，继续使用内置元数据：{exc}', file=sys.stderr)
    return metas


def file_state(path: Path, kind: str) -> tuple[bool, str]:
    """检查有效的 PDF/HTML 文件，防止 0 字节和错误页面被计为成功。"""
    if not path.is_file():
        return False, '不存在'
    try:
        size = path.stat().st_size
        with path.open('rb') as f:
            header = f.read(4096)
        if kind == 'PDF':
            if size < 500 or not header.startswith(b'%PDF-'):
                return False, '文件疑似损坏/不是PDF'
        else:
            h = header.lower()
            if size < 200 or not any(sig in h for sig in (b'<html', b'<!doctype html', b'<body')):
                return False, '文件疑似损坏/不是HTML'
            if size < 10000 and any(bad in h for bad in (
                b'access denied', b'captcha', b'just a moment', b'error 403', b'403 forbidden'
            )):
                return False, '疑似网站拦截页面'
        return True, f'{size / 1024:.1f} KB'
    except OSError as exc:
        return False, f'无法读取：{exc}'


def safe_url(url: str) -> str:
    return html.escape(url, quote=True)


def link(label: str, url: str, css: str = '') -> str:
    if not url:
        return ''
    return (f'<a class="{css}" target="_blank" rel="noopener noreferrer" '
            f'href="{safe_url(url)}">{html.escape(label)}</a>')


def render(rows: list[dict], out_path: Path) -> str:
    count = len(rows)
    done = sum(x['done'] for x in rows)
    missing_pdf = sum(not x['pdf_ok'] for x in rows)
    missing_html = sum(not x['html_ok'] for x in rows)
    missing_both = sum(not x['html_ok'] and not x['pdf_ok'] for x in rows)
    missing_any = count - done
    issue_count = sum(bool(x['pdf_issue']) + bool(x['html_issue']) for x in rows)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    trs = []
    for r in rows:
        pmid, order = r['pmid'], r['order']
        meta = r['meta']
        title = html.escape(meta.get('title') or '未缓存标题（点击 PubMed 查看）')
        pmcid = str(meta.get('pmcid') or '').strip().upper()
        doi = str(meta.get('doi') or '').strip()
        if not re.fullmatch(r'PMC\d+', pmcid):
            pmcid = ''
        pubmed = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
        pmc = f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/' if pmcid else ''
        pdf_link = f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/' if pmcid else ''
        doi_link = 'https://doi.org/' + quote(doi, safe='/') if doi else ''
        resources = ' '.join(filter(None, [
            link('PubMed', pubmed), link('PMC', pmc),
            link('PMC PDF', pdf_link), link('DOI', doi_link)
        ]))
        # 相对链接：HTML 与 PDF/FullText 文件夹位于同一级目录。
        pdf_status = (link('✓ 已有 PDF', f'PDF/{pmid}.pdf', 'local')
                      if r['pdf_ok'] else '<span class="bad">✗ 缺 PDF</span>')
        html_status = (link('✓ 已有 HTML', f'FullText/{pmid}.html', 'local')
                       if r['html_ok'] else '<span class="bad">✗ 缺 HTML</span>')
        problems = []
        if r['pdf_issue']:
            problems.append('PDF：' + r['pdf_issue'])
        if r['html_issue']:
            problems.append('HTML：' + r['html_issue'])
        issue_display = html.escape('；'.join(problems)) if problems else '—'
        state = ('done' if r['done'] else
                 'both' if not r['pdf_ok'] and not r['html_ok'] else
                 'pdf' if not r['pdf_ok'] else 'html')
        trs.append(f'''<tr data-state="{state}" data-search="{html.escape((pmid + ' ' + order + ' ' + (meta.get('title') or '')).lower(), quote=True)}">
          <td>{html.escape(order)}</td><td class="mono">{pmid}</td>
          <td class="article">{title}</td>
          <td>{pdf_status}</td><td>{html_status}</td>
          <td class="links">{resources}</td>
          <td class="note">{issue_display}</td></tr>''')

    # 文件中的 JS 只做本地筛选，不发起任何网络请求。
    template = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PubMed 文献缺失统计 · 本地检查</title>
<style>
:root{font-family:system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;color:#223044;background:#f5f7fb}
*{box-sizing:border-box}body{margin:0}main{max-width:1540px;margin:auto;padding:28px 22px 48px}
h1{font-size:28px;margin:0 0 8px}p{color:#576579;line-height:1.8;margin:6px 0 16px}
.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:21px 0}
.card{background:#fff;border:1px solid #e3eaf3;border-radius:13px;padding:16px;box-shadow:0 2px 8px #172b4d08}
.card .num{font-size:28px;font-weight:750;color:#1459a9}.card .label{color:#52627b;font-size:13px;margin-top:5px}
.toolbar{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin:18px 0}
.filters{display:flex;gap:7px;flex-wrap:wrap}.filters button{cursor:pointer;background:#fff;color:#345272;border:1px solid #d4ddea;border-radius:8px;padding:9px 12px}
.filters button.selected{background:#1459a9;color:#fff;border-color:#1459a9}
#search{padding:10px 12px;width:min(330px,100%);font:inherit;border:1px solid #d4ddea;border-radius:8px}
.tablewrap{background:#fff;border:1px solid #dfe6f0;border-radius:12px;overflow:auto;box-shadow:0 3px 12px #172b4d08}
table{border-collapse:collapse;width:100%;min-width:1140px;font-size:13.5px}th,td{padding:13px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #e9edf4}
th{position:sticky;top:0;background:#edf4fc;color:#294869;z-index:2;white-space:nowrap}
tbody tr:hover{background:#f6fafe}td.article{min-width:230px;max-width:360px;line-height:1.5}
td.links{min-width:260px;max-width:380px}a{color:#1459a9;text-decoration:none}a:hover{text-decoration:underline}
td.links a{display:inline-block;padding:3px 6px;margin:0 4px 5px 0;background:#eaf3fe;border-radius:4px}
a.local{color:#067242}.bad{font-weight:650;color:#b32323}.note{color:#7b8390;max-width:220px;font-size:12px}
.mono{font-variant-numeric:tabular-nums;white-space:nowrap}small{color:#65758b}
@media(max-width:850px){.cards{grid-template-columns:repeat(2,1fr)}main{padding:20px 12px}h1{font-size:24px}}
@media print{.toolbar{display:none}main{padding:0}.tablewrap{overflow:visible}th{position:static}}
</style></head><body><main>
<h1>📚 PubMed 文献缺失统计与手动下载导航</h1>
<p>生成时间：__NOW__。数据仅来自 <strong>zzx.xlsx</strong> 和本地 <strong>PDF/、FullText/</strong> 文件夹；<strong>本脚本不联网、不自动下载。</strong><br>
默认只显示缺失文献。点击「PMC」查看完整正文并保存 HTML，点击「PMC PDF」尝试获取 PDF；如果缺少 PMC 或 DOI 链接，可从 PubMed 页面进入全文来源。</p>
<div class="cards">
 <div class="card"><div class="num">__TOTAL__</div><div class="label">Excel 总文献（篇）</div></div>
 <div class="card"><div class="num">__DONE__</div><div class="label">PDF + HTML 均齐全（篇）</div></div>
 <div class="card"><div class="num">__ANY__</div><div class="label">至少缺一种格式（篇）</div></div>
 <div class="card"><div class="num">__PDF__</div><div class="label">缺 PDF（份）</div></div>
 <div class="card"><div class="num">__HTML__</div><div class="label">缺 HTML（份）</div></div>
</div>
<div class="toolbar"><div class="filters">
<button class="selected" data-filter="missing">待补全部 (__ANY__)</button>
<button data-filter="pdf">只缺 PDF</button><button data-filter="html">只缺 HTML</button>
<button data-filter="both">两种都缺 (__BOTH__)</button>
<button data-filter="done">已完成</button><button data-filter="all">全部 __TOTAL__ 篇</button>
</div><input id="search" placeholder="搜索 PMID / 序号 / 标题" aria-label="搜索文献"></div>
<p id="visibleCount" aria-live="polite"></p>
<div class="tablewrap"><table><thead><tr><th>序号</th><th>PMID</th><th>文章标题</th><th>PDF</th><th>Full Text HTML</th><th>网页链接</th><th>文件检查备注</th></tr></thead>
<tbody>__ROWS__</tbody></table></div>
<p><small>注意：本地检查只能判断文件形式是否合理，不能保证论文正文内容完整。HTML 请保存 PMC / 期刊的完整正文页，不要保存仅有摘要的 PubMed 页面。检测到异常文件不会自动删除；异常文件计为缺失。生成页面请保存于 reference 根目录，以保证已有文件的相对链接可用。</small></p>
</main><script>
(()=>{let filter='missing';const tbody=document.querySelector('tbody'), trs=[...tbody.querySelectorAll('tr')], search=document.getElementById('search');
function apply(){const q=search.value.trim().toLowerCase();let n=0;for(const tr of trs){let state=tr.dataset.state;let fits=filter==='all'||(filter==='missing'&&state!=='done')||state===filter;let visible=fits&&tr.dataset.search.includes(q);tr.hidden=!visible;tr.style.display=visible?'':'none';if(visible)n++}document.getElementById('visibleCount').textContent='当前显示 '+n+' 篇 · 文件状态以页面生成时为准，手动补完后重新运行 Python 脚本刷新。'}
for(const b of document.querySelectorAll('button[data-filter]'))b.addEventListener('click',()=>{document.querySelector('.selected')?.classList.remove('selected');b.classList.add('selected');filter=b.dataset.filter;apply()});search.addEventListener('input',apply);apply();})();
</script></body></html>'''
    for key, value in {
        '__NOW__': html.escape(now), '__TOTAL__': str(count), '__DONE__': str(done),
        '__ANY__': str(missing_any), '__PDF__': str(missing_pdf),
        '__HTML__': str(missing_html), '__BOTH__': str(missing_both),
        '__ROWS__': '\n'.join(trs)
    }.items():
        template = template.replace(key, value)
    print(f'异常/损坏文件：{issue_count} 个（如果有，已计入缺失）')
    return template


def main() -> int:
    ap = argparse.ArgumentParser(description='本地扫描文献 PDF/HTML，生成可交互的缺失统计网页（不联网）')
    ap.add_argument('--excel', type=Path, default=DEFAULT_EXCEL, help=f'默认：{DEFAULT_EXCEL}')
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT, help=f'默认：{DEFAULT_ROOT}')
    args = ap.parse_args()
    try:
        papers = read_pmids_xlsx(args.excel)
        root = args.root
        pdf_dir = root / 'PDF'
        html_dir = root / 'FullText'
        if not pdf_dir.is_dir() or not html_dir.is_dir():
            raise FileNotFoundError(f'请确认目录存在：\n  {pdf_dir}\n  {html_dir}')
        metas = get_local_metadata(root)
        rows = []
        for order, pmid in papers:
            pdf_ok, pdf_detail = file_state(pdf_dir / f'{pmid}.pdf', 'PDF')
            html_ok, html_detail = file_state(html_dir / f'{pmid}.html', 'HTML')
            rows.append({
                'order': order, 'pmid': pmid, 'meta': metas.get(pmid, {}),
                'pdf_ok': pdf_ok, 'html_ok': html_ok,
                'pdf_issue': pdf_detail if not pdf_ok and pdf_detail != '不存在' else '',
                'html_issue': html_detail if not html_ok and html_detail != '不存在' else '',
                'done': pdf_ok and html_ok,
            })
        output = root / OUTPUT_NAME
        output.write_text(render(rows, output), encoding='utf-8')
        c = Counter({
            'pdf_ok': sum(r['pdf_ok'] for r in rows),
            'html_ok': sum(r['html_ok'] for r in rows),
            'done': sum(r['done'] for r in rows),
        })
        print(f'Excel 要求总数：{len(rows)} 篇')
        print(f'PDF 已有：{c["pdf_ok"]} 篇；缺少：{len(rows) - c["pdf_ok"]} 篇')
        print(f'HTML 已有：{c["html_ok"]} 篇；缺少：{len(rows) - c["html_ok"]} 篇')
        print(f'两种都齐全：{c["done"]} 篇；仍需人工补充：{len(rows) - c["done"]} 篇')
        print(f'HTML 已生成：{output}')
        return 0
    except (OSError, ValueError, ET.ParseError) as exc:
        print(f'生成失败：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
