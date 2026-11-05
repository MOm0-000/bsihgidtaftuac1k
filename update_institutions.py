#!/usr/bin/env python3
"""按 PMID 补充人工核实的机构信息，生成 papers_metadata_reviewed.csv。"""

import csv
from pathlib import Path

FOLDER = Path('/home/zxx/bigdata/exp1/reference/features')
INPUT = FOLDER / 'papers_metadata.csv'
OUTPUT = FOLDER / 'papers_metadata_reviewed.csv'

# PMID: (机构名称, 核查来源, 特别说明)
# 注意：会议摘要合集没有统一机构，保留 institutions 为空。
CORRECTIONS = {
    '24937928': ('Clinical Center of Banja Luka; Health Center, Banja Luka', 'manual_pdf', ''),
    '40011409': ('', 'manual_no_unified_institution', '会议摘要合集：各摘要有自己的作者及机构，没有统一所属机构'),
    '14450716': ('Columbia University School of Public Health and Administrative Medicine (Division of Sanitary Science)', 'manual_pdf', ''),
    '16995156': ('Physiological Laboratory, Cambridge', 'manual_pdf', '原文未明确写大学全称'),
    '21048818': ('Paul-Ehrlich-Institut', 'manual_correspondence', '通讯联系人/工作组关联机构；并非全部作者的所属机构'),
    '7057412': ("St Thomas' Hospital, London", 'manual_pdf', ''),
    '21433778': ('Yale University', 'manual_external', '根据作者同期任职资料核实；非本论文直接署名'),
    '32625401': ('European Food Safety Authority (EFSA)', 'manual_group_author', 'PPR 专家组署名'),
    '6487732': ('Hosei University; Juntendo University; FUJITSU Limited', 'manual_pdf', ''),
}


def main():
    if not INPUT.is_file():
        raise SystemExit(f'找不到输入文件：{INPUT}')

    with INPUT.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if not {'pmid', 'institutions'}.issubset(headers):
            raise SystemExit('CSV 必须包含 pmid 和 institutions 两列')
        rows = list(reader)

    found = set()
    for row in rows:
        pmid = row['pmid'].strip()
        if pmid not in CORRECTIONS:
            continue
        institution, source, note = CORRECTIONS[pmid]
        row['institutions'] = institution
        if 'institutions_source' in headers:
            row['institutions_source'] = source
        if note and 'notes' in headers:
            original = row['notes'].strip()
            row['notes'] = (original + ' | ' if original else '') + note
        found.add(pmid)
        print(f'{pmid}: {institution or "无统一机构（保持空值）"}')

    with OUTPUT.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n共 {len(rows)} 篇，核对 {len(found)}/{len(CORRECTIONS)} 篇。')
    if missing := set(CORRECTIONS) - found:
        print('未匹配 PMID：' + ', '.join(sorted(missing)))
    print(f'输出：{OUTPUT}')
    print('原始 papers_metadata.csv 未修改。')


if __name__ == '__main__':
    main()
