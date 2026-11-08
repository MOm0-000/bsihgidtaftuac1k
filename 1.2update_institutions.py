#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充已人工核实的机构和说明，规范缺失来源；不修改原始元数据CSV。

目录：脚本位于项目根目录，输入 extracted/papers_metadata.csv，
输出 extracted/papers_metadata_reviewed.csv。无需命令行参数。
"""

import csv
from pathlib import Path

FOLDER = Path(__file__).resolve().parent / "extracted"
INPUT = FOLDER / "papers_metadata.csv"
OUTPUT = FOLDER / "papers_metadata_reviewed.csv"

# PMID: (核实后的机构, 核实来源, 必要的人工说明)
CORRECTIONS = {
    "24937928": ("Clinical Center of Banja Luka; Health Center, Banja Luka", "manual_pdf", ""),
    "40011409": ("", "manual_no_unified_institution", "会议摘要合集：各摘要有自己的作者及机构，没有统一所属机构"),
    "14450716": ("Columbia University School of Public Health and Administrative Medicine (Division of Sanitary Science)", "manual_pdf", ""),
    "16995156": ("Physiological Laboratory, Cambridge", "manual_pdf", "原文未明确写大学全称"),
    "21048818": ("Paul-Ehrlich-Institut", "manual_correspondence", "通讯联系人/工作组关联机构；并非全部作者的所属机构"),
    "7057412": ("St Thomas' Hospital, London", "manual_pdf", ""),
    "21433778": ("Yale University", "manual_external", "根据作者同期任职资料核实；非本论文直接署名"),
    "32625401": ("European Food Safety Authority (EFSA)", "manual_group_author", "PPR 专家组署名"),
    "6487732": ("Hosei University; Juntendo University; FUJITSU Limited", "manual_pdf", ""),
}


def main():
    if not INPUT.is_file():
        raise SystemExit(f"找不到输入文件：{INPUT}")

    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        required = {"pmid", "institutions", "institutions_source"}
        if not required.issubset(headers):
            raise SystemExit(f"CSV缺少字段：{', '.join(sorted(required - set(headers)))}")
        rows = list(reader)

    if "notes" not in headers:
        headers.append("notes")

    seen, updated, missing_sources = set(), 0, 0
    source_columns = [col for col in headers if col.endswith("_source")]

    for row in rows:
        pmid = (row.get("pmid") or "").strip()
        if not pmid or pmid in seen:
            raise SystemExit(f"PMID为空或重复：{pmid!r}")
        seen.add(pmid)
        row["notes"] = row.get("notes") or ""

        if pmid in CORRECTIONS:
            institution, source, note = CORRECTIONS[pmid]
            row["institutions"] = institution
            row["institutions_source"] = source
            if note:
                # 多次运行不应重复追加同一条说明。
                existing = [x.strip() for x in row["notes"].split(" | ") if x.strip()]
                if note not in existing:
                    existing.append(note)
                row["notes"] = " | ".join(existing)
            updated += 1

        # 只对真正缺失的值标记 missing；有值但来源不明不能标记为缺失。
        # 人工确认“无统一机构”等明确来源标记不受影响。
        for source_col in source_columns:
            if (row.get(source_col) or "").strip():
                continue
            field = source_col.removesuffix("_source")
            if not (row.get(field) or "").strip():
                row[source_col] = "missing"
                missing_sources += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    unmatched = sorted(set(CORRECTIONS) - seen)
    print(f"处理 {len(rows)} 篇；人工机构修订 {updated} 篇；补全缺失来源标记 {missing_sources} 处。")
    if unmatched:
        print("提示：未找到的人工修订PMID：" + ", ".join(unmatched))
    print(f"输出：{OUTPUT}")
    print("输入 papers_metadata.csv 未修改。")


if __name__ == "__main__":
    main()
