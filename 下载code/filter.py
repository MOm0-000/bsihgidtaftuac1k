import pandas as pd

# 1. 读取 sheet1
input_file = "/home/zxx/bigdata/exp1/下载code/pubmedid_for-stu.xlsx"  # 改成你的文件名
output_file = "/home/zxx/bigdata/exp1/下载code/zzx.xlsx"     # 输出文件名

df = pd.read_excel(input_file, sheet_name="Sheet1", dtype={"Student ID": str})

# 2. 保留 Student ID 为 2024113212 的行
df_filtered = df[df["Student ID"].astype(str).str.strip() == "2024113212"]

# 3. 写出到新文件（保留原 sheet 结构，只写入筛选后的行）
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_filtered.to_excel(writer, sheet_name="Sheet1", index=False)

print(f"筛选完成，共保留 {len(df_filtered)} 行，已保存到 {output_file}")
