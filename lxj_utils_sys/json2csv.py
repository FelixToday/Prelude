import json
import csv
import os

from tabulate import tabulate


def get_by_path(data, path):
    """支持中间层为 list 时，对每个元素继续取字段"""
    keys = path.split('.')
    cur = data

    for i, k in enumerate(keys):
        if isinstance(cur, list):
            # list 中每个元素继续取后续路径
            rest_path = '.'.join(keys[i:])
            return [get_by_path(item, rest_path) for item in cur]

        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None

    return cur


def normalize_value(v):
    """把任意值转换成最终用于列的 list"""
    if isinstance(v, list):
        return v
    if isinstance(v, (int, float, str)):
        return [v]
    return [None]


def build_table(data, paths):
    """根据 paths 构造二维表（list[dict]）"""
    columns = {}
    max_len = 0

    for p in paths:
        v = get_by_path(data, p)
        v = normalize_value(v)

        columns[p] = v
        max_len = max(max_len, len(v))

    # 构造表格（list of dict rows）
    table = []
    for i in range(max_len):
        row = {}
        for p, col in columns.items():
            row[p] = col[i] if i < len(col) else None
        table.append(row)

    return table


# --------------------------------------------------------
#                 新增主接口：写 CSV
# --------------------------------------------------------

def extract_json_to_csv(json_path_or_dict, paths, output_csv_path="./output.csv", verbose=True):
    """
    从 JSON 文件或 dict 中读取数据，
    根据 paths 提取表格并输出到 CSV。
    """
    # 1. 加载 JSON
    if isinstance(json_path_or_dict, dict):
        data = json_path_or_dict
        if verbose:
            print("[INFO] 直接使用传入的 dict 数据")
    else:
        if verbose:
            print(f"[INFO] 从文件加载 JSON: {json_path_or_dict}")
        with open(json_path_or_dict, "r", encoding="utf-8") as f:
            data = json.load(f)

    # 2. 构造 table
    if verbose:
        print(f"[INFO] 提取路径: {paths}")
    table = build_table(data, paths)

    if verbose:
        print(f"[INFO] 共生成 {len(table)} 行")

    # 3. 写入 CSV
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=paths)
        writer.writeheader()
        writer.writerows(table)

    if verbose:
        print(f"[INFO] CSV 已写入: {output_csv_path}")

    return table  # 可选：返回表格给调用者


# --------------------------------------------------------
# 示例调用
# --------------------------------------------------------
if __name__ == "__main__":
    json_path = "../cache/result.json"
    paths = [
        "epochs.epoch", "epochs.train.p@5", "epochs.train.ap@5", "epochs.train.loss",
        "test2.epoch", "test2.p@5", "test2.ap@5", "test2.loss",
        "test3.epoch", "test3.p@5", "test3.ap@5", "test3.loss",
        "test4",  # list[primitive]
        "test5"  # scalar
    ]
    table = extract_json_to_csv(
        json_path_or_dict=json_path,
        paths=paths,
        output_csv_path="../cache/output.csv",
        verbose=True
    )

    # 打印预览
    print(tabulate(table, headers="keys", tablefmt="simple"))
