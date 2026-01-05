import json
import csv


def load_go_terms(year):
    """加载指定年份的GO术语数据并返回字典（id到名称的映射）"""
    file_path = f"data/go_{year}/go_terms.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {item['id']: item['name'] for item in data}


def analyze_go_pairs(csv_file, data_2022, data_2023):
    """
    分析CSV文件中的GO对，返回：
    - A列发生名称变化的GOID集合
    - B列发生名称变化的GOID集合
    """
    a_changed_ids = set()  # 存储A列变化的GOID（自动去重）
    b_changed_ids = set()  # 存储B列变化的GOID（自动去重）

    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) != 2:
                continue

            go_a, go_b = row

            # 检查A列GOID名称是否变化
            name_a_2022 = data_2022.get(go_a)
            name_a_2023 = data_2023.get(go_a)
            if (name_a_2022 is not None and
                name_a_2023 is not None and
                name_a_2022 != name_a_2023):
                a_changed_ids.add(go_a)  # 使用集合自动去重

            # 检查B列GOID名称是否变化
            name_b_2022 = data_2022.get(go_b)
            name_b_2023 = data_2023.get(go_b)
            if (name_b_2022 is not None and
                name_b_2023 is not None and
                name_b_2022 != name_b_2023):
                b_changed_ids.add(go_b)  # 使用集合自动去重

    return a_changed_ids, b_changed_ids


def save_ids_to_file(ids, filename):
    """将GOID集合保存到文件"""
    with open(filename, 'w') as f:
        for go_id in sorted(ids):
            f.write(f"{go_id}\n")


if __name__ == "__main__":
    # 加载数据
    data_2022 = load_go_terms(2022)
    data_2023 = load_go_terms(2023)

    # 分析并获取变化的GOID
    a_ids, b_ids = analyze_go_pairs(
        "new_go_pairs_2023.csv",
        data_2022,
        data_2023
    )

    # 保存结果到文件
    save_ids_to_file(a_ids, "changed_go_ids_column_a.txt")
    save_ids_to_file(b_ids, "changed_go_ids_column_b.txt")

    # 输出统计信息
    print(f"📊 统计结果：")
    print(f"1. 第一列（A）名称变化的唯一GOID数量：{len(a_ids)} → 已保存到 changed_go_ids_column_a.txt")
    print(f"2. 第二列（B）名称变化的唯一GOID数量：{len(b_ids)} → 已保存到 changed_go_ids_column_b.txt\n")
    print("✅ 完成！已记录所有发生名称变化的GOID。")