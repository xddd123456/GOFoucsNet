import json
import numpy as np


def add_sequence_numbers_to_relationships(data_isa_path, class_json_path, output_npy_path):
    # 读取 data_isa.json 文件
    with open(data_isa_path, "r") as data_file:
        data_isa = json.load(data_file)

    # 读取 class.json 文件
    with open(class_json_path, "r") as class_file:
        class_data = json.load(class_file)

    # 处理 data_isa 并根据 class.json 进行编号
    numbered_relationships = []

    for relationship in data_isa:
        numbered_pair = []
        for go_id, related_go_id in relationship.items():
            # 查找 class.json 中的序号，如果存在则编号
            if go_id in class_data:
                numbered_pair.append(class_data[go_id])
            else:
                # 如果 class.json 中没有这个 GO ID，保留原样
                numbered_pair.append(go_id)

            if related_go_id in class_data:
                numbered_pair.append(class_data[related_go_id])
            else:
                # 如果 class.json 中没有这个 GO ID，保留原样
                numbered_pair.append(related_go_id)

        # 将编号的关系存储为 NumPy 数组
        numbered_relationships.append(np.array(numbered_pair))

    # 将所有关系保存为 .npy 文件
    np.save(output_npy_path, numbered_relationships)


def save_data_relation_as_npy(data_relation_path, class_json_path, relation_json_path, output_npy_path):
    """
    读取 data_relation.json，class.json 和 relation.json，并根据 class.json 和 relation.json 中的数据
    对 GO ID 和关系名称进行编号，然后将编号后的数据保存为 .npy 文件。
    """
    # 读取 data_relation.json 文件
    with open(data_relation_path, "r") as data_file:
        data_relation = json.load(data_file)

    # 读取 class.json 文件
    with open(class_json_path, "r") as class_file:
        class_data = json.load(class_file)

    # 读取 relation.json 文件
    with open(relation_json_path, "r") as relation_file:
        relation_data = json.load(relation_file)

    # 调试：输出数据结构，检查数据是否正确加载
    print("data_relation:", data_relation)
    print("class_data:", class_data)
    print("relation_data:", relation_data)

    # 处理 data_relation，并对 GO ID 和关系名称进行编号
    numbered_relationships = []

    for item in data_relation:
        go_id = item.get("id")  # 获取 GO ID
        relationships = item.get("relationships", [])

        # 获取 GO ID 的编号
        go_id_number = class_data.get(go_id, None)  # 获取 GO ID 编号

        if go_id_number is None:
            print(f"GO ID {go_id} not found in class.json")
            continue

        # 处理每个关系
        for relationship in relationships:
            # 使用 " GO" 来分割关系字符串
            parts = relationship.split(" GO")

            # 确保分割后的结果包含至少两个部分
            if len(parts) < 2:
                print(f"Skipping invalid relationship: {relationship}")
                continue

            relation_name = parts[0].strip()  # 获取关系名称
            related_go_id = "GO" + parts[1].strip()  # 获取相关的 GO ID

            # 获取相关 GO ID 的编号
            related_go_id_number = class_data.get(related_go_id, None)

            if related_go_id_number is None:
                print(f"Related GO ID {related_go_id} not found in class.json")
                continue

            # 查找 relation.json 中的关系名称编号
            relation_number = relation_data.get(relation_name, None)

            if relation_number is None:
                print(f"Relation {relation_name} not found in relation.json")
                continue

            # 将编号后的关系存储为 [GO ID 编号, 关系编号, 相关的 GO ID 编号]
            numbered_pair = [go_id_number, relation_number, related_go_id_number]

            # 添加到列表中
            numbered_relationships.append(np.array(numbered_pair))

    # 将所有关系保存为 .npy 文件
    np.save(output_npy_path, np.array(numbered_relationships))

    print(f"Data has been saved to {output_npy_path}")


def save_data_intersection_as_npy(input_json_path, class_json_path, output_npy_path):
    # 读取提取的 JSON 数据
    with open(input_json_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # 读取 class.json 文件
    with open(class_json_path, "r", encoding="utf-8") as class_file:
        class_mapping = json.load(class_file)

    result = []

    for term in data:
        term_id = term.get("id", "")
        intersection_of = term.get("intersection_of", [])

        if len(intersection_of) >= 2:
            go_id_1 = intersection_of[0]
            go_id_2 = intersection_of[1].split()[-1]  # 提取第二个部分的 GO ID

            # 根据 class.json 获取编号
            term_id_number = class_mapping.get(term_id, -1)
            go_id_1_number = class_mapping.get(go_id_1, -1)
            go_id_2_number = class_mapping.get(go_id_2, -1)

            # 检查是否成功映射到编号
            if term_id_number != -1 and go_id_1_number != -1 and go_id_2_number != -1:
                result.append([term_id_number, go_id_1_number, go_id_2_number])
            else:
                print(f"Warning: ID mapping failed for {term_id}, {go_id_1}, or {go_id_2}")

    # 打印转换结果
    print("Mapped IDs to numbers:")
    print(result)

    # 保存为 NPY 文件
    np.save(output_npy_path, np.array(result, dtype=int))
    print(f"Saved numbered data to {output_npy_path}")



def load_data_from_npy(npy_file_path):
    """
    从 .npy 文件加载数据
    """
    return np.load(npy_file_path, allow_pickle=True)

# 文件路径
data_isa_path = "data/go_2025/data_isa.json"  # 修改为实际的 data_isa.json 文件路径
class_json_path = "data/go_2025/class.json"  # 修改为实际的 class.json 文件路径
output_txt_path = "data/go_2025/isa_data.npy"
relation_json_path = "data/go_2025/relation.json"  # 修改为实际的 relation.json 文件路径
output_relation_path = "data/go_2025/relation_data.npy"
data_relation_path = "data/go_2025/data_relation.json"
data_intersection_path = "data/go_2025/data_intersection.json"
output_intersection_path = "data/go_2025/intersection_data.npy"
# 添加序号并保存
add_sequence_numbers_to_relationships(data_isa_path, class_json_path, output_txt_path)

save_data_relation_as_npy(data_relation_path, class_json_path, relation_json_path, output_relation_path)

save_data_intersection_as_npy(data_intersection_path, class_json_path, output_intersection_path)
loaded_data = load_data_from_npy(output_intersection_path)
print(loaded_data)