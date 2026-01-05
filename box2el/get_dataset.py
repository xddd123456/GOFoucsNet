import json

def extract_is_a_relationships(obo_file_path, output_json_path):
    relationships = []

    with open(obo_file_path, "r") as obo_file:
        current_id = None
        for line in obo_file:
            line = line.strip()
            if line.startswith("id:"):
                current_id = line.split("id:")[1].strip()
            elif line.startswith("is_a:"):
                if current_id:
                    related_id = line.split("is_a:")[1].split("!")[0].strip()
                    relationships.append({current_id : related_id})
            elif line == "[Term]":
                current_id = None  # Reset for the next term

    with open(output_json_path, "w") as json_file:
        json.dump(relationships, json_file, indent=4)


def extract_intersection_of(obo_file_path, output_json_path):
    term_data = []
    current_term = {}

    with open(obo_file_path, "r") as obo_file:
        for line in obo_file:
            line = line.strip()

            # 处理每个新的术语块
            if line == "[Term]":
                # 如果当前术语包含 intersection_of，添加到结果中
                if current_term.get("intersection_of"):
                    term_data.append(current_term)
                current_term = {"intersection_of": []}  # 清空当前术语的数据

            elif line.startswith("id:"):
                current_term["id"] = line.split("id:")[1].strip()  # 提取 id

            elif line.startswith("intersection_of:"):
                # 提取 intersection_of 条目，只保留第一个 GO 术语
                intersection_term = line.split("intersection_of:")[1].split("!")[0].strip()
                if intersection_term not in current_term["intersection_of"]:
                    current_term["intersection_of"].append(intersection_term)

        # 处理最后一个术语
        if current_term.get("intersection_of"):
            term_data.append(current_term)

    # 写入 JSON 文件
    with open(output_json_path, "w") as json_file:
        json.dump(term_data, json_file, indent=4)

def extract_relationships(obo_file_path, output_json_path):
    relationships_data = []
    current_term = {}

    with open(obo_file_path, "r") as obo_file:
        for line in obo_file:
            line = line.strip()

            # 处理每个新的术语块
            if line == "[Term]":
                # 如果当前术语包含 relationship，添加到结果中
                if current_term.get("relationships"):
                    relationships_data.append(current_term)
                current_term = {"relationships": []}  # 清空当前术语的数据

            elif line.startswith("id:"):
                current_term["id"] = line.split("id:")[1].strip()  # 提取 id

            elif line.startswith("relationship:"):
                # 提取 relationship 条目，并去掉 ! 后面的部分
                relationship = line.split("relationship:")[1].strip()
                # 去除 ! 后面的部分，只保留 GO 术语和关系标识符
                if "!" in relationship:
                    relationship = relationship.split("!")[0].strip()
                current_term["relationships"].append(relationship)

        # 处理最后一个术语
        if current_term.get("relationships"):
            relationships_data.append(current_term)

    # 写入 JSON 文件
    with open(output_json_path, "w") as json_file:
        json.dump(relationships_data, json_file, indent=4)

# 文件路径
obo_file_path = "data/go_2025/go2025.obo"  # 修改为实际的 go.obo 文件路径
output_json_isa = "data/go_2025/data_isa.json"
output_json_intsection_of = "data/go_2025/data_intersection.json"
output_json_relation = "data/go_2025/data_relation.json"

# 提取并保存
extract_is_a_relationships(obo_file_path, output_json_isa)
extract_intersection_of(obo_file_path, output_json_intsection_of)
extract_relationships(obo_file_path, output_json_relation)


