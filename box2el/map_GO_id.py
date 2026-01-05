import csv
import json

# Step 1: 读取 class.json 并反转为 index -> id 的映射字典
with open("data/go_2025/class.json", "r") as jsonfile:
    id_to_index = json.load(jsonfile)
    # 反转字典：从 {GO_ID: 编号} 变为 {编号: GO_ID}
    index_to_id = {v: k for k, v in id_to_index.items()}

# Step 2: 读取 top2_GO_id.csv 并将编号转换回 GO ID
go_id_relations = []
with open("Result/go_2025/top2_nf1.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        source_index = int(row[0])  # 第一个编号
        target_index = int(row[1])  # 第二个编号

        # 将编号转换回 GO ID，如果不存在则跳过
        if source_index in index_to_id and target_index in index_to_id:
            go_id_relations.append([
                index_to_id[source_index],  # 编号转为 GO ID
                index_to_id[target_index]  # 编号转为 GO ID
            ])

# Step 3: 将转换回来的 GO ID 保存为 CSV 文件
output_csv_path = "./Result/go_2025/top2_GO_id_to_GO.csv"
with open(output_csv_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(["GO_ID", "Related_GO_ID"])  # 写入表头
    writer.writerows(go_id_relations)

print(f"转换完成，结果已保存到 {output_csv_path}")
