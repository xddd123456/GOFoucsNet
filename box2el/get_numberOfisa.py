import numpy as np
import csv

# 加载 .npy 文件
# npy_file = "data/go_2023/isa_data.npy"  # 替换为你的 .npy 文件路径
# csv_file = "data/go_2023/isa_relations_index.csv"    # 替换为你的目标 .csv 文件路径
npy_file = "Result/go_2022/top3_nf1.npy"
csv_file = "top3_nf1.csv"
# 读取 npy 文件
data = np.load(npy_file)

# 检查数据形状
print(f"数据形状: {data.shape}")

# 将数据写入 CSV 文件
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    # 如果数据是二维的，直接逐行写入
    if len(data.shape) == 2:
        writer.writerows(data)
    # 如果数据是一维的，逐个写入
    elif len(data.shape) == 1:
        for value in data:
            writer.writerow([value])
    # 如果数据是多维的，展开成二维后写入
    else:
        flattened_data = data.reshape(data.shape[0], -1)
        writer.writerows(flattened_data)



print(f"成功将 {npy_file} 转换为 {csv_file}！")
