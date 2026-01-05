import numpy as np
import csv

# 加载 .npy 文件
npy_file = "data/go_2023/relation_data.npy"  # 替换为你的 .npy 文件路径
csv_file = "data/go_2023/relations_indexed.csv"  # 替换为你的目标 .csv 文件路径

# 读取 npy 文件
data = np.load(npy_file)

# 检查数据形状
print(f"数据形状: {data.shape}")

# 确保数据至少是二维的（以便提取列）
if len(data.shape) < 2 or data.shape[1] < 3:
    raise ValueError("数据需要至少有两列以提取第一列和第三列！")

# 提取第一列和第三列
data_subset = data[:, [0, 2]]  # 选取第一列和第三列

# 将数据写入 CSV 文件
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(data_subset)  # 写入提取后的两列数据

print(f"成功将 {npy_file} 的第一列和第三列转换为 {csv_file}！")