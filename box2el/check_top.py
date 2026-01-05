import numpy as np
import pandas as pd

# 假设加载的数据是一个二维数组，每行是类似 "47867 47868 13696 1948" 的形式
data = np.load("Result/go_2022/top2_nf1.npy")  # 替换为实际的文件路径
print("Loaded Data:", data)

# 将每一行分解为第一个元素和其余元素的组合
result = []
for row in data:
    first_element = row[0]
    for other_element in row[1:]:
        result.append([first_element, other_element])  # 添加 Relation 列，值为 0

# 转换为 Pandas DataFrame
df = pd.DataFrame(result, columns=["First", "Other"])
df = df.drop_duplicates()
# 保存为 CSV 文件
output_csv_path = "top2_nf1.csv"
df.to_csv(output_csv_path, index=False, sep='\t')

print(f"Data saved to {output_csv_path}")
