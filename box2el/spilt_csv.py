import numpy as np
import pandas as pd


def split_npy_keep_one_duplicate(input_npy, output_csv):
    """
    读取 `.npy` 文件，将每一行的第一个元素拆分到多个行，与后续元素分别组成新行，
    如果有重复的整行，则只保留一行，而不是全部删除。

    :param input_npy: 输入 `.npy` 文件路径
    :param output_csv: 输出 `.csv` 文件路径
    """
    data = np.load(input_npy, allow_pickle=True)  # 读取 `.npy` 文件

    new_rows = []  # 用列表存储拆分后的数据

    for row in data:
        first_value = row[0]  # 取第一列的值
        for value in row[1:]:  # 遍历后续列
            new_rows.append((first_value, value))  # 组成新的 (第一列, 其他列) 对

    # 转换为 DataFrame，并使用 drop_duplicates() 保留第一行重复数据
    new_df = pd.DataFrame(new_rows).drop_duplicates()

    # 保存结果
    new_df.to_csv(output_csv, sep='\t', index=False, header=False)


# 示例调用
input_npy_file = "Result/go_2025/top2_nf1.npy"  # 你的 `.npy` 文件路径
output_csv_file = "Result/go_2025/top2_nf1.csv"  # 目标 `.csv` 文件路径
split_npy_keep_one_duplicate(input_npy_file, output_csv_file)

print(f"转换完成，已保存到 {output_csv_file}")
