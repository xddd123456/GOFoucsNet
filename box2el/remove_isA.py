import pandas as pd

# 读取 CSV 文件
is_a_relation = pd.read_csv("data/go_2023/isa_relations_index.csv", header=None, sep='\t', usecols=[0, 1])
top1_nf1_relation = pd.read_csv("Result/go_2023/top1_nf1.csv", header=None, sep='\t', skiprows=1)
# top2_nf1_relation = pd.read_csv("top2_nf1_relation.csv", header=None, sep='\t', skiprows=1)

# 重命名列以便对比
is_a_relation.columns = ["Col1", "Col2"]
top1_nf1_relation.columns = ["Col1", "Col2"]
# top2_nf1_relation.columns = ["Col1", "Col2"]

# 转换为集合以提高查找效率
is_a_set = set(is_a_relation.apply(tuple, axis=1))

# 定义一个函数，用于去除重复项和反向关系，并逐条打印
def remove_duplicates_and_reverse_with_logging(relation_df, relation_name):
    filtered_relation = []
    processed_pairs = set()  # 用于记录已处理的关系（无方向性）
    for idx, row in relation_df.iterrows():
        row_tuple = tuple(row)
        reverse_tuple = (row_tuple[1], row_tuple[0])  # 反向关系

        # 如果当前关系或反向关系已存在于 `is_a_set` 或已处理集合中，则跳过
        if row_tuple in is_a_set or reverse_tuple in is_a_set or reverse_tuple in processed_pairs:
            print(f"{relation_name}: Removed duplicate or reverse row {row_tuple}")
        else:
            filtered_relation.append(row_tuple)
            processed_pairs.add(row_tuple)  # 添加当前关系到已处理集合

    return pd.DataFrame(filtered_relation, columns=["Col1", "Col2"])


# 去掉重复部分和反向关系
filtered_top1_nf1_relation = remove_duplicates_and_reverse_with_logging(top1_nf1_relation, "Top1")
# filtered_top2_nf1_relation = remove_duplicates_and_reverse_with_logging(top2_nf1_relation, "Top2")

# 保存去重后的结果
filtered_top1_nf1_relation.to_csv("Result/go_2023/filtered_top1_def.csv", index=False, header=False, sep='\t')
#filtered_top2_nf1_relation.to_csv("filtered_top2_relation.csv", index=False, header=False, sep='\t')

print("去重完成，结果已保存")
