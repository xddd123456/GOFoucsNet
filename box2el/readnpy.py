import numpy as np


def load_data_from_npy(npy_file_path):
    """
    从 .npy 文件加载数据并打印
    """
    # 加载 .npy 文件
    data = np.load(npy_file_path, allow_pickle=True)

    # 打印加载的数据
    print("Loaded data from .npy file:")
    print(data)

    # 返回数据（如果需要进一步处理）
    return data


# 文件路径
# npy_file_path = "data/intersection_data.npy"  # 修改为实际的 .npy 文件路径
# npy_file_path = "data/isa_data.npy"  # 修改为实际的 .npy 文件路径
# npy_file_path = "data/relation_data.npy"
npy_file_path = "data/go_2025/isa_data.npy"
# 调用函数并读取数据
loaded_data = load_data_from_npy(npy_file_path)
print(loaded_data)
txt_file_path = "data/go_2025/isa_data.txt"
# 保存为 .txt 文件
# np.savetxt(txt_file_path, loaded_data, fmt="%d")
