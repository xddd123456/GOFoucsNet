import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import os
from model.BoxSquaredEL import BoxSquaredEL


def compute_nf1_ranks(model, batch_data):
    class_boxes = model.get_boxes(model.class_embeds)  # 获取模型的class_boxes
    centers = class_boxes.centers  # 获取每个类的中心
    batch_centers = centers[batch_data[:, 0].long()]  # 获取当前批次数据中对应类别的中心
    current_batch_size = batch_data.shape[0]
    # 计算每个样本和所有类的距离
    dists = batch_centers[:, None, :] - torch.tile(centers, (current_batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)  # 计算欧氏距离

    # 使用scatter_来排除自我匹配的元素
    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1).long(), torch.inf)

    # 获取top3和top5
    top1 = torch.topk(dists, k=1, dim=1, largest=False).indices  # 获取top1排名
    top2 = torch.topk(dists, k=3, dim=1, largest=False).indices  # 获取top3排名

    # 返回格式为[[x], [a, b, c]]
    # x 是当前类别的索引，top3 是排名前3的索引
    top3_with_class = torch.cat([batch_data[:, 0].reshape(-1, 1), top1], dim=1)  # 类别与top1合并
    top5_with_class = torch.cat([batch_data[:, 0].reshape(-1, 1), top2], dim=1)  # 类别与top2合并

    return dists_to_ranks(dists, batch_data[:, 1]), top3_with_class, top5_with_class


def compute_nf2_ranks(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    c_boxes = class_boxes[batch_data[:, 0]]
    d_boxes = class_boxes[batch_data[:, 1]]

    intersection, _, _ = c_boxes.intersect(d_boxes)
    dists = intersection.centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)
    dists.scatter_(1, batch_data[:, 1].reshape(-1, 1), torch.inf)
    return dists_to_ranks(dists, batch_data[:, 2])


def compute_nf3_ranks(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    bumps = model.bumps
    head_boxes = model.get_boxes(model.relation_heads)
    tail_boxes = model.get_boxes(model.relation_tails)

    centers = class_boxes.centers
    d_centers = centers[batch_data[:, 2]]
    d_bumps = bumps[batch_data[:, 2]]
    batch_heads = head_boxes.centers[batch_data[:, 1]]
    batch_tails = tail_boxes.centers[batch_data[:, 1]]

    bumped_c_centers = torch.tile(centers, (batch_size, 1, 1)) + d_bumps[:, None, :]
    bumped_d_centers = d_centers[:, None, :] + torch.tile(bumps, (batch_size, 1, 1))

    c_dists = bumped_c_centers - batch_heads[:, None, :]
    c_dists = torch.linalg.norm(c_dists, dim=2, ord=2)
    d_dists = bumped_d_centers - batch_tails[:, None, :]
    d_dists = torch.linalg.norm(d_dists, dim=2, ord=2)
    dists = c_dists + d_dists
    return dists_to_ranks(dists, batch_data[:, 0])


def dists_to_ranks(dists, targets):
    index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
    index = index.long()
    targets = targets.long()
    return torch.take_along_dim(index, targets.reshape(-1, 1), dim=1).flatten()


# def evaluate_model(model, val_data, batch_size, device):
#     """
#     在验证集上评估模型的性能，计算 Accuracy, MRR, AUPR 和 AUC 等指标。
#
#     参数:
#     - model: BoxSquaredEL 实例，训练好的模型
#     - val_data: 包含验证数据的字典
#     - batch_size: 批次大小
#     - device: 设备信息（'cuda' 或 'cpu'）
#
#     返回:
#     - 各种评估指标（Accuracy, MRR, AUPR, AUC）
#     """
#     model.eval()  # 设置为评估模式
#     all_ranks_nf1 = []
#     # all_ranks_nf2 = []
#     # all_ranks_nf3 = []
#     all_top1_nf1 = []  # 用于存储每个节点的top3
#     all_top2_nf1 = []  # 用于存储每个节点的top5
#     # 遍历验证数据
#     with torch.no_grad():
#         total_batches = len(val_data) // batch_size + (1 if len(val_data) % batch_size > 0 else 0)
#         for i in range(total_batches):
#             batch_data = torch.tensor(val_data[i * batch_size: (i + 1) * batch_size], device=device)
#                 # 'nf2': val_data['nf2'][i * batch_size: (i + 1) * batch_size].to(device),
#                 # 'nf3': val_data['nf3'][i * batch_size: (i + 1) * batch_size].to(device),
#             # 计算各个类型的排名
#             ranks_nf1, top1_nf1, top2_nf1 = compute_nf1_ranks(model, batch_data)
#             torch.cuda.empty_cache()
#
#             print(i)
#             # ranks_nf2 = compute_nf2_ranks(model, batch_data['nf2'], batch_size)
#             # ranks_nf3 = compute_nf3_ranks(model, batch_data['nf3'], batch_size)
#
#             all_ranks_nf1.append(ranks_nf1)
#             all_top1_nf1.append(top1_nf1)  # 保存top3
#             all_top2_nf1.append(top2_nf1)  # 保存top5
#             # all_ranks_nf2.append(ranks_nf2)
#             # all_ranks_nf3.append(ranks_nf3)
#
#
#     # 合并所有批次的排名
#     all_ranks_nf1 = torch.cat(all_ranks_nf1, dim=0)
#     all_top1_nf1 = torch.cat(all_top1_nf1, dim=0)  # 合并top3
#     all_top2_nf1 = torch.cat(all_top2_nf1, dim=0)  # 合并top5
#     # all_ranks_nf2 = torch.cat(all_ranks_nf2, dim=0)
#     # all_ranks_nf3 = torch.cat(all_ranks_nf3, dim=0)
#
#     # 打印评估结果
#     print("Evaluation Results for NF1:")
#
#     # print("Evaluation Results for NF2:")
#     # print(f"AUC: {auc_nf2:.4f}")
#     #
#     # print("Evaluation Results for NF3:")
#     # print(f"AUC: {auc_nf3:.4f}")
#     # 保存 top3 和 top5 排名
#     np.save("Result/go_2023/top1_nf1.npy", all_top1_nf1.cpu().numpy())  # 保存 top3 排名
#     # np.save("top2_nf1.npy", all_top2_nf1.cpu().numpy())  # 保存 top5 排名
#
#     return {
#         'top1_nf1': all_top1_nf1, 'top2_nf1': all_top2_nf1
#     }


def evaluate_model(model, val_data, batch_size, device):
    """
    在验证集上评估模型的性能，计算 Accuracy, MRR, AUPR 和 AUC 等指标。

    参数:
    - model: BoxSquaredEL 实例，训练好的模型
    - val_data: 包含验证数据的字典
    - batch_size: 批次大小
    - device: 设备信息（'cuda' 或 'cpu'）

    返回:
    - 各种评估指标（Accuracy, MRR, AUPR, AUC）
    """
    model.eval()  # 设置为评估模式
    all_ranks_nf1 = []
    all_top1_nf1 = []  # 用于存储每个节点的top3
    all_top2_nf1 = []  # 用于存储每个节点的top5
    all_top2_only_nf1 = []  # 用于存储前 2 的排名

    # 遍历验证数据
    with torch.no_grad():
        total_batches = len(val_data) // batch_size + (1 if len(val_data) % batch_size > 0 else 0)
        for i in range(total_batches):
            batch_data = torch.tensor(val_data[i * batch_size: (i + 1) * batch_size], device=device)

            # 计算各个类型的排名
            ranks_nf1, top1_nf1, top2_nf1 = compute_nf1_ranks(model, batch_data)

            # 提取前 2 的排名
            top2_only_nf1 = top1_nf1[:, :2]  # 只取前 3
            torch.cuda.empty_cache()
            print(i)

            all_ranks_nf1.append(ranks_nf1)
            all_top1_nf1.append(top1_nf1)  # 保存top3
            all_top2_nf1.append(top2_nf1)  # 保存top5
            all_top2_only_nf1.append(top2_only_nf1)  # 保存前 2

    # 合并所有批次的排名
    all_ranks_nf1 = torch.cat(all_ranks_nf1, dim=0)
    all_top1_nf1 = torch.cat(all_top1_nf1, dim=0)  # 合并top1
    all_top2_nf1 = torch.cat(all_top2_nf1, dim=0)  # 合并top2
    all_top2_only_nf1 = torch.cat(all_top2_only_nf1, dim=0)  # 合并前 2

    # 打印评估结果
    print("Evaluation Results for NF1:")

    # 保存 top3 和 top5 排名以及前 2 排名
    np.save("Result/go_2025/top1_nf1.npy", all_top1_nf1.cpu().numpy())  # 保存 1 排名
    np.save("Result/go_2025/top2_nf1.npy", all_top2_nf1.cpu().numpy())  # 保存 前 2 排名

    return {
        'top1_nf1': all_top1_nf1,
        'top2_nf1': all_top2_nf1,
        'top2_only_nf1': all_top2_only_nf1  # 返回前 2 的排名
    }


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = BoxSquaredEL(
    device=device,
    embedding_dim=384,  # 根据你的模型设置调整
    num_classes=47994,  # 根据实际情况调整
    num_roles=9,  # 根据实际情况调整
    batch_size=64,  # 根据实际情况调整
)  # 确保模型类 BoxSquaredEL 已经定义好
model.load_state_dict(torch.load('Result/go_2025/boxe_model.pt', map_location=device))
model.to(device)
isa_file = "data/go_2025/isa_data.npy"
# 从 .npy 文件中加载数据

isa_data = np.load(isa_file)
# isa_data_subset = isa_data[:128]
result = evaluate_model(model, isa_data, batch_size=64, device=device)
print(result)
