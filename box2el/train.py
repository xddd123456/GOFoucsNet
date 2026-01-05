import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import os
from model.BoxSquaredEL import BoxSquaredEL
from sklearn.metrics import roc_auc_score, average_precision_score

# 计算 Accuracy
def calculate_accuracy(predictions, targets):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

# 计算 MRR (Mean Reciprocal Rank)
def calculate_mrr(predictions, targets):
    # 假设 predictions 是一个排序后的概率或相关度分数
    _, indices = torch.sort(predictions, dim=1, descending=True)
    rank = (indices == targets.view(-1, 1)).nonzero(as_tuple=True)[1] + 1
    mrr = torch.mean(1.0 / rank.float())
    return mrr

# 计算 AUPR (Area Under Precision-Recall Curve)
def calculate_aupr(predictions, targets):
    return average_precision_score(targets.cpu().numpy(), predictions.cpu().numpy())

# 计算 AUC (Area Under ROC Curve)
def calculate_auc(predictions, targets):
    return roc_auc_score(targets.cpu().numpy(), predictions.cpu().numpy())

def train_box_squared_el_model(model, train_data, val_data=None, num_epochs=100, learning_rate=0.001, save_path="model.pth"):
    """
    训练 BoxSquaredEL 模型的函数。

    参数:
    - model: BoxSquaredEL 实例
    - train_data: 包含训练数据的字典，键包括 'nf1', 'nf2', 'nf3', 'disjoint', 'nf3_neg' 等
    - val_data: 包含验证数据的字典，可选
    - num_epochs: 训练的总轮数
    - learning_rate: 学习率
    - save_path: 模型保存路径

    返回:
    - 训练损失列表
    - 验证损失列表（如果提供了验证数据）
    """
    device = model.device
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # 按批次训练
        for _ in range(len(train_data['nf1']) // model.batch_size):
            optimizer.zero_grad()

            # 前向传播并计算损失
            batch_loss = model(train_data)
            train_loss += batch_loss.item()

            # 反向传播并更新参数
            batch_loss.backward()
            optimizer.step()

        train_loss /= len(train_data['nf1']) // model.batch_size
        train_loss_list.append(train_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        # 验证模型（如果提供了验证数据）
        # if val_data:
        #     model.eval()
        #     val_loss = 0
        #     with torch.no_grad():
        #         for _ in range(len(val_data['nf1']) // model.batch_size):
        #             batch_val_loss = model(val_data)
        #             val_loss += batch_val_loss.item()
        #
        #     val_loss /= len(val_data['nf1']) // model.batch_size
        #     val_loss_list.append(val_loss)
        #     print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # 每轮保存模型
        torch.save(model.state_dict(), save_path)

    return train_loss_list, val_loss_list if val_data else train_loss_list




# 数据示例：假设已预处理成字典形式
isa_file = "data/go_2025/isa_data.npy"
intersection_file = "data/go_2025/intersection_data.npy"
relation_file = "data/go_2025/relation_data.npy"

# 从 .npy 文件中加载数据
isa_data = np.load(isa_file)  # 形状 (num_samples, 2)
intersection_data = np.load(intersection_file)  # 形状 (num_samples, 3)
relation_data = np.load(relation_file)  # 形状 (num_samples, 3)

# 划分训练和验证集
# def split_data(data, train_ratio=0.8):
#     train_size = int(len(data) * train_ratio)
#     train_data = data[:train_size]
#     val_data = data[train_size:]
#     return train_data, val_data
#
# isa_train, isa_val = split_data(isa_data)
# intersection_train, intersection_val = split_data(intersection_data)
# relation_train, relation_val = split_data(relation_data)

# 构造训练和验证数据字典
train_data = {
    # "nf1": torch.tensor(isa_data, dtype=torch.long),  # 确保是整数索引
    # "nf2": torch.tensor(intersection_data, dtype=torch.long),
    # "nf3": torch.tensor(relation_data, dtype=torch.long),
    "nf1": torch.tensor(isa_data, dtype=torch.long),  # 确保是整数索引
    "nf2": torch.tensor(intersection_data, dtype=torch.long),
    "nf3": torch.tensor(relation_data, dtype=torch.long),
}

# val_data = {
#     "nf1": torch.tensor(isa_val, dtype=torch.long),
#     "nf2": torch.tensor(intersection_val, dtype=torch.long),
#     "nf3": torch.tensor(relation_val, dtype=torch.long),
# }

print("Training data:", {k: v.shape for k, v in train_data.items()})
# print("Validation data:", {k: v.shape for k, v in val_data.items()})

# 模型实例
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_dim = 384
num_classes = 47994
num_roles = 9

# embedding_matrix_name = np.load('data/go_term_name_embeddings_weight.npy')
# embedding_tensor_name = torch.from_numpy(embedding_matrix_name).float()
embedding_matrix_def = np.load('data/go_2025/go_name_embeddings_weight.npy')
embedding_matrix_def = torch.from_numpy(embedding_matrix_def).float()

model = BoxSquaredEL(
    device=device,
    embedding_dim=embedding_dim,
    num_classes=num_classes,
    num_roles=num_roles,
    batch_size=256,
    # pretrained_embeddings=embedding_tensor_name
    pretrained_embeddings = embedding_matrix_def
)

# 训练模型
train_loss = train_box_squared_el_model(
    model,
    train_data,
    num_epochs=50,
    learning_rate=0.001,
    save_path="Result/go_2025/boxe_model.pt"
)
