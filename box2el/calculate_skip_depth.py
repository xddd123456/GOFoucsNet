import pandas as pd
import networkx as nx

# =======================  1. 读取 is_a 关系并构建 GO DAG  =======================
print("正在读取 is_a 关系数据...")
df = pd.read_csv("data/go_2023/is_a_relations.csv", delimiter='\t', dtype=str)

# 构建有向无环图（DAG），父节点指向子节点
G_directed = nx.DiGraph()
for _, row in df.iterrows():
    G_directed.add_edge(row["related_id"], row["id"])  # Parent -> Child

print("GO DAG 构建完成！")
print(f"总节点数: {G_directed.number_of_nodes()}, 总边数: {G_directed.number_of_edges()}")

# 计算所有根节点（没有父节点的节点）
roots = [node for node in G_directed.nodes if G_directed.in_degree(node) == 0]
print(f"GO 图中的根节点数量: {len(roots)}")

# 计算所有叶子节点（没有子节点的节点）
leaves = [node for node in G_directed.nodes if G_directed.out_degree(node) == 0]
print(f"GO 图中的叶子节点数量: {len(leaves)}")


# =======================  2. 计算 GO 术语的层级（到最近根节点的最短路径）  =======================
def get_min_depth(go_id):
    """计算某个 GO 术语到最近根节点的最短路径（层级数）"""
    if go_id not in G_directed:
        return None
    min_depth = float("inf")
    for root in roots:
        try:
            depth = nx.shortest_path_length(G_directed, source=root, target=go_id)
            min_depth = min(min_depth, depth)
        except nx.NetworkXNoPath:
            continue
    return min_depth if min_depth != float("inf") else None


# =======================  3. 计算到最近叶子节点的最短路径  =======================
def get_leaf_distance(go_id):
    """计算某个 GO 术语到最近叶子节点的最短路径"""
    if go_id not in G_directed:
        return None
    queue = [(go_id, 0)]
    visited = set()
    while queue:
        node, depth = queue.pop(0)
        successors = list(G_directed.successors(node))
        if not successors:  # 叶子节点
            return depth
        for child in successors:
            if child not in visited:
                queue.append((child, depth + 1))
                visited.add(child)
    return None


# =======================  4. 转换为无向图并计算最短跳数  =======================
G_undirected = G_directed.to_undirected()

print("无向 GO 图构建完成！")
print(f"总节点数: {G_undirected.number_of_nodes()}, 总边数: {G_undirected.number_of_edges()}")

def get_go1_to_go2_hops_undirected(go1, go2):
    """计算 GO_1 到 GO_2 在无向图中的最短路径跳数"""
    if go1 in G_undirected and go2 in G_undirected:
        try:
            return nx.shortest_path_length(G_undirected, source=go1, target=go2)
        except nx.NetworkXNoPath:
            return None  # 没有路径
    return None  # GO_1 或 GO_2 不在图中


# =======================  5. 读取 new_go_pairs_2023.csv 并计算所有指标  =======================
print("正在读取 new_go_pairs_2024.csv...")
go_pairs = pd.read_csv("Result/go_2023/analysis_results.csv", delimiter=',', names=["GO_1", "GO_2"], dtype=str)

# 计算层级数（深度）、到叶子节点的最短距离和无向图最短跳数
print("正在计算 GO 术语的层级数、叶子距离和无向跳数...")
go_pairs["GO_1_Depth"] = go_pairs["GO_1"].apply(get_min_depth)
go_pairs["GO_1_Leaf_Distance"] = go_pairs["GO_1"].apply(get_leaf_distance)
go_pairs["GO_2_Depth"] = go_pairs["GO_2"].apply(get_min_depth)
go_pairs["GO_2_Leaf_Distance"] = go_pairs["GO_2"].apply(get_leaf_distance)
go_pairs["GO_1_to_GO_2_Hops_Undirected"] = go_pairs.apply(lambda row: get_go1_to_go2_hops_undirected(row["GO_1"], row["GO_2"]), axis=1)

# =======================  6. 保存最终结果  =======================
output_file = "analysis_results_2024.csv"
go_pairs.to_csv(output_file, index=False, sep='\t')
print(f"所有计算完成，结果已保存到 {output_file}！")
