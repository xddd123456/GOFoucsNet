import pandas as pd
from collections import defaultdict, deque


# 读取数据
def load_data(is_a_file, relation_file):
    # 构建关系图
    df = pd.read_csv(is_a_file, sep='\t')
    tree = defaultdict(list)
    parents = defaultdict(list)

    for _, row in df.iterrows():
        tree[row['related_id']].append(row['id'])  # 构造子节点查找
        parents[row['id']].append(row['related_id'])  # 构造父节点查找

    # 加载待分析关系
    relations = []
    with open(relation_file, 'r', encoding='utf-8-sig') as f:  # 防止 UTF-8 BOM 问题
        for line in f:
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) == 2:  # 确保每行只有两个元素
                relations.append(parts)
            else:
                print(f"⚠️ 警告：跳过格式错误的行 -> {line.strip()}")

    return tree, parents, relations


# 找到所有祖先
def get_ancestors(node, parents):
    ancestors = set()
    queue = deque([node])
    while queue:
        current = queue.popleft()
        for parent in parents[current]:
            if parent not in ancestors:
                ancestors.add(parent)
                queue.append(parent)
    return ancestors


# 找到所有后代
def get_descendants(node, tree):
    descendants = set()
    queue = deque([node])
    while queue:
        current = queue.popleft()
        for child in tree[current]:
            if child not in descendants:
                descendants.add(child)
                queue.append(child)
    return descendants


# 判断关系
def check_relation(node1, node2, tree, parents):
    ancestors1 = get_ancestors(node1, parents)
    ancestors2 = get_ancestors(node2, parents)

    # 判断爷孙
    if node2 in get_descendants(node1, tree):
        return '爷孙'
    elif node1 in get_descendants(node2, tree):
        return '孙子'

    # 判断兄弟
    if ancestors1 & ancestors2:
        common_parents = ancestors1 & ancestors2
        for parent in common_parents:
            if node1 in tree[parent] and node2 in tree[parent]:
                return '兄弟'

    # 判断叔侄
    for ancestor in ancestors2:
        if node1 in tree[ancestor]:
            return '叔侄'

    return '无直接亲属关系'


# 主程序
def main():
    is_a_file = 'data/go_2022/is_a_relations.csv'
    relation_file = 'top2_new.csv'

    tree, parents, relations = load_data(is_a_file, relation_file)

    # 判断每对关系
    for node1, node2 in relations:
        relation = check_relation(node1, node2, tree, parents)
        print(f"{node1} 和 {node2} 的关系是：{relation}")


if __name__ == "__main__":
    main()
