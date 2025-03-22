import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def load_hypergraph():
    """加载tcmbank_pre_ingredients.txt构建超图"""
    # 读取文件
    hyperedges = []
    with open('../data/processed/tcmbank/tcmbank_pre_ingredients.txt', 'r', encoding='utf-8') as f:
        for line in f:
            # 保留空行，但将其转换为空列表
            if not line.strip():
                hyperedges.append([])
            else:
                # 将每行的序号转换为整数列表
                nodes = [int(x) for x in line.strip().split()]
                hyperedges.append(nodes)
    
    # 获取所有唯一的节点ID
    all_nodes = set()
    for edge in hyperedges:
        all_nodes.update(edge)
    num_nodes = max(all_nodes) + 1  # 节点ID从0开始
    
    # 构建超边索引
    edge_index = []
    for edge_idx, edge in enumerate(hyperedges):
        for node in edge:
            edge_index.append([node, edge_idx])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # 构建节点特征（这里使用one-hot编码）
    x = torch.eye(num_nodes, dtype=torch.float)
    
    # 构建超边特征（这里使用one-hot编码）
    edge_attr = torch.eye(len(hyperedges), dtype=torch.float)
    
    # 创建Data对象
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )
    
    # 转换为无向图
    data.edge_index = to_undirected(data.edge_index)
    
    # 保存处理后的数据
    os.makedirs('../data/processed/tcmbank', exist_ok=True)
    torch.save(data, '../data/processed/tcmbank/hypergraph.pt')
    
    print(f"构建完成超图：")
    print(f"- 节点数量：{num_nodes}")
    print(f"- 超边数量：{len(hyperedges)}")
    print(f"- 边数量：{data.edge_index.size(1)}")
    print(f"- 空超边数量：{sum(1 for edge in hyperedges if not edge)}")
    
    return data

def main():
    # 构建超图
    data = load_hypergraph()

if __name__ == "__main__":
    main() 