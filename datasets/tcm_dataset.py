import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import ToUndirected

class TCMDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        中医处方推荐数据集
        
        Args:
            root: 数据根目录，包含processed文件夹
        """
        super(TCMDataset, self).__init__(root, transform, pre_transform)
        
        # 加载边索引数据
        processed_dir = os.path.join(root, 'processed')
        self.herb_herb_edge_index = torch.from_numpy(
            np.load(os.path.join(processed_dir, 'herb_herb_edge_index.npy'))
        ).long()
        self.symptom_symptom_edge_index = torch.from_numpy(
            np.load(os.path.join(processed_dir, 'symptom_symptom_edge_index.npy'))
        ).long()
        self.herb_symptom_edge_index = torch.from_numpy(
            np.load(os.path.join(processed_dir, 'herb_symptom_edge_index.npy'))
        ).long()
        
        # 获取节点数量
        self.num_herbs = self.herb_herb_edge_index.max().item() + 1
        self.num_symptoms = self.symptom_symptom_edge_index.max().item() + 1
        
        # 创建异构图数据
        self.data = self._create_heterogeneous_graph()
        
        # 加载处方数据
        self.prescriptions = np.load(os.path.join(processed_dir, 'prescriptions.npy'), allow_pickle=True)
    
    def _create_heterogeneous_graph(self):
        # 创建节点特征（这里使用简单的one-hot编码，您可以根据需要修改）
        herb_x = torch.eye(self.num_herbs)
        symptom_x = torch.eye(self.num_symptoms)
        
        # 创建异构图数据对象
        data = Data()
        
        # 添加节点特征
        data.herb_x = herb_x
        data.symptom_x = symptom_x
        
        # 添加边索引
        data.herb_herb_edge_index = self.herb_herb_edge_index
        data.symptom_symptom_edge_index = self.symptom_symptom_edge_index
        data.herb_symptom_edge_index = self.herb_symptom_edge_index
        
        # 确保所有边都是无向的
        to_undirected = ToUndirected()
        data = to_undirected(data)
        
        return data
    
    def len(self):
        return 1
    
    def get(self, idx):
        return self.data