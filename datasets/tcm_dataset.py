import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import ToUndirected
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
        
        self.herb_graph = self._create_herb_graph()
        self.symptom_graph = self._create_symptom_graph()
        self.cross_graph = self._create_cross_graph()
        
        # 加载处方数据
        self.prescriptions = np.load(os.path.join(processed_dir, 'prescriptions.npy'), allow_pickle=True)
    
    def _create_herb_graph(self):
        """药材-药材图 (ID从0开始)"""
        herb_features = torch.randn(self.num_herbs, 64)
        edge_index = self.herb_herb_edge_index
        return ToUndirected()(Data(x=herb_features, edge_index=edge_index))
    
    def _create_symptom_graph(self):
        """症状-症状图 (ID从0开始)"""
        symptom_features = torch.randn(self.num_symptoms, 64)
        edge_index = self.symptom_symptom_edge_index
        return ToUndirected()(Data(x=symptom_features, edge_index=edge_index))
    
    def _create_cross_graph(self):
        """跨域二分图 (药材和症状ID统一编码)"""
        cross_features = torch.randn(self.num_herbs + self.num_symptoms, 64)
        edge_index = self.herb_symptom_edge_index
        return ToUndirected()(Data(x=cross_features, edge_index=edge_index))
    
    def len(self):
        return 1
    
    def get(self, idx):
        return self.data