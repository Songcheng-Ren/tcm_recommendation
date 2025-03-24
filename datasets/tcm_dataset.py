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
        
        self.ingredient_edge_index = torch.from_numpy(
            np.load(os.path.join(processed_dir, 'herb_ingredient_edge_index.npy'))
        ).long()
        # 获取节点数量
        self.num_herbs = self.herb_herb_edge_index.max().item() + 1
        self.num_symptoms = self.symptom_symptom_edge_index.max().item() + 1
        self.num_ingredients = self.ingredient_edge_index.max().item() + 1

        self.herb_graph = self._create_herb_graph()
        self.symptom_graph = self._create_symptom_graph()
        self.cross_graph = self._create_cross_graph()
        self.hyper_graph = self._create_hyper_graph()
        
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

    def _create_hyper_graph(self):
        def create_hyper_edge_mapping(edge_index, num_ingredients):
            # 初始化映射，-1 表示未赋值
            mapping = torch.full((num_ingredients,), -1, dtype=torch.long)
            # 遍历 edge_index 的每一列
            for i in range(edge_index.size(1)):
                ing = edge_index[0, i].item()   # 成分节点索引
                herb = edge_index[1, i].item()  # 对应的草药（超边）索引
                # 如果该成分还没有映射，则记录第一个出现的超边编号
                if mapping[ing] == -1:
                    mapping[ing] = herb
            return mapping
        hyper_edge_mapping = create_hyper_edge_mapping(self.ingredient_edge_index, self.num_ingredients)
        """成分-药材超图 (ID从0开始)"""
        hyper_features = torch.randn(self.num_ingredients, 64)
        edge_index = self.ingredient_edge_index
        data = Data(x=hyper_features, edge_index=edge_index, num_hyperedges=self.num_herbs, hyper_edge_mapping=hyper_edge_mapping)
        data = ToUndirected()(data)
        data.hyper_edge_mapping = hyper_edge_mapping  # 确保自定义属性被保留
        return data
    
    def len(self):
        return 1
    
    def get(self):
        return self.data
    