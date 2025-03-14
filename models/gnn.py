import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TCMRecommender(nn.Module):
    def __init__(self, num_herbs, num_symptoms, hidden_channels=64, num_heads=4, dropout=0.2):
        """
        基于GAT的中医处方推荐模型
        
        Args:
            num_herbs (int): 药材节点数量
            num_symptoms (int): 症状节点数量
            hidden_channels (int): 隐藏层维度
            num_heads (int): 注意力头数
            dropout (float): Dropout率
        """
        super(TCMRecommender, self).__init__()
        
        # 初始特征嵌入层
        self.herb_embedding = nn.Linear(num_herbs, hidden_channels)
        self.symptom_embedding = nn.Linear(num_symptoms, hidden_channels)
        
        # 药材子图的GAT层 - 注意输出维度
        self.herb_gat1 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.herb_gat2 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        
        # 症状子图的GAT层 - 注意输出维度
        self.symptom_gat1 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.symptom_gat2 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        
        # 药材-症状跨域GAT层 - 注意输出维度
        self.cross_gat = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        
        # 症状注意力层
        self.symptom_attention = nn.MultiheadAttention(
            hidden_channels, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # 线性变换层，用于调整维度
        self.herb_transform = nn.Linear(hidden_channels, hidden_channels)
        self.symptom_transform = nn.Linear(hidden_channels, hidden_channels)
        
        # 修改预测层的维度
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # 将第一层的输出维度改为hidden_channels
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_herbs)  # 第二层的输入维度相应改变
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data, symptom_indices, symptom_mask=None):
        """
        前向传播
        
        Args:
            data: 图数据对象，包含herb_herb_edge_index, symptom_symptom_edge_index, herb_symptom_edge_index
            symptom_indices: 输入症状的索引列表 [batch_size, max_symptoms]
            symptom_mask: 症状的mask [batch_size, max_symptoms]
        """
        # 初始节点嵌入
        herb_x = self.herb_embedding(data.herb_x)
        symptom_x = self.symptom_embedding(data.symptom_x)
        
        # 药材子图消息传递
        herb_x = F.elu(self.herb_gat1(herb_x, data.herb_herb_edge_index))
        herb_x = self.dropout(herb_x)
        herb_x = F.elu(self.herb_gat2(herb_x, data.herb_herb_edge_index))
        herb_x = self.herb_transform(herb_x)  # 调整维度
        
        # 症状子图消息传递
        symptom_x = F.elu(self.symptom_gat1(symptom_x, data.symptom_symptom_edge_index))
        symptom_x = self.dropout(symptom_x)
        symptom_x = F.elu(self.symptom_gat2(symptom_x, data.symptom_symptom_edge_index))
        symptom_x = self.symptom_transform(symptom_x)  # 调整维度
        
        # 药材-症状跨域消息传递
        edge_index = data.herb_symptom_edge_index
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        bidir_edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        
        # 合并药材和症状特征
        all_x = torch.cat([herb_x, symptom_x], dim=0)
        all_x = F.elu(self.cross_gat(all_x, bidir_edge_index))
        
        # 分离回药材和症状特征
        herb_x = all_x[:herb_x.size(0)]
        symptom_x = all_x[herb_x.size(0):]
        
        if symptom_indices is not None:
            # 获取指定症状的特征
            query_symptoms = symptom_x[symptom_indices]
            
            if symptom_mask is not None:
                query_symptoms = query_symptoms * symptom_mask.unsqueeze(-1)
            
            # 症状自注意力
            attended_symptoms, _ = self.symptom_attention(
                query_symptoms,
                query_symptoms,
                query_symptoms
            )
            
            # 使用symptom_mask来正确聚合症状特征
            if symptom_mask is not None:
                mask_expanded = symptom_mask.unsqueeze(-1).float()
                symptom_query = (attended_symptoms * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                symptom_query = attended_symptoms.mean(dim=1)
            
            # 获取全局药材图特征
            global_herb = herb_x.mean(dim=0, keepdim=True).expand(symptom_query.size(0), -1)
            
            # 合并症状查询和全局药材特征
            combined_features = torch.cat([symptom_query, global_herb], dim=1)
            
            # 预测药材概率
            herb_scores = self.predictor(combined_features)
            
            return herb_scores
        
        return herb_x, symptom_x
    
    def recommend_herbs(self, data, symptom_indices, symptom_mask=None, top_k=10):
        """
        根据输入症状推荐药材
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data, symptom_indices, symptom_mask)
            probs = torch.sigmoid(logits)
            values, indices = torch.topk(probs, k=top_k, dim=1)
        return indices, values
