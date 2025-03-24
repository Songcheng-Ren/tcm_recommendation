import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import HypergraphConv
from torch_scatter import scatter_mean


class HyperGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
class TCMRecommender(nn.Module):
    def __init__(self, herb_graph, symptom_graph, cross_graph, hyper_graph, num_herbs, num_symptoms, num_ingredients, hidden_channels=64, num_heads=4, dropout=0.2):
        """
        基于GAT的中医处方推荐模型
        
        Args:
            herb_graph (torch_geometric.data.Data): 药材图数据
            symptom_graph (torch_geometric.data.Data): 症状图数据
            cross_graph (torch_geometric.data.Data): 跨域图数据
            num_herbs (int): 药材节点数量
            num_symptoms (int): 症状节点数量
            hidden_channels (int): 隐藏层维度
            num_heads (int): 注意力头数
            dropout (float): Dropout率
        """
        super(TCMRecommender, self).__init__()
        
        # 保存配置参数
        self.herb_graph = herb_graph
        self.symptom_graph = symptom_graph
        self.cross_graph = cross_graph
        self.hyper_graph = hyper_graph
        self.num_herbs = num_herbs
        self.num_symptoms = num_symptoms
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        
        # 初始特征嵌入层
        self.herb_embedding = nn.Linear(hidden_channels, hidden_channels)
        self.symptom_embedding = nn.Linear(hidden_channels, hidden_channels)
        self.cross_embedding = nn.Linear(hidden_channels, hidden_channels)
        
        # 药材-药材图的GAT层 - 注意输出维度
        self.herb_gat1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.herb_gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout)
        
        # 症状-症状图的GAT层 - 注意输出维度
        self.symptom_gat1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.symptom_gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout)
        
        # 药材-症状跨域GAT层 - 注意输出维度
        self.cross_gat1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.cross_gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout)
        
        # 症状注意力层
        self.symptom_attention = nn.MultiheadAttention(
            hidden_channels, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # 新增超图处理分支
        self.hyper_gnn = HyperGNN(hidden_channels, hidden_channels, hidden_channels)
        self.hyper_proj = nn.Linear(hidden_channels*2, hidden_channels)  # 特征融合层
        
        self.predictor = nn.Sequential(
            nn.Linear(self.num_herbs, self.num_herbs * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.num_herbs * 2, self.num_herbs)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入的症状多热编码
        """
        # 确保输入张量是浮点类型
        x = x.float()
        
        # 初始节点嵌入
        herb_x = self.herb_graph.x
        symptom_x = self.symptom_graph.x
        cross_x = self.cross_graph.x
        # 药材-药材消息传递
        herb_x = F.elu(self.herb_gat1(herb_x, self.herb_graph.edge_index))
        herb_x = self.dropout(herb_x)
        herb_x = F.elu(self.herb_gat2(herb_x, self.herb_graph.edge_index))
        
        # 症状-症状消息传递
        symptom_x = F.elu(self.symptom_gat1(symptom_x, self.symptom_graph.edge_index))
        symptom_x = self.dropout(symptom_x)
        symptom_x = F.elu(self.symptom_gat2(symptom_x, self.symptom_graph.edge_index))
        
        # 药材-症状跨域消息传递
        cross_x = F.elu(self.cross_gat1(cross_x, self.cross_graph.edge_index))
        cross_x = self.dropout(cross_x)
        cross_x = F.elu(self.cross_gat2(cross_x, self.cross_graph.edge_index))

        # 超图学习分支
        hyper_x = self.hyper_gnn(self.hyper_graph.x, self.hyper_graph.edge_index)

        # 超边聚合（将成分特征聚合为草药特征）
        herb_from_hyper = scatter_mean(hyper_x, 
                                 self.hyper_graph.hyper_edge_mapping, 
                                 dim=0, dim_size=self.num_herbs) 
        
        # --- 特征融合 ---
        # 分割跨域特征
        cross_symptoms = cross_x[:self.num_symptoms]     # 症状部分
        cross_herbs = cross_x[self.num_symptoms:]        # 药材部分
        
        # 加权融合
        final_symptoms = symptom_x + cross_symptoms      # [num_symptoms, hidden]
        final_herbs = herb_x + cross_herbs + herb_from_hyper               # [num_herbs, hidden]
        
        # --- 多热编码交互 ---
        # 症状特征加权平均 [batch_size, hidden]
        symptom_emb = torch.matmul(x, final_symptoms) 
        
        # --- 计算推荐分数 ---
        # [batch_size, num_herbs]
        logits = torch.matmul(symptom_emb, final_herbs.T)  
        
        # 最终预测
        return self.predictor(logits)
    
    def recommend_herbs(self, symptom_indices, max_k=15):
        self.eval()
        with torch.no_grad():
            logits = self.forward(symptom_indices)          # [batch_size, num_herbs]
            probs = torch.sigmoid(logits)                   # 计算概率
            batch_k = torch.sum(probs > 0.5, dim=1)        # 动态确定每样本的k值
            batch_k = torch.clamp(batch_k, max=max_k)      # 限制最大推荐数
            indices = [torch.topk(p, k=k.item()).indices for p, k in zip(probs, batch_k)]
        return indices
