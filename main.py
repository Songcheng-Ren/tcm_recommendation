import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datasets.tcm_dataset import TCMDataset
from models.gnn import TCMRecommender
from sklearn.model_selection import train_test_split
import random
from datasets.prescription_dataset import PrescriptionDataset

class TrainingConfig:
    def __init__(self):
        self.num_epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.001
        self.hidden_channels = 64
        self.num_heads = 4
        self.dropout = 0.2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.early_stopping_patience = 10
        self.seed = 42

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def calculate_pos_weight(dataset):
    """计算正样本权重"""
    total_herbs = torch.zeros(dataset.num_herbs)
    for p in dataset.prescriptions:
        total_herbs[p['herbs']] += 1
    
    # 计算正负样本比例
    num_prescriptions = len(dataset.prescriptions)
    pos_weight = (num_prescriptions - total_herbs) / total_herbs.clamp(min=1)
    return pos_weight

def evaluate(model, data_loader, config):  
    """支持多阈值k值的评估函数"""
    model.eval()
    device = config.device
    
    # 初始化结果容器
    metrics = {
        'p@5': 0.0, 'p@10': 0.0, 'p@20': 0.0,
        'r@5': 0.0, 'r@10': 0.0, 'r@20': 0.0,
        'total_samples': 0
    }
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['symptoms'].to(device)
            labels = batch['herbs'].to(device).float()
            
            # 模型预测
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            
            # 遍历不同k值
            for k in [5, 10, 20]:
                # 获取topk预测结果
                _, topk_indices = torch.topk(probs, k=k, dim=1)
                
                # 生成预测矩阵
                preds = torch.zeros_like(probs)
                for i in range(preds.size(0)):
                    preds[i, topk_indices[i]] = 1.0
                
                # 计算正确预测数
                correct = (preds * labels).sum(dim=1)
                
                # 计算精确率和召回率
                precision = correct / k
                recall = correct / (labels.sum(dim=1) + 1e-8)  # 避免除零
                
                # 累计结果
                metrics[f'p@{k}'] += precision.sum().item()
                metrics[f'r@{k}'] += recall.sum().item()
            
            metrics['total_samples'] += inputs.size(0)
    
    # 计算最终指标
    for k in [5, 10, 20]:
        metrics[f'p@{k}'] /= metrics['total_samples']
        metrics[f'r@{k}'] /= metrics['total_samples']
    
    return metrics

def main():
    # 加载配置
    config = TrainingConfig()
    set_seed(config.seed)
    print(f"使用设备: {config.device}")
    
    # 加载数据集
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data')
    dataset = TCMDataset(root=data_dir)

    # 获取图数据并转移到设备
    herb_graph = dataset.herb_graph.to(config.device)
    symptom_graph = dataset.symptom_graph.to(config.device)
    cross_graph = dataset.cross_graph.to(config.device)

    # 准备处方数据集
    prescription_data = PrescriptionDataset(
        prescriptions=dataset.prescriptions,
        num_herbs=dataset.num_herbs,
        num_symptoms=dataset.num_symptoms
    )

    # 划分数据集
    train_idx, test_idx = train_test_split(
        range(len(prescription_data)),
        test_size=0.2,
        random_state=config.seed
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        random_state=config.seed
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        torch.utils.data.Subset(prescription_data, train_idx),
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(prescription_data, val_idx),
        batch_size=config.batch_size
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(prescription_data, test_idx),
        batch_size=config.batch_size,
    )
    
    print(f"训练集大小: {len(train_idx)}")
    print(f"验证集大小: {len(val_idx)}")
    print(f"测试集大小: {len(test_idx)}")
    
    # 初始化模型
    model = TCMRecommender(
        herb_graph=herb_graph,
        symptom_graph=symptom_graph,
        cross_graph=cross_graph,
        num_herbs=dataset.num_herbs,
        num_symptoms=dataset.num_symptoms,
        hidden_channels=config.hidden_channels,
        num_heads=config.num_heads,
        dropout=config.dropout
    ).to(config.device)
    
    # 计算正样本权重
    # pos_weight = calculate_pos_weight(dataset).to(config.device)
    
    # 使用加权BCE损失
    criterion = nn.BCEWithLogitsLoss()
    
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 训练循环
    best_metric = -float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = batch['symptoms'].to(config.device)
            labels = batch['herbs'].to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 评估阶段
        eval_metrics = evaluate(model, val_loader, config)
        print(f"Epoch {epoch+1}/{config.num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Metrics: Precision={eval_metrics['p@5']:.4f}, {eval_metrics['p@10']:.4f}, {eval_metrics['p@20']:.4f}"
              f"Recall={eval_metrics['r@5']:.4f}, {eval_metrics['r@10']:.4f}, {eval_metrics['r@20']:.4f}")
        
        # 验证阶段后添加早停判断
        current_metric = eval_metrics['r@10']  # 选择召回率@10作为监控指标
        
        if current_metric > best_metric:
            print(f"发现新的最佳模型：{current_metric:.4f} -> {best_metric:.4f}")
            best_metric = current_metric
            patience_counter = 0
            torch.save(model, os.path.join(project_root, 'checkpoints/best_model_checkpoint.pth'))
            torch.save(dataset, os.path.join(project_root, 'checkpoints/best_dataset_checkpoint.pth'))
        else:
            patience_counter += 1
            print(f"早停计数器：{patience_counter}/{config.early_stopping_patience}")
            if patience_counter >= config.early_stopping_patience:
                print("达到早停条件，终止训练")
                break
        
        
    
    # 加载最佳模型进行测试
    dataset = torch.load(os.path.join(project_root, 'checkpoints/best_dataset_checkpoint.pth'))
    model.load_state_dict(torch.load(os.path.join(project_root, 'checkpoints/best_model_checkpoint.pth')))
    test_metrics = evaluate(model, test_loader, config)
    
    print("\n测试集结果:")
    print(f"Precision={test_metrics['p@5']:.4f}, {test_metrics['p@10']:.4f}, {test_metrics['p@20']:.4f}"
          f"Recall={test_metrics['r@5']:.4f}, {test_metrics['r@10']:.4f}, {test_metrics['r@20']:.4f}")

if __name__ == "__main__":
    main()
