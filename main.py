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

def prepare_prescription_data(dataset):
    """
    直接使用预处理好的处方数据
    """
    return dataset.prescriptions

def create_herb_label_vector(herbs, num_herbs):
    """创建药材的one-hot标签向量"""
    label = torch.zeros(num_herbs)
    label[herbs] = 1.0
    return label

def calculate_pos_weight(dataset):
    """计算正样本权重"""
    total_herbs = torch.zeros(dataset.num_herbs)
    for p in dataset.prescriptions:
        total_herbs[p['herbs']] += 1
    
    # 计算正负样本比例
    num_prescriptions = len(dataset.prescriptions)
    pos_weight = (num_prescriptions - total_herbs) / total_herbs.clamp(min=1)
    return pos_weight

def train_epoch(model, data, train_prescriptions, optimizer, criterion, config, dataset):
    model.train()
    total_loss = 0
    
    # 随机打乱训练数据
    random.shuffle(train_prescriptions)
    
    # 批次训练
    for i in range(0, len(train_prescriptions), config.batch_size):
        batch = train_prescriptions[i:i + config.batch_size]
        
        # 准备批次数据
        # 将每个处方的症状列表转换为tensor，不需要统一长度
        symptom_indices = [torch.tensor(p['symptoms']) for p in batch]
        # 将列表打包成一个tensor，每个样本可能长度不同
        symptom_indices = torch.nn.utils.rnn.pad_sequence(
            symptom_indices, 
            batch_first=True, 
            padding_value=-1
        ).to(config.device)
        
        # 创建mask来处理填充的值
        symptom_mask = (symptom_indices != -1).to(config.device)
        
        herb_labels = torch.stack([
            create_herb_label_vector(p['herbs'], dataset.num_herbs)
            for p in batch
        ]).to(config.device)
        
        # 前向传播（需要传入mask）
        optimizer.zero_grad()
        herb_scores = model(data, symptom_indices, symptom_mask)
        
        # 计算损失
        loss = criterion(herb_scores, herb_labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_prescriptions)

def evaluate(model, data, prescriptions, config):
    model.eval()
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    with torch.no_grad():
        for prescription in prescriptions:
            # 处理单个样本的症状
            symptom_indices = torch.tensor([prescription['symptoms']]).to(config.device)
            symptom_mask = torch.ones_like(symptom_indices, dtype=torch.bool).to(config.device)
            
            true_herbs = set(prescription['herbs'])
            
            # 获取推荐结果（需要传入mask）
            pred_indices, _ = model.recommend_herbs(
                data, 
                symptom_indices, 
                symptom_mask,
                top_k=len(true_herbs)
            )
            pred_herbs = set(pred_indices[0].cpu().numpy())
            
            # 计算评估指标
            intersection = len(true_herbs & pred_herbs)
            precision = intersection / len(pred_herbs) if pred_herbs else 0
            recall = intersection / len(true_herbs) if true_herbs else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
    
    n = len(prescriptions)
    return {
        'precision': total_precision / n,
        'recall': total_recall / n,
        'f1': total_f1 / n
    }

def main():
    # 加载配置
    config = TrainingConfig()
    set_seed(config.seed)
    print(f"使用设备: {config.device}")
    
    # 加载数据集
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data')
    dataset = TCMDataset(root=data_dir)
    data = dataset[0].to(config.device)
    
    # 准备处方数据
    prescription_data = prepare_prescription_data(dataset)
    # 划分训练集和测试集
    train_data, test_data = train_test_split(
        prescription_data, 
        test_size=0.2, 
        random_state=config.seed
    )
    
    # 从训练集中划分验证集
    train_data, val_data = train_test_split(
        train_data, 
        test_size=0.2, 
        random_state=config.seed
    )
    
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 初始化模型
    model = TCMRecommender(
        num_herbs=dataset.num_herbs,
        num_symptoms=dataset.num_symptoms,
        hidden_channels=config.hidden_channels,
        num_heads=config.num_heads,
        dropout=config.dropout
    ).to(config.device)
    
    # 计算正样本权重
    pos_weight = calculate_pos_weight(dataset).to(config.device)
    
    # 使用加权BCE损失
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 修改模型的最后一层，移除sigmoid
    model.predictor = nn.Sequential(
        nn.Linear(config.hidden_channels, config.hidden_channels * 2),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_channels * 2, dataset.num_herbs)
    ).to(config.device)
    
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 训练循环
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # 训练
        train_loss = train_epoch(model, data, train_data, optimizer, criterion, config, dataset)
        
        # 验证
        val_metrics = evaluate(model, data, val_data, config)
        
        # 打印进度
        print(f'Epoch {epoch+1}/{config.num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Metrics: Precision={val_metrics["precision"]:.4f}, '
              f'Recall={val_metrics["recall"]:.4f}, F1={val_metrics["f1"]:.4f}')
        
        # 早停检查
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(project_root, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print("Early stopping triggered!")
                break
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(os.path.join(project_root, 'best_model.pth')))
    test_metrics = evaluate(model, data, test_data, config)
    
    print("\n测试集结果:")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-score: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
