import numpy as np
from collections import defaultdict
import os

def build_graphs_from_prescriptions(data_dir):
    """
    从处方数据构建herb-herb、symptom-symptom和herb-symptom的无向图，
    并保存处方数据（症状组合-药材组合对）
    
    Args:
        data_dir (str): 数据目录路径
    """
    # 读取数据文件
    with open(os.path.join(data_dir, 'raw', 'pre_herbs.txt'), 'r') as f:
        herbs_data = [list(map(int, line.strip().split())) for line in f.readlines()]
    
    with open(os.path.join(data_dir, 'raw', 'pre_symptoms.txt'), 'r') as f:
        symptoms_data = [list(map(int, line.strip().split())) for line in f.readlines()]
    
    with open(os.path.join(data_dir, 'processed', 'tcmbank/tcmbank_pre_ingredients.txt'), 'r') as f:
        ingredients_data = [list(map(int, line.strip().split())) for line in f.readlines()]

    # 计算全局ID偏移量（症状的最大ID + 1）
    max_symptom_id = max([s for sub in symptoms_data for s in sub]) if symptoms_data else 0
    GLOBAL_ID_OFFSET = max_symptom_id + 1  # 关键修改点

    # 初始化共现频率字典
    herb_herb_freq = defaultdict(int)
    symptom_symptom_freq = defaultdict(int)
    herb_symptom_freq = defaultdict(int)
    
    # 保存处方数据
    prescriptions = []
    
    # 统计共现频率和收集处方数据
    for herbs, symptoms in zip(herbs_data, symptoms_data):
        adjusted_herbs = [hid + GLOBAL_ID_OFFSET for hid in herbs]
        # 保存处方数据
        prescriptions.append({
            'symptoms': symptoms,
            'herbs': herbs
        })
        
        # herb-herb关系
        for i in range(len(herbs)):
            for j in range(i+1, len(herbs)):
                edge = tuple(sorted([herbs[i], herbs[j]]))
                herb_herb_freq[edge] += 1
        
        # symptom-symptom关系
        for i in range(len(symptoms)):
            for j in range(i+1, len(symptoms)):
                edge = tuple(sorted([symptoms[i], symptoms[j]]))
                symptom_symptom_freq[edge] += 1
        
        # herb-symptom关系（药材使用偏移后的ID）
        for herb in adjusted_herbs:
            for symptom in symptoms:
                edge = (herb, symptom)  # 药材ID已偏移
                herb_symptom_freq[edge] += 1
    
    # 构建成分-草药超图的边索引
    ingredient_edges_src = []  # 成分id
    ingredient_edges_dst = []  # 草药id
    
    for herb_id, ingredients in enumerate(ingredients_data):
        for ingredient_id in ingredients:
            ingredient_edges_src.append(ingredient_id)
            ingredient_edges_dst.append(herb_id)
    
    ingredient_edge_index = np.array([ingredient_edges_src, ingredient_edges_dst])
    # 转换为edge_index格式
    def freq_to_edge_index(freq_dict):
        edges = list(freq_dict.keys())
        if edges:
            return np.array([[e[0] for e in edges], [e[1] for e in edges]])
        return np.array([[],[]])
    
    herb_herb_edge_index = freq_to_edge_index(herb_herb_freq)
    symptom_symptom_edge_index = freq_to_edge_index(symptom_symptom_freq)
    herb_symptom_edge_index = freq_to_edge_index(herb_symptom_freq)
    
    # 保存结果
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # 保存图数据
    np.save(os.path.join(processed_dir, 'herb_herb_edge_index.npy'), herb_herb_edge_index)
    np.save(os.path.join(processed_dir, 'symptom_symptom_edge_index.npy'), symptom_symptom_edge_index)
    np.save(os.path.join(processed_dir, 'herb_symptom_edge_index.npy'), herb_symptom_edge_index)
    np.save(os.path.join(processed_dir, 'herb_ingredient_edge_index.npy'), ingredient_edge_index)
    # 保存处方数据
    np.save(os.path.join(processed_dir, 'prescriptions.npy'), prescriptions)
    
    print("图数据统计:")
    print(f"药材-药材边数量: {herb_herb_edge_index.shape[1]}")
    print(f"症状-症状边数量: {symptom_symptom_edge_index.shape[1]}")
    print(f"药材-症状边数量: {herb_symptom_edge_index.shape[1]}")
    print(f"处方数量: {len(prescriptions)}")
    print("关键修改验证:")
    print(f"症状ID范围: 0 ~ {max_symptom_id}")
    print(f"药材ID偏移量: {GLOBAL_ID_OFFSET}")
    print(f"调整后的药材-症状边示例: {herb_symptom_edge_index[:,0] if herb_symptom_edge_index.size > 0 else '无'}")

    # 超图统计
    num_ingredients = max(ingredient_edges_src) + 1
    num_herbs = max(ingredient_edges_dst) + 1
    print("\n超图统计:")
    print(f"成分节点数量: {num_ingredients}")
    print(f"草药超边数量: {num_herbs}")
    print(f"总边数: {len(ingredient_edges_src)}")
    print(f"平均每个草药包含成分数: {len(ingredient_edges_src)/num_herbs:.2f}")