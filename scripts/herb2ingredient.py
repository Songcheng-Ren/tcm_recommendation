import pandas as pd
import os

def load_herbs_contains():
    """加载herbs_contains.txt中的所有草药"""
    herbs = []
    with open('../data/raw/herbs_contains.txt', 'r', encoding='utf-8') as f:
        for line in f:
            herbs.append(line.strip())
    return herbs

def load_mapping():
    """加载已生成的映射文件"""
    mapping = {}
    with open('../data/processed/tcmbank/tcmbank_herb_mapping.txt', 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            herb_name, contains_idx, all_idx = line.strip().split('\t')
            if all_idx != "NA":
                mapping[all_idx] = (herb_name, int(contains_idx))
    return mapping

def process_herb2ingredient(herb_dict, all_herbs):
    """处理herb2ingredient.xlsx，筛选出映射文件中存在的草药对应的成分"""
    # 读取herb2ingredient.xlsx
    df = pd.read_excel('../data/raw/tcmbank/relation/herb2ingredient.xlsx')
    
    # 获取所有有效的草药ID
    valid_herb_ids = set(herb_dict.keys())
    
    # 筛选出有效草药对应的成分
    valid_ingredients = df[df['Herb_ID'].isin(valid_herb_ids)]
    
    # 获取所有成分ID并排序
    all_ingredient_ids = set()
    for ingredient_str in valid_ingredients['Ingredient_IDs']:
        if pd.notna(ingredient_str):  # 检查是否为空
            ingredient_ids = ingredient_str.split(';')
            all_ingredient_ids.update(ingredient_ids)
    
    ingredient_ids = sorted(list(all_ingredient_ids))
    
    # 创建成分ID到新序号的映射
    ingredient_mapping = {old_id: new_idx for new_idx, old_id in enumerate(ingredient_ids)}
    
    # 创建输出目录
    os.makedirs('../data/processed/tcmbank', exist_ok=True)
    
    # 保存所有成分ID
    with open('../data/processed/tcmbank/tcmbank_ingredients_contains.txt', 'w', encoding='utf-8') as f:
        for ingredient_id in ingredient_ids:
            f.write(f"{ingredient_id}\n")
    
    # 创建草药-成分映射文件
    with open('../data/processed/tcmbank/tcmbank_pre_ingredients.txt', 'w', encoding='utf-8') as f:
        # 按herbs_contains.txt的顺序处理每个草药
        for herb_name in all_herbs:
            # 查找该草药在映射文件中的ID
            herb_id = None
            for id_, (name, _) in herb_dict.items():
                if name == herb_name:
                    herb_id = id_
                    break
            
            if herb_id is not None:
                # 获取该草药的所有成分
                herb_row = valid_ingredients[valid_ingredients['Herb_ID'] == herb_id]
                if not herb_row.empty:
                    ingredient_str = herb_row['Ingredient_IDs'].iloc[0]
                    if pd.notna(ingredient_str):  # 检查是否为空
                        # 将成分ID转换为新序号
                        ingredient_ids = ingredient_str.split(';')
                        new_indices = [ingredient_mapping[ing_id] for ing_id in ingredient_ids]
                        # 写入文件：成分序号1 成分序号2 ...
                        f.write(f"{' '.join(map(str, new_indices))}\n")
                    else:
                        # 如果没有成分，写入空行
                        f.write("\n")
                else:
                    # 如果没有找到草药，写入空行
                    f.write("\n")
            else:
                # 如果草药不在映射文件中，写入空行
                f.write("\n")
    
    return len(ingredient_ids)

def main():
    # 创建输出目录
    os.makedirs('../data/processed/tcmbank', exist_ok=True)
    
    # 加载所有草药和映射文件
    all_herbs = load_herbs_contains()
    herb_dict = load_mapping()
    
    # 处理herb2ingredient.xlsx并统计成分
    num_ingredients = process_herb2ingredient(herb_dict, all_herbs)
    print(f"总共找到 {num_ingredients} 个成分ID")

if __name__ == "__main__":
    main()