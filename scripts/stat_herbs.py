import pandas as pd
import os

def load_herbs_contains():
    """加载herbs_contains.txt中的草药列表"""
    with open('../data/raw/herbs_contains.txt', 'r', encoding='utf-8') as f:
        herbs = [line.strip() for line in f if line.strip()]
    return herbs

def load_herb_all():
    """加载herb_all.xlsx中的草药列表，返回id和名字的字典"""
    df = pd.read_excel('../data/raw/tcmbank/entity/herb_all.xlsx')
    # 获取id和名字列
    herb_dict = dict(zip(df.iloc[:, 3], df.iloc[:, 0]))  # 名字作为键，id作为值
    return herb_dict

def main():
    # 加载两个文件中的草药
    herbs_contains = load_herbs_contains()
    herb_dict = load_herb_all()
    
    # 统计信息
    herbs_contains_set = set(herbs_contains)
    herb_all_set = set(herb_dict.keys())
    common_herbs = herbs_contains_set.intersection(herb_all_set)
    
    print("\n统计信息:")
    print(f"herbs_contains.txt中的草药数量: {len(herbs_contains_set)}")
    print(f"herb_all.xlsx中的草药数量: {len(herb_all_set)}")
    print(f"两个文件中都存在的草药数量: {len(common_herbs)}")
    print(f"仅在contains中存在的草药数量: {len(herbs_contains_set - herb_all_set)}")
    print(f"仅在all中存在的草药数量: {len(herb_all_set - herbs_contains_set)}")

    # 创建输出目录
    os.makedirs('../data/processed', exist_ok=True)
    
    # 创建输出文件
    with open('../data/processed/tcmbank/tcmbank_herb_mapping.txt', 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("草药名\tcontains序号\tall序号\n")
        
        # 写入所有草药信息
        for i, herb in enumerate(herbs_contains):
            contains_idx = i
            all_idx = herb_dict.get(herb, "NA")  # 如果草药存在则获取其id，否则返回NA
            f.write(f"{herb}\t{contains_idx}\t{all_idx}\n")

if __name__ == "__main__":
    main() 