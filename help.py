import pandas as pd
import os

def load_herbs_contains():
    """加载herbs_contains.txt中的草药列表"""
    with open('data/raw/herbs_contains.txt', 'r', encoding='utf-8') as f:
        herbs = [line.strip() for line in f if line.strip()]
    return set(herbs)

def load_herb_all():
    """加载herb_all.xlsx中的草药列表"""
    df = pd.read_excel('data/raw/tcmbank/entity/herb_all.xlsx')
    # 假设草药名称在第一列，根据实际列名调整
    herbs = set(df.iloc[:, 3].tolist())
    return herbs

def main():
    # 加载两个文件中的草药
    herbs_contains = load_herbs_contains()
    herbs_all = load_herb_all()
    
    # 统计重复情况
    common_herbs = herbs_contains.intersection(herbs_all)
    herbs_only_in_contains = herbs_contains - herbs_all
    herbs_only_in_all = herbs_all - herbs_contains
    
    # 打印统计结果
    print(f"herbs_contains.txt中的草药总数: {len(herbs_contains)}")
    print(f"herb_all.xlsx中的草药总数: {len(herbs_all)}")
    print(f"两个文件中重复的草药数量: {len(common_herbs)}")
    print(f"仅在herbs_contains.txt中的草药数量: {len(herbs_only_in_contains)}")
    print(f"仅在herb_all.xlsx中的草药数量: {len(herbs_only_in_all)}")
    
    # 保存结果到文件
    with open('data/processed/herb_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("=== 草药统计结果 ===\n")
        f.write(f"herbs_contains.txt中的草药总数: {len(herbs_contains)}\n")
        f.write(f"herb_all.xlsx中的草药总数: {len(herbs_all)}\n")
        f.write(f"两个文件中重复的草药数量: {len(common_herbs)}\n")
        f.write(f"仅在herbs_contains.txt中的草药数量: {len(herbs_only_in_contains)}\n")
        f.write(f"仅在herb_all.xlsx中的草药数量: {len(herbs_only_in_all)}\n\n")
        
        f.write("=== 重复的草药列表 ===\n")
        for herb in sorted(common_herbs):
            f.write(f"{herb}\n")
        
        f.write("\n=== 仅在herbs_contains.txt中的草药 ===\n")
        for herb in sorted(herbs_only_in_contains):
            f.write(f"{herb}\n")
        
        f.write("\n=== 仅在herb_all.xlsx中的草药 ===\n")
        for herb in sorted(herbs_only_in_all):
            f.write(f"{herb}\n")

if __name__ == "__main__":
    main()