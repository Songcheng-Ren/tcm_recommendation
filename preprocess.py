from utils.data_utils import build_graphs_from_prescriptions
import os

def main():
    # 获取当前文件所在目录作为项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 数据目录路径
    data_dir = os.path.join(project_root, 'data')
    
    print(f"项目根目录: {project_root}")
    print(f"数据目录: {data_dir}")
    
    print("开始构建图...")
    build_graphs_from_prescriptions(data_dir)
    print("图构建完成！")
    print(f"处理后的数据已保存至: {os.path.join(data_dir, 'processed')}")

if __name__ == "__main__":
    main()
