# main.py

from src.data_loader import load_train_data, load_test_data
from src.model import train_model
from src.predict import predict_and_save


import os

def check_paths():
    print("当前工作目录:", os.getcwd())
    file_path = "data/train.jsonl"
    print(f"相对路径 '{file_path}' 是否存在？", os.path.exists(file_path))
    print(f"绝对路径: {os.path.abspath(file_path)}")
    print(f"绝对路径是否存在？", os.path.exists(os.path.abspath(file_path)))

def main():
    # 1. 读取训练数据
    print("正在加载训练数据...")
    train_df = load_train_data("data/train.jsonl")

    # 2. 训练模型并保存
    print("正在训练模型...")
    train_model(train_df)

    # 3. 读取测试数据
    print("正在加载测试数据...")
    test_df = load_test_data("data/test.jsonl")

    # 4. 加载模型并预测
    print("正在进行预测...")
    predict_and_save(test_df, output_path="output/submit.txt")

    print("全部流程完成 ✅，提交文件为 submit.txt")

if __name__ == "__main__":
    #check_paths()
    main()
