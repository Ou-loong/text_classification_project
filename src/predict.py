# 预测模块

import joblib
import pandas as pd
from .utils import clean_text

def predict_and_save(test_df, model_path="models/tfidf_model.pkl", vectorizer_path="models/vectorizer.pkl", output_path="submit.txt"):
    """
    使用训练好的模型对测试集进行预测，并保存结果为 submit.txt
    :param test_df: 测试集 DataFrame，仅包含 'text'
    :param model_path: 模型文件路径
    :param vectorizer_path: 向量器文件路径
    :param output_path: 提交文件路径
    """
    # 加载模型和向量器
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # 文本清洗
    texts = test_df["text"].apply(clean_text)

    # 特征转换
    X_test_tfidf = vectorizer.transform(texts)

    # 预测
    predictions = model.predict(X_test_tfidf)

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        for label in predictions:
            f.write(f"{int(label)}\n")

    print(f"预测结果已保存至 {output_path}")
