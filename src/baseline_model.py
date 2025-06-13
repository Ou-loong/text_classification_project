# 训练 TF-IDF + LR 模型

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
import joblib
from .utils import clean_text  # 清洗文本

def train_model(train_df):
    """
    训练 TF-IDF + LR 模型，并返回训练好的模型和向量器
    """
    print("正在清洗文本...")
    texts = train_df["text"].apply(clean_text)
    labels = train_df["label"]

    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)

    # TF-IDF 特征提取 + 逻辑回归
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",    
        max_features=15000,      # 增大特征数量
        ngram_range=(1, 3),       # 使用 uni+bi+tri-gram
        sublinear_tf=True           # 缩小高频词影响
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    print("正在训练逻辑回归模型...")
    model = LogisticRegression(
        max_iter=1000,
        C=2.0,                      # 更少正则，减小欠拟合
        class_weight="balanced",   # 平衡类别分布
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)

    # 验证集上评估 F1-score
    y_pred = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    print(f"验证集准确率 Accuracy: {acc:.4f}")
    print(f"验证集 F1-score: {f1:.4f}")
    print("分类报告：\n", classification_report(y_val, y_pred))

    # 保存模型
    joblib.dump(model, "models/tfidf_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("模型已保存至 models/ 目录。")

    return model, vectorizer
