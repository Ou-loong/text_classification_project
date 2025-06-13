# 训练 TF-IDF + LR 模型

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib
from .utils import clean_text  # 清洗文本

def train_model(train_df):
    """
    训练 TF-IDF + LR 模型，并返回训练好的模型和向量器
    """
    texts = train_df["text"].apply(clean_text)
    labels = train_df["label"]

    # 划分验证集（可选）
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)

    # TF-IDF 特征提取 + 逻辑回归
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # 验证集上评估 F1-score
    y_pred = model.predict(X_val_tfidf)
    f1 = f1_score(y_val, y_pred)
    print(f"Validation F1-score: {f1:.4f}")

    # 保存模型
    joblib.dump(model, "models/tfidf_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    return model, vectorizer
