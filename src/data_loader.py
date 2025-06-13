# 负责读取和预处理数据

import pandas as pd
import json

def load_train_data(path):
    """
    加载训练集（带标签）
    :param path: 训练集路径 train.jsonl
    :return: pd.DataFrame，包含 'text' 和 'label' 两列
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({"text": item["text"], "label": item["label"]})
    return pd.DataFrame(data)


def load_test_data(path):
    """
    加载测试集（不带标签）
    :param path: 测试集路径 test.jsonl
    :return: pd.DataFrame，包含 'text' 一列
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({"text": item["text"]})
    return pd.DataFrame(data)
