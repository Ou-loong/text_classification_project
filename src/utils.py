# 具函数模块（如文本清洗）

import re

def clean_text(text):
    # 简单清洗文本
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
