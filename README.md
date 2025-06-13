 项目简介
本项目用于识别给定文本是由大语言模型生成（如 GPT-4o, Claude-3 等），还是由真实人类撰写。
项目基于传统机器学习方法（TF-IDF + 分类器）实现二分类模型，适用于天池比赛的数据格式与提交要求。

项目结构
text_classification_project/
├── data/                  # 存放训练和测试数据（JSONL 格式）
│   ├── train.jsonl
│   └── test.jsonl
├── models/                # 保存训练好的模型
│   ├── tfidf_model.pkl
│   └── vectorizer.pkl
├── output/                # 存放测试集预测结果 submit.txt
├── src/                   # 核心源码文件
│   ├── baseline_model.py  # 训练 TF-IDF + LR 模型
│   ├── data_loader.py     # 数据加载与预处理
│   ├── predict.py         # 预测模块
│   └── utils.py           # 工具函数
├── main.py                # 项目入口文件
├── requirements.txt       # Python依赖库
└── README.md              # 项目说明文档
 

环境依赖
使用 Python 3.8+，推荐创建虚拟环境。
安装依赖：
pip install -r requirements.txt

使用说明
准备数据
将 train.jsonl 和 test.jsonl 放入 data/ 目录，格式如下：
{"text": "This is an example text.", "label": 1}   # 训练数据
{"text": "Another example without label"}          # 测试数据
训练模型并预测
python main.py
运行后将在：
models/ 中保存训练好的分类器和 TF-IDF 向量器；
output/submit.txt 中保存测试集的预测结果（0 或 1，每行一个标签）。
提交结果
将 output/submit.txt 文件提交至比赛平台，格式如下：
1
0
0
1
...

文件说明
train.jsonl: 包含字段 text 和 label
test.jsonl: 仅包含 text 字段
submit.txt: 每行是一个预测标签（0 或 1）

模型说明
使用 TF-IDF 提取文本特征，并采用以下分类器：
Logistic Regression


比赛项目：CCKS2025-大模型生成文本检测