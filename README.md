# 图书馆图书推荐系统

> 全球校园人工智能算法精英大赛——基于高校图书馆借阅数据的用户潜在图书借阅预测推荐。本项目实现了 8 大类推荐算法，并配有详细的中文技术文档。

---

## 目录

- [项目概述](#项目概述)
- [赛题介绍](#赛题介绍)
- [仓库结构](#仓库结构)
- [算法架构总览](#算法架构总览)
- [快速开始](#快速开始)
- [算法家族详解](#算法家族详解)
- [实验结果](#实验结果)
- [关键技术总结](#关键技术总结)
- [License](#license)

---

## 项目概述

本项目是全球校园人工智能算法精英大赛——基于图书借阅数据的用户潜在借阅预测推荐的多种完整解决方案。基于高校图书馆的真实借阅数据，预测每个用户下一本可能借阅的图书。

**核心亮点：**
- **8 大算法家族全覆盖**：XGBoost、GNN、Transformer、GRU、DeepFM、LightGBM、Item2Vec、RecBole 框架
- **丰富的特征工程**：28 维特征向量，融合行为特征、图特征、内容特征和时序特征
- **多路召回策略**：12 路召回通道（分类、关键词、作者、出版社、主题等）
- **智能集成融合**：基于多样性增强的加权融合策略

**版本演进路线**：
```
v4 (7维特征) → v5 (15维特征) → xgboost-600 (10维+GNN) → v8 (28维+二分图+Transformer)
```

---

## 赛题介绍

**赛题名称**：基于高校图书馆借阅数据的用户潜在图书借阅预测推荐

**任务描述**：
给定图书馆历史借阅数据，为每个用户预测其下一本可能借阅的图书，输出 Top-1 推荐。

**数据规模**：
- 88,378 条借阅记录
- 1,451 个用户
- 58,549 本图书

**评估指标**：F1 Score（精确率和召回率的调和平均数）

**数据文件**：
- `item.csv`：图书信息（book_id、题名、作者、出版社、一级分类、二级分类）
- `inter.csv`：借阅交互记录（user_id、book_id、借阅时间、还书时间、续借次数）
- `user.csv`：用户信息（借阅人、性别、DEPT、年级、类型）

---

## 仓库结构

```
library-recommendation-competition/
|__ README.md                          # 项目总览（本文档）
|__ requirements.txt                   # Python 依赖包
|__ .gitignore                         # Git 忽略规则
|__ docs/                              # 技术文档
|   |__ algorithms/                    # 算法详细文档（8篇）
|   |   |__ 01-xgboost-gnn-pipeline.md  # XGBoost+GNN 混合方案
|   |   |__ 02-gnn-models.md           # 图神经网络
|   |   |__ 03-transformer-models.md   # Transformer 注意力模型
|   |   |__ 04-gru4rec.md              # GRU 序列推荐
|   |   |__ 05-recbole-framework.md    # RecBole 多模型集成框架
|   |   |__ 06-item2vec-lightgbm.md    # Item2Vec + LightGBM 排序
|   |   |__ 07-deepfm.md               # 深度因子分解机
|   |   |__ 08-multimodal-dualtower.md # 多模态双塔模型
|   |__ experiments/                   # 实验记录与分析
|       |__ technical-report.md        # 技术报告摘要
|__ src/                               # 源代码
|   |__ xgboost_gnn_pipeline/          # XGBoost + GNN 方案
|   |   |__ v8.py                      # 最终版（28维特征）
|   |   |__ xgboost-600.py             # GNN 增强 XGBoost
|   |   |__ deprecated/                # 早期版本（v4、v5）
|   |__ gnn_models/                    # 图神经网络模型
|   |   |__ gnn_feature.py             # 增强版多关系 GNN
|   |   |__ gnn_v5.py                  # LightGCN 基础实现
|   |__ transformer_models/            # Transformer 模型
|   |   |__ transformer_v8.py          # 轻量 Transformer 编码器
|   |   |__ graph_utils_v8.py          # 二分图工具类
|   |__ gru4rec/                       # RNN 序列模型
|   |   |__ GRU4Rec 0.0158.py          # 兴趣场景感知的 GRU4Rec
|   |__ recbole_framework/             # RecBole 集成框架
|   |   |__ main.py                    # 主入口
|   |   |__ configs.py                 # 模型配置
|   |   |__ data_processor.py          # 数据预处理
|   |   |__ model_trainer.py           # 模型训练封装
|   |   |__ ensemble.py                # 智能集成器
|   |   |__ evaluator.py               # 续借感知评估器
|   |   |__ smart_weights.py           # 智能权重计算
|   |   |__ quality_analyzer.py        # 质量分析器
|   |   |__ model_classifier.py        # 模型分类器
|   |   |__ utils.py                   # 工具函数
|   |__ item2vec_lightgbm/             # Item2Vec + LightGBM 排序
|   |   |__ 01_train_item2vec.py       # 训练 Item2Vec 嵌入
|   |   |__ 02_build_rank_samples.py   # 构建排序样本
|   |   |__ 03_train_lightgbm.py       # 训练 LightGBM 排序模型
|   |   |__ 04_inference.py            # 推理预测
|   |   |__ 05_multi_recalle.py        # 多路召回
|   |   |__ 06_build_rank2.py          # 二阶段排序
|   |   |__ 08_blend_inference.py      # 混合推理
|   |__ deepfm/                        # DeepFM 深度学习模型
|   |   |__ train_deepfm.py            # DeepFM PyTorch 实现
|   |__ multimodal_dualtower/          # 多模态双塔模型
|       |__ data_loader.py             # 数据加载与预处理
|       |__ user_interesting.py        # 用户兴趣建模
|       |__ model.py                   # 核心模型实现
|       |__ result.py                  # 评估与结果生成
|__ results/                           # 比赛结果
|__ scripts/                           # 辅助脚本
```

---

## 算法架构总览

```
输入数据（用户 + 图书 + 交互记录）
    |
    |-- [XGBoost+GNN 方案] ----- 12路召回 --> 特征工程 --> XGBoost 分类器
    |-- [GNN 模型] ------------- 多层图 --> LightGCN/LGConv 嵌入
    |-- [Transformer] ---------- 二分图 --> Transformer 编码器 --> 打分
    |-- [GRU4Rec] -------------- 用户序列 --> GRU + 场景嵌入
    |-- [RecBole 框架] --------- ItemKNN/BPR/SASRec/GRU4Rec/BERT4Rec --> 集成
    |-- [Item2Vec+LightGBM] ---- Word2Vec 嵌入 --> LightGBM 排序
    |-- [DeepFM] --------------- 稀疏/稠密特征 --> FM + 深度网络
    |-- [多模态双塔] ------------- M3E 文本嵌入 --> 双塔召回
    |
    v
最终推荐结果（每个用户 Top-1 图书）
```

---

## 快速开始

### 1. 环境搭建

```bash
# 克隆仓库
git clone https://github.com/yourusername/library-recommendation-competition.git
cd library-recommendation-competition

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将比赛数据文件放入 `data/` 目录：
- `inter_preliminary.csv` — 交互记录
- `item.csv` — 图书元数据
- `user.csv` — 用户画像

### 3. 运行各算法

**XGBoost + GNN 方案（推荐）**
```bash
cd src/xgboost_gnn_pipeline
python v8.py
```

**Item2Vec + LightGBM 完整流程**
```bash
cd src/item2vec_lightgbm
python 01_train_item2vec.py      # 训练图书嵌入
python 02_build_rank_samples.py  # 构建排序样本
python 03_train_lightgbm.py      # 训练排序模型
python 04_inference.py           # 生成预测结果
```

**RecBole 集成框架**
```bash
cd src/recbole_framework
python main.py
```

**GRU4Rec 序列模型**
```bash
cd src/gru4rec
python "GRU4Rec 0.0158.py"
```

**DeepFM**
```bash
cd src/deepfm
python train_deepfm.py
```

---

## 算法详解

| # | 算法 | 类型 | 核心特点 | 文档 |
|---|------|------|---------|------|
| 1 | **XGBoost + GNN** | 梯度提升+图特征 | 28维特征、12路召回 | [详情](docs/algorithms/01-xgboost-gnn-pipeline.md) |
| 2 | **GNN 模型** | 图神经网络 | 多关系异构图、LightGCN | [详情](docs/algorithms/02-gnn-models.md) |
| 3 | **Light Transformer** | 注意力机制 | 二分图+Transformer编码器 | [详情](docs/algorithms/03-transformer-models.md) |
| 4 | **GRU4Rec** | RNN序列 | 兴趣-场景时序建模 | [详情](docs/algorithms/04-gru4rec.md) |
| 5 | **RecBole 框架** | 集成系统 | 8个模型+智能融合 | [详情](docs/algorithms/05-recbole-framework.md) |
| 6 | **Item2Vec + LightGBM** | 嵌入+梯度提升 | Skip-gram嵌入+排序学习 | [详情](docs/algorithms/06-item2vec-lightgbm.md) |
| 7 | **DeepFM** | 深度学习 | 因子分解机+深度网络 | [详情](docs/algorithms/07-deepfm.md) |
| 8 | **多模态双塔** | 多模态 | M3E文本嵌入+双塔召回 | [详情](docs/algorithms/08-multimodal-dualtower.md) |


---

## 关键技术总结

### 多路召回策略（12路）

1. 二级分类匹配
2. 院系一级分类匹配
3. 关键词匹配
4. 作者偏好
5. 出版社偏好
6. 细化分类匹配
7. 复合分类匹配
8. 主题标签匹配
9. 图嵌入相似度
10. 时间衰减热门图书
11. 负样本过滤
12. 共现模式（Apriori）

### 特征工程（28维）

- **行为特征**：续借分数、热度分数
- **内容匹配**：分类、关键词、作者、出版社、主题
- **交叉特征**：院系×分类、关键词×分类
- **图特征**：用户-图书嵌入点积
- **类别特征**：17维哈希桶编码

### 多样性增强

- 热门图书惩罚
- 分类多样性约束
- 智能权重计算（性能+多样性+覆盖率）
- 基尼系数集中度检测

---

## License

本项目采用 MIT License。

---

## 致谢

- 赛题主办方：**全球校园人工智能算法精英大赛**
- 出题单位：**模式分析与机器智能工信部重点实验室**
- 本项目使用了 [RecBole](https://github.com/RUCAIBox/RecBole)、[PyTorch](https://pytorch.org/)、[XGBoost](https://xgboost.ai/)、[LightGBM](https://lightgbm.readthedocs.io/)
