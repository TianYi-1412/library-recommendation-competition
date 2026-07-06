# 算法一：XGBoost + GNN 混合方案（最终版 v8）

> **文件**：`src/xgboost_gnn_pipeline/v8.py`
>
> **类型**：梯度提升树 + 图特征工程
>
> **版本演进**：v4（7维特征）→ v5（15维特征）→ xgboost-600（10维+GNN）→ **v8（28维+二分图+Transformer）**

---

## 目录

- [算法原理](#算法原理)
- [系统架构](#系统架构)
- [代码结构分析](#代码结构分析)
- [特征工程详解](#特征工程详解)
- [多路召回策略](#多路召回策略)
- [核心代码解析](#核心代码解析)
- [执行流程](#执行流程)
- [性能分析](#性能分析)

---

## 算法原理

### 核心思想

本算法将**梯度提升树（XGBoost）**与**图神经网络嵌入（GNN）**相结合，构建了一个强大的混合推荐系统。采用"召回+排序"的两阶段架构：

1. **多路召回**：从12个不同通道生成候选图书（分类、关键词、作者等）
2. **特征提取**：为每个用户-图书对提取28维特征向量
3. **Transformer 打分**：使用轻量 Transformer 模型对候选集打分
4. **图相似度融合**：结合二分图嵌入的余弦相似度

### 为什么用 XGBoost + 图特征？

| 组件 | 作用 |
|------|------|
| XGBoost | 处理非线性特征交互，抗过拟合能力强 |
| GNN 嵌入 | 捕获用户-图书图结构中的高阶协同信号 |
| 二分图 | 直接建模用户-图书交互结构 |
| Transformer | 捕获交互序列中的时序模式 |

### 数学公式

**最终得分** = 0.6 × Transformer_得分 + 0.4 × Cosine_相似度(用户嵌入, 图书嵌入)

其中：
- `Transformer_得分`：LightTransformer 对节点序列的输出
- `Cosine_相似度`：用户和图书图嵌入之间的余弦相似度

---

## 系统架构

```
原始数据（用户 + 图书 + 交互记录）
    |
    v
[数据预处理]
    - ID 统一（字符串类型）
    - 文本清洗（小写、去除空白）
    - 核心关键词提取
    - 时间特征计算
    |
    v
[二分图构建]
    - 用户节点 + 图书节点
    - PCA 降维嵌入（64维）
    - 邻接矩阵表示
    |
    v
[多路召回]（12路）
    - 分类/关键词/作者/出版社/主题匹配
    - 过滤已借阅图书
    |
    v
[特征提取]（28维）
    - 11维数值特征（行为、内容、图）
    - 17维类别特征（哈希桶 5万）
    |
    v
[Transformer 打分]
    - LightTransformer 编码器
    - BCE 损失训练
    |
    v
[最终排序]
    - Transformer 得分 + 图相似度的加权融合
    - 每个用户选 Top-1
```

---

## 代码结构分析

### 主类：`PureNewBreakthroughV8`

```python
class PureNewBreakthroughV8:
    # 数据结构
    ├── user_train_data        # 每个用户的训练记录
    ├── user_val_data          # 验证数据（正样本 + 3个负样本）
    ├── val_labels             # 标签 [1, 0, 0, 0]
    ├── book_info              # 图书元数据字典
    ├── user_profile           # 用户院系映射
    ├── user_dept_top_cates    # 每个院系的热门分类
    ├── user_secondary_cates   # 用户的 Top-5 二级分类
    ├── user_core_keywords     # 用户的 Top-7 关键词
    ├── user_book_renew        # 用户-图书续借次数
    ├── user_book_repeat       # 用户-图书重复借阅次数
    ├── book_hot               # 时间衰减加权热度
    │
    # 多路召回通道（9个映射表）
    ├── secondary_cate_books   # 二级分类 → 图书集合
    ├── keyword_books          # 关键词 → 图书集合
    ├── author_books           # 作者 → 图书集合
    ├── publisher_books        # 出版社 → 图书集合
    ├── detail_cate_books      # 细化分类 → 图书集合
    ├── compound_cate_books    # 复合分类 → 图书集合
    ├── theme_books            # 主题标签 → 图书集合
    │
    # 图结构组件
    ├── user2idx / book2idx    # ID → 索引映射
    ├── edges                  # 用户-图书边
    ├── graph (BipartiteGraph) # PCA 嵌入
    ├── user_embed / book_embed # 64维嵌入
    │
    # 模型
    ├── transformer (LightTransformer) # 打分模型
    └── device (GPU/CPU)
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embed_dim` | 64 | 图嵌入维度 |
| `time_decay_half_life` | 90天 | 时间衰减半衰期 |
| `hot_decay` | 0.15 | 热度衰减因子 |
| `weak_feat_boost` | 8 | 弱特征增强倍数 |
| `renew_score_upper` | 1.5 | 续借分数上限 |

---

## 特征工程详解

### 28维特征向量

```
特征索引 | 特征名                  | 类型     | 说明
----------|-------------------------|----------|------------------------------
0         | renew_score             | 浮点     | 归一化续借次数
1         | hot_score               | 浮点     | 时间衰减热度（归一化）
2         | secondary_match         | 浮点     | 二级分类匹配（×8）
3         | dept_match              | 浮点     | 院系分类匹配（×8）
4         | keyword_match           | 浮点     | 关键词匹配（×8）
5         | dept_secondary_cross    | 浮点     | 交叉特征：院系×分类
6         | keyword_secondary_cross | 浮点     | 交叉特征：关键词×分类
7         | graph_sim               | 浮点     | 用户-图书嵌入点积
8         | author_match            | 二元     | 作者偏好匹配
9         | publisher_match         | 二元     | 出版社偏好匹配
10        | theme_tag_match         | 二元     | 主题标签重叠
11-27     | cat_feats (17维)       | 整数     | 哈希桶特征（模 50000）
```

### 哈希桶特征（17维）

```python
cat_feats = [
    hash(作者) % 50000,
    hash(出版社) % 50000,
    hash(一级分类) % 50000,
    hash(二级分类) % 50000,
    hash(主题标签) % 50000,
    hash(用户ID) % 50000,
    hash(图书ID) % 50000,
    hash(用户主题) % 50000,
    hash(细化分类) % 50000,
    hash(复合分类) % 50000,
    hash(书名关键词) % 50000,
    hash(用户核心关键词) % 50000,
    hash(用户院系) % 50000,
    hash(作者+出版社) % 50000,
    hash(细化分类+复合分类) % 50000,
    hash(主题标签+用户主题) % 50000,
    hash(作者+图书ID) % 50000
]
```

### 特征设计 rationale

1. **续借分数**：高续借次数表示强兴趣
2. **热度分数**：热门图书更可能被借阅
3. **分类匹配**：用户偏好其感兴趣的分类
4. **关键词匹配**：语义内容相似度
5. **交叉特征**：捕获特征交互（如院系×分类）
6. **图相似度**：来自 GNN 的协同过滤信号
7. **哈希特征**：处理高基数类别变量

---

## 多路召回策略

### 9路召回通道

```python
def recommend_for_user(self, user_id):
    # 通道 1：二级分类图书
    for cate in user_secondary:
        candidates.update(secondary_cate_books[cate])

    # 通道 2：院系热门分类
    for cate in dept_top:
        candidates.update(该分类下的所有图书)

    # 通道 3：关键词匹配
    for kw in user_kws:
        candidates.update(keyword_books[kw])

    # 通道 4：作者偏好
    for a in user_authors(user_id):
        candidates.update(author_books[a])

    # 通道 5：出版社偏好
    for p in user_publishers(user_id):
        candidates.update(publisher_books[p])

    # 通道 6：细化分类
    for d in user_detail_cates(user_id):
        candidates.update(detail_cate_books[d])

    # 通道 7：复合分类
    for c in user_compound_cates(user_id):
        candidates.update(compound_cate_books[c])

    # 通道 8：主题标签
    for t in user_themes(user_id):
        candidates.update(theme_books[t])
```

### 负采样策略

每个正样本（用户, 图书, 标签=1）搭配 3 个负样本：
- 从用户未借阅的图书中采样
- 正负样本比例 1:3

---

## 核心代码解析

### 1. 数据加载与预处理

```python
def load_data(self, inter_path, book_path, user_path):
    # 加载三张表
    self.inter_df = pd.read_csv(inter_path, 
        usecols=['user_id', 'book_id', '借阅时间', '续借次数'],
        parse_dates=['借阅时间'])
    self.book_df = pd.read_csv(book_path)
    self.user_df = pd.read_csv(user_path, usecols=['借阅人', 'DEPT'])

    # 链式预处理
    self._unify_ids()              # 所有 ID 转为字符串
    self._clean_text()             # 清洗文本字段
    self._extract_core_keywords()  # 从标题提取关键词
    self.latest_time = self.inter_df['借阅时间'].max()
    self._build_book_mappings()    # 构建图书相关字典
    self._build_user_mappings()    # 构建用户相关字典
    self._expand_validation_set()  # 生成带负样本的验证集
    self._build_bipartite_graph()  # 构建用户-图书二分图
```

### 2. 验证集与负采样

```python
def _expand_validation_set(self):
    for user_id, group in inter_sorted.groupby('user_id'):
        # 正样本：最后一次借阅的图书
        val_row = group.tail(1).iloc[0]
        true_book = val_row['book_id']

        # 从同分类生成 3 个负样本
        same_cate_books = [b for b in books 
                           if same_category(b, true_book) and b not in user_borrowed]
        # 不足时用热门图书补足
        if len(same_cate_books) < 3:
            same_cate_books += hot_books[:3 - len(same_cate_books)]

        neg_books = np.random.choice(same_cate_books, 3, replace=False)

        # 存储：[正样本, 负1, 负2, 负3]，标签 [1, 0, 0, 0]
        self.user_val_data[user_id] = [true_book] + list(neg_books)
        self.val_labels[user_id] = [1, 0, 0, 0]
```

### 3. 二分图构建

```python
def _build_bipartite_graph(self):
    # 创建索引映射
    users = sorted(self.inter_df['user_id'].unique())
    books = sorted(self.inter_df['book_id'].unique())
    self.user2idx = {u: i for i, u in enumerate(users)}
    self.book2idx = {b: i for i, b in enumerate(books)}

    # 构建边列表
    self.edges = [(user2idx[u], book2idx[b]) 
                  for u, b in zip(inter_df['user_id'], inter_df['book_id'])]

    # PCA 嵌入
    self.graph = BipartiteGraph(self.edges, len(users), len(books), embed_dim=64)
    self.user_embed = self.graph.user_embed  # [用户数, 64]
    self.book_embed = self.graph.book_embed  # [图书数, 64]
```

### 4. Transformer 训练

```python
def _build_transformer_model(self, features, labels):
    # 从交互记录创建节点序列 [user_id, book_id]
    node_seq = np.column_stack([user_ids, book_ids])  # [N, 2]

    # 初始化 LightTransformer
    num_nodes = len(self.user2idx) + len(self.book2idx)
    self.transformer = LightTransformer(num_nodes, embed_dim=64, num_layers=2)

    # 训练循环
    for epoch in range(10):
        optimizer.zero_grad()
        out = self.transformer(seq_tensor).squeeze()   # [N]
        loss = nn.BCELoss()(out, y_tensor)              # 二元交叉熵
        loss.backward()
        optimizer.step()
```

### 5. 推荐打分

```python
def recommend_for_user(self, user_id):
    # 多路召回
    candidates = self._multi_channel_recall(user_id)
    candidates = [b for b in candidates if b not in self.user_borrow_history[user_id]]

    # 提取 28 维特征
    feats = np.array([self._extract_feature_vector(user_id, b) for b in candidates])
    base_new = feats[:, :11].astype(np.float32)      # 11维数值
    cat_feats = feats[:, -17:].astype(np.int32)      # 17维类别

    # Transformer 打分
    with torch.no_grad():
        node_seq = np.column_stack([[user_idx] * len(candidates),
                                    [book_idx for b in candidates]])
        node_tensor = torch.tensor(node_seq, dtype=torch.long).to(device)
        deep_score = self.transformer(node_tensor).cpu().numpy().squeeze()

    # 加权融合：60% Transformer + 40% 图相似度
    final_score = 0.6 * deep_score + 0.4 * cosine_similarity(user_embed, book_embeds)[0]

    return candidates[np.argmax(final_score)]
```

---

## 执行流程

```
main()
  ├── load_data()              # ~2-5 秒
  │   ├── _unify_ids()
  │   ├── _clean_text()
  │   ├── _extract_core_keywords()
  │   ├── _build_book_mappings()
  │   ├── _build_user_mappings()
  │   ├── _expand_validation_set()
  │   └── _build_bipartite_graph()
  ├── build_train_features()   # ~10-30 秒
  │   ├── _extract_feature_vector() × N
  │   └── _build_transformer_model()
  ├── evaluate()               # ~30-60 秒
  │   └── recommend_for_user() × 用户数量
  │       ├── _multi_channel_recall()
  │       ├── _extract_feature_vector()
  │       └── Transformer 打分
  └── save_recommendations()   # ~1 秒
```

---

## 性能分析

### 优势

1. **特征丰富**：28维特征从多个角度刻画用户-图书关联
2. **多路召回**：9路召回通道确保候选集覆盖率高
3. **图嵌入**：基于 PCA 的二分图捕获协同信号
4. **Transformer 打分**：注意力机制捕获序列模式
5. **负采样**：平衡训练数据，引入难负样本

### 劣势

1. **冷启动**：需要足够的用户交互历史
2. **计算成本**：特征提取是 O(N × M) 复杂度
3. **哈希冲突**：17维哈希特征在 5 万桶中可能有冲突
4. **无时序建模**：时间衰减是静态的，未建模动态兴趣

### 各版本对比

| 版本 | 特征维度 | GNN | Transformer | F1 分数 |
|------|----------|-----|-------------|---------|
| v4 | 7维 | 基础 | 无 | 基线 |
| v5 | 15维 | 作者/出版社/主题 | 无 | 提升 |
| xgboost-600 | 10维 | 增强 GNN | 无 | 更好 |
| **v8** | **28维** | **二分图+PCA** | **有** | **最佳** |

---

## 相关文件

- `src/xgboost_gnn_pipeline/v8.py` — 主实现（最终版）
- `src/xgboost_gnn_pipeline/xgboost-600.py` — GNN + XGBoost 混合
- `src/transformer_models/transformer_v8.py` — LightTransformer 模型
- `src/transformer_models/graph_utils_v8.py` — BipartiteGraph 工具类
