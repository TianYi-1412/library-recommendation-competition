# 算法二：图神经网络（GNN）模型

> **文件**：`src/gnn_models/gnn_feature.py`、`src/gnn_models/gnn_v5.py`
>
> **类型**：图神经网络
>
> **模型**：LightGCN（基础版）→ 增强多关系 GNN（最终版）

---

## 目录

- [算法原理](#算法原理)
- [模型演进](#模型演进)
- [代码结构分析](#代码结构分析)
- [LightGCN 基础实现](#lightgcn-基础实现)
- [增强版 GNN](#增强版-gnn)
- [多关系图构建](#多关系图构建)
- [BPR 损失函数](#bpr-损失函数)
- [核心代码解析](#核心代码解析)
- [性能分析](#性能分析)

---

## 算法原理

### GNN 在推荐中的作用

图神经网络将用户-物品交互建模为**二分图**：
- **用户节点**表示借阅人
- **物品（图书）节点**表示图书
- **边**表示借阅交互

GNN 沿边传播信息，使用户嵌入捕获其交互过的图书信息，反之亦然。经过 K 层传播后，嵌入捕获了**K 阶协同信号**。

### LightGCN 架构

LightGCN 简化了 GCN，去除了特征变换和非线性激活：

```
h_u^(k+1) = sum(h_i^(k) / sqrt(|N_u| * |N_i|))  for i in N_u
```

其中：
- `h_u^(k)`：第 k 层的用户嵌入
- `N_u`：用户 u 的邻居（借过的图书）
- `N_i`：物品 i 的邻居（借过它的用户）
- `1/sqrt(|N_u| * |N_i|)`：归一化项，防止嵌入幅度爆炸

### 最终嵌入与预测

```
e_u = sum(alpha_k * h_u^(k)) for k = 0, 1, ..., K
score(u, i) = e_u^T * e_i  （点积）
```

---

## 模型演进

```
GNNV5 (gnn_v5.py)              EnhancedGNN (gnn_feature.py)
    |                                  |
    | 单关系（仅用户-图书）            | 多关系（4 种图类型）
    v                                  v
[LightGCN]                    [MultiRelationGNN]
- 基础传播                     - 用户-图书交互
- 简单推荐损失                  - 用户-用户相似度
                               - 图书-图书相似度
                               - 用户-院系归属
                               - BPR 损失
                               - LGConv 层
```

---

## 代码结构分析

### GNNV5（基础实现）

```python
class GNNV5:
    """简单的 LightGCN 包装器，用于二分图嵌入。"""
    
    def __init__(self, edges, num_nodes, embed_dim=64, num_layers=2):
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.data = Data(edge_index=edge_index, num_nodes=num_nodes)
        self.model = LightGCN(num_nodes=num_nodes, embedding_dim=embed_dim, 
                              num_layers=num_layers)
        self._train()
    
    def _train(self, epochs=20, lr=5e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(self.data.edge_index)
            loss = self.model.recommendation_loss(self.data.edge_index, out)
            loss.backward()
            optimizer.step()
    
    def get_embed_dict(self, idx2entity):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.edge_index)
        return {idx2entity[i]: out[i].numpy() for i in idx2entity}
```

### EnhancedGNN（多关系）

```python
class EnhancedGNNEmbedder:
    """多关系异构图神经网络。构建 4 种图类型。"""
    
    def __init__(self, inter_df, book_df, user_df, embed_dim=64, num_layers=2):
        self._build_mapping(inter_df, book_df, user_df)
        self._build_multi_layer_graph(inter_df, book_df, user_df)
        self.model = self._build_enhanced_model()
        self._train()
```

---

## LightGCN 基础实现

### 使用方式

```python
# 构建作者-图书边
author_edges = []
author2idx = {}
for _, row in book_df.iterrows():
    a = row['作者']
    b = row['book_id']
    if a not in author2idx:
        author2idx[a] = len(author2idx)
    author_edges.append([author2idx[a], int(b)])

# 训练 GNN
gnn_author = GNNV5(author_edges, num_nodes=len(author2idx) + book_df.shape[0])
author_embed = gnn_author.get_embed_dict({i: a for a, i in author2idx.items()})
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embed_dim` | 64 | 嵌入维度 |
| `num_layers` | 2 | 传播深度 |
| `epochs` | 20 | 训练轮数 |
| `lr` | 5e-3 | 学习率 |

---

## 增强版 GNN

### 多关系图构建

```python
def _build_multi_layer_graph(self, inter_df, book_df, user_df):
    self.hetero_data = HeteroData()
    
    # 1. 用户-图书交互（主关系）
    user_idx = inter_df['user_id'].map(self.user2idx).values
    book_idx = inter_df['book_id'].map(self.book2idx).values
    self.hetero_data['user', 'borrows', 'book'].edge_index = 
        torch.tensor([user_idx, book_idx], dtype=torch.long)
    
    # 加权边（续借次数 + 时间衰减）
    edge_weights = self._compute_edge_weights(inter_df)
    self.hetero_data['user', 'borrows', 'book'].edge_weight = edge_weights
    
    # 2. 用户-用户相似度图
    user_user_edges = self._build_user_similarity_graph(inter_df)
    if user_user_edges is not None:
        self.hetero_data['user', 'similar_to', 'user'].edge_index = user_user_edges
    
    # 3. 图书-图书相似度图
    book_book_edges = self._build_book_similarity_graph(book_df)
    if book_book_edges is not None:
        self.hetero_data['book', 'similar_to', 'book'].edge_index = book_book_edges
    
    # 4. 用户-院系图
    user_dept_edges = self._build_user_dept_graph(user_df)
    if user_dept_edges is not None:
        self.hetero_data['user', 'belongs_to', 'dept'].edge_index = user_dept_edges
```

### 边权重计算

```python
def _compute_edge_weights(self, inter_df):
    """结合续借次数和时间衰减计算边权重。"""
    weights = []
    latest_time = inter_df['借阅时间'].max()
    
    for _, row in inter_df.iterrows():
        renew_weight = 1.0 + 0.5 * row['续借次数']
        days_diff = (latest_time - row['借阅时间']).days
        time_weight = np.exp(-days_diff / 90)  # 90天半衰期
        weights.append(renew_weight * time_weight)
    
    return torch.tensor(weights, dtype=torch.float)
```

### 用户相似度图（基于余弦相似度）

```python
def _build_user_similarity_graph(self, inter_df, threshold=0.3, top_k=10):
    # 构建用户-图书交互矩阵
    user_book_matrix = np.zeros((num_users, num_books))
    for _, row in inter_df.iterrows():
        u_idx = self.user2idx[row['user_id']]
        b_idx = self.book2idx[row['book_id']]
        user_book_matrix[u_idx, b_idx] = 1
    
    # 计算余弦相似度
    user_similarity = cosine_similarity(user_book_matrix)
    
    # 保留每个用户的 Top-K 相似用户
    edges = []
    for i in range(num_users):
        similarities = user_similarity[i]
        similarities[i] = 0  # 排除自己
        top_indices = np.argsort(similarities)[-top_k:]
        for j in top_indices:
            if similarities[j] > threshold:
                edges.append([i, j])
    
    return torch.tensor(edges, dtype=torch.long).t()
```

### 图书相似度图（基于 TF-IDF）

```python
def _build_book_similarity_graph(self, book_df, threshold=0.4, top_k=15):
    # 构建图书特征：分类 + 标题关键词
    book_features = []
    for book_id, idx in self.book2idx.items():
        book_info = book_df[book_df['book_id'] == book_id].iloc[0]
        feature_str = f"{book_info['二级分类']} {book_info.get('分词题名', '')}"
        book_features.append(feature_str)
    
    # TF-IDF 向量化
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(book_features)
    book_similarity = cosine_similarity(tfidf_matrix)
    
    # 保留 Top-K 相似图书
    edges = []
    for i, idx_i in enumerate(book_ids):
        top_indices = np.argsort(book_similarity[i])[-top_k:]
        for j in top_indices:
            if j != i and book_similarity[i, j] > threshold:
                edges.append([idx_i, book_ids[j]])
    
    return torch.tensor(edges, dtype=torch.long).t()
```

---

## BPR 损失函数

### 贝叶斯个性化排序

```python
def _compute_hetero_loss(self, out):
    """BPR 损失：最大化正样本得分 - 负样本得分。"""
    
    # 正样本：实际存在的交互
    pos_edge_index = self.hetero_data['user', 'borrows', 'book'].edge_index
    pos_users = out['user'][pos_edge_index[0]]
    pos_books = out['book'][pos_edge_index[1]]
    pos_scores = (pos_users * pos_books).sum(dim=1)
    
    # 负采样：随机选择未交互的图书
    num_pos = pos_edge_index.shape[1]
    neg_books = torch.randint(0, len(self.book2idx), (num_pos,))
    neg_scores = (pos_users * out['book'][neg_books]).sum(dim=1)
    
    # BPR 损失
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    return loss
```

### 损失公式

```
L_BPR = -sum(log(sigma(e_u^T * e_i - e_u^T * e_j))) / |D|
```

---

## 核心代码解析

### MultiRelationGNN 模型架构

```python
class MultiRelationGNN(nn.Module):
    def __init__(self, num_users, num_books, num_depts, embed_dim=64, num_layers=2):
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.book_embed = nn.Embedding(num_books, embed_dim)
        self.dept_embed = nn.Embedding(num_depts, embed_dim // 2)
        
        self.borrow_layers = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.user_sim_layers = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.book_sim_layers = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, hetero_data):
        user_x = self.user_embed.weight
        book_x = self.book_embed.weight
        user_final = user_x.clone()
        book_final = book_x.clone()
        
        # 用户-图书交互传播
        for layer in self.borrow_layers:
            user_borrow_out = layer(user_final, edge_index.flip([0]))
            book_borrow_out = layer(book_final, edge_index)
            user_final = user_final + self.dropout(user_borrow_out)
            book_final = book_final + self.dropout(book_borrow_out)
        
        # 用户-用户相似度传播
        for layer in self.user_sim_layers:
            user_sim_out = layer(user_final, user_edge_index)
            user_final = user_final + self.dropout(user_sim_out)
        
        # 图书-图书相似度传播
        for layer in self.book_sim_layers:
            book_sim_out = layer(book_final, book_edge_index)
            book_final = book_final + self.dropout(book_sim_out)
        
        return {'user': user_final, 'book': book_final}
```

---

## 性能分析

### 优势

1. **多关系建模**：捕获用户-用户、图书-图书、用户-院系多种相似信号
2. **加权边**：融入续借次数和时间信息
3. **BPR 损失**：直接优化排序目标
4. **LightGCN**：简单有效，无过度平滑

### 劣势

1. **内存占用**：多关系图需要更多内存
2. **随机负样本**：简单随机负采样可能不够难
3. **无内容特征**：纯协同过滤，没有利用图书文本内容

### 使用建议

- **GNNV5**：需要快速简单的作者/出版社/主题嵌入
- **EnhancedGNN**：需要丰富的多关系图信号

---

## 相关文件

- `src/gnn_models/gnn_feature.py` — 增强多关系 GNN
- `src/gnn_models/gnn_v5.py` — 基础 LightGCN 包装器
- `src/xgboost_gnn_pipeline/v8.py` — 使用 GNN 嵌入作为特征
