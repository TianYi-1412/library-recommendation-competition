# 算法三：轻量 Transformer 模型

> **文件**：`src/transformer_models/transformer_v8.py`、`src/transformer_models/graph_utils_v8.py`
>
> **类型**：Transformer（自注意力机制）
>
> **集成**：作为 v8.py XGBoost 流水线中的打分模型

---

## 目录

- [算法原理](#算法原理)
- [模型架构](#模型架构)
- [代码结构分析](#代码结构分析)
- [二分图工具类](#二分图工具类)
- [核心代码解析](#核心代码解析)
- [与 v8 流水线集成](#与-v8-流水线集成)
- [性能分析](#性能分析)

---

## 算法原理

### 为什么用 Transformer 做推荐？

传统协同过滤只捕获静态的用户-物品关系。Transformer 通过自注意力机制建模**序列模式**：

1. **自注意力**：每个节点（用户/图书）关注序列中的所有其他节点
2. **位置编码**：通过序列顺序隐式编码
3. **多头注意力**：捕获不同类型的关系

### 在本项目中的应用

不同于处理文本序列，这里的 Transformer 处理**二分图的节点序列**：

```
输入：[user_id, book_id] 作为节点索引
嵌入层：node_index -> embedding_vector
Transformer 编码器：在所有节点间注意力计算
输出：二元得分（是否会交互）
```

### 数学公式

**自注意力**：
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

其中 Q（查询）、K（键）、V（值）是输入嵌入的线性投影，`d_k` 是键维度（embed_dim / nhead）。

**多头注意力**：
```
MultiHead(X) = Concat(head_1, ..., head_h) * W^O
```

**最终输出**：
```
output = mean(transformer_output, dim=1)   # 平均池化
score = sigmoid(Linear(output))             # 二分类
```

---

## 模型架构

```
输入：node_indices [batch_size, seq_len=2]  # [user_id, book_id]

步骤 1：嵌入层
  node_indices -> nn.Embedding(num_nodes, embed_dim)
  输出：[batch_size, seq_len, embed_dim]

步骤 2：Transformer 编码器
  输入：[batch_size, seq_len, embed_dim]
  |
  +-- TransformerEncoderLayer x num_layers
  |     +-- MultiHeadAttention (nhead=4)
  |     +-- FeedForward (embed_dim -> embed_dim)
  |     +-- LayerNorm + Dropout
  |
  输出：[batch_size, seq_len, embed_dim]

步骤 3：池化 + 分类
  平均池化：[batch_size, seq_len, embed_dim] -> [batch_size, embed_dim]
  线性层：[batch_size, embed_dim] -> [batch_size, 1]
  Sigmoid：概率得分

输出：[batch_size]  # 交互概率
```

---

## 代码结构分析

### LightTransformer

```python
class LightTransformer(nn.Module):
    """
    用于二分图节点打分的轻量 Transformer。
    
    架构：
    - 嵌入层：num_nodes x embed_dim
    - TransformerEncoder：2 层，4 头
    - 序列平均池化
    - 线性投影到 1 维得分
    - Sigmoid 激活
    
    参数：
        num_nodes：节点总数（用户数 + 图书数）
        embed_dim：嵌入维度（默认 64）
        num_layers：Transformer 层数（默认 2）
    """
    
    def __init__(self, num_nodes, embed_dim=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        emb = self.embedding(x)              # [B, N, 64]
        out = self.transformer(emb)          # [B, N, 64]
        out = out.mean(dim=1)                # [B, 64]
        return torch.sigmoid(self.fc(out).squeeze(1))  # [B]
```

### BipartiteGraph（二分图工具类）

```python
class BipartiteGraph:
    """
    带 PCA 嵌入的二分图。
    
    从用户-图书边构建邻接矩阵，
    然后应用 PCA 降维。
    
    参数：
        edges：(user_idx, book_idx) 元组列表
        num_users：用户节点数
        num_books：图书节点数
        embed_dim：目标嵌入维度（默认 64）
    """
    
    def __init__(self, edges, num_users, num_books, embed_dim=64):
        self.edges = np.array(edges)
        self.num_users = num_users
        self.num_books = num_books
        self.embed_dim = embed_dim
        self._build_adj()
    
    def _build_adj(self):
        # 构建用户-图书邻接矩阵
        adj = np.zeros((self.num_users, self.num_books))
        for u, b in self.edges:
            adj[u, b] = 1
        
        # PCA 嵌入（用户侧）
        self.user_embed = PCA(n_components=self.embed_dim).fit_transform(adj)
        # [num_users, 64]
        
        # PCA 嵌入（图书侧）
        self.book_embed = PCA(n_components=self.embed_dim).fit_transform(adj.T)
        # [num_books, 64]
```

---

## 二分图工具类

### 二分图构建

```python
# 在 v8.py 中
users = sorted(self.inter_df['user_id'].unique())
books = sorted(self.inter_df['book_id'].unique())
self.user2idx = {u: i for i, u in enumerate(users)}
self.book2idx = {b: i for i, b in enumerate(books)}

# 构建边
self.edges = [(self.user2idx[u], self.book2idx[b]) 
              for u, b in zip(inter_df['user_id'], inter_df['book_id'])]

# PCA 嵌入
self.graph = BipartiteGraph(self.edges, len(users), len(books), embed_dim=64)
```

---

## 核心代码解析

### 1. Transformer 训练

```python
def _build_transformer_model(self, features, labels):
    # 从交互记录创建节点序列 [user_id, book_id]
    user_ids = [self.user2idx.get(self.inter_df.iloc[i]['user_id'], 0) 
                for i in range(len(self.inter_df))]
    book_ids = [self.book2idx.get(self.inter_df.iloc[i]['book_id'], 0) 
                for i in range(len(self.inter_df))]
    node_seq = np.column_stack([user_ids, book_ids])  # [N, 2]
    
    # 初始化模型
    num_nodes = len(self.user2idx) + len(self.book2idx)
    self.transformer = LightTransformer(num_nodes, embed_dim=64, num_layers=2).to(device)
    
    # 训练
    optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-3)
    seq_tensor = torch.tensor(node_seq, dtype=torch.long).to(device)
    y_tensor = torch.tensor([1] * len(node_seq), dtype=torch.float32).to(device)
    
    for epoch in range(10):
        optimizer.zero_grad()
        out = self.transformer(seq_tensor).squeeze()   # [N]
        loss = nn.BCELoss()(out, y_tensor)              # 全部正样本
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Transformer epoch {epoch} | loss {loss.item():.4f}")
```

### 2. 候选集打分

```python
def recommend_for_user(self, user_id):
    # 多路召回
    candidates = self._multi_channel_recall(user_id)
    candidates = [b for b in candidates if b not in self.user_borrow_history[user_id]]
    
    # 提取 28 维特征
    feats = np.array([self._extract_feature_vector(user_id, b) for b in candidates])
    
    # Transformer 打分
    with torch.no_grad():
        node_seq = np.column_stack([[user_idx] * len(candidates),
                                    [self.book2idx.get(b, 0) for b in candidates]])
        node_tensor = torch.tensor(node_seq, dtype=torch.long).to(device)
        deep_score = self.transformer(node_tensor).cpu().numpy().squeeze()
    
    # 加权融合：60% Transformer + 40% 图相似度
    final_score = (0.6 * deep_score + 
                   0.4 * cosine_similarity([self.user_embed[user_idx]],
                                            self.book_embed[[...]])[0])
    return candidates[np.argmax(final_score)]
```

---

## 与 v8 流水线集成

Transformer 是 v8 系统的组件之一：

```
v8 流水线组件：
  +-- BipartiteGraph (graph_utils_v8.py)   # PCA 嵌入
  +-- LightTransformer (transformer_v8.py)  # 打分模型
  +-- XGBoost / 特征工程                   # 最终排序

数据流：
  原始数据 -> BipartiteGraph -> PCA 嵌入 ----+
                                             +---> 最终得分
  原始数据 -> LightTransformer -> 深度得分 --+
```

---

## 性能分析

### 优势

1. **捕获序列模式**：自注意力建模交互顺序
2. **轻量高效**：仅 2 层 4 头，训练和推理速度快
3. **与图特征互补**：Transformer + PCA 相似度效果强大
4. **架构简单**：易于理解和修改

### 劣势

1. **序列短**：seq_len=2 限制了序列模式捕获
2. **无位置编码**：仅依赖序列顺序
3. **全正样本训练**：只用正样本训练
4. **容量有限**：2 层可能无法捕获复杂模式

### 设计决策

| 决策 | 理由 |
|------|------|
| seq_len=2 | 用户-图书对是基本交互单元 |
| nhead=4 | 头数少，效率高 |
| num_layers=2 | 防止在小数据集上过拟合 |
| embed_dim=64 | 与 GNN 嵌入维度一致 |
| 平均池化 | 简单有效的聚合方式 |
| Sigmoid 输出 | 可解释的概率得分 |

---

## 相关文件

- `src/transformer_models/transformer_v8.py` — Transformer 核心模型
- `src/transformer_models/graph_utils_v8.py` — 二分图 PCA 工具
- `src/xgboost_gnn_pipeline/v8.py` — 集成流水线
