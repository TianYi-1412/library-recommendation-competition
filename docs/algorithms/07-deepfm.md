# 算法七：DeepFM（深度因子分解机）

> **文件**：`src/deepfm/train_deepfm.py`
>
> **类型**：深度学习 + 因子分解机
>
> **框架**：PyTorch

---

## 目录

- [算法原理](#算法原理)
- [模型架构](#模型架构)
- [代码结构分析](#代码结构分析)
- [FM 组件](#fm-组件)
- [Deep 组件](#deep-组件)
- [核心代码解析](#核心代码解析)
- [性能分析](#性能分析)

---

## 算法原理

### 什么是 DeepFM？

DeepFM 将**因子分解机（FM）**和**深度神经网络（DNN）**融合为统一的单一模型：

- **FM 组件**：通过低维嵌入学习特征两两交互。对稀疏类别特征非常高效。
- **Deep 组件**：通过深度神经网络学习高阶非线性特征交互。捕获复杂模式。

### 为什么用 DeepFM 做推荐？

| 组件 | 处理 | 示例 |
|------|------|------|
| FM | 低阶（两两）交互 | 用户 ID x 图书 ID 亲和度 |
| Deep | 高阶（复杂）交互 | 用户 ID x 图书 ID x 分类 x 时间 |
| 结合 | 全阶交互 | 两全其美 |

### 数学公式

**FM 输出**：
```
y_FM = w_0 + sum(w_i * x_i) + sum(v_i^T * v_j * x_i * x_j)
```

- `w_0`：全局偏置
- `w_i`：线性权重
- `v_i`：特征 i 的隐向量
- `x_i * x_j`：两两交互

**Deep 输出**：
```
y_Deep = MLP(concatenate([v_1, v_2, ..., v_n]))
```

**最终输出**：
```
y = sigmoid(y_FM + y_Deep)
```

---

## 模型架构

```
输入：稀疏特征（user_id、book_id、history_books、categories）
      稠密特征（renewal_mean）

步骤 1：嵌入层
  +-- 对每个稀疏特征：
  +--     +-- Embedding(field_size, embed_dim=64)
  +-- 输出：[B, 64] 张量列表

步骤 2：FM 组件
  +-- 拼接嵌入：[B, num_fields*64]
  +-- FM 交互：0.5 * (sum^2 - sum_of_squares)
  +-- 输出：每样本标量

步骤 3：Deep 组件（MLP）
  +-- 拼接嵌入：[B, num_fields*64]
  +-- 线性层：num_fields*64 -> 256
  +-- ReLU
  +-- 拼接稠密特征
  +-- 线性层：256+len(dense) -> 128
  +-- ReLU
  +-- 线性层：128 -> 1
  +-- 输出：每样本标量

步骤 4：融合
  +-- Final = sigmoid(Deep_output + FM_output)
```

---

## 代码结构分析

### DeepFM 模型

```python
class DeepFM(nn.Module):
    """
    用于图书推荐的深度因子分解机。
    
    架构：
    - 每个稀疏字段一个嵌入
    - FM 组件：两两交互
    - Deep 组件：MLP
    - 输出：sigmoid(Deep + FM)
    
    参数：
        field_dims：字段大小列表 [num_users, num_books, ...]
        embed_dim：嵌入维度（默认 64）
        mlp_dims：MLP 隐藏层大小（默认 (256, 128)）
    """
    
    def __init__(self, field_dims, embed_dim=64, mlp_dims=(256, 128)):
        super(DeepFM, self).__init__()
        
        # 每个稀疏字段一个嵌入
        self.embedding = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])
        
        # Deep 组件：MLP
        self.fc = nn.Linear(len(field_dims) * embed_dim, mlp_dims[0])
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dims[0] + len(dense_cols), mlp_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[1], 1)
        )
    
    def forward(self, x_sparse, x_dense):
        # x_sparse: [B, num_fields]
        # x_dense: [B, num_dense]
        
        # 步骤 1：查嵌入
        embeds = [self.embedding[i](x_sparse[:, i]) 
                  for i in range(len(field_dims))]
        
        # 步骤 2：FM 组件
        embed_vec = torch.cat(embeds, dim=1)
        embed_3d = embed_vec.view(-1, len(field_dims), embed_dim)
        
        sum_embed = embed_3d.sum(dim=1)           # [B, K]
        sum_sq_embed = (embed_3d ** 2).sum(dim=1) # [B, K]
        fm_part = 0.5 * (sum_embed ** 2 - sum_sq_embed).sum(dim=1, keepdim=True)
        
        # 步骤 3：Deep 组件
        deep_part = self.fc(embed_vec)           # [B, 256]
        deep_part = torch.cat([deep_part, x_dense], dim=1)
        deep_part = self.mlp(deep_part)          # [B, 1]
        
        # 步骤 4：FM + Deep 结合
        out = torch.sigmoid(deep_part + fm_part).squeeze()
        return out
```

### 特征定义

```python
# 稀疏（类别）特征
sparse_cols = [
    'user_id',
    'cand_book_id',
    'last1_book_id', 'last2_book_id', 'last3_book_id',
    'last4_book_id', 'last5_book_id',
    'last1_cate2', 'last2_cate2', 'last3_cate2',
    'last4_cate2', 'last5_cate2',
    'cand_cate2'
]

# 稠密（连续）特征
dense_cols = ['cand_renew_mean']

# 用 LabelEncoder 编码稀疏特征
encoders = {}
for col in sparse_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# 嵌入层字段维度
field_dims = [X_sparse_train[:, i].max() + 1 
              for i in range(X_sparse_train.shape[1])]
```

---

## FM 组件

### FM 公式实现

```python
# FM：建模两两交互
# 对 n 个特征，FM 捕获所有 (i,j) 对

embed_3d = embed_vec.view(batch_size, num_fields, embed_dim)

# 方法：0.5 * (sum(v)^2 - sum(v^2))
sum_embed = embed_3d.sum(dim=1)           # [B, K]
sum_sq_embed = (embed_3d ** 2).sum(dim=1) # [B, K]

fm_output = 0.5 * (sum_embed ** 2 - sum_sq_embed)  # [B, K]
fm_output = fm_output.sum(dim=1, keepdim=True)      # [B, 1]
```

### 为什么这样计算

```
(sum(v))^2 = (v1 + v2 + ... + vf)^2 
           = v1^2 + v2^2 + ... + vf^2 + 2*v1*v2 + 2*v1*v3 + ...

sum(v^2) = v1^2 + v2^2 + ... + vf^2

(sum(v))^2 - sum(v^2) = 2*(v1*v2 + v1*v3 + ... + vf-1*vf)

0.5 * (...) = 所有两两点积之和
```

这样以 O(n*K) 时间计算所有两两交互，而非 O(n^2)。

---

## Deep 组件

### MLP 架构

```python
# 输入：拼接的嵌入 [B, F*K]
self.fc = nn.Linear(len(field_dims) * embed_dim, 256)

# 带稠密特征拼接的 MLP
self.mlp = nn.Sequential(
    nn.Linear(256 + len(dense_cols), 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
```

### 训练循环

```python
# 超参数
embed_dim = 64
batch_size = 4096
epochs = 5
learning_rate = 1e-3

# 模型和优化器
model = DeepFM(field_dims, embed_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# 训练
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_sparse, x_dense, y in train_loader:
        x_sparse = x_sparse.to(device)
        x_dense = x_dense.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        preds = model(x_sparse, x_dense)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{epochs} loss: {total_loss/len(train_loader):.4f}')
```

---

## 核心代码解析

### 1. 数据准备

```python
# 读取并合并训练/验证
train = pd.read_csv('../output+hj/train_rank2.csv')
valid = pd.read_csv('../output+hj/valid_rank2.csv')
df = pd.concat([train, valid], ignore_index=True)

# 定义特征列
sparse_cols = ['user_id', 'cand_book_id', 
               'last1_book_id', ..., 'last5_book_id',
               'last1_cate2', ..., 'last5_cate2',
               'cand_cate2']
dense_cols = ['cand_renew_mean']

# 稀疏特征标签编码
encoders = {}
for col in sparse_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# 拆分回训练/验证
train = df.iloc[:len(train)]
valid = df.iloc[len(train):]
```

### 2. 模型训练

```python
# 准备张量
X_sparse_train = train[sparse_cols].values  # [N, num_sparse]
X_dense_train = train[dense_cols].values    # [N, num_dense]
y_train = train['label'].values             # [N]

# 计算字段维度
field_dims = [X_sparse_train[:, i].max() + 1 
              for i in range(X_sparse_train.shape[1])]

# DataLoader
class DeepFMDataset(Dataset):
    def __getitem__(self, idx):
        return (self.X_sparse[idx], self.X_dense[idx], self.y[idx])

train_loader = DataLoader(DeepFMDataset(X_sparse_train, X_dense_train, y_train),
                          batch_size=4096, shuffle=True)
```

### 3. 模型保存/加载

```python
# 保存模型
torch.save(model.state_dict(), '../output+hj/deepfm_pytorch.pt')

# 保存元数据（编码器、特征信息）
joblib.dump({
    'encoders': encoders,
    'sparse_cols': sparse_cols,
    'dense_cols': dense_cols,
    'field_dims': field_dims
}, '../output+hj/deepfm_pytorch_meta.pkl')
```

---

## 性能分析

### 优势

1. **FM + Deep**：结合低阶和高阶特征交互
2. **统一模型**：联合训练，无需手工特征工程
3. **处理稀疏特征**：嵌入处理高基数类别特征
4. **推理快**：单次前向传播

### 劣势

1. **仍需特征设计**：需要手工特征选择
2. **无序列建模**：不捕获时序模式
3. **仅限表格数据**：不直接使用文本内容

### 适用场景

- **最佳场景**：类别特征丰富的场景
- **好基线**：通常优于纯 FM 或纯 MLP

---

## 相关文件

- `src/deepfm/train_deepfm.py` — 完整实现
- `src/recbole_framework/configs.py` — RecBole DeepFM 配置
