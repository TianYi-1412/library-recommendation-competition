# 算法八：多模态双塔模型

> **文件**：`src/multimodal_dualtower/user_interesting.py`、`data_loader.py`、`model.py`、`result.py`
>
> **类型**：多模态深度学习（文本 + ID）
>
> **核心特色**：M3E-base 中文文本嵌入，用于图书内容理解

---

## 目录

- [算法原理](#算法原理)
- [系统架构](#系统架构)
- [代码结构分析](#代码结构分析)
- [M3E 文本嵌入](#m3e-文本嵌入)
- [用户兴趣建模](#用户兴趣建模)
- [双塔架构](#双塔架构)
- [特征工程](#特征工程)
- [核心代码解析](#核心代码解析)
- [性能分析](#性能分析)

---

## 算法原理

### 什么是双塔模型？

双塔模型由两个并行的神经网络组成：
- **用户塔**：将用户特征（ID、 demographics、历史）编码为向量
- **物品塔**：将物品特征（ID、文本、分类）编码为向量

**打分**：cosine_similarity(用户向量, 物品向量)

### 多模态方面

本实现添加了**文本理解**能力，使用 M3E-base（中文文本嵌入模型）：
- 图书标题、作者、分类被编码为语义向量
- 这些与基于 ID 的嵌入结合，形成更丰富的表示

### 为什么用多模态？

| 模态 | 捕获 | 示例 |
|------|------|------|
| 基于 ID | 协同模式 | 借过 A 的用户也借过 B |
| 基于文本 | 语义内容 | "机器学习"和"深度学习"相似 |
| 结合 | 两者 | 更好的冷启动处理 |

---

## 系统架构

```
输入：用户 + 图书 + 交互数据

步骤 1：数据预处理（data_loader.py）
  +-- 加载和清洗数据
  +-- 编码 ID（用户、图书）
  +-- 处理类别特征（性别、年级、院系、类型）
  +-- 生成正/负样本
  +-- 保存处理后的数据

步骤 2：文本嵌入（user_interesting.py）
  +-- 加载 M3E-base 模型
  +-- 提取图书文本（标题 + 作者 + 分类）
  +-- 批量编码（batch_size=64）
  +-- PCA 降维（768 -> 128）
  +-- 保存文本嵌入

步骤 3：用户兴趣建模（user_interesting.py）
  +-- 构建用户历史序列
  +-- 提取用户分类偏好
  +-- 保存用户兴趣特征

步骤 4：模型训练（model.py）
  +-- 双塔架构
  +-- 多种损失函数（BPR、对比损失）
  +-- 保存训练好的模型

步骤 5：评估与推理（result.py）
  +-- 生成候选排序
  +-- 计算 F1 得分
  +-- 创建提交文件
```

---

## 代码结构分析

### 模块依赖

```
data_loader.py
  +-- load_and_clean_data()      # 加载和预处理
  +-- build_encoders()           # ID 编码
  +-- generate_pos_neg_samples() # 样本生成
  +-- save_processed_data()      # 保存为 pickle

user_interesting.py
  +-- extract_book_text_embedding()  # M3E 编码
  +-- build_user_history_sequence()  # 用户序列
  +-- extract_user_category_preference()  # 分类偏好

model.py
  +-- 双塔模型架构
  +-- 训练循环
  +-- 嵌入层

result.py
  +-- get_features()             # 特征提取
  +-- generate_recommendations() # 推理
  +-- f1_at_k()                  # 评估
```

---

## M3E 文本嵌入

### M3E-base 模型

M3E（Moka Massive Mixed Embedding）是中文文本嵌入模型，将文本映射到 768 维向量。

```python
from sentence_transformers import SentenceTransformer

# 加载 M3E 模型
m3e_model = SentenceTransformer('moka-ai/m3e-base')
# 或从本地加载
m3e_model = SentenceTransformer(LOCAL_M3E_PATH)

# 如有 GPU
if torch.cuda.is_available():
    m3e_model = m3e_model.to('cuda')
```

### 图书文本编码

```python
def extract_book_text_embedding(books, item_encoder):
    """
    为所有图书提取文本嵌入。
    
    流程：
    1. 拼接标题 + 作者 + 分类作为文本
    2. M3E 批量编码（batch_size=64）
    3. PCA：768维 -> 128维
    """
    
    # 1. 构建图书文本字典
    book_text_dict = books.set_index('book_id').apply(
        lambda r: f"{r['题名']} {r['作者']} {r['一级分类']}", axis=1
    ).to_dict()
    
    # 2. M3E 批量编码
    batch_size = 64
    book_ids = [str(bid) for bid in item_encoder.keys()]
    all_texts = [book_text_dict.get(bid, "未知") for bid in book_ids]
    
    item_vec_list = []
    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            embeddings = m3e_model.encode(
                batch, convert_to_tensor=True,
                normalize_embeddings=True, device=m3e_model.device
            )
            item_vec_list.append(embeddings.cpu().numpy())
    
    item_vec_mat = np.concatenate(item_vec_list, axis=0)
    # Shape: [num_books, 768]
    
    # 3. PCA 降维
    EMB_DIM = 128
    pca = PCA(n_components=EMB_DIM, random_state=42)
    item_text_emb = pca.fit_transform(item_vec_mat)
    # Shape: [num_books, 128]
    
    return item_text_emb
```

### 为什么用 PCA？

| 维度 | 优点 | 缺点 |
|------|------|------|
| 768（原始） | 信息完整 | 内存大、计算慢 |
| 128（PCA） | 紧凑、快速 | 有信息损失 |

PCA 保留约 90% 方差，同时减少 6 倍大小。

---

## 用户兴趣建模

### 用户历史序列

```python
def build_user_history_sequence(train_data, num_users, user_encoder, 
                                 item_encoder, HIST_MAX=50):
    """
    构建用户交互历史序列。
    
    返回：
    - user_seq_dict: {user_idx: tensor([pad, pad, ..., book1, book2])}
    - PAD: 填充索引
    """
    PAD = num_users  # 避免与物品索引冲突
    
    # 收集用户交互
    user_hist = defaultdict(list)
    for _, row in train_data.iterrows():
        u_enc = user_encoder[str(row['user_id'])]
        i_enc = item_encoder[str(row['book_id'])]
        user_hist[u_enc].append(i_enc)
    
    # 填充/截断序列
    user_seq_dict = {}
    for u_enc in range(num_users):
        hist = user_hist.get(u_enc, [])
        hist_truncated = hist[-HIST_MAX:] if len(hist) > HIST_MAX else hist
        hist_padded = [PAD] * (HIST_MAX - len(hist_truncated)) + hist_truncated
        user_seq_dict[u_enc] = torch.tensor(hist_padded, dtype=torch.long)
    
    return user_seq_dict, PAD
```

### 用户分类偏好

```python
def extract_user_category_preference(inter_train, user_encoder, uid2raw):
    """
    提取用户分类偏好分布。
    
    返回：{user_idx: {category: proportion}}
    """
    user_cat_dist = {}
    
    for u_enc in range(len(user_encoder)):
        u_raw = uid2raw[u_enc]
        user_inter = inter_train[inter_train['user_id'] == u_raw]
        
        if len(user_inter) == 0:
            user_cat_dist[u_enc] = {"未知分类": 1.0}
            continue
        
        # 分类分布
        cat_count = user_inter['一级分类'].value_counts(normalize=True)
        user_cat_dist[u_enc] = cat_count.to_dict()
    
    return user_cat_dist
```

---

## 双塔架构

### 模型设计

```
用户塔：
  输入：user_id（稀疏）
  嵌入：user_id -> [128]
  可选：demographics -> 拼接
  输出：user_vector [128]

物品塔：
  输入：book_id（稀疏）+ text_embedding [128]
  嵌入：book_id -> [128]
  拼接：[book_emb || text_emb] -> [256]
  线性层：256 -> [128]
  输出：item_vector [128]

打分：
  similarity = cosine(user_vector, item_vector)
  或：dot_product(user_vector, item_vector)
```

### 训练目标

```python
# BPR 损失（贝叶斯个性化排序）
loss = -log(sigmoid(score(positive) - score(negative)))

# 对比损失
loss = -log(exp(score(positive)/temperature) / 
             sum(exp(score(negative_i)/temperature)))
```

---

## 特征工程

### 特征向量（result.py）

```python
def get_features(u_idx, raw_book_id, model_dict):
    """
    提取 6*EMB_DIM + 3 + 4 维特征向量。
    
    组成：
    - u_vec: 用户嵌入 [EMB_DIM]
    - i_vec: 物品嵌入 [EMB_DIM]
    - text_vec: 文本嵌入 [EMB_DIM]
    - u_i_dot: 逐元素积 [EMB_DIM]
    - u_text_dot: 逐元素积 [EMB_DIM]
    - i_text_dot: 逐元素积 [EMB_DIM]
    - book_pop: log 热度 [1]
    - cat_sim: 分类相似度 [1]
    - same_author: 二元 [1]
    - user_features: 性别、年级、院系、类型 [4]
    """
    u_vec = user_vecs[u_idx]              # [EMB_DIM]
    i_vec = item_vecs[i_idx]              # [EMB_DIM]
    text_vec = item_text_emb[i_idx]       # [EMB_DIM]
    
    # 逐元素积（特征交互）
    u_i_dot = u_vec * i_vec               # [EMB_DIM]
    u_text_dot = u_vec * text_vec         # [EMB_DIM]
    i_text_dot = i_vec * text_vec         # [EMB_DIM]
    
    # 手工特征
    book_pop = np.log1p(item_pop.get(raw_book_id, 1e-5))  # [1]
    cat_sim = category_similarity                         # [1]
    same_author = 1.0 if author_match else 0.0            # [1]
    user_feat = [gender, grade, dept, type]               # [4]
    
    # 拼接：[6*EMB_DIM + 3 + 4]
    feat = np.concatenate([
        u_vec, i_vec, text_vec, u_i_dot, u_text_dot, i_text_dot,
        [book_pop, cat_sim, same_author], user_feat
    ])
    
    # 标准化
    feat = (feat - mean) / (std + 1e-8)
    return feat
```

---

## 核心代码解析

### 1. 数据加载与编码

```python
# data_loader.py

def load_and_clean_data():
    # 读取数据
    inter = pd.read_csv(f'{PATH}/pre_inter.csv')
    books = pd.read_csv(f'{PATH}/book_data.csv')
    users = pd.read_csv(f'{PATH}/pre_user.csv')
    
    # 清洗和预处理
    inter['book_id'] = inter['book_id'].astype(str).str.strip()
    books['book_id'] = books['book_id'].astype(str).str.strip()
    inter['借阅时间'] = pd.to_datetime(inter['借阅时间'])
    
    # 展开续借（将续借视为独立交互）
    inter_ext = []
    for _, row in inter.iterrows():
        inter_ext.append(row.to_dict())
        for _ in range(row['续借次数']):
            new_entry = row.copy()
            new_entry['is_renewal'] = True
            inter_ext.append(new_entry)
    
    # 编码类别特征
    gender_encoder = {g: i for i, g in enumerate(users['性别'].unique())}
    grade_encoder = {g: i for i, g in enumerate(users['年级'].unique())}
    dept_encoder = {d: i for i, d in enumerate(users['DEPT'].unique())}
    type_encoder = {t: i for i, t in enumerate(users['类型'].unique())}
    
    return inter_train, books, users, encoders...

def build_encoders(inter_train, books, target_users, all_book_ids):
    # 用户编码
    user_encoder = {u: i for i, u in enumerate(sorted(target_users))}
    uid2raw = {i: u for u, i in user_encoder.items()}
    
    # 物品编码
    item_encoder = {b: i for i, b in enumerate(sorted(all_book_ids))}
    idx2book = {i: b for b, i in item_encoder.items()}
    
    return user_encoder, item_encoder, uid2raw, idx2book, num_users, num_items

def generate_pos_neg_samples(inter_train, user_encoder, item_encoder, idx2book, NEG_K=4):
    # 正样本
    pos_samples = [(user_encoder[u], item_encoder[b], 1) 
                   for _, u, b in inter_train[['user_id', 'book_id']].itertuples()]
    
    # 跟踪每个用户见过的物品
    user_seen = defaultdict(set)
    for u, i, _ in pos_samples:
        user_seen[u].add(i)
    
    # 负采样（逆频率加权）
    train_data = pos_samples.copy()
    for u in user_seen:
        candidate_items = [i for i in range(num_items) if i not in user_seen[u]]
        probs = inv_freq[candidate_items]
        probs /= probs.sum()
        neg_items = np.random.choice(candidate_items, size=min(NEG_K, len(candidate_items)),
                                     p=probs, replace=False)
        train_data.extend([(u, j, 0) for j in neg_items])
    
    random.shuffle(train_data)
    return train_data
```

### 2. 文本嵌入流水线

```python
# user_interesting.py

def extract_book_text_embedding(books, item_encoder):
    EMB_DIM = 128
    LOCAL_M3E_PATH = "m3e-base"  # 或下载：moka-ai/m3e-base
    
    # 加载 M3E 模型
    m3e_model = SentenceTransformer(LOCAL_M3E_PATH)
    if torch.cuda.is_available():
        m3e_model = m3e_model.to('cuda')
    
    # 构建图书文本
    book_text_dict = books.set_index('book_id').apply(
        lambda r: f"{r['题名']} {r['作者']} {r['一级分类']}", axis=1
    ).to_dict()
    
    # 批量编码
    book_ids = [str(bid) for bid in item_encoder.keys()]
    all_texts = [book_text_dict.get(bid, "未知") for bid in book_ids]
    
    item_vec_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_texts), 64)):
            batch = all_texts[i:i + 64]
            embeddings = m3e_model.encode(batch, convert_to_tensor=True, 
                                           normalize_embeddings=True)
            item_vec_list.append(embeddings.cpu().numpy())
    
    item_vec_mat = np.concatenate(item_vec_list, axis=0)
    
    # PCA：768 -> 128
    pca = PCA(n_components=EMB_DIM, random_state=42)
    item_text_emb = pca.fit_transform(item_vec_mat)
    
    # 清理
    del m3e_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return item_text_emb
```

### 3. 推荐生成

```python
# result.py

def generate_recommendations(model_dict, valid_uid2last=None, is_submit=False):
    ranker = model_dict["ranker"]
    rec_candidates = model_dict["rec_candidates"]
    seen_by_user = model_dict["seen_by_user"]
    
    pred_dict = {}
    for raw_uid in tqdm(target_users):
        u_idx = raw2uid[raw_uid]
        seen = seen_by_user.get(raw_uid, set())
        
        # 获取候选（排除已见图书）
        cands = [b for b in rec_candidates[u_idx] if b not in seen]
        
        # 不足时补足
        if len(cands) < TOP_K:
            all_books = set(idx2book.values())
            unseen = list(all_books - seen - set(cands))
            cands += list(np.random.choice(unseen, min(TOP_K - len(cands), len(unseen))))
        
        # 提取特征并打分
        X = np.array([get_features(u_idx, b, model_dict) for b in cands])
        scores = ranker.predict(X)
        
        # 选 Top-K
        top_books = [cands[i] for i in np.argsort(-scores)[:TOP_K]]
        pred_dict[raw_uid] = top_books
    
    return pred_dict
```

---

## 性能分析

### 优势

1. **文本理解**：M3E 捕获图书间的语义相似度
2. **多模态融合**：结合基于 ID 和基于内容的信号
3. **丰富特征**：6*EMB_DIM + 7 个特征捕获多方面信息
4. **冷启动友好**：文本嵌入帮助处理未见过的新图书

### 劣势

1. **M3E 依赖**：需要下载大模型（约 400MB）
2. **内存占用**：存储所有文本嵌入消耗内存
3. **PCA 损失**：降维会损失一些信息
4. **流水线复杂**：多步骤（嵌入 -> PCA -> 模型）

### 设计决策

| 决策 | 理由 |
|------|------|
| M3E-base | 当时最好的开源中文嵌入模型 |
| PCA 768->128 | 6 倍大小缩减，保留约 90% 方差 |
| 逐元素积 | 低成本的特征交互 |
| Log 热度 | 减少热门图书的偏斜 |
| 展开续借 | 将续借视为隐式正信号 |

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `data_loader.py` | 数据加载、编码、采样 |
| `user_interesting.py` | 文本嵌入、用户兴趣建模 |
| `model.py` | 双塔模型定义 |
| `result.py` | 评估和提交生成 |
