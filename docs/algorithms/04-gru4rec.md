# 算法四：GRU4Rec —— 兴趣场景时序推荐器

> **文件**：`src/gru4rec/GRU4Rec 0.0158.py`
>
> **类型**：RNN 序列模型（基于 GRU）
>
> **核心特色**：兴趣强度 + 学期场景建模
>
> **得分**：F1 = 0.0158（最佳序列模型）

---

## 目录

- [算法原理](#算法原理)
- [系统架构](#系统架构)
- [代码结构分析](#代码结构分析)
- [兴趣强度计算](#兴趣强度计算)
- [学期场景建模](#学期场景建模)
- [核心代码解析](#核心代码解析)
- [性能分析](#性能分析)

---

## 算法原理

### 什么是 GRU4Rec？

GRU4Rec 是一种**基于会话的推荐模型**，使用门控循环单元（GRU）建模用户交互序列。与传统协同过滤不同，它捕获用户行为的**时序顺序**。

### 本实现的创新点

1. **兴趣强度**：每次交互根据借阅时长和重复次数赋予加权得分
2. **学期场景**：根据校历上下文（假期、春/秋学期、期末）调整推荐
3. **时间衰减**：近期交互权重更高，使用指数衰减

### 数学公式

**GRU 更新方程**：
```
r_t = sigmoid(W_r * [h_{t-1}, x_t])      # 重置门
z_t = sigmoid(W_z * [h_{t-1}, x_t])      # 更新门
n_t = tanh(W_n * [r_t * h_{t-1}, x_t])   # 新门
h_t = (1 - z_t) * n_t + z_t * h_{t-1}    # 隐藏状态
```

**兴趣强度**：
```
intensity = min(borrow_duration_normalized + repeat_weight, 1.0)
```

**时间衰减权重**：
```
weight = exp(-days_since_interaction / half_life)
```

---

## 系统架构

```
输入：用户借阅历史（按时间排序）

步骤 1：数据预处理
  +-- 计算每次交互的兴趣强度
  +-- 分配学期场景（0=假期, 1=春季, 2=秋季, 3=期末）
  +-- 计算时间衰减权重
  +-- 过滤出至少有 2 次交互的用户

步骤 2：序列构建
  +-- 固定序列长度 seq_len=3
  +-- 滑动窗口：[book_1, book_2, book_3] -> 预测 book_4
  +-- 短序列补零

步骤 3：模型输入
  +-- 图书嵌入：book_id -> embedding_dim
  +-- 场景嵌入：scene_id -> embedding_dim/2
  +-- 兴趣强度：标量值
  +-- 拼接：[book_emb || scene_emb || intensity]

步骤 4：GRU 处理
  +-- GRU 层：hidden_dim=128
  +-- Dropout：0.25
  +-- 输出：最终隐藏状态

步骤 5：预测
  +-- 线性投影：hidden_dim -> num_books
  +-- Softmax：在所有图书上的概率分布
  +-- 选择概率最高的图书
```

---

## 代码结构分析

### InterestSceneTemporalRecommender（主类）

```python
class InterestSceneTemporalRecommender:
    """
    基于兴趣强度和场景感知的 GRU 序列推荐器。
    
    核心组件：
    - InterestSceneGRU：GRU 核心模型
    - InterestSceneDataset：序列数据加载器
    - 多通道候选生成
    - 时间衰减加权打分
    """
    
    # 数据结构
    +-- user_train_data       # 每个用户的训练记录
    +-- user_val_data         # 验证集：{user_id: last_book_id}
    +-- user_borrow_history   # 每个用户已借的图书集合
    +-- book_info             # 图书元数据
    +-- secondary_cate_books  # 分类 -> 图书映射
    +-- keyword_books         # 关键词 -> 图书映射
    +-- user_book_renew       # 续借次数
    +-- user_book_repeat      # 重复借阅次数
    +-- user_high_int_cates   # 用户的 Top-2 分类
    +-- book_sliding_int      # 滑动窗口兴趣得分
    +-- scene2id              # 场景映射
    
    # 超参数
    +-- seq_len = 3           # 序列长度
    +-- embedding_dim = 64    # 嵌入维度
    +-- hidden_dim = 128      # GRU 隐藏层维度
    +-- time_decay_half_life = 30  # 天数
    +-- stop_words = {'高等', '教程', '基础', '入门', '数学', '原理'}
```

### InterestSceneGRU（模型）

```python
class InterestSceneGRU(nn.Module):
    """
    融合图书、场景和兴趣强度的 GRU 模型。
    
    架构：
    - 用户嵌入：[num_users, 64]
    - 图书嵌入：[num_books, 64]
    - 场景嵌入：[4, 32]（4 种场景类型）
    - GRU：input=64+32+1=97, hidden=128, layers=1
    - FC：128 -> num_books
    """
    
    def __init__(self, user_num, book_num, scene_num=4, 
                 embedding_dim=64, hidden_dim=128, dropout=0.2):
        self.user_emb = nn.Embedding(user_num, embedding_dim)
        self.book_emb = nn.Embedding(book_num, embedding_dim)
        self.scene_emb = nn.Embedding(scene_num, embedding_dim // 2)
        
        self.gru = nn.GRU(
            input_size=embedding_dim + embedding_dim // 2 + 1,  # 97
            hidden_size=hidden_dim,  # 128
            batch_first=True,
            dropout=dropout  # 0.25
        )
        self.fc = nn.Linear(hidden_dim, book_num)
    
    def forward(self, user_ids, seq_books, seq_scenes, seq_intensities):
        book_emb = self.book_emb(seq_books)          # [B, seq_len, 64]
        scene_emb = self.scene_emb(seq_scenes)       # [B, seq_len, 32]
        int_emb = seq_intensities.unsqueeze(-1)      # [B, seq_len, 1]
        
        input_emb = torch.cat([book_emb, scene_emb, int_emb], dim=-1)  # [B, seq_len, 97]
        gru_out, _ = self.gru(input_emb)             # [B, seq_len, 128]
        final_out = gru_out[:, -1, :]                # [B, 128]
        logits = self.fc(final_out)                  # [B, num_books]
        return logits
```

---

## 兴趣强度计算

```python
def _compute_interest_intensity(self):
    """计算每次交互的兴趣强度。"""
    
    # 归一化借阅时长
    scaler_duration = MinMaxScaler()
    inter_df['借阅时长_标准化'] = scaler_duration.fit_transform(inter_df[['借阅时长']])
    
    # 重复借阅权重
    borrow_cnt = inter_df.groupby(['user_id', 'book_id']).size().reset_index(name='重复次数')
    borrow_cnt['重复权重'] = borrow_cnt['重复次数'].apply(
        lambda x: min(0.3 * (x - 1), 0.6)  # 上限 0.6
    )
    
    # 合并计算兴趣强度
    inter_df = inter_df.merge(borrow_cnt, on=['user_id', 'book_id'])
    inter_df['兴趣强度'] = (
        inter_df['借阅时长_标准化'] + inter_df['重复权重']
    ).clip(0, 1)
```

### 兴趣强度组成

| 组件 | 范围 | 说明 |
|------|------|------|
| 借阅时长 | [0, 1] | 借得越久 = 兴趣越高 |
| 重复权重 | [0, 0.6] | 重复借阅表示强烈兴趣 |
| 最终强度 | [0, 1] | 裁剪后的总和 |

---

## 学期场景建模

### 场景映射

```python
# 基于月份的场景映射
scene2id = {
    0: '假期',      # 1月, 2月, 7月, 8月
    1: '春季学期',   # 3月, 4月, 5月
    2: '秋季学期',   # 9月, 10月, 11月
    3: '期末考试'    # 6月, 12月
}

def _get_scene(self, month):
    if month in [1, 2, 7, 8]:
        return 0  # 假期
    elif month in [3, 4, 5]:
        return 1  # 春季学期
    elif month in [9, 10, 11]:
        return 2  # 秋季学期
    else:
        return 3  # 期末考试
```

### 为什么场景很重要？

| 场景 | 借阅模式 |
|------|---------|
| 假期 | 休闲阅读、非学术类 |
| 春季学期 | 课程教材、专业书籍 |
| 秋季学期 | 新课程、参考书 |
| 期末考试 | 复习资料、习题集 |

---

## 核心代码解析

### 1. 数据加载

```python
def load_data(self, inter_path, book_path, user_path):
    # 加载图书数据（含核心关键词）
    self.book_df = pd.read_csv(book_path, 
        usecols=['book_id', '一级分类', '二级分类', '分词题名', '题名'])
    
    # 加载交互数据
    self.inter_df = pd.read_csv(inter_path,
        usecols=['user_id', 'book_id', '借阅时间', '还书时间', '续借次数', '借阅时长'],
        parse_dates=['借阅时间', '还书时间'])
    
    # 处理兴趣强度
    self._compute_interest_intensity()
    
    # 分配学期场景
    for month in self.inter_df['借阅时间'].dt.month:
        scene_list.append(self._get_scene(month))
    self.inter_df['学期场景'] = scene_list
    
    # 时间衰减
    self.inter_df['距离最新天数'] = (latest_time - borrow_time).days
    self.inter_df['时间衰减权重'] = np.exp(-days / self.time_decay_half_life)
```

### 2. 序列数据集

```python
class InterestSceneDataset(Dataset):
    def __init__(self, user_train_data, book2id, user2id, scene2id, seq_len=3):
        self.data = self._build_sequences()
    
    def _build_sequences(self):
        sequences = []
        for user_id, records in user_train_data.items():
            records_sorted = sorted(records, key=lambda x: x['借阅时间'])
            book_ids = [book2id[rec['book_id']] for rec in records_sorted]
            scenes = [scene2id[rec['学期场景']] for rec in records_sorted]
            intensities = [rec['兴趣强度'] for rec in records_sorted]
            
            # 滑动窗口：需要 seq_len+1 条记录
            if len(book_ids) < self.seq_len + 1:
                continue
            
            for i in range(len(book_ids) - self.seq_len):
                seq_books = book_ids[i:i + self.seq_len]
                seq_scenes = scenes[i:i + self.seq_len]
                seq_ints = intensities[i:i + self.seq_len]
                target = book_ids[i + self.seq_len]
                sequences.append((user_id, seq_books, seq_scenes, seq_ints, target))
        return sequences
```

### 3. 推荐逻辑

```python
def recommend_for_user(self, user_id):
    # 1. 检查重复借阅（最强信号）
    user_repeat = [(b, cnt) for (u, b), cnt in self.user_book_repeat.items() if u == user_id]
    if user_repeat:
        return max(user_repeat, key=lambda x: x[1])[0]
    
    # 2. 检查续借
    user_renew = [(b, cnt, intensity) for (u, b), cnt in self.user_book_renew.items() 
                  if u == user_id and cnt >= 1]
    if user_renew:
        user_renew.sort(key=lambda x: x[1] * x[2], reverse=True)
        return user_renew[0][0]
    
    # 3. GRU 模型预测
    records = sorted(self.user_train_data[user_id], key=lambda x: x['借阅时间'])
    # 提取序列特征...
    
    # 4. 候选生成
    candidates = self._get_interest_candidates(user_id)
    
    # 5. 用 GRU + 兴趣权重打分
    candidate_scores = []
    for book_id in candidates:
        model_score = probs[book_idx].item()
        book_int_score = self.book_sliding_int.get(book_id, 0.0)
        total_score = model_score * 0.8 + book_int_score * 0.2
        candidate_scores.append((book_id, total_score))
    
    return candidates[np.argmax([s for _, s in candidate_scores])]
```

### 4. 训练循环

```python
def train_temporal_model(self):
    dataset = InterestSceneDataset(
        user_train_data=self.user_train_data,
        book2id=self.book2id, user2id=self.user2id,
        scene2id=self.scene2id, seq_len=self.seq_len
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    self.temporal_model = InterestSceneGRU(...)
    optimizer = torch.optim.Adam(self.temporal_model.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(15):
        for batch in dataloader:
            user_ids, seq_books, seq_scenes, seq_intensities, targets = batch
            optimizer.zero_grad()
            logits = self.temporal_model(user_ids, seq_books, seq_scenes, seq_intensities)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
```

---

## 性能分析

### 优势

1. **序列建模**：GRU 捕获时序借阅模式
2. **兴趣强度**：量化用户参与程度
3. **场景感知**：适应校历周期
4. **时间衰减**：近期交互权重更高
5. **最佳单模型**：F1 = 0.0158

### 劣势

1. **序列短**：seq_len=3 限制长期模式捕获
2. **冷启动**：每个用户至少需要 2 次交互
3. **候选有限**：约 80 个候选可能遗漏好选项
4. **无内容特征**：未直接利用图书文本/作者信息

### 为什么效果好

1. **强基线规则**：重复/续借规则捕获明显模式
2. **上下文感知**：场景建模捕获学术周期
3. **软信号**：兴趣强度区分交互强弱
4. **GRU 时序性**：自然地建模"接下来借什么"

---

## 相关文件

- `src/gru4rec/GRU4Rec 0.0158.py` — 完整实现
