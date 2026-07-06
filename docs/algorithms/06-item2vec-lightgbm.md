# 算法六：Item2Vec + LightGBM 排序流水线

> **文件**：`src/item2vec_lightgbm/01_train_item2vec.py`、`02_build_rank_samples.py`、`03_train_lightgbm.py`、`04_inference.py`、`05_multi_recalle.py`
>
> **类型**：嵌入 + 梯度提升排序
>
> **流水线**：Item2Vec 嵌入 -> 特征构建 -> LightGBM 排序 -> 推理

---

## 目录

- [算法原理](#算法原理)
- [系统架构](#系统架构)
- [代码结构分析](#代码结构分析)
- [步骤一：Item2Vec 训练](#步骤一item2vec-训练)
- [步骤二：排序样本构建](#步骤二排序样本构建)
- [步骤三：LightGBM 训练](#步骤三lightgbm-训练)
- [步骤四：推理流水线](#步骤四推理流水线)
- [步骤五：多路召回](#步骤五多路召回)
- [性能分析](#性能分析)

---

## 算法原理

### 核心思想

本算法将推荐视为一个**排序学习（Learning-to-Rank）**问题：

1. **Item2Vec**：将用户借阅序列视为"句子"，用 Skip-gram 模型学习图书嵌入
2. **特征工程**：用 Item2Vec 相似度、热度、用户历史构建排序特征
3. **LightGBM**：训练梯度提升模型对候选集排序
4. **推理**：打分候选集，为每个用户选 Top-1

### 为什么用 Item2Vec + LightGBM？

| 组件 | 作用 | 优势 |
|------|------|------|
| Item2Vec | 嵌入 | 捕获物品共现模式 |
| LightGBM | 排序器 | 快、支持混合特征、排序能力强 |
| 结合 | 端到端 | 嵌入相似度 + 手工特征 |

### 数学公式

**Item2Vec（Skip-gram）**：
```
P(book_j | book_i) = exp(v_j'^T * v_i) / sum(exp(v_k'^T * v_i))
```

**LightGBM 排序**：
```
score = LightGBM(features(user, book))
features = [Item2Vec_sim, popularity, category_match, ...]
```

---

## 系统架构

```
输入：原始借阅数据

步骤 1：Item2Vec 训练（01_train_item2vec.py）
  +-- 加载交互数据
  +-- 按用户分组 -> 借阅序列
  +-- 在序列上训练 Word2Vec（Skip-gram）
  +-- 提取嵌入矩阵 [num_books, 64]
  +-- 保存嵌入

步骤 2：构建排序样本（02_build_rank_samples.py）
  +-- 加载 Item2Vec 模型
  +-- 对每个用户：
  +--     +-- 提取最近 3 本（历史）
  +--     +-- 正样本：下一本
  +--     +-- 负样本：4 本随机未借阅图书
  +--     +-- 为每个候选计算特征
  +-- 保存训练/验证 CSV

步骤 3：LightGBM 训练（03_train_lightgbm.py）
  +-- 加载排序样本
  +-- 定义类别特征
  +-- 训练 LightGBM（二分类）
  +-- 保存模型

步骤 4：推理（04_inference.py）
  +-- 加载训练好的模型
  +-- 对每个用户：
  +--     +-- 构建候选池（同分类 + 相似）
  +--     +-- 为候选计算特征
  +--     +-- 用 LightGBM 打分
  +--     +-- 选 Top-1
  +-- 保存提交文件

步骤 5：多路召回（05_multi_recalle.py）
  +-- 环境热门图书
  +-- Item2Vec 相似图书
  +-- Apriori 关联规则
  +-- 合并并过滤候选集
```

---

## 代码结构分析

### 流水线文件

| 文件 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `01_train_item2vec.py` | 训练嵌入 | inter.csv | book_emb_64.npy、book2vec_dict.pkl |
| `02_build_rank_samples.py` | 构建排序数据 | 嵌入 + inter.csv | train_rank.csv、valid_rank.csv |
| `03_train_lightgbm.py` | 训练排序器 | train_rank.csv | lgb_rank.model |
| `04_inference.py` | 生成预测 | 模型 + 数据 | result.csv |
| `05_multi_recalle.py` | 多路召回 | 数据 + 嵌入 | cand_dict_80.pkl |
| `06_build_rank2.py` | 二阶段样本 | 候选集 | train_rank2.csv |
| `08_blend_inference.py` | 混合推理 | 多模型 | 最终提交 |

---

## 步骤一：Item2Vec 训练

### 核心逻辑

```python
# 1. 加载交互数据
inter = pd.read_csv(INTER_PATH)
inter['borrow_time'] = pd.to_datetime(inter['借阅时间'])
inter = inter.sort_values(['user_id', 'borrow_time'])

# 2. 构建用户借阅序列
# 每个用户的借阅列表 = 一个"句子"
user_seq = inter.groupby('user_id')['book_id'].apply(list).tolist()
# 结果：[[book1, book2, ...], [book3, book4, ...], ...]

# 3. 训练 Item2Vec（Skip-gram）
model = Word2Vec(
    sentences=user_seq,       # 用户序列列表
    vector_size=64,           # 嵌入维度
    window=5,                 # 上下文窗口
    min_count=1,              # 保留冷门图书
    sg=1,                     # 1=Skip-gram, 0=CBOW
    workers=8,                # 并行训练
    epochs=20,                # 迭代轮数
    seed=42
)

# 4. 导出嵌入矩阵
books = pd.read_csv(BOOK_PATH)['book_id'].tolist()
max_id = max(books)
emb_matrix = np.zeros((max_id + 1, 64), dtype=np.float32)

oov_cnt = 0
for bid in books:
    if str(bid) in model.wv:     # gensim 以字符串存储键
        emb_matrix[bid] = model.wv[str(bid)]
    else:
        oov_cnt += 1

np.save(f'{OUT_DIR}/book_emb_64.npy', emb_matrix)
```

### Item2Vec 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `vector_size` | 64 | 嵌入维度 |
| `window` | 5 | 上下文窗口 |
| `min_count` | 1 | 最低频次（保留全部） |
| `sg` | 1 | Skip-gram |
| `epochs` | 20 | 训练轮数 |
| `workers` | 8 | CPU 线程数 |

### Item2Vec 工作原理

```
用户借阅序列：[A, B, C, D, E]

window=2 时的训练对：
  中心：B -> 上下文：[A, C]
  中心：C -> 上下文：[A, B, D, E]
  中心：D -> 上下文：[B, C, E]

模型学习从中心图书预测上下文图书。
结果：相似的图书有相似的嵌入。
```

---

## 步骤二：排序样本构建

### 特征设计

```python
# 对每个（用户, 候选图书）对：

def gen_samples(uid, seq):
    samples = []
    for i in range(SEQ_LEN, len(seq)):
        hist = seq[i-SEQ_LEN:i]       # 最近 3 本
        next_book, next_cate, next_renew = seq[i]
        
        # 负采样：4 本随机未借阅图书
        neg_pool = list(all_books - set(seq[:, 0]))
        neg_books = np.random.choice(neg_pool, NEG_SAMPLES, replace=False)
        
        # 合并：1 正 + 4 负
        for cand_book, label in [(next_book, 1)] + [(nb, 0) for nb in neg_books]:
            row = [
                uid,                    # user_id
                cand_book,              # 候选 book_id
                hist[0][0],             # last1_book_id
                hist[1][0],             # last2_book_id
                hist[2][0],             # last3_book_id
                hist[0][1],             # last1_category
                hist[1][1],             # last2_category
                hist[2][1],             # last3_category
                hist[0][2],             # last1_renewal
                hist[1][2],             # last2_renewal
                hist[2][2],             # last3_renewal
                cand_cate,              # 候选分类
                cand_renew_mean,        # 候选平均续借
                cand_sim,               # Item2Vec 相似度
                cand_pop_dept,          # 院系热度
                label                   # 1=正, 0=负
            ]
            samples.append(row)
    return samples
```

### 特征表

| 列名 | 类型 | 说明 |
|------|------|------|
| `user_id` | int | 用户标识 |
| `cand_book_id` | int | 候选图书 |
| `last1_book_id` | int | 最近一本 |
| `last2_book_id` | int | 第二近 |
| `last3_book_id` | int | 第三近 |
| `last1_cate2` | cat | last1 的分类 |
| `last2_cate2` | cat | last2 的分类 |
| `last3_cate2` | cat | last3 的分类 |
| `last1_renew` | int | last1 的续借次数 |
| `last2_renew` | int | last2 的续借次数 |
| `last3_renew` | int | last3 的续借次数 |
| `cand_cate2` | cat | 候选分类 |
| `cand_renew_mean` | float | 候选平均续借 |
| `cand_sim_last1` | float | Item2Vec 相似度(last1, 候选) |
| `cand_pop_dept` | int | 用户院系中该书的借阅次数 |
| `label` | int | 1=借过, 0=未借 |

---

## 步骤三：LightGBM 训练

### 训练配置

```python
import lightgbm as lgb

# 定义特征和类别列
feat_cols = [c for c in train_df.columns 
             if c not in ['user_id', 'cand_book_id', 'label']]
cate_cols = [c for c in feat_cols if 'cate2' in c]

# 标记类别特征
for col in cate_cols:
    train_df[col] = train_df[col].astype('category')
    valid_df[col] = valid_df[col].astype('category')

# 创建数据集
dtrain = lgb.Dataset(train_df[feat_cols], label=train_df['label'])
dvalid = lgb.Dataset(valid_df[feat_cols], label=valid_df['label'], reference=dtrain)

# 参数
params = {
    'objective': 'binary',      # 二分类
    'metric': 'auc',            # AUC 评估
    'learning_rate': 0.05,
    'num_leaves': 127,          # 每棵树最大叶子数
    'min_child_samples': 20,    # 叶节点最小样本数
    'feature_fraction': 0.8,    # 列采样
    'bagging_fraction': 0.8,    # 行采样
    'bagging_freq': 1,
    'verbose': -1,
    'seed': 42
}

# 训练
model = lgb.train(
    params, dtrain, num_boost_round=5000,
    valid_sets=[dvalid],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)
```

### LightGBM 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `objective` | binary | 二分类 |
| `metric` | auc | ROC 曲线下面积 |
| `learning_rate` | 0.05 | 收缩步长 |
| `num_leaves` | 127 | 控制模型复杂度 |
| `min_child_samples` | 20 | 防止过拟合 |
| `feature_fraction` | 0.8 | 列子采样 |
| `bagging_fraction` | 0.8 | 行子采样 |

---

## 步骤四：推理流水线

### 候选生成

```python
def build_candidates(uid, hist_books, dept):
    last_book = hist_books[-1]
    
    # 1. 同分类图书
    last_cate = book.loc[book.book_id == last_book, '二级分类'].values[0]
    same_cate_books = book.loc[book['二级分类'] == last_cate, 'book_id'].tolist()
    
    # 排除已借阅
    cand_pool = list(set(same_cate_books) - set(hist_books))
    
    # 不足时补足
    if len(cand_pool) < 20:
        extra = set(book.sample(50)['book_id']) - set(hist_books)
        cand_pool.extend(list(extra))
    
    return cand_pool[:50]  # 最多 50 个候选
```

### 打分与选择

```python
for uid in tqdm(test_users):
    hist = inter[inter.user_id == uid]['book_id'].tolist()
    dept = user.loc[user['借阅人'] == uid, 'DEPT'].values[0]
    
    # 构建候选
    cand_df = build_candidates(uid, hist, dept)
    
    # 为所有候选计算特征
    for col in ['last1_cate2', 'last2_cate2', 'last3_cate2', 'cand_cate2']:
        cand_df[col] = cand_df[col].astype('category')
    
    # LightGBM 打分
    X = cand_df[model.feature_name()]
    scores = model.predict(X)
    cand_df['score'] = scores
    
    # 选 Top-1
    top1 = cand_df.loc[cand_df['score'].idxmax(), 'cand_book_id']
    results.append({'user_id': uid, 'book_id': int(top1)})
```

---

## 步骤五：多路召回

### 三路召回策略

```python
# 通道 1：环境热门图书
def env_top50(uid, dept):
    """获取用户所在院系和学术阶段的热门 Top-50 图书。"""
    last_time = inter[inter.user_id == uid]['borrow_time'].iloc[-1]
    ph = time2phase(last_time)  # autumn_start、autumn_final 等
    sub = env_hot[(env_hot.phase == ph) & (env_hot.DEPT == dept)]
    return sub.nlargest(50, 'env_hot')['book_id'].tolist()

# 通道 2：Item2Vec 相似图书
def ivec_top50(uid, hist):
    """获取与最近借阅最相似的 Top-50 图书。"""
    last = hist[-1]
    if str(last) not in model.wv:
        # 回退到全局热门
        global_top = inter['book_id'].value_counts().head(100).index.tolist()
        return [int(b) for b in global_top if int(b) not in hist][:50]
    return [int(b) for b, _ in model.wv.most_similar(str(last), topn=100)
            if int(b) not in hist][:50]

# 通道 3：Apriori 关联规则
rules = apriori(final_seq, min_support=0.003, min_confidence=0.3)
apriori_top30 = [int(r.items_add[0]) for r in rules if r.confidence >= 0.3][:30]

# 合并所有通道
for uid in user_hist:
    pool = list(set(env_top50(uid, dept) + ivec_top50(uid, hist) + apriori_top30) 
                - set(hist))[:80]
    cand_dict[uid] = pool
```

### 学期阶段检测

```python
def time2phase(dt):
    month, day = dt.month, dt.day
    if (month == 9 and day >= 1) or month == 10:
        return 'autumn_start'    # 新学期
    if (month == 11 and day >= 15) or month == 12:
        return 'autumn_final'    # 期末
    if month == 3 and day <= 15:
        return 'spring_start'    # 春季学期
    if month == 6:
        return 'spring_final'    # 春季期末
    return 'normal'
```

---

## 性能分析

### 优势

1. **强基线**：Item2Vec 很好地捕获共借模式
2. **训练快**：LightGBM 非常高效
3. **可解释特征**：手工特征含义清晰
4. **多路召回**：三路召回确保高覆盖率
5. **可扩展**：高效处理大规模候选集

### 劣势

1. **特征工程**：需要领域知识
2. **负采样**：随机负样本可能太简单
3. **无深度交互**：相比深度神经网络
4. **冷启动**：新用户/图书无嵌入

### 流水线执行顺序

```bash
# 按顺序执行：
python 01_train_item2vec.py    # ~2 分钟
python 02_build_rank_samples.py # ~5 分钟
python 03_train_lightgbm.py     # ~3 分钟
python 04_inference.py          # ~5 分钟
# 总计：~15 分钟
```

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `01_train_item2vec.py` | 训练 Word2Vec 嵌入 |
| `02_build_rank_samples.py` | 构建排序数据集 |
| `03_train_lightgbm.py` | 训练 LightGBM 排序器 |
| `04_inference.py` | 生成预测 |
| `05_multi_recalle.py` | 多路候选生成 |
| `06_build_rank2.py` | 二阶段排序样本 |
| `08_blend_inference.py` | 多模型混合推理 |
