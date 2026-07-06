# 算法五：RecBole 集成框架

> **文件**：`src/recbole_framework/*.py`（main.py、configs.py、data_processor.py、model_trainer.py、ensemble.py、evaluator.py 等）
>
> **类型**：多模型集成框架
>
> **支持模型**：ItemKNN、BPR、NeuMF、SASRec、GRU4Rec、BERT4Rec、DeepFM、Pop

---

## 目录

- [算法原理](#算法原理)
- [系统架构](#系统架构)
- [代码结构分析](#代码结构分析)
- [模型配置](#模型配置)
- [数据处理流水线](#数据处理流水线)
- [集成策略](#集成策略)
- [智能权重计算](#智能权重计算)
- [核心代码解析](#核心代码解析)
- [性能分析](#性能分析)

---

## 算法原理

### 什么是 RecBole？

[RecBole](https://github.com/RUCAIBox/RecBole) 是 RUCAIBox 开发的统一、全面、高效的推荐算法库，提供：
- **80+ 推荐算法**
- **统一数据格式**（原子文件）
- **标准化评估协议**
- **GPU 加速**

### 为什么用集成？

没有单一模型是完美的。不同模型捕获不同模式：

| 模型类型 | 擅长捕获 | 容易遗漏 |
|----------|---------|---------|
| 协同过滤 | 用户-物品交互 | 内容、序列 |
| 序列模型 | 时序模式 | 全局热门 |
| 基于内容 | 物品特征 | 协同信号 |
| 图模型 | 高阶连接 | 直接特征 |

**集成的力量**：加权融合多样化模型，获得更稳健的推荐。

### 框架设计

```
数据 -> 预处理 -> 训练多个模型 -> 评估 -> 
    智能权重 -> 多样性融合 -> 最终推荐
```

---

## 系统架构

```
输入：原始 CSV 数据（user.csv、book.csv、inter.csv）

步骤 1：数据处理（data_processor.py）
  +-- 加载并重命名列
  +-- 解析时间戳
  +-- 提取时间特征（小时、星期、月份）
  +-- 基于续借次数计算评分
  +-- 去重和过滤
  +-- 保存为 RecBole 原子文件

步骤 2：时序感知的划分
  +-- 按用户+时间排序
  +-- 测试集：每个用户的最后一次交互
  +-- 训练集：其余交互
  +-- 验证集：训练用户的子集

步骤 3：模型训练（model_trainer.py + configs.py）
  +-- 遍历配置中的每个模型
  +--     +-- 初始化 RecBole 配置
  +--     +-- 创建数据集和划分
  +--     +-- 训练模型
  +--     +-- 在测试集上评估
  +-- 收集所有预测

步骤 4：智能集成（ensemble.py + smart_weights.py）
  +-- 计算每个模型的质量指标
  +-- 计算智能权重（性能+多样性）
  +-- 检测并惩罚有偏模型
  +-- 融合推荐
  +-- 应用多样性增强

步骤 5：续借感知调整（evaluator.py）
  +-- 分析续借行为
  +-- 根据用户续借率调整得分
  +-- 输出最终推荐
```

---

## 代码结构分析

### 模块依赖

```
main.py
  +-- configs.py              <- FixedBaseConfig（模型超参数）
  +-- data_processor.py       <- EnhancedDataProcessor（数据准备）
  +-- model_trainer.py        <- FixedModelTrainer（RecBole 包装器）
  +-- evaluator.py            <- EnhancedRenewAwareEvaluator
  +-- ensemble.py             <- IntelligentEnsembler
  +-- smart_weights.py        <- SmartWeightCalculator
  +-- quality_analyzer.py     <- RecommendationQualityAnalyzer
  +-- model_classifier.py     <- ModelClassifier
  +-- utils.py                <- save_candidates、calculate_f1_score
```

### 配置（configs.py）

```python
class FixedBaseConfig:
    """所有 RecBole 模型的中央配置。"""
    
    # 数据路径
    data_path = './data'
    dataset_name = 'library'
    
    # 字段映射
    user_id_field = 'user_id'
    item_id_field = 'item_id'
    rating_field = 'rating'
    time_field = 'timestamp_unix'
    
    # 训练
    epochs = 30
    learning_rate = 0.001
    train_batch_size = 512
    eval_batch_size = 1024
    early_stop_patience = 3
    
    # 评估
    topk = [1, 5, 10]
    metrics = ['Recall', 'Precision', 'Hit', 'NDCG', 'MRR']
    valid_metric = 'Recall@10'
    
    def to_dict(self, model_name):
        """为指定模型生成配置字典。"""
        config = {基础设置...}
        
        if model_name == 'SASRec':
            config.update({
                'embedding_size': 64, 'n_layers': 2, 'n_heads': 2,
                'hidden_size': 64, 'MAX_ITEM_LIST_LENGTH': 20,
            })
        elif model_name == 'BERT4Rec':
            config.update({
                'n_layers': 3, 'n_heads': 4, 'hidden_size': 128,
                'MAX_ITEM_LIST_LENGTH': 80, 'mask_ratio': 0.15,
            })
        elif model_name == 'GRU4Rec':
            config.update({
                'embedding_size': 64, 'hidden_size': 128, 'num_layers': 1,
            })
        # ... 更多模型
        return config
```

---

## 模型配置

### 支持的模型

| 模型 | 类别 | 关键配置 | 说明 |
|------|------|----------|------|
| **ItemKNN** | 协同过滤 | k=100 | 基于物品的最近邻 |
| **BPR** | 矩阵分解 | embedding_size=64 | 贝叶斯个性化排序 |
| **NeuMF** | 深度学习 | mlp_hidden=[64,32,16] | 神经矩阵分解 |
| **SASRec** | 序列 | n_layers=2, n_heads=2 | 自注意力序列推荐 |
| **GRU4Rec** | 序列 | hidden_size=128 | 基于 GRU 的序列推荐 |
| **BERT4Rec** | 序列 | n_layers=3, n_heads=4 | 基于 BERT 的序列推荐 |
| **DeepFM** | 上下文感知 | mlp_hidden=[400,400,400] | 深度因子分解机 |
| **Pop** | 基线 | - | 基于热门度 |

### 序列模型特殊处理

```python
SEQUENTIAL_MODELS = {'SASRec', 'GRU4Rec', 'Caser', 'BERT4Rec'}

# 序列模型需要物品序列
if model_name in SEQUENTIAL_MODELS:
    # 获取用户交互历史
    item_seq = user_inter_df[item_id_field].tolist()
    
    # 填充/截断到固定长度
    max_len = config['MAX_ITEM_LIST_LENGTH']
    if len(item_seq) >= max_len:
        item_seq = item_seq[-max_len:]
    else:
        item_seq = [0] * (max_len - len(item_seq)) + item_seq
    
    # 创建带序列的交互
    interaction = Interaction({
        uid_field: torch.tensor([uid]),
        'item_id_list': torch.tensor([item_seq]),
        'item_length': torch.tensor([max_len]),
    })
```

---

## 数据处理流水线

### EnhancedDataProcessor

```python
class EnhancedDataProcessor:
    def load_data(self):
        """加载 CSV 文件并返回 dataframe。"""
        self.book_df = pd.read_csv(f'{data_path}/item.csv')
        self.user_df = pd.read_csv(f'{data_path}/user.csv')
        self.inter_df = pd.read_csv(f'{data_path}/inter.csv')
    
    def enhanced_preprocess_data(self):
        """为 RecBole 预处理。"""
        # 重命名为 RecBole 格式
        book_df.rename(columns={'book_id': 'item_id', '题名': 'title', ...})
        user_df.rename(columns={'借阅人': 'user_id', 'DEPT': 'dept', ...})
        
        # 处理时间戳
        inter_df['timestamp'] = pd.to_datetime(inter_df['timestamp'])
        inter_df['timestamp_unix'] = inter_df['timestamp'].astype(int) // 10**9
        
        # 提取时间特征
        inter_df['borrow_hour'] = inter_df['timestamp'].dt.hour
        inter_df['borrow_dayofweek'] = inter_df['timestamp'].dt.dayofweek
        inter_df['borrow_month'] = inter_df['timestamp'].dt.month
        
        # 基于续借的评分
        inter_df['has_renewed'] = (inter_df['renew_count'] > 0).astype(int)
        inter_df['rating'] = 1.0 + has_renewed * 0.8 + (renew_count > 1) * 0.4
        
        # 去重和过滤
        inter_df = inter_df.drop_duplicates(subset=['user_id', 'item_id', 'timestamp'])
    
    def save_data_for_recbole(self, train_df, valid_df, test_df):
        """保存为 RecBole 原子文件（.user、.item、.inter）。"""
        # 保存用户数据
        user_df_out.to_csv(f'{dataset_dir}/{dataset}.user', sep='\t')
        # 保存物品数据
        book_df_out.to_csv(f'{dataset_dir}/{dataset}.item', sep='\t')
        # 保存交互数据
        inter_df_out.to_csv(f'{dataset_dir}/{dataset}.inter', sep='\t')
```

---

## 集成策略

### IntelligentEnsembler

```python
class IntelligentEnsembler:
    """
    多阶段集成，带多样性增强。
    
    阶段：
    1. 每个模型的质量分析
    2. 智能权重计算
    3. 有偏模型检测
    4. 加权融合
    5. 多样性增强
    """
    
    def intelligent_fusion(self, all_recommendations, model_results, 
                          user_df, item_df, top_k=1):
        # 分析推荐质量
        quality_report = self.quality_analyzer.analyze(...)
        
        # 检测有偏模型（过于集中）
        biased_models = self.quality_analyzer.detect_biased_models()
        
        # 计算智能权重
        smart_weights = self.weight_calculator.calculate_smart_weights(...)
        
        # 执行融合
        ensemble = self._execute_fusion(all_recommendations, smart_weights, top_k)
        
        return ensemble
```

### 融合公式

```python
# 对每个用户的每个候选物品：
item_score = sum(
    model_weight[model] * exp(-rank * 0.5) 
    for model, rank in model_recommendations[item]
)

# model_weight：智能计算的权重
# rank：在模型推荐列表中的位置
# exp(-rank * 0.5)：指数位置衰减
```

---

## 智能权重计算

### 四阶段权重计算

```python
def calculate_smart_weights(self, model_results, all_recommendations, 
                           all_items, item_popularity):
    # 阶段 1：基于性能的权重
    perf_weights = _calculate_performance_weights(model_results)
    # 使用：recall@10、ndcg@10、precision@10
    # 公式：0.5*recall + 0.3*ndcg + 0.2*precision
    
    # 阶段 2：基于多样性的调整
    div_weights = _calculate_diversity_weights(all_recommendations, perf_weights)
    # 高多样性模型获得奖励：weight *= (1 + diversity * 0.5)
    
    # 阶段 3：基于质量的调整（覆盖率 + 新颖性）
    quality_weights = _calculate_quality_weights(all_recommendations, div_weights,
                                                  all_items, item_popularity)
    # 奖励：1 + coverage*0.3 + novelty*0.2
    
    # 阶段 4：特殊规则
    final_weights = _apply_special_rules(quality_weights, all_recommendations)
    # Pop 模型惩罚（过于集中）
    # 低多样性模型惩罚（diversity < 0.1 -> weight *= 0.5）
    
    return _normalize_weights(final_weights)
```

### 多样性得分

```python
def calculate_diversity_score(self, recommendations):
    # diversity = unique_items / total_recommendations
    item_counts = Counter(all_recommended_items)
    diversity = unique_items / total_recommendations
    
    # 集中度惩罚（基尼系数启发式）
    max_count = max(item_counts.values())
    concentration_penalty = max_count / total_recommendations
    
    final_diversity = diversity * (1 - concentration_penalty * 0.5)
    return max(0.0, final_diversity)
```

---

## 核心代码解析

### 主入口

```python
def fixed_diversity_main():
    # 步骤 1：配置
    base_config = FixedBaseConfig()
    
    # 步骤 2：数据加载
    data_processor = EnhancedDataProcessor(base_config.data_path)
    book_df, user_df, inter_df = data_processor.load_data()
    book_df, user_df, inter_df = data_processor.enhanced_preprocess_data()
    
    # 步骤 3：数据划分
    train_df, valid_df, test_df = data_processor.time_aware_split()
    full_inter_df = data_processor.save_data_for_recbole(train_df, valid_df, test_df)
    
    # 步骤 4：分析续借行为
    evaluator = EnhancedRenewAwareEvaluator()
    renew_stats = evaluator.analyze_enhanced_renew_behavior(full_inter_df, book_df)
    
    # 步骤 5：训练模型
    stable_models = ['ItemKNN']  # 可添加更多模型
    for model_name in stable_models:
        config_dict = base_config.to_dict(model_name)
        trainer = FixedModelTrainer(config_dict)
        trainer.prepare_data()
        valid_score, valid_result = trainer.train()
        test_result = trainer.evaluate()
        recommendations = trainer.generate_recommendations(test_users, top_k=10)
    
    # 步骤 6：集成
    ensemble_recommendations = fixed_diversity_fusion(
        all_recommendations, model_results, book_df, top_k=5)
    final_recommendations = evaluator.enhanced_adjust_recommendations(
        ensemble_recommendations, book_df, top_k=1)
    
    # 步骤 7：保存结果
    save_candidates(final_recommendations, 'candidates_10.csv')
```

### 模型训练器

```python
class FixedModelTrainer:
    def prepare_data(self):
        self.config = Config(model=self.config_dict['model'], ...)
        init_seed(self.config['seed'], self.config['reproducibility'])
        self.dataset = create_dataset(self.config)
        self.train_data, self.valid_data, self.test_data = 
            data_preparation(self.config, self.dataset)
    
    def init_model(self):
        if model_name == 'SASRec':
            from recbole.model.sequential_recommender import SASRec
            self.model = SASRec(self.config, self.dataset)
        elif model_name == 'BERT4Rec':
            from recbole.model.sequential_recommender import BERT4Rec
            self.model = BERT4Rec(self.config, self.dataset)
        # ... 更多模型
    
    def train(self):
        self.trainer = Trainer(self.config, self.model)
        best_valid_score, best_valid_result = self.trainer.fit(
            self.train_data, self.valid_data, verbose=True, saved=True)
        return best_valid_score, best_valid_result
```

---

## 性能分析

### 优势

1. **模型多样性**：支持 8+ 种不同类型的模型
2. **标准化框架**：RecBole 处理数据格式和评估
3. **智能集成**：性能 + 多样性 + 覆盖率 + 新颖性
4. **续借感知**：根据用户续借行为调整
5. **质量分析**：检测有偏/过于集中的模型

### 劣势

1. **复杂性**：许多文件和组件需要管理
2. **RecBole 依赖**：锁定在 RecBole 的数据格式
3. **内存**：同时加载多个模型
4. **调参**：跨模型的大量超参数

### 使用建议

- **适合**：想快速实验多种算法时
- **最佳用途**：快速原型、比较不同模型家族
- **可与**其他流水线结合用于最终集成

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `main.py` | 主入口 |
| `configs.py` | 模型超参数 |
| `data_processor.py` | 数据预处理 |
| `model_trainer.py` | RecBole 模型包装器 |
| `ensemble.py` | 智能融合 |
| `evaluator.py` | 续借感知评估 |
| `smart_weights.py` | 权重计算 |
| `quality_analyzer.py` | 质量指标 |
| `model_classifier.py` | 模型分类 |
| `utils.py` | 工具函数 |
