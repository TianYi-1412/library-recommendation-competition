import pandas as pd
import numpy as np
import warnings
from collections import defaultdict, Counter
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
torch.manual_seed(42)  # 固定随机种子，确保结果可复现


class InterestSceneTemporalRecommender:
    def __init__(self):
        # 基础数据结构（仅保留有交互用户相关）
        self.user_train_data = defaultdict(list)  # 有交互用户的训练集记录
        self.user_val_data = {}  # 有交互用户的验证集（600个，核心推荐对象）
        self.user_borrow_history = defaultdict(set)  # 有交互用户的已借阅图书

        # 图书核心信息
        self.book_info = {}  # 图书基础信息（ID→分类/关键词）
        self.secondary_cate_books = defaultdict(set)  # 二级分类→图书集合
        self.keyword_books = defaultdict(set)  # 关键词→图书集合

        # 用户兴趣相关（仅针对有交互用户）
        self.user_book_renew = {}  # 用户-图书→续借次数
        self.user_book_repeat = {}  # 用户-图书→重复借阅次数（≥2）
        self.user_high_int_cates = {}  # 用户→高兴趣二级分类（Top2）

        # 时序与场景相关
        self.book_sliding_int = defaultdict(float)  # 图书滑动窗口兴趣得分
        self.scaler_int = MinMaxScaler()  # 兴趣强度归一化器
        self.scene2id = {0: 0, 1: 1, 2: 2, 3: 3}  # 场景映射：0=假期，1=春学期，2=秋学期，3=期末
        self.latest_time = None  # 数据集中最新借阅时间

        # 模型相关
        self.book2id = {}  # 图书ID→整数映射（Embedding用）
        self.id2book = {}  # 整数→图书ID映射
        self.user2id = {}  # 有交互用户ID→整数映射（仅处理600个用户）
        self.temporal_model = None  # 时序模型实例
        self.optimizer = None
        self.criterion = None

        # 超参数（适配有交互用户的时序特征）
        self.seq_len = 3  # 序列长度（短期兴趣更精准）
        self.embedding_dim = 64  # 嵌入维度
        self.hidden_dim = 128  # GRU隐藏层维度
        self.time_decay_half_life = 30  # 时间衰减半衰期
        self.stop_words = {'高等', '教程', '基础', '入门', '数学', '原理'}  # 关键词停用词

    def load_data(self, inter_path, book_path, user_path):
        """加载数据+预处理：仅保留有交互的600个用户"""
        start = time.time()

        # -------------------------- 1. 加载图书数据 --------------------------
        print("1/4 加载图书数据...")
        self.book_df = pd.read_csv(
            book_path,
            usecols=['book_id', '一级分类', '二级分类', '分词题名', '题名']
        )
        self.book_df['book_id'] = self.book_df['book_id'].astype(str)
        # 处理图书文本缺失值（带进度条）
        text_cols = ['一级分类', '二级分类', '分词题名', '题名']
        for col in tqdm(text_cols, desc="  处理图书文本缺失值"):
            self.book_df[col] = self.book_df[col].fillna('未知')
        # 提取图书核心关键词（带进度条）
        print("  提取图书核心关键词...")
        core_kws = []
        for _, row in tqdm(self.book_df.iterrows(), total=len(self.book_df), desc="  生成关键词"):
            kws = row['分词题名'].split() if row['分词题名'] != '未知' else []
            core_kw = ' '.join([kw for kw in kws if len(kw) >= 2 and kw not in self.stop_words][:5])
            core_kws.append(core_kw)
        self.book_df['核心关键词'] = core_kws

        # -------------------------- 2. 加载交互数据+关联二级分类（仅保留有交互用户） --------------------------
        print("\n2/4 加载交互数据并筛选有交互用户...")
        self.inter_df = pd.read_csv(
            inter_path,
            usecols=['user_id', 'book_id', '借阅时间', '还书时间', '续借次数', '借阅时长'],
            parse_dates=['借阅时间', '还书时间']
        )
        self.inter_df['user_id'] = self.inter_df['user_id'].astype(str)
        self.inter_df['book_id'] = self.inter_df['book_id'].astype(str)
        self.latest_time = self.inter_df['借阅时间'].max()

        # 2.1 关联图书二级分类
        print("  关联交互数据与图书二级分类...")
        self.inter_df = self.inter_df.merge(
            self.book_df[['book_id', '二级分类']],
            on='book_id',
            how='left'
        )
        self.inter_df['二级分类'] = self.inter_df['二级分类'].fillna('未知分类')

        # 2.2 处理交互特征（兴趣强度+场景）
        print("  处理交互数据特征（兴趣强度+学期场景）...")
        # 借阅时长标准化
        scaler_duration = MinMaxScaler()
        self.inter_df['借阅时长_标准化'] = scaler_duration.fit_transform(self.inter_df[['借阅时长']])
        # 计算重复借阅权重
        borrow_cnt = self.inter_df.groupby(['user_id', 'book_id']).size().reset_index(name='重复次数')
        borrow_cnt['重复权重'] = borrow_cnt['重复次数'].apply(lambda x: min(0.3 * (x - 1), 0.6))
        # 合并权重并计算兴趣强度
        self.inter_df = self.inter_df.merge(borrow_cnt[['user_id', 'book_id', '重复次数', '重复权重']],
                                            on=['user_id', 'book_id'], how='left')
        self.inter_df['重复次数'] = self.inter_df['重复次数'].fillna(1)
        self.inter_df['重复权重'] = self.inter_df['重复权重'].fillna(0)
        self.inter_df['兴趣强度'] = (self.inter_df['借阅时长_标准化'] + self.inter_df['重复权重']).clip(0, 1)

        # 生成学期场景（带进度条）
        scene_list = []
        for month in tqdm(self.inter_df['借阅时间'].dt.month, total=len(self.inter_df), desc="  生成学期场景"):
            if month in [1, 2, 7, 8]:
                scene_list.append(0)
            elif month in [3, 4, 5]:
                scene_list.append(1)
            elif month in [9, 10, 11]:
                scene_list.append(2)
            else:
                scene_list.append(3)
        self.inter_df['学期场景'] = scene_list

        # 2.3 计算时间衰减权重
        self.inter_df['距离最新天数'] = (self.latest_time - self.inter_df['借阅时间']).dt.days
        self.inter_df['时间衰减权重'] = np.exp(-self.inter_df['距离最新天数'] / self.time_decay_half_life)

        # -------------------------- 3. 加载用户数据（仅用于验证有交互用户） --------------------------
        print("\n3/4 加载用户数据（仅验证有交互用户）...")
        self.user_df = pd.read_csv(
            user_path,
            usecols=['借阅人', 'DEPT', '年级'],
            encoding='utf-8'
        )
        self.user_df.rename(columns={'借阅人': 'user_id'}, inplace=True)
        self.user_df['user_id'] = self.user_df['user_id'].astype(str)

        # -------------------------- 4. 构建核心映射表（仅针对有交互用户） --------------------------
        print("\n4/4 构建核心映射表（仅处理有交互用户）...")
        self._split_loo_train_val()  # 先分割训练/验证集，确定600个有交互用户
        self._build_book_mappings()  # 图书-分类/关键词映射
        self._build_user_mappings()  # 有交互用户-兴趣映射
        self._compute_book_sliding_int()  # 图书滑动窗口兴趣得分

        # -------------------------- 数据验证打印（重点显示有交互用户数量） --------------------------
        print(f"\n数据加载完成，总耗时：{time.time() - start:.2f}秒")
        print(f"=== 核心数据量统计 ===")
        print(f"交互记录数：{len(self.inter_df)} | 图书数：{len(self.book_df)}")
        print(f"有交互用户数（核心推荐对象）：{len(self.user_val_data)}（目标600个）")
        print(f"=== 关键字段验证 ===")
        print(f"inter_df是否包含'二级分类'：{'二级分类' in self.inter_df.columns}")
        print(f"兴趣强度范围：{self.inter_df['兴趣强度'].min():.2f}~{self.inter_df['兴趣强度'].max():.2f}")

    def _build_book_mappings(self):
        """构建图书相关映射：基础信息、分类-图书、关键词-图书"""
        # 图书基础信息
        self.book_info = self.book_df.set_index('book_id')[
            ['一级分类', '二级分类', '核心关键词', '题名']
        ].to_dict('index')

        # 二级分类→图书集合（带进度条）
        print("  构建二级分类-图书映射...")
        cate_book_int = self.inter_df.groupby(['二级分类', 'book_id'])['兴趣强度'].mean().reset_index()
        unique_cates = self.book_df['二级分类'].unique()
        for cate in tqdm(unique_cates, total=len(unique_cates), desc="  处理分类映射"):
            cate_books = cate_book_int[cate_book_int['二级分类'] == cate].sort_values('兴趣强度', ascending=False)[
                'book_id'].tolist()
            self.secondary_cate_books[cate] = set(cate_books[:100])

        # 关键词→图书集合（带进度条）
        print("  构建关键词-图书映射...")
        kw_book_map = defaultdict(list)
        for book_id, info in tqdm(self.book_info.items(), total=len(self.book_info), desc="  收集关键词"):
            for kw in info['核心关键词'].split():
                kw_book_map[kw].append(book_id)
        # 为每个关键词筛选高兴趣图书
        for kw, books in tqdm(kw_book_map.items(), total=len(kw_book_map), desc="  筛选高兴趣图书"):
            book_int = [(b, self.inter_df[self.inter_df['book_id'] == b]['兴趣强度'].mean())
                        for b in books if b in self.inter_df['book_id'].unique()]
            book_int.sort(key=lambda x: x[1], reverse=True)
            self.keyword_books[kw] = set([b for b, _ in book_int[:50]])

    def _build_user_mappings(self):
        """构建有交互用户的兴趣映射（仅处理600个用户）"""
        print("  构建有交互用户的兴趣映射...")
        # 1. 有交互用户的高兴趣二级分类
        user_cate_int = self.inter_df.groupby(['user_id', '二级分类'])['兴趣强度'].sum().reset_index()
        # 仅处理有交互的用户（self.user_val_data中的用户）
        valid_users = list(self.user_val_data.keys())
        for user_id in tqdm(valid_users, total=len(valid_users), desc="  计算用户高兴趣分类"):
            user_cates = user_cate_int[user_cate_int['user_id'] == user_id].sort_values('兴趣强度', ascending=False)
            self.user_high_int_cates[user_id] = user_cates['二级分类'].tolist()[:2] if len(user_cates) >= 2 else [
                '未知分类']

        # 2. 续借/重复借阅统计（仅针对有交互用户）
        self.user_book_renew = self.inter_df.groupby(['user_id', 'book_id'])['续借次数'].sum().to_dict()
        self.user_book_repeat = self.inter_df.groupby(['user_id', 'book_id']).size()
        self.user_book_repeat = self.user_book_repeat[self.user_book_repeat >= 2].to_dict()

    def _split_loo_train_val(self):
        """留一法分割：仅保留“至少2次交互”的用户（最终600个）"""
        print("  分割训练/验证集（仅保留有2次以上交互的用户）...")
        inter_sorted = self.inter_df.sort_values(['user_id', '借阅时间'])
        unique_users = inter_sorted['user_id'].unique()
        # 仅保留至少2次交互的用户
        for user_id in tqdm(unique_users, total=len(unique_users), desc="  筛选有交互用户"):
            group = inter_sorted[inter_sorted['user_id'] == user_id]
            if len(group) < 2:  # 仅保留至少2次交互的用户（1次训练+1次验证）
                continue
            # 验证集：最后一次借阅
            val_row = group.tail(1).iloc[0]
            self.user_val_data[user_id] = val_row['book_id']
            # 训练集：其余借阅
            train_rows = group.head(-1)
            self.user_train_data[user_id] = train_rows.to_dict('records')
            # 用户已借阅图书集合
            self.user_borrow_history[user_id] = set(group['book_id'])
        # 打印筛选结果（确认是否为600个）
        print(f"  筛选后有交互用户数：{len(self.user_val_data)}（目标600个）")

    def _compute_book_sliding_int(self):
        """计算图书滑动窗口兴趣得分（基于有交互用户的行为）"""
        print("  计算图书滑动窗口兴趣得分...")
        inter_sorted = self.inter_df.sort_values(['book_id', '借阅时间'])
        inter_sorted['窗口'] = (self.latest_time - inter_sorted['借阅时间']).dt.days // 7
        # 按图书+窗口统计兴趣得分
        book_window_int = inter_sorted.groupby(['book_id', '窗口']).apply(
            lambda x: (x['兴趣强度'] * x['时间衰减权重']).sum()
        ).reset_index(name='窗口得分')
        # 累加窗口得分（带进度条）
        unique_books = self.inter_df['book_id'].unique()
        for book_id in tqdm(unique_books, total=len(unique_books), desc="  计算图书兴趣得分"):
            book_data = book_window_int[book_window_int['book_id'] == book_id]
            self.book_sliding_int[book_id] = sum(book_data['窗口得分'] * (book_data['窗口'] + 1))
        # 归一化兴趣得分
        if len(self.book_sliding_int) > 0:
            int_vals = np.array(list(self.book_sliding_int.values())).reshape(-1, 1)
            self.scaler_int.fit(int_vals)
            for book_id in tqdm(self.book_sliding_int.keys(), total=len(self.book_sliding_int),
                                desc="  归一化兴趣得分"):
                self.book_sliding_int[book_id] = self.scaler_int.transform([[self.book_sliding_int[book_id]]])[0][0]

    # ------------------------------ 核心时序模型（仅针对有交互用户） ------------------------------
    class InterestSceneGRU(nn.Module):
        """GRU时序模型：输入=图书序列+场景序列+兴趣强度序列"""

        def __init__(self, user_num, book_num, scene_num=4, embedding_dim=64, hidden_dim=128, dropout=0.2):
            super().__init__()
            self.user_emb = nn.Embedding(user_num, embedding_dim, padding_idx=0)
            self.book_emb = nn.Embedding(book_num, embedding_dim, padding_idx=0)
            self.scene_emb = nn.Embedding(scene_num, embedding_dim // 2, padding_idx=0)
            self.gru = nn.GRU(
                input_size=embedding_dim + embedding_dim // 2 + 1,
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=dropout
            )
            self.fc = nn.Linear(hidden_dim, book_num)

        def forward(self, user_ids, seq_books, seq_scenes, seq_intensities):
            if user_ids.dim() == 0:
                user_ids = user_ids.unsqueeze(0)
            batch_size = user_ids.size(0)
            seq_len = seq_books.size(1)

            book_emb = self.book_emb(seq_books)
            scene_emb = self.scene_emb(seq_scenes)
            int_emb = seq_intensities.unsqueeze(-1).float()

            input_emb = torch.cat([book_emb, scene_emb, int_emb], dim=-1)
            gru_out, _ = self.gru(input_emb)
            final_out = gru_out[:, -1, :]
            logits = self.fc(final_out)
            return logits

    class InterestSceneDataset(Dataset):
        """序列数据集：仅包含有交互用户的训练数据"""

        def __init__(self, user_train_data, book2id, user2id, scene2id, seq_len=3):
            self.user_train_data = user_train_data  # 仅600个有交互用户的训练数据
            self.book2id = book2id
            self.user2id = user2id
            self.scene2id = scene2id
            self.seq_len = seq_len
            self.data = self._build_sequences()

        def _build_sequences(self):
            sequences = []
            # 仅遍历有交互用户的训练数据
            for user_id, records in self.user_train_data.items():
                records_sorted = sorted(records, key=lambda x: x['借阅时间'])
                book_ids = [self.book2id.get(rec['book_id'], 0) for rec in records_sorted]
                scenes = [self.scene2id.get(rec['学期场景'], 0) for rec in records_sorted]
                intensities = [rec['兴趣强度'] for rec in records_sorted]
                user_id_int = self.user2id.get(user_id, 0)

                # 至少需要seq_len+1条记录（seq_len输入+1个目标）
                if len(book_ids) < self.seq_len + 1:
                    continue
                # 滑动窗口生成样本
                for i in range(len(book_ids) - self.seq_len):
                    seq_books = book_ids[i:i + self.seq_len]
                    seq_scenes = scenes[i:i + self.seq_len]
                    seq_intensities = intensities[i:i + self.seq_len]
                    target_book = book_ids[i + self.seq_len]
                    sequences.append((user_id_int, seq_books, seq_scenes, seq_intensities, target_book))
            return sequences

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            user_id_int, seq_books, seq_scenes, seq_intensities, target_book = self.data[idx]
            # 序列padding到固定长度
            seq_books = torch.tensor(seq_books + [0] * (self.seq_len - len(seq_books)), dtype=torch.long)
            seq_scenes = torch.tensor(seq_scenes + [0] * (self.seq_len - len(seq_scenes)), dtype=torch.long)
            seq_intensities = torch.tensor(seq_intensities + [0.0] * (self.seq_len - len(seq_intensities)),
                                           dtype=torch.float32)
            return (
                torch.tensor(user_id_int, dtype=torch.long),
                seq_books,
                seq_scenes,
                seq_intensities,
                torch.tensor(target_book, dtype=torch.long)
            )

    def train_temporal_model(self):
        """训练时序模型：仅用有交互用户的序列数据"""
        start = time.time()
        print("\n" + "=" * 50)
        print("【开始训练GRU时序模型（仅用有交互用户数据）】")
        print("=" * 50)

        # 1. 构建ID映射（仅包含有交互用户和有交互的图书）
        # 图书映射（仅包含有交互的图书）
        interacted_books = self.inter_df['book_id'].unique()
        self.book2id = {b: i + 1 for i, b in enumerate(interacted_books)}
        self.id2book = {i + 1: b for i, b in enumerate(interacted_books)}
        book_num = len(self.book2id) + 1

        # 用户映射（仅包含600个有交互用户）
        valid_users = list(self.user_val_data.keys())
        self.user2id = {u: i + 1 for i, u in enumerate(valid_users)}
        user_num = len(self.user2id) + 1

        # 2. 加载数据集（仅600个用户的训练数据）
        dataset = self.InterestSceneDataset(
            user_train_data=self.user_train_data,
            book2id=self.book2id,
            user2id=self.user2id,
            scene2id=self.scene2id,
            seq_len=self.seq_len
        )
        if len(dataset) == 0:
            raise ValueError("训练样本数为0，请检查有交互用户的交互记录（至少2次）")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        print(f"训练样本总数：{len(dataset)} | 总批次：{len(dataloader)}")
        print(f"有交互用户数：{len(valid_users)} | 有交互图书数：{len(interacted_books)}")

        # 3. 初始化模型
        self.temporal_model = self.InterestSceneGRU(
            user_num=user_num,
            book_num=book_num,
            scene_num=len(self.scene2id),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout=0.25
        )
        self.optimizer = torch.optim.Adam(self.temporal_model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()

        # 4. 模型训练（带批次进度条）
        epochs = 15
        self.temporal_model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs} | 训练批次"):
                user_ids, seq_books, seq_scenes, seq_intensities, targets = batch

                self.optimizer.zero_grad()
                logits = self.temporal_model(user_ids, seq_books, seq_scenes, seq_intensities)
                loss = self.criterion(logits, targets)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * user_ids.size(0)

            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch + 1}/{epochs} | 平均损失：{avg_loss:.4f} | 剩余轮次：{epochs - epoch - 1}")

        print(f"\n模型训练完成，总耗时：{time.time() - start:.2f}秒")
        print("=" * 50)

    # ------------------------------ 推荐逻辑（仅针对有交互的600个用户） ------------------------------
    def _get_user_avg_intensity(self, user_id, book_id):
        """计算有交互用户对某本图书的平均兴趣强度"""
        records = self.user_train_data.get(user_id, [])
        ints = [rec['兴趣强度'] for rec in records if rec['book_id'] == book_id]
        return np.mean(ints) if ints else 0.0

    def _get_interest_candidates(self, user_id):
        """为有交互用户生成高兴趣候选集（带进度条）"""
        candidates = set()
        records_sorted = sorted(self.user_train_data[user_id], key=lambda x: x['借阅时间'])
        user_high_cates = self.user_high_int_cates.get(user_id, ['未知分类'])
        current_scene = self._get_current_scene()

        # 分步骤生成候选集（带进度条）
        steps = [
            ("高兴趣分类图书", self._add_cate_candidates, (user_high_cates, current_scene)),
            ("高兴趣关键词图书", self._add_kw_candidates, (records_sorted, current_scene)),
            ("近期相似图书", self._add_similar_candidates, (records_sorted,))
        ]

        for step_name, step_func, step_args in tqdm(steps, desc=f"为用户{user_id}生成候选集"):
            step_candidates = step_func(*step_args)
            candidates.update(step_candidates)

        # 过滤低质量候选（已借阅、低兴趣强度）
        borrow_history = self.user_borrow_history.get(user_id, set())
        valid_candidates = []
        for b in tqdm(candidates, desc=f"过滤用户{user_id}候选集"):
            if b not in self.book_info or b in borrow_history:
                continue
            # 过滤低兴趣强度图书（仅保留≥0.2的图书，放宽条件提高候选集覆盖率）
            book_avg_int = self.inter_df[self.inter_df['book_id'] == b]['兴趣强度'].mean() if b in self.inter_df[
                'book_id'].unique() else 0.0
            if book_avg_int >= 0.2:
                valid_candidates.append(b)

        # 限制候选集大小（避免排序效率低）
        return valid_candidates[:80] if len(valid_candidates) >= 80 else valid_candidates

    # 候选集生成辅助函数
    def _add_cate_candidates(self, user_high_cates, current_scene):
        """添加高兴趣分类候选图书"""
        cate_candidates = set()
        for cate in user_high_cates:
            cate_books = list(self.secondary_cate_books.get(cate, set()))
            scene_cate_books = [b for b in cate_books if self._is_book_in_scene(b, current_scene)]
            cate_candidates.update(scene_cate_books[:20])
        return cate_candidates

    def _add_kw_candidates(self, records_sorted, current_scene):
        """添加高兴趣关键词候选图书"""
        kw_candidates = set()
        recent_kws = []
        for rec in records_sorted[-2:]:
            book_id = rec['book_id']
            recent_kws.extend(self.book_info[book_id]['核心关键词'].split()[:3])
        recent_kws = list(set(recent_kws))[:3]
        for kw in recent_kws:
            kw_books = list(self.keyword_books.get(kw, set()))
            scene_kw_books = [b for b in kw_books if self._is_book_in_scene(b, current_scene)]
            kw_candidates.update(scene_kw_books[:15])
        return kw_candidates

    def _add_similar_candidates(self, records_sorted):
        """添加近期相似候选图书"""
        similar_candidates = set()
        if len(records_sorted) >= 1:
            latest_book = records_sorted[-1]['book_id']
            latest_kws = self.book_info[latest_book]['核心关键词'].split()
            for kw in latest_kws:
                similar_books = list(self.keyword_books.get(kw, set()))[:10]
                similar_candidates.update(similar_books)
        return similar_candidates

    def _get_current_scene(self):
        """获取当前场景（基于数据集中最新借阅时间）"""
        current_month = self.latest_time.month
        return 0 if current_month in [1, 2, 7, 8] else 1 if current_month <= 6 else 2 if current_month <= 11 else 3

    def _is_book_in_scene(self, book_id, scene):
        """判断图书是否在目标场景中被借阅过"""
        book_inter = self.inter_df[self.inter_df['book_id'] == book_id]
        if len(book_inter) == 0:
            return True  # 无交互记录的图书默认场景匹配
        return scene in book_inter['学期场景'].unique()

    def recommend_for_user(self, user_id):
        """为有交互用户推荐（仅处理self.user_val_data中的用户）"""
        # 仅为有交互的用户提供推荐（过滤冷启动用户）
        if user_id not in self.user_val_data:
            raise ValueError(f"用户{user_id}无交互记录，不生成推荐")

        # 1. 强兴趣：重复借阅图书（优先级最高）
        user_repeat = [(b, cnt) for (u, b), cnt in self.user_book_repeat.items() if u == user_id]
        if user_repeat:
            return max(user_repeat, key=lambda x: x[1])[0]

        # 2. 高兴趣：续借≥1次且兴趣强度高的图书
        user_renew = [(b, cnt, self._get_user_avg_intensity(user_id, b))
                      for (u, b), cnt in self.user_book_renew.items() if u == user_id and cnt >= 1]
        if user_renew:
            # 按“续借次数×兴趣强度”排序，取得分最高的
            user_renew.sort(key=lambda x: x[1] * x[2], reverse=True)
            return user_renew[0][0]

        # 3. 时序模型推荐（核心逻辑）
        records_sorted = sorted(self.user_train_data[user_id], key=lambda x: x['借阅时间'])
        # 提取序列字段（图书、场景、兴趣强度）
        book_ids = [self.book2id.get(rec['book_id'], 0) for rec in records_sorted]
        scenes = [self.scene2id.get(rec['学期场景'], 0) for rec in records_sorted]
        intensities = [rec['兴趣强度'] for rec in records_sorted]

        # 截取最后seq_len条记录（不足则padding）
        seq_books = book_ids[-self.seq_len:] if len(book_ids) >= self.seq_len else book_ids + [0] * (
                    self.seq_len - len(book_ids))
        seq_scenes = scenes[-self.seq_len:] if len(scenes) >= self.seq_len else scenes + [0] * (
                    self.seq_len - len(scenes))
        seq_intensities = intensities[-self.seq_len:] if len(intensities) >= self.seq_len else intensities + [0.0] * (
                    self.seq_len - len(intensities))

        # 转换为张量
        seq_books = torch.tensor(seq_books, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
        seq_scenes = torch.tensor(seq_scenes, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
        seq_intensities = torch.tensor(seq_intensities, dtype=torch.float32).unsqueeze(0)  # (1, seq_len)
        user_id_int = torch.tensor(self.user2id.get(user_id, 0), dtype=torch.long).unsqueeze(0)  # (1,)

        # 模型预测
        self.temporal_model.eval()
        with torch.no_grad():
            logits = self.temporal_model(user_id_int, seq_books, seq_scenes, seq_intensities)
            probs = torch.softmax(logits, dim=1).squeeze(0)  # 概率分布

        # 4. 候选集与综合得分
        candidates = self._get_interest_candidates(user_id)
        if not candidates:
            # 候选集为空时，推荐用户高兴趣分类的热门图书
            user_high_cates = self.user_high_int_cates.get(user_id, ['未知分类'])
            fallback_books = []
            for cate in user_high_cates:
                fallback_books.extend(list(self.secondary_cate_books.get(cate, set()))[:10])
            fallback_books = [b for b in fallback_books if b not in self.user_borrow_history[user_id]]
            return fallback_books[0] if fallback_books else self.id2book[1]

        # 计算综合得分（模型分主导）
        candidate_scores = []
        for book_id in candidates:
            book_idx = self.book2id.get(book_id, 0)
            model_score = probs[book_idx].item() if book_idx < len(probs) else 0.0
            # 图书全局兴趣强度（辅助得分）
            book_int_score = self.book_sliding_int.get(book_id, 0.0)
            # 综合得分：模型分(0.8) + 兴趣强度分(0.2)
            total_score = model_score * 0.8 + book_int_score * 0.2
            candidate_scores.append((book_id, total_score))

        # 排序返回最优推荐
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return candidate_scores[0][0]

    # ------------------------------ 评估与输出（仅针对600个有交互用户） ------------------------------
    def evaluate(self):
        """评估有交互用户的推荐效果（带进度条）"""
        start = time.time()
        print("\n" + "=" * 50)
        print("【开始评估有交互用户的推荐效果】")
        print("=" * 50)
        total = len(self.user_val_data)  # 仅评估600个有交互用户
        if total == 0:
            print("无有交互用户，评估终止")
            return 0.0

        total_hit = 0
        interest_hit = 0  # 兴趣匹配命中数
        scene_hit = 0  # 场景匹配命中数
        candidate_coverage = 0

        # 遍历有交互用户评估（带进度条）
        for user_id, true_book in tqdm(self.user_val_data.items(), total=total, desc="评估有交互用户"):
            # 1. 检查候选集是否覆盖真实图书
            candidates = self._get_interest_candidates(user_id)
            if str(true_book) in candidates:
                candidate_coverage += 1

            # 2. 预测与命中判断
            pred_book = self.recommend_for_user(user_id)
            if pred_book and str(pred_book) == str(true_book):
                total_hit += 1

                # 3. 兴趣与场景匹配判断
                user_high_cates = self.user_high_int_cates.get(user_id, [])
                true_book_cate = self.book_info.get(str(true_book), {}).get('二级分类', '未知')
                if true_book_cate in user_high_cates:
                    interest_hit += 1

                current_scene = self._get_current_scene()
                if self._is_book_in_scene(str(true_book), current_scene):
                    scene_hit += 1

        # 计算核心评估指标
        precision = total_hit / total if total > 0 else 0.0
        recall = total_hit / total if total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-9)  # 避免分母为0
        coverage_rate = candidate_coverage / total if total > 0 else 0.0
        interest_match_rate = interest_hit / total_hit if total_hit > 0 else 0.0
        scene_match_rate = scene_hit / total_hit if total_hit > 0 else 0.0

        # 打印评估报告（重点突出600个用户的效果）
        print("\n【有交互用户推荐效果评估报告】")
        print(f"有交互用户总数：{total}（目标600个）")
        print(f"总命中数：{total_hit} | 命中率：{total_hit / total:.2%}")
        print(f"F1值：{f1:.4f} | 候选集覆盖率：{coverage_rate:.2%}")
        print(f"兴趣匹配命中率：{interest_match_rate:.2%} | 场景匹配命中率：{scene_match_rate:.2%}")
        print(f"评估耗时：{time.time() - start:.2f}秒")
        print("=" * 50)

        return f1

    def generate_outputs(self):
        """生成推荐结果文件（仅包含600个有交互用户）"""
        start = time.time()
        print("\n" + "=" * 50)
        print("【开始生成有交互用户的推荐结果】")
        print("=" * 50)

        # 1. 生成详细推荐结果（含预测/真实图书、命中标记）
        print("1/2 生成详细推荐结果（仅600个有交互用户）...")
        title_map = {b: self.book_info[b]['题名'] for b in self.book_info}
        recommendations = []
        # 遍历有交互用户生成详细结果（带进度条）
        for user_id, true_book in tqdm(self.user_val_data.items(), total=len(self.user_val_data),
                                       desc="  处理有交互用户结果"):
            pred_book = self.recommend_for_user(user_id)
            # 兴趣/场景匹配标记
            user_high_cates = self.user_high_int_cates.get(user_id, [])
            true_cate = self.book_info.get(str(true_book), {}).get('二级分类', '未知')
            pred_cate = self.book_info.get(pred_book, {}).get('二级分类', '未知')
            true_interest_match = 1 if true_cate in user_high_cates else 0
            pred_interest_match = 1 if pred_cate in user_high_cates else 0

            current_scene = self._get_current_scene()
            true_scene_match = 1 if self._is_book_in_scene(str(true_book), current_scene) else 0
            pred_scene_match = 1 if self._is_book_in_scene(pred_book, current_scene) else 0

            recommendations.append({
                'user_id': user_id,
                'pred_book_id': pred_book,
                'true_book_id': true_book,
                'pred_title': title_map.get(pred_book, '未知图书'),
                'true_title': title_map.get(str(true_book), '未知图书'),
                'is_hit': 1 if str(pred_book) == str(true_book) else 0,
                'pred_interest_match': pred_interest_match,
                'pred_scene_match': pred_scene_match,
                'true_interest_match': true_interest_match,
                'true_scene_match': true_scene_match
            })
        # 保存详细结果
        detail_df = pd.DataFrame(recommendations)
        detail_df.to_csv('interacted_user_recommendation_detail.csv', index=False, encoding='utf-8-sig')
        print(f"  详细结果已保存：interacted_user_recommendation_detail.csv")

        # 2. 生成提交文件（仅600个有交互用户）
        print("\n2/2 生成提交文件（仅600个有交互用户）...")
        submission = []
        # 仅遍历有交互用户生成提交数据（带进度条）
        for user_id in tqdm(self.user_val_data.keys(), total=len(self.user_val_data), desc="  生成提交数据"):
            pred_book = self.recommend_for_user(user_id)
            submission.append({
                'user_id': user_id,
                'book_id': pred_book if pred_book else ''
            })
        # 保存提交文件（确保列顺序正确）
        submission_df = pd.DataFrame(submission)
        submission_df = submission_df[['user_id', 'book_id']].drop_duplicates(subset='user_id')
        submission_df.to_csv('interacted_user_recommendation_submission.csv', index=False, encoding='utf-8-sig')
        print(f"  提交文件已保存：interacted_user_recommendation_submission.csv")

        print(f"\n输出文件生成完成，总耗时：{time.time() - start:.2f}秒")
        print("=" * 50)

    @classmethod
    def run(cls, inter_path, book_path, user_path):
        """完整运行流程（仅处理有交互的600个用户）"""
        start = time.time()
        print("=" * 60)
        print("【有交互用户时序推荐系统启动】")
        print(f"数据路径：")
        print(f"- 交互数据：{inter_path}")
        print(f"- 图书数据：{book_path}")
        print(f"- 用户数据：{user_path}")
        print("核心目标：仅为有交互的600个用户生成推荐")
        print("=" * 60)

        # 初始化推荐器
        recommender = cls()
        # 加载数据（仅保留有交互用户）
        recommender.load_data(inter_path, book_path, user_path)
        # 训练模型（仅用有交互用户数据）
        recommender.train_temporal_model()
        # 生成推荐结果（仅600个有交互用户）
        recommender.generate_outputs()
        # 评估效果（仅评估有交互用户）
        f1 = recommender.evaluate()

        print(f"\n【系统运行完成】")
        print(f"总耗时：{time.time() - start:.2f}秒")
        print(f"有交互用户数：{len(recommender.user_val_data)}（目标600个）")
        print(f"最终F1值：{f1:.4f}")
        print("=" * 60)

        return recommender, f1


# ------------------------------ 运行入口 ------------------------------
if __name__ == "__main__":
    # 数据路径（请确认与你的数据集实际路径一致）
    INTER_PATH = './datasets/inter_400.csv'
    BOOK_PATH = './datasets/item.csv'
    USER_PATH = './datasets/pre_user.csv'

    # 启动系统（仅为有交互的600个用户生成推荐）
    _, final_f1 = InterestSceneTemporalRecommender.run(INTER_PATH, BOOK_PATH, USER_PATH)