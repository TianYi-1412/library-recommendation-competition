# v8.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings
from collections import defaultdict, Counter
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from transformer_v8 import LightTransformer
from graph_utils_v8 import BipartiteGraph
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

class PureNewBreakthroughV8:
    def __init__(self):
        # 数据
        self.user_train_data = defaultdict(list)
        self.user_val_data = {}
        self.user_borrow_history = defaultdict(set)
        self.book_info = {}
        self.book_hot = defaultdict(float)
        self.user_profile = {}
        self.user_dept_top_cates = {}
        self.user_secondary_cates = defaultdict(list)
        self.user_core_keywords = defaultdict(list)
        self.user_book_renew = {}
        self.user_book_repeat = {}

        # 多路召回（必须初始化）
        self.author_books = defaultdict(set)
        self.publisher_books = defaultdict(set)
        self.detail_cate_books = defaultdict(set)
        self.compound_cate_books = defaultdict(set)
        self.theme_books = defaultdict(set)
        self.keyword_books = defaultdict(set)

        # 图结构
        self.user2idx = {}
        self.book2idx = {}
        self.edges = []
        self.user_embed = None
        self.book_embed = None

        # 模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_hot = MinMaxScaler()

        # 超参
        self.time_decay_half_life = 90
        self.latest_time = None
        self.hot_decay = 0.15
        self.weak_feat_boost = 8
        self.core_kw_min_len = 2
        self.stop_words = {'高等', '教程', '基础', '入门'}
        self.renew_score_upper = 1.5

    # -------------------- 数据加载 --------------------
    def load_data(self, inter_path, book_path, user_path):
        start = time.time()
        self.inter_df = pd.read_csv(inter_path, usecols=['user_id', 'book_id', '借阅时间', '续借次数'], parse_dates=['借阅时间'])
        self.book_df = pd.read_csv(book_path)
        self.user_df = pd.read_csv(user_path, usecols=['借阅人', 'DEPT'])

        self._unify_ids()
        self._clean_text()
        self._extract_core_keywords()
        self.latest_time = self.inter_df['借阅时间'].max()
        self._build_book_mappings()
        self._build_user_mappings()
        self._expand_validation_set()
        self._build_bipartite_graph()
        print(f"数据加载完成，耗时：{time.time() - start:.2f}秒")

    def _unify_ids(self):
        for col in ['user_id', 'book_id']:
            self.inter_df[col] = self.inter_df[col].astype(str)
        self.book_df['book_id'] = self.book_df['book_id'].astype(str)
        self.user_df['借阅人'] = self.user_df['借阅人'].astype(str)
        self.user_df.rename(columns={'借阅人': 'user_id'}, inplace=True)

    def _clean_text(self):
        for col in ['一级分类', '二级分类', '分词题名', '题名', '作者', '出版社', '细化分类', '复合分类', '书名关键词', '主题标签']:
            self.book_df[col] = self.book_df[col].astype(str).str.replace(r'[\n\t\s]', '', regex=True).str.lower()

    def _extract_core_keywords(self):
        self.book_df['核心关键词'] = self.book_df['书名关键词'].fillna('').str.lower()

    def _build_book_mappings(self):
        self.book_info = self.book_df.set_index('book_id')[['一级分类', '二级分类', '核心关键词', '题名', '作者', '出版社', '细化分类', '复合分类', '主题标签']].to_dict('index')
        self.secondary_cate_books = self.book_df.groupby('二级分类')['book_id'].apply(set).to_dict()
        for _, row in self.book_df.iterrows():
            for kw in row['核心关键词'].split():
                self.keyword_books[kw].add(row['book_id'])
            self.author_books[row['作者']].add(row['book_id'])
            self.publisher_books[row['出版社']].add(row['book_id'])
            self.detail_cate_books[row['细化分类']].add(row['book_id'])
            self.compound_cate_books[row['复合分类']].add(row['book_id'])
            for theme in row['主题标签'].split(' '):
                self.theme_books[theme].add(row['book_id'])

        for _, row in self.inter_df.iterrows():
            days_since = (self.latest_time - row['借阅时间']).days
            self.book_hot[row['book_id']] += np.exp(-days_since / self.time_decay_half_life)
        hot_vals = np.array(list(self.book_hot.values())).reshape(-1, 1)
        self.scaler_hot.fit(hot_vals)

    def _build_user_mappings(self):
        self.user_profile = self.user_df.set_index('user_id')['DEPT'].to_dict()
        inter_dept_cate = self.inter_df.merge(self.user_df[['user_id', 'DEPT']], on='user_id').merge(
            self.book_df[['book_id', '一级分类']], on='book_id')
        dept_cate_cnt = inter_dept_cate.groupby(['DEPT', '一级分类']).size().to_dict()
        for dept in self.user_df['DEPT'].unique():
            dept_cates = [(cate, cnt) for (d, cate), cnt in dept_cate_cnt.items() if d == dept]
            dept_cates.sort(key=lambda x: x[1], reverse=True)
            self.user_dept_top_cates[dept] = [cate for cate, _ in dept_cates[:2]]

    def _expand_validation_set(self):
        self.user_val_data = {}
        self.val_labels = {}
        inter_sorted = self.inter_df.sort_values(['user_id', '借阅时间'])

        for user_id, group in inter_sorted.groupby('user_id'):
            # 1. 正样本：最后一次借阅
            val_row = group.tail(1).iloc[0]
            true_book = val_row['book_id']

            # 2. 生成 3 个负样本（保底：同类别 → 热门补位）
            user_borrowed = set(group['book_id'])
            same_cate_books = [
                b for b, info in self.book_info.items()
                if info.get('二级分类') == self.book_info.get(true_book, {}).get('二级分类', '')
                   and b not in user_borrowed
            ]
            # 不足 3 本 → 用热门书补位
            if len(same_cate_books) < 3:
                hot_books = [b for b, _ in sorted(self.book_hot.items(), key=lambda x: x[1], reverse=True)[:100]
                             if b not in user_borrowed]
                same_cate_books += hot_books[:3 - len(same_cate_books)]

            neg_books = np.random.choice(same_cate_books, 3, replace=False)

            # 3. 记录正负样本
            self.user_val_data[user_id] = [true_book] + list(neg_books)
            self.val_labels[user_id] = [1, 0, 0, 0]

            # 4. 训练集 = 剩余记录
            train_rows = group.head(-1)
            self.user_train_data[user_id] = train_rows.to_dict('records')
            self.user_borrow_history[user_id] = set(group['book_id'])

    def _compute_time_decay_preferences(self):
        for user_id, records in self.user_train_data.items():
            secondary_counter = Counter()
            keyword_counter = Counter()
            for r in records:
                days_since = (self.latest_time - r['借阅时间']).days
                weight = np.exp(-days_since / self.time_decay_half_life)
                book_info = self.book_info.get(r['book_id'], {})
                secondary_counter[book_info.get('二级分类', '未知')] += weight
                for kw in book_info.get('核心关键词', '').split():
                    keyword_counter[kw] += weight
            self.user_secondary_cates[user_id] = [cate for cate, _ in secondary_counter.most_common(5)]
            self.user_core_keywords[user_id] = [kw for kw, _ in keyword_counter.most_common(7)]

    def _compute_train_renew_repeat(self, inter_sorted):
        train_users = set(self.user_train_data.keys())
        train_inter = inter_sorted[inter_sorted['user_id'].isin(train_users)]
        self.user_book_renew = train_inter.groupby(['user_id', 'book_id'])['续借次数'].sum().to_dict()
        borrow_cnt = train_inter.groupby(['user_id', 'book_id']).size()
        self.user_book_repeat = borrow_cnt[borrow_cnt >= 2].to_dict()

    # -------------------- 二分图 + Transformer --------------------
    def _build_bipartite_graph(self):
        print("⏳ 构建二分图 + Transformer...")
        # 建索引
        users = sorted(self.inter_df['user_id'].unique())
        books = sorted(self.inter_df['book_id'].unique())
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.book2idx = {b: i for i, b in enumerate(books)}
        self.edges = [(self.user2idx[u], self.book2idx[b]) for u, b in zip(self.inter_df['user_id'], self.inter_df['book_id'])]

        # 二分图 + PCA 嵌入
        self.graph = BipartiteGraph(self.edges, len(users), len(books), embed_dim=64)
        self.user_embed = self.graph.user_embed
        self.book_embed = self.graph.book_embed
        print("✅ 二分图完成")

    # -------------------- 特征：行为 + 图 + 内容 --------------------
    def build_train_features(self):
        start = time.time()
        features, labels = [], []
        for user_id, records in self.user_train_data.items():
            for r in records:
                book_id = r['book_id']
                feat = self._extract_feature_vector(user_id, book_id)
                label = 1 if (user_id, book_id) in self.user_book_repeat else 0
                features.append(feat)
                labels.append(label)
        self._build_transformer_model(features, labels)
        print(f"特征构建完成，训练样本数：{len(features)}，耗时：{time.time() - start:.2f}秒")
        return np.array(features), np.array(labels)

    def _extract_feature_vector(self, user_id, book_id):
        book_info = self.book_info.get(book_id, {})
        renew_cnt = self.user_book_renew.get((user_id, book_id), 0)
        renew_score = min(renew_cnt / self.renew_score_upper, 1.0)
        hot_raw = self.book_hot.get(book_id, 0)
        hot_score = (self.scaler_hot.transform([[hot_raw]])[0][0] * self.hot_decay) / 2

        secondary_match = 1 if book_info.get('二级分类') in self.user_secondary_cates.get(user_id, []) else 0
        dept_match = 1 if book_info.get('一级分类') in self.user_dept_top_cates.get(self.user_profile.get(user_id, '未知'), []) else 0
        keyword_match = 1 if any(kw in self.user_core_keywords.get(user_id, []) for kw in book_info.get('核心关键词', '').split()) else 0

        secondary_match *= self.weak_feat_boost
        dept_match *= self.weak_feat_boost
        keyword_match *= self.weak_feat_boost

        dept_secondary_cross = (dept_match * secondary_match) / self.weak_feat_boost * 2
        keyword_secondary_cross = (keyword_match * secondary_match) / self.weak_feat_boost * 2

        # 图嵌入（64D → 取均值）
        user_idx = self.user2idx.get(user_id, 0)
        book_idx = self.book2idx.get(book_id, 0)
        user_vec = self.user_embed[user_idx]
        book_vec = self.book_embed[book_idx]
        graph_sim = float(np.dot(user_vec, book_vec))

        # 内容相似（作者 + 主题）
        author = book_info.get('作者', '')
        publisher = book_info.get('出版社', '')
        theme_tags = set(book_info.get('主题标签', '').split())
        user_authors = set(self._user_authors(user_id))
        user_publishers = set(self._user_publishers(user_id))
        user_themes = set(self._user_themes(user_id))

        author_match = 1 if author in user_authors else 0
        publisher_match = 1 if publisher in user_publishers else 0
        theme_tag_match = 1 if theme_tags & user_themes else 0

        # 17 维类别（哈希桶 5 万）
        cat_feats = [hash(author) % 50000, hash(publisher) % 50000, hash(book_info.get('一级分类', '')) % 50000,
                     hash(book_info.get('二级分类', '')) % 50000, hash(tuple(theme_tags)) % 50000,
                     hash(user_id) % 50000, hash(book_id) % 50000, hash(tuple(user_themes)) % 50000,
                     hash(book_info.get('细化分类', '')) % 50000, hash(book_info.get('复合分类', '')) % 50000,
                     hash(tuple(set(book_info.get('书名关键词', '').split()))) % 50000,
                     hash(tuple(self.user_core_keywords.get(user_id, []))) % 50000,
                     hash(self.user_profile.get(user_id, '')) % 50000, hash(author + publisher) % 50000,
                     hash(book_info.get('细化分类', '') + book_info.get('复合分类', '')) % 50000,
                     hash(tuple(theme_tags) + tuple(user_themes)) % 50000, hash(author + book_id) % 50000]

        return [renew_score, hot_score, secondary_match, dept_match, keyword_match,
                dept_secondary_cross, keyword_secondary_cross,
                graph_sim, author_match, publisher_match, theme_tag_match] + cat_feats  # 11 + 17 = 28 维

    # -------------------- 用户偏好缓存 --------------------
    def _user_authors(self, user_id):
        if not hasattr(self, '_ua_cache'):
            self._ua_cache = defaultdict(set)
            for r in self.user_train_data.get(user_id, []):
                self._ua_cache[user_id].add(self.book_info.get(r['book_id'], {}).get('作者', ''))
        return self._ua_cache[user_id]

    def _user_publishers(self, user_id):
        if not hasattr(self, '_up_cache'):
            self._up_cache = defaultdict(set)
            for r in self.user_train_data.get(user_id, []):
                self._up_cache[user_id].add(self.book_info.get(r['book_id'], {}).get('出版社', ''))
        return self._up_cache[user_id]

    def _user_themes(self, user_id):
        if not hasattr(self, '_ut_cache'):
            self._ut_cache = defaultdict(set)
            for r in self.user_train_data.get(user_id, []):
                for t in self.book_info.get(r['book_id'], {}).get('主题标签', '').split(' '):
                    self._ut_cache[user_id].add(t)
        return self._ut_cache[user_id]

    # -------------------- 用户偏好缓存（缺失函数）--------------------
    def _user_detail_cates(self, user_id):
        if not hasattr(self, '_ud_cache'):
            self._ud_cache = defaultdict(set)
            for r in self.user_train_data.get(user_id, []):
                self._ud_cache[user_id].add(self.book_info.get(r['book_id'], {}).get('细化分类', ''))
        return self._ud_cache[user_id]

    def _user_compound_cates(self, user_id):
        if not hasattr(self, '_uc_cache'):
            self._uc_cache = defaultdict(set)
            for r in self.user_train_data.get(user_id, []):
                self._uc_cache[user_id].add(self.book_info.get(r['book_id'], {}).get('复合分类', ''))
        return self._uc_cache[user_id]

    def _user_themes(self, user_id):
        if not hasattr(self, '_ut_cache'):
            self._ut_cache = defaultdict(set)
            for r in self.user_train_data.get(user_id, []):
                for t in self.book_info.get(r['book_id'], {}).get('主题标签', '').split(' '):
                    self._ut_cache[user_id].add(t)
        return self._ut_cache[user_id]

    # -------------------- Transformer 训练 --------------------
    def _build_transformer_model(self, features, labels):
        start = time.time()
        # 构造节点序列（用户-图书交互）
        user_ids = [self.user2idx.get(self.inter_df.iloc[i]['user_id'], 0) for i in range(len(self.inter_df))]
        book_ids = [self.book2idx.get(self.inter_df.iloc[i]['book_id'], 0) for i in range(len(self.inter_df))]
        node_seq = np.column_stack([user_ids, book_ids])  # [N, 2]

        # 轻量 Transformer
        num_nodes = len(self.user2idx) + len(self.book2idx)
        self.transformer = LightTransformer(num_nodes, embed_dim=64, num_layers=2).to(self.device)
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-3)

        seq_tensor = torch.tensor(node_seq, dtype=torch.long).to(self.device)
        y_tensor = torch.tensor([1] * len(node_seq), dtype=torch.float32).to(self.device)

        self.transformer.train()
        for epoch in range(10):
            optimizer.zero_grad()
            out = self.transformer(seq_tensor).squeeze()
            loss = nn.BCELoss()(out, y_tensor)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                print(f"   Transformer epoch {epoch} | loss {loss.item():.4f}")
        self.transformer.eval()
        print("✅ Transformer 训练完成")

    # -------------------- 推荐 --------------------
    def recommend_for_user(self, user_id):
        user_repeat = [(b, cnt) for (u, b), cnt in self.user_book_repeat.items() if u == user_id]
        if user_repeat:
            return max(user_repeat, key=lambda x: x[1])[0]

        user_dept = self.user_profile.get(user_id, '未知')
        user_secondary = self.user_secondary_cates.get(user_id, [])
        user_kws = self.user_core_keywords.get(user_id, [])
        dept_top = self.user_dept_top_cates.get(user_dept, [])

        # 12 路召回（不裁剪）
        candidates = set()
        for cate in user_secondary:
            candidates.update(self.secondary_cate_books.get(cate, []))
        for cate in dept_top:
            candidates.update([b for b, info in self.book_info.items() if info.get('一级分类') == cate])
        for kw in user_kws:
            candidates.update(self.keyword_books.get(kw, []))
        for a in self._user_authors(user_id):
            candidates.update(self.author_books.get(a, []))
        for p in self._user_publishers(user_id):
            candidates.update(self.publisher_books.get(p, []))
        for d in self._user_detail_cates(user_id):
            candidates.update(self.detail_cate_books.get(d, []))
        for c in self._user_compound_cates(user_id):
            candidates.update(self.compound_cate_books.get(c, []))
        for t in self._user_themes(user_id):
            candidates.update(self.theme_books.get(t, []))

        candidates = [b for b in candidates if b not in self.user_borrow_history[user_id]]
        if not candidates:
            return None

        # 构造特征 + Transformer 打分
        feats = np.array([self._extract_feature_vector(user_id, b) for b in candidates])
        base_new = feats[:, :11].astype(np.float32)  # 11 维基础
        cat_feats = feats[:, -17:].astype(np.int32)  # 17 维类别

        with torch.no_grad():
            node_seq = np.column_stack([[self.user2idx.get(user_id, 0)] * len(candidates),
                                        [self.book2idx.get(b, 0) for b in candidates]])
            node_tensor = torch.tensor(node_seq, dtype=torch.long).to(self.device)
            deep_score = self.transformer(node_tensor).cpu().numpy().squeeze()

        # 加权融合（可调）
        final_score = 0.6 * deep_score + 0.4 * cosine_similarity([self.user_embed[self.user2idx.get(user_id, 0)]],
                                                                  self.book_embed[[self.book2idx.get(b, 0) for b in candidates]])[0]
        return candidates[np.argmax(final_score)]

    # -------------------- 全量评估 + 实时进度 --------------------
    def evaluate(self):
        start = time.time()
        total = len(self.user_val_data)
        total_hit = 0
        check_step = max(100, total // 20)  # 每 5% 打印一次
        pbar = tqdm(self.user_val_data.items(), desc="全量评估中", total=total, mininterval=1.0)

        for i, (user_id, candidate_books) in enumerate(pbar, 1):
            true_book = candidate_books[0]
            pred_book = self.recommend_for_user(user_id)
            if pred_book == true_book:
                total_hit += 1

            # 每 100 用户打印一次实时命中率
            if i % check_step == 0 or i == total:
                current_precision = total_hit / i if i > 0 else 0
                pbar.set_postfix({"实时命中率": f"{current_precision:.2%}"})

        precision = total_hit / total if total > 0 else 0
        recall = total_hit / total if total > 0 else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        print(f"【全量评估】命中率：{precision:.2%} | F1：{f1:.4f} | 耗时：{time.time() - start:.2f}秒")
        return f1

    def save_recommendations(self):
        start = time.time()
        recommendations = []
        title_map = {b: info['题名'] for b, info in self.book_info.items()}
        for user_id in self.user_val_data.keys():
            pred_book = self.recommend_for_user(user_id)
            recommendations.append({
                'user_id': user_id,
                'book_id': pred_book if pred_book else '',
                'title': title_map.get(pred_book, '未知')
            })
        df = pd.DataFrame(recommendations)
        df.to_csv('v8_recommendations.csv', index=False, encoding='utf-8-sig')
        print(f"✅ 推荐结果已保存为 v8_recommendations.csv，耗时：{time.time() - start:.2f}秒")

    @classmethod
    def main(cls, inter_path, book_path, user_path):
        start = time.time()
        rec = cls()
        rec.load_data(inter_path, book_path, user_path)
        rec.build_train_features()
        rec.evaluate()
        rec.save_recommendations()
        print(f"总耗时：{time.time() - start:.2f}秒")
        return

if __name__ == "__main__":
    PureNewBreakthroughV8.main(
        inter_path='../datasets/pre_inter.csv',
        book_path='../datasets/item.csv',
        user_path='../datasets/pre_user.csv'
    )