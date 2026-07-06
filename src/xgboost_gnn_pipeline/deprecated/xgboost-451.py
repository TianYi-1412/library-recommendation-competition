# v1_gnn_minimal.py
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
from collections import defaultdict, Counter
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from gnn_feature import GNNEmbedder  # 仅新增

warnings.filterwarnings('ignore')

class PureNewBreakthroughRecommenderV1GNN:
    def __init__(self):
        # 原始结构保留
        self.user_borrow_seq = defaultdict(list)
        self.user_train_data = defaultdict(list)
        self.user_val_data = {}
        self.user_borrow_history = defaultdict(set)

        self.book_info = {}
        self.secondary_cate_books = defaultdict(set)
        self.keyword_books = defaultdict(set)
        self.book_hot = defaultdict(float)
        self.user_profile = {}
        self.user_dept_top_cates = {}
        self.user_core_keywords = defaultdict(list)
        self.user_secondary_cates = defaultdict(list)
        self.user_book_renew = {}
        self.user_book_repeat = {}

        # === 新增：demo 嵌入 ===
        self.user_gnn_embed = {}
        self.book_gnn_embed = {}

        self.scaler_hot = MinMaxScaler()
        self.model = xgb.XGBClassifier(
            n_estimators=250, max_depth=2, learning_rate=0.01,
            min_child_weight=10, scale_pos_weight=8, reg_lambda=2.0, random_state=42
        )

        # 原超参
        self.time_decay_half_life = 90
        self.latest_time = None
        self.hot_decay = 0.15
        self.weak_feat_boost = 6
        self.core_kw_min_len = 2
        self.stop_words = {'高等', '教程', '基础', '入门'}
        self.high_hot_threshold = 90
        self.renew_score_upper = 1.5

    # -------------------- 数据加载（仅新增GNN） --------------------
    def load_data(self, inter_path, book_path, user_path):
        start = time.time()
        self.inter_df = pd.read_csv(inter_path, usecols=['user_id', 'book_id', '借阅时间', '续借次数'], parse_dates=['借阅时间'])
        self.book_df = pd.read_csv(book_path, usecols=['book_id', '一级分类', '二级分类', '分词题名', '题名'])
        self.user_df = pd.read_csv(user_path, usecols=['借阅人', 'DEPT'])

        self._unify_ids()
        self._clean_text()
        self._extract_core_keywords()
        self.latest_time = self.inter_df['借阅时间'].max()
        self._build_book_mappings()
        self._build_user_mappings()
        self._split_loo_train_val()

        # === 新增：demo 嵌入 ===
        print("⏳ 构建 demo 嵌入...")
        gnn = GNNEmbedder(self.inter_df, self.book_df, self.user_df)
        self.user_gnn_embed, self.book_gnn_embed = gnn.export()
        print("✅ demo 嵌入完成")

        print(f"数据加载完成，耗时：{time.time() - start:.2f}秒")

    # -------------------- 原函数不变（省略复制） --------------------
    def _unify_ids(self):
        for col in ['user_id', 'book_id']:
            self.inter_df[col] = self.inter_df[col].astype(str)
        self.book_df['book_id'] = self.book_df['book_id'].astype(str)
        self.user_df['借阅人'] = self.user_df['借阅人'].astype(str)
        self.user_df.rename(columns={'借阅人': 'user_id'}, inplace=True)

    def _clean_text(self):
        for col in ['一级分类', '二级分类', '分词题名', '题名']:
            self.book_df[col] = self.book_df[col].astype(str).str.replace(r'[\n\t\s]', '', regex=True).str.lower()
        self.user_df['DEPT'] = self.user_df['DEPT'].astype(str).str.replace(r'[\n\t]', '', regex=True).str.lower()

    def _extract_core_keywords(self):
        self.book_df['核心关键词'] = self.book_df['分词题名'].apply(
            lambda x: ' '.join([w for w in x.split() if len(w) >= self.core_kw_min_len and w not in self.stop_words])
        )

    def _build_book_mappings(self):
        self.book_info = self.book_df.set_index('book_id')[['一级分类', '二级分类', '核心关键词', '题名']].to_dict('index')
        self.secondary_cate_books = self.book_df.groupby('二级分类')['book_id'].apply(set).to_dict()
        for _, row in self.book_df.iterrows():
            for kw in row['核心关键词'].split():
                self.keyword_books[kw].add(row['book_id'])
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

    def _split_loo_train_val(self):
        inter_sorted = self.inter_df.sort_values(['user_id', '借阅时间'])
        for user_id, group in inter_sorted.groupby('user_id'):
            if len(group) < 2: continue
            val_row = group.tail(1).iloc[0]
            self.user_val_data[user_id] = val_row['book_id']
            train_rows = group.head(-1)
            self.user_train_data[user_id] = train_rows.to_dict('records')
            self.user_borrow_history[user_id] = set(group['book_id'])
        self._compute_time_decay_preferences()
        self._compute_train_renew_repeat(inter_sorted)

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

    # -------------------- 特征：7 维 → 10 维 --------------------
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
        self.model.fit(features, labels)
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

        # === 新增 3 维 ===
        # 1. demo 嵌入相似度
        user_vec = self.user_gnn_embed.get(user_id, np.zeros(64))
        book_vec = self.book_gnn_embed.get(book_id, np.zeros(64))
        gnn_sim = float(np.dot(user_vec, book_vec).sum())

        # 2. 书名关键词匹配（新增字段）
        title_kws = set(book_info.get('题名', '').split())
        user_kws_set = set(self.user_core_keywords.get(user_id, []))
        title_kw_match = 1 if title_kws & user_kws_set else 0

        # 3. 主题标签匹配（用一级分类当主题）
        theme_match = 1 if book_info.get('一级分类') in self.user_dept_top_cates.get(self.user_profile.get(user_id, '未知'), []) else 0

        return [renew_score, hot_score, secondary_match, dept_match, keyword_match,
                dept_secondary_cross, keyword_secondary_cross,
                gnn_sim, title_kw_match, theme_match]

    # -------------------- 推荐 & 评估 & 保存 --------------------
    def recommend_for_user(self, user_id):
        user_repeat = [(b, cnt) for (u, b), cnt in self.user_book_repeat.items() if u == user_id]
        if user_repeat:
            return max(user_repeat, key=lambda x: x[1])[0]

        user_dept = self.user_profile.get(user_id, '未知')
        user_secondary = self.user_secondary_cates.get(user_id, [])
        user_kws = self.user_core_keywords.get(user_id, [])
        dept_top = self.user_dept_top_cates.get(user_dept, [])

        candidates = set()
        for cate in user_secondary:
            candidates.update(self.secondary_cate_books.get(cate, []))
        for cate in dept_top:
            candidates.update([b for b, info in self.book_info.items() if info.get('一级分类') == cate])
        for kw in user_kws:
            candidates.update(self.keyword_books.get(kw, []))

        candidates = [b for b in candidates if b not in self.user_borrow_history[user_id]]
        if not candidates:
            return None

        features = [self._extract_feature_vector(user_id, b) for b in candidates]
        scores = self.model.predict_proba(features)[:, 1]
        return candidates[np.argmax(scores)]

    def evaluate(self):
        start = time.time()
        total = len(self.user_val_data)
        total_hit = 0
        for user_id, true_book in tqdm(self.user_val_data.items(), desc="评估中"):
            pred_book = self.recommend_for_user(user_id)
            if pred_book and pred_book == str(true_book):
                total_hit += 1
        precision = total_hit / total if total > 0 else 0
        recall = total_hit / total if total > 0 else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        print(f"【评估结果】命中率：{precision:.2%} | F1：{f1:.4f} | 耗时：{time.time() - start:.2f}秒")
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
        df.to_csv('v2.csv', index=False, encoding='utf-8-sig')
        print(f"✅ 推荐结果已保存为 v2.csv，耗时：{time.time() - start:.2f}秒")

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
    PureNewBreakthroughRecommenderV1GNN.main(
        inter_path='./datasets/inter_451.csv',
        book_path='./datasets/item.csv',
        user_path='./datasets/pre_user.csv'
    )