import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lightgbm import LGBMRanker
import faiss
import torch.nn.functional as F
from user_interesting import *  # 确保导入正确的文本特征模块

# -------------------------- 全局配置 --------------------------
SEED = 42
BATCH = 256
EPOCHS = 8
LR = 2e-4
WEIGHT_DECAY = 1e-4
RECALL_TOP_K = 180
RERANK_TOP_K = 80
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化种子
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


# -------------------------- 1. 召回模型：Two-Tower + Transformer --------------------------
class SeqDataset(Dataset):
    """数据集类：适配用户-物品-序列输入"""

    def __init__(self, train_data, user_seq_dict):
        self.train_data = train_data  # 格式：[(user_idx, item_idx, label), ...]
        self.user_seq_dict = user_seq_dict  # 用户序列字典

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        u, i, l = self.train_data[idx]
        seq = self.user_seq_dict[u]  # 获取用户u的历史序列
        return torch.tensor(u), torch.tensor(i), torch.tensor(l, dtype=torch.float), seq


class UserTower(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, pad_idx, hist_max,
                 num_genders, num_grades, num_depts, num_types):
        super().__init__()
        # 基础embedding层
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.seq_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=pad_idx)

        # 多粒度序列编码 - 修复维度问题
        self.attention_pool = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)

        # GRU输出维度调整为emb_dim，避免双向导致的维度翻倍
        self.gru = nn.GRU(emb_dim, emb_dim // 2, batch_first=True, bidirectional=True)

        # 用户特征融合
        self.gender_emb = nn.Embedding(num_genders, emb_dim // 4)
        self.grade_emb = nn.Embedding(num_grades, emb_dim // 4)
        self.dept_emb = nn.Embedding(num_depts, emb_dim // 4)
        self.type_emb = nn.Embedding(num_types, emb_dim // 4)

        # 特征交叉网络
        self.cross_net = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim * 2, emb_dim)
        )

        # 投影层确保维度一致
        self.gru_proj = nn.Linear(emb_dim, emb_dim)  # GRU输出投影到emb_dim
        self.attn_proj = nn.Linear(emb_dim, emb_dim)  # Attention输出投影

        self.pad_idx = pad_idx
        self.hist_max = hist_max
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, uid, seq, gender, grade, dept, type_):
        # 序列编码
        seq_emb = self.seq_emb(seq)
        mask = (seq == self.pad_idx)

        # 多头注意力池化
        attn_out, _ = self.attention_pool(seq_emb, seq_emb, seq_emb, key_padding_mask=mask)
        attn_pool = self.attn_proj(attn_out.mean(dim=1))  # [batch, emb_dim]

        # GRU序列建模
        gru_out, _ = self.gru(seq_emb)
        gru_pool = self.gru_proj(gru_out.mean(dim=1))  # [batch, emb_dim]

        # 平均池化
        mask_float = (~mask).float().unsqueeze(-1)
        mean_pool = (seq_emb * mask_float).sum(1) / mask_float.sum(1).clamp(min=1)  # [batch, emb_dim]

        # 确保所有特征维度一致
        assert attn_pool.size(1) == gru_pool.size(1) == mean_pool.size(1) == self.user_emb.embedding_dim

        # 序列特征融合（加权平均）
        seq_weights = F.softmax(torch.randn(3, 1).to(seq_emb.device), dim=0)
        seq_features = torch.stack([attn_pool, gru_pool, mean_pool], dim=1)  # [batch, 3, emb_dim]
        seq_fusion = (seq_features * seq_weights.view(1, 3, 1)).sum(dim=1)  # [batch, emb_dim]

        # 用户特征
        u_emb = self.user_emb(uid)
        user_feat = torch.cat([
            self.gender_emb(gender),
            self.grade_emb(grade),
            self.dept_emb(dept),
            self.type_emb(type_)
        ], dim=1)

        # 特征交叉融合
        fusion_input = torch.cat([u_emb, seq_fusion, user_feat], dim=1)
        fusion_emb = self.cross_net(fusion_input)
        fusion_emb = self.layer_norm(fusion_emb)

        return F.normalize(fusion_emb, p=2, dim=1)
class ItemTower(nn.Module):
    """物品塔：物品embedding"""

    def __init__(self, num_items, emb_dim):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, iid):
        emb = self.item_emb(iid)
        emb = self.layer_norm(emb)
        return nn.functional.normalize(emb, p=2, dim=1)


class TwoTower(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, pad_idx, hist_max,
                 num_genders, num_grades, num_depts, num_types, temperature=0.1):
        super().__init__()
        self.user_tower = UserTower(num_users, num_items, emb_dim, pad_idx, hist_max,
                                            num_genders, num_grades, num_depts, num_types)
        self.item_tower = ItemTower(num_items, emb_dim)
        self.temperature = temperature
        self.renew_head = nn.Linear(emb_dim, 1)

    def contrastive_loss(self, u_emb, i_emb, labels):
        """对比学习损失"""
        # 简单的对比学习实现
        batch_size = u_emb.size(0)
        logits = torch.matmul(u_emb, i_emb.T) / self.temperature
        targets = torch.arange(batch_size).to(u_emb.device)
        loss = F.cross_entropy(logits, targets)
        return loss

    def forward(self, uid, iid, seq, gender, grade, dept, type_, labels=None):
        u_emb = self.user_tower(uid, seq, gender, grade, dept, type_)
        i_emb = self.item_tower(iid)
        score = (u_emb * i_emb).sum(dim=1)
        renew_pred = self.renew_head(u_emb).squeeze()

        if labels is not None:
            cl_loss = self.contrastive_loss(u_emb, i_emb, labels)
            return score, renew_pred, cl_loss

        return score, renew_pred


def train_two_tower(train_data, user_seq_dict, num_users, num_items, emb_dim, pad_idx, hist_max,
                             inter_train, user_encoder,
                             gender_encoder, grade_encoder, dept_encoder, type_encoder):
    """训练改进的Two-Tower模型"""
    dataset = SeqDataset(train_data, user_seq_dict)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        drop_last=False,
        pin_memory=True
    )

    model = TwoTower(num_users, num_items, emb_dim, pad_idx, hist_max,
                             len(gender_encoder), len(grade_encoder),
                             len(dept_encoder), len(type_encoder)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 修复版本兼容性
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    loss_clf = nn.BCEWithLogitsLoss()
    loss_reg = nn.MSELoss()

    # 准备续借次数标签
    uid2raw = {i: u for u, i in user_encoder.items()}
    renew_total = inter_train.groupby('user_id')['续借次数'].sum().to_dict()
    renew_label = []
    for u, i, l in train_data:
        raw_uid = uid2raw[u]
        renew_label.append(renew_total.get(raw_uid, 0.0))

    # 标准化回归标签
    renew_label = np.array(renew_label)
    renew_mean, renew_std = renew_label.mean(), renew_label.std() + 1e-8
    renew_label = (renew_label - renew_mean) / renew_std
    renew_label = torch.tensor(renew_label, dtype=torch.float).to(device)

    # 构建用户特征映射
    users = inter_train[['user_id', 'gender_idx', 'grade_idx', 'dept_idx', 'type_idx']].drop_duplicates()
    user_id_to_gender = users.set_index('user_id')['gender_idx'].to_dict()
    user_id_to_grade = users.set_index('user_id')['grade_idx'].to_dict()
    user_id_to_dept = users.set_index('user_id')['dept_idx'].to_dict()
    user_id_to_type = users.set_index('user_id')['type_idx'].to_dict()

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for idx, batch in enumerate(tqdm(train_loader, desc=f"改进Two-Tower训练 Epoch {epoch + 1}/{EPOCHS}")):
            uid, iid, label, seq = [x.to(device) for x in batch]
            batch_renew_label = renew_label[idx * BATCH: (idx + 1) * BATCH]

            # 获取用户原始 ID
            uid_list = uid.cpu().numpy()
            raw_uid_list = [uid2raw[u] for u in uid_list]

            # 获取用户特征
            gender = torch.tensor([user_id_to_gender[u] for u in raw_uid_list], dtype=torch.long).to(device)
            grade = torch.tensor([user_id_to_grade[u] for u in raw_uid_list], dtype=torch.long).to(device)
            dept = torch.tensor([user_id_to_dept[u] for u in raw_uid_list], dtype=torch.long).to(device)
            type_ = torch.tensor([user_id_to_type[u] for u in raw_uid_list], dtype=torch.long).to(device)

            # 修复autocast使用
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    score, renew_pred, cl_loss = model(uid, iid, seq, gender, grade, dept, type_, labels=uid)
                    loss_c = loss_clf(score, label)
                    loss_r = loss_reg(renew_pred, batch_renew_label)
                    cl_weight = 0.1  # 对比学习权重
                    loss = loss_c + 0.3 * loss_r + cl_weight * cl_loss
            else:
                score, renew_pred, cl_loss = model(uid, iid, seq, gender, grade, dept, type_, labels=uid)
                loss_c = loss_clf(score, label)
                loss_r = loss_reg(renew_pred, batch_renew_label)
                cl_weight = 0.1
                loss = loss_c + 0.3 * loss_r + cl_weight * cl_loss

            optimizer.zero_grad()

            if scaler:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * uid.size(0)
            total_samples += uid.size(0)

        avg_loss = total_loss / total_samples
        print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

    return model


# -------------------------- 2. Faiss向量召回 --------------------------
def generate_recall_candidates(model, num_users, num_items, user_seq_dict, idx2book, PAD):
    """用Faiss生成召回候选"""
    model.eval()
    user_vecs, item_vecs = [], []

    with torch.no_grad():
        # 提取物品向量（分批处理）
        item_vecs = []
        batch_size = 512
        for i in range(0, num_items, batch_size):
            batch_ids = torch.arange(i, min(i + batch_size, num_items)).to(device)
            batch_vecs = model.item_tower(batch_ids).cpu().numpy()
            item_vecs.append(batch_vecs)
        item_vecs = np.concatenate(item_vecs, axis=0)

        # 提取用户向量
        for u in tqdm(range(num_users), desc="提取用户向量"):
            uid = torch.tensor([u]).to(device)
            seq = user_seq_dict[u].unsqueeze(0).to(device)

            # 构造 dummy 用户特征（与训练时一致）
            dummy_gender = torch.tensor([0]).to(device)
            dummy_grade = torch.tensor([0]).to(device)
            dummy_dept = torch.tensor([0]).to(device)
            dummy_type = torch.tensor([0]).to(device)

            u_vec = model.user_tower(uid, seq, dummy_gender, dummy_grade, dummy_dept, dummy_type).cpu().numpy()
            user_vecs.append(u_vec[0])

    user_vecs = np.stack(user_vecs)

    # 向量质量检查
    print("用户向量统计:")
    print(f"- 均值: {user_vecs.mean():.4f}")
    print(f"- 标准差: {user_vecs.std():.4f}")
    print(f"- 范数均值: {np.linalg.norm(user_vecs, axis=1).mean():.4f}")

    print("物品向量统计:")
    print(f"- 均值: {item_vecs.mean():.4f}")
    print(f"- 标准差: {item_vecs.std():.4f}")
    print(f"- 范数均值: {np.linalg.norm(item_vecs, axis=1).mean():.4f}")

    # Faiss检索
    faiss_index = faiss.IndexFlatIP(user_vecs.shape[1])
    faiss_index.add(item_vecs.astype('float32'))
    _, rec_indices = faiss_index.search(user_vecs.astype('float32'), RECALL_TOP_K)

    # 索引过滤
    rec_indices = np.clip(rec_indices, 0, num_items - 1)

    # 映射到原始ID
    idx2book_arr = np.array([idx2book[i] for i in range(num_items)])
    rec_candidates = [idx2book_arr[indices].tolist() for indices in rec_indices]

    print(f"Faiss召回完成：每个用户召回{RECALL_TOP_K}本候选书")
    return rec_candidates, user_vecs, item_vecs


# -------------------------- 3. LightGBM重排模型（核心修复） --------------------------
def build_rerank_features(rec_candidates, user_vecs, item_vecs, item_text_emb,
                           inter_train, books, user_encoder, idx2book, num_users,
                           train_data, users):  # 去掉 raw_uid
    """构建重排特征，修复'一级分类'列缺失问题"""
    import numpy as np
    from collections import defaultdict
    from tqdm import tqdm

    inter_with_cat = inter_train.copy()
    inter_with_cat['一级分类'] = inter_with_cat['一级分类'].fillna('未知分类')

    # 用户特征查询函数
    def get_user_feat(raw_uid):
        return np.array([
            users.loc[users['user_id'] == raw_uid, 'gender_idx'].values[0],
            users.loc[users['user_id'] == raw_uid, 'grade_idx'].values[0],
            users.loc[users['user_id'] == raw_uid, 'dept_idx'].values[0],
            users.loc[users['user_id'] == raw_uid, 'type_idx'].values[0]
        ])

    # ID 映射
    all_book_ids = set()
    all_book_ids.update(inter_train['book_id'].astype(str).unique())
    all_book_ids.update(books['book_id'].astype(str).unique())
    for candidates in rec_candidates:
        all_book_ids.update(str(b) for b in candidates)
    all_book_ids.update(str(b) for b in idx2book.values())

    raw2idx = {str(b): i for i, b in idx2book.items()}
    next_available_idx = len(raw2idx)
    unknown_book_id = "unknown_book"
    if unknown_book_id not in raw2idx:
        raw2idx[unknown_book_id] = next_available_idx
        next_available_idx += 1
    for book_id in all_book_ids:
        str_id = str(book_id)
        if str_id not in raw2idx:
            raw2idx[str_id] = next_available_idx
            next_available_idx += 1

    # 向量填充
    item_vec_dim = item_vecs.shape[1]
    item_vec_mean = item_vecs.mean(axis=0)
    if len(item_vecs) < next_available_idx:
        need = next_available_idx - len(item_vecs)
        item_vecs = np.pad(item_vecs, ((0, need), (0, 0)), constant_values=item_vec_mean)

    text_vec_dim = item_text_emb.shape[1]
    text_vec_mean = item_text_emb.mean(axis=0)
    if len(item_text_emb) < next_available_idx:
        need = next_available_idx - len(item_text_emb)
        item_text_emb = np.pad(item_text_emb, ((0, need), (0, 0)), constant_values=text_vec_mean)

    # 元数据
    item_pop = inter_train['book_id'].astype(str).value_counts().to_dict()
    seen_by_user = defaultdict(set)
    for _, row in inter_train.iterrows():
        seen_by_user[str(row['user_id'])].add(str(row['book_id']))

    uid2raw = {i: u for u, i in user_encoder.items()}
    user_cat_dist = {}
    for u_idx in range(num_users):
        raw_uid = uid2raw.get(u_idx, str(u_idx))
        user_inter = inter_with_cat[inter_with_cat['user_id'] == raw_uid]
        user_cat_dist[u_idx] = user_inter['一级分类'].value_counts(normalize=True).to_dict() if len(user_inter) else {"未知分类": 1.0}

    # 单样本特征生成
    def get_features(u_idx, raw_book_id):
        try:
            str_book_id = str(raw_book_id).strip()
            if str_book_id not in raw2idx:
                str_book_id = unknown_book_id
            i_idx = raw2idx[str_book_id]
            i_idx = max(0, min(i_idx, len(item_vecs) - 1))

            u_vec = user_vecs[u_idx] if 0 <= u_idx < len(user_vecs) else np.zeros_like(item_vec_mean)
            i_vec = item_vecs[i_idx] if 0 <= i_idx < len(item_vecs) else item_vec_mean
            text_vec = item_text_emb[i_idx] if 0 <= i_idx < len(item_text_emb) else text_vec_mean

            u_i_dot = u_vec * i_vec
            u_text_dot = u_vec * text_vec
            i_text_dot = i_vec * text_vec

            book_pop = np.log1p(item_pop.get(str_book_id, 1e-5))

            book_cat = "未知分类"
            book_mask = books['book_id'].astype(str) == str_book_id
            if np.any(book_mask):
                book_info = books[book_mask].iloc[0]
                if '一级分类' in book_info and pd.notna(book_info['一级分类']):
                    book_cat = book_info['一级分类']
            cat_sim = user_cat_dist[u_idx].get(book_cat, 0.0)

            same_author = 0.0
            if np.any(book_mask) and '作者' in book_info and pd.notna(book_info['作者']):
                book_author = book_info['作者']
                author_mask = (inter_with_cat['user_id'] == uid2raw.get(u_idx, str(u_idx))) & (inter_with_cat['作者'] == book_author)
                same_author = 1.0 if np.any(author_mask) else 0.0

            # 每个用户用自己的特征
            raw_uid = uid2raw.get(u_idx, str(u_idx))
            user_feat = get_user_feat(raw_uid)

            return np.concatenate([
                u_vec, i_vec, text_vec,
                u_i_dot, u_text_dot, i_text_dot,
                [book_pop, cat_sim, same_author],
                user_feat
            ])
        except Exception as e:
            base_dim = len(item_vec_mean)
            return np.concatenate([
                np.zeros(base_dim), item_vec_mean, text_vec_mean,
                np.zeros(base_dim), np.zeros(base_dim), np.zeros(base_dim),
                [0.0, 0.0, 0.0], np.zeros(4)
            ])

    # 构建训练集
    rank_train, rank_label, group_train = [], [], []
    pos_samples_dict = defaultdict(set)
    for u, i, l in train_data:
        if l == 1:
            pos_samples_dict[u].add(str(idx2book.get(i, unknown_book_id)))

    for u_idx in tqdm(range(num_users), desc="构建重排特征"):
        raw_uid = uid2raw.get(u_idx, str(u_idx))
        seen = seen_by_user.get(raw_uid, set())

        valid_cands = []
        seen_in_cands = set()
        for b in rec_candidates[u_idx]:
            str_b = str(b).strip()
            if str_b not in seen and str_b not in seen_in_cands:
                valid_cands.append(str_b)
                seen_in_cands.add(str_b)
            if len(valid_cands) >= 200:
                break

        if len(valid_cands) < 10:
            all_possible = [b for b in raw2idx.keys() if b not in seen and b not in seen_in_cands]
            if all_possible:
                add_count = min(200 - len(valid_cands), len(all_possible))
                valid_cands += np.random.choice(all_possible, add_count, replace=False).tolist()
            while len(valid_cands) < 10:
                valid_cands.append(unknown_book_id)

        pos_books = pos_samples_dict.get(u_idx, set())
        labels = [1 if b in pos_books else 0 for b in valid_cands]
        features_list = [get_features(u_idx, b) for b in valid_cands]

        rank_train.extend(features_list)
        rank_label.extend(labels)
        group_train.append(len(features_list))

    X_rerank = np.array(rank_train)
    feature_mean = X_rerank.mean(axis=0)
    feature_std = X_rerank.std(axis=0) + 1e-8
    X_rerank = (X_rerank - feature_mean) / feature_std
    X_rerank = np.nan_to_num(X_rerank)

    print(f"重排特征构建完成：总样本数={len(rank_train)}，特征维度={X_rerank.shape[1]}")
    return X_rerank, np.array(rank_label), np.array(group_train), seen_by_user, raw2idx, feature_mean, feature_std

def train_reranker(X_train, y_train, group_train):
    """训练LightGBM重排模型"""
    # 调整参数解决无有效特征问题
    ranker = LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=20,
        min_data_in_leaf=1,
        min_child_weight=0.001,
        # 关键：使用LightGBM官方推荐的参数名，避免别名冲突
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=5, 
        verbosity=1,
        random_state=SEED
    )

    ranker.fit(
        X=X_train,
        y=y_train,
        group=group_train
    )

    # 输出特征重要性
    if hasattr(ranker, 'feature_importances_'):
        importances = ranker.feature_importances_
        print("重排模型特征重要性（前10）:")
        for i in np.argsort(importances)[-10:][::-1]:
            print(f"特征 {i}: {importances[i]}")
    else:
        print("无法获取特征重要性")

    print("LightGBM重排模型训练完成")
    return ranker


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    PATH = './datasets'
    # 加载数据
    with open(f"{PATH}/processed_data.pkl", "rb") as f:
        data_dict = pickle.load(f)
    with open(f"{PATH}/user_interest_features.pkl", "rb") as f:
        interest_dict = pickle.load(f)

    # 解析数据
    inter_train = data_dict["inter_train"]
    books = data_dict["books"]
    user_encoder = data_dict["user_encoder"]
    item_encoder = data_dict["item_encoder"]
    train_data = data_dict["train_data"]  # 正负样本数据
    idx2book = data_dict["idx2book"]
    num_users = data_dict["num_users"]
    num_items = data_dict["num_items"]
    users = data_dict["users"]

    user_seq_dict = interest_dict["user_seq_dict"]
    PAD = interest_dict["PAD"]
    item_text_emb = interest_dict["item_text_emb"]
    EMB_DIM = interest_dict["EMB_DIM"]
    HIST_MAX = interest_dict["HIST_MAX"]
    gender_encoder = data_dict["gender_encoder"]
    grade_encoder = data_dict["grade_encoder"]
    dept_encoder = data_dict["dept_encoder"]
    type_encoder = data_dict["type_encoder"]

    # 训练Two-Tower模型
    two_tower_model = train_two_tower(
        train_data, user_seq_dict, num_users, num_items,
        EMB_DIM, PAD, HIST_MAX, inter_train, user_encoder,
        gender_encoder, grade_encoder, dept_encoder, type_encoder
    )

    # 生成召回候选
    rec_candidates, user_vecs, item_vecs = generate_recall_candidates(
        two_tower_model, num_users, num_items, user_seq_dict, idx2book, PAD
    )

    # 构建重排特征并训练重排模型（新增train_data参数）
    X_rerank, y_rerank, group_rerank, seen_by_user, raw2idx, feature_mean, feature_std = build_rerank_features(
        rec_candidates, user_vecs, item_vecs, item_text_emb,
        inter_train, books, user_encoder, idx2book, num_users,
        train_data, users
    )
    ranker = train_reranker(X_rerank, y_rerank, group_rerank)

    # 保存模型
    model_dict = {
        "two_tower_model": two_tower_model.state_dict(),
        "ranker": ranker,
        "rec_candidates": rec_candidates,
        "user_vecs": user_vecs,
        "item_vecs": item_vecs,
        "item_text_emb": item_text_emb,
        "seen_by_user": seen_by_user,
        "raw2idx": raw2idx,
        "user_encoder": user_encoder,
        "idx2book": idx2book,
        "num_users": num_users,
        "num_items": num_items,
        "EMB_DIM": EMB_DIM,
        "books": books,
        "inter_train": inter_train,
        "feature_mean": feature_mean,
        "feature_std": feature_std
    }

    with open(f"{PATH}/models.pkl", "wb") as f:
        pickle.dump(model_dict, f)

    print("模型构建模块完成！模型已保存至 ./datasets/models.pkl")