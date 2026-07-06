import random
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

# -------------------------- 全局配置 --------------------------
SEED = 42
PATH = './datasets'  # 数据存储路径
HIST_MAX = 50  # 用户历史序列最大长度
NEG_K = 4  # 负采样比例（1正:4负）
TARGET_USER_NUM = 600  # 目标用户数量

# 初始化随机种子
random.seed(SEED)
np.random.seed(SEED)


def load_and_clean_data():
    """加载并清洗原始数据，合并必要特征"""
    # 读取原始数据
    inter = pd.read_csv(f'{PATH}/pre_inter.csv')
    books = pd.read_csv(f'{PATH}/book_data.csv')
    users = pd.read_csv(f'{PATH}/pre_user.csv')

    users = users.rename(columns={'借阅人': 'user_id'})

    # 数据类型统一处理
    inter['book_id'] = inter['book_id'].astype(str).str.strip()
    books['book_id'] = books['book_id'].astype(str).str.strip()

    # 时间格式处理
    inter['借阅时间'] = pd.to_datetime(inter['借阅时间'], errors='coerce')
    inter = inter.dropna(subset=['借阅时间'])

    # 扩展续借行为
    inter['续借次数'] = inter['续借次数'].fillna(0).astype(int)
    inter_ext = []
    for _, row in inter.iterrows():
        base_entry = row.to_dict()
        inter_ext.append(base_entry)
        for _ in range(base_entry['续借次数']):
            new_entry = base_entry.copy()
            new_entry['is_renewal'] = True
            inter_ext.append(new_entry)
    inter = pd.DataFrame(inter_ext).sort_values(['user_id', '借阅时间'])
    inter['is_renewal'] = inter['is_renewal'].fillna(False)

    # 合并书籍信息（包含一级分类）
    books_info = books[['book_id', '作者', '一级分类', '题名']].drop_duplicates()
    inter = inter.merge(books_info, on='book_id', how='left')
    inter['作者'] = inter['作者'].fillna('未知作者')
    inter['一级分类'] = inter['一级分类'].fillna('未知分类')
    inter['题名'] = inter['题名'].fillna('未知书名')

    # 筛选有交互记录的600个用户
    user_inter_count = inter['user_id'].value_counts().reset_index()
    user_inter_count.columns = ['user_id', 'count']
    valid_users_with_inter = user_inter_count[user_inter_count['count'] > 0]


    if len(valid_users_with_inter) < TARGET_USER_NUM:
        raise ValueError(f"错误：有效用户不足 {TARGET_USER_NUM} 个，仅找到 {len(valid_users_with_inter)} 个")

    target_users = set(valid_users_with_inter.head(TARGET_USER_NUM)['user_id'])
    inter = inter[inter['user_id'].isin(target_users)]
    users = users[users['user_id'].isin(target_users)]

    users['性别'] = users['性别'].map({'F': 0, 'M': 1})
    users['年级'] = users['年级'].astype(int)
    users['类型'] = users['类型'].astype(str)

    gender_encoder = {g: i for i, g in enumerate(users['性别'].unique())}
    grade_encoder = {g: i for i, g in enumerate(users['年级'].unique())}
    dept_encoder = {d: i for i, d in enumerate(users['DEPT'].unique())}
    type_encoder = {t: i for i, t in enumerate(users['类型'].unique())}

    users['gender_idx'] = users['性别'].map(gender_encoder)
    users['grade_idx'] = users['年级'].map(grade_encoder)
    users['dept_idx'] = users['DEPT'].map(dept_encoder)
    users['type_idx'] = users['类型'].map(type_encoder)

    # 合并用户特征到交互数据
    inter = inter.merge(users[['user_id', 'gender_idx', 'grade_idx', 'dept_idx', 'type_idx']], on='user_id', how='left')
    print(f"已筛选 {len(target_users)} 个目标用户")
    print(f"交互数据包含的列：{inter.columns.tolist()}")

    # 时间切分验证集
    def split_validation_by_time(inter, time_ratio=0.2):
        valid_uid2last = defaultdict(list)
        train_indices = []
        for uid, grp in inter.groupby('user_id'):
            sorted_grp = grp.sort_values('借阅时间')
            split_idx = max(1, int(len(sorted_grp) * (1 - time_ratio)))
            valid_items = sorted_grp.iloc[split_idx:]['book_id'].tolist()
            valid_uid2last[uid] = valid_items
            train_indices.extend(sorted_grp.iloc[:split_idx].index)
        inter_train = inter.loc[train_indices]
        return inter_train, dict(valid_uid2last)

    inter_train, valid_uid2last = split_validation_by_time(inter)
    all_book_ids = set(inter['book_id'].unique()) | set(books['book_id'].unique())
    print(f"数据清洗完成：用户={len(target_users)}, 书籍={len(all_book_ids)}, 训练样本={len(inter_train)}")

    return inter_train, books, users, target_users, valid_uid2last, all_book_ids, \
        gender_encoder, grade_encoder, dept_encoder, type_encoder


def build_encoders(inter_train, books, target_users, all_book_ids):
    """构建用户和物品的编码字典"""
    # 用户编码及反向映射
    user_encoder = {u: i for i, u in enumerate(sorted(target_users))}
    uid2raw = {i: u for u, i in user_encoder.items()}

    # 物品编码及反向映射
    all_sorted_books = sorted(all_book_ids)
    item_encoder = {book_id: idx for idx, book_id in enumerate(all_sorted_books)}
    idx2book = {i: b for b, i in item_encoder.items()}

    # 处理未知书籍
    unknown_book_id = "unknown_book"
    if unknown_book_id not in item_encoder:
        item_encoder[unknown_book_id] = len(item_encoder)
        idx2book[len(idx2book)] = unknown_book_id

    num_users = len(user_encoder)
    num_items = len(item_encoder)
    print(f"编码完成：用户数={num_users}, 物品数={num_items}")

    return user_encoder, item_encoder, uid2raw, idx2book, num_users, num_items


def generate_pos_neg_samples(inter_train, user_encoder, item_encoder, idx2book, NEG_K):
    """生成正负样本用于模型训练"""
    # 预计算物品逆频率（用于负采样）
    all_raw_books = [idx2book[i] for i in range(len(item_encoder))]
    item_pop = inter_train['book_id'].value_counts()
    inv_freq = 1.0 / (np.array([item_pop.get(b, 0) + 1 for b in all_raw_books]))
    inv_freq /= inv_freq.sum()

    # 生成正样本
    pos_samples = [(user_encoder[u], item_encoder[b], 1)
                   for _, u, b in inter_train[['user_id', 'book_id']].itertuples()]
    pos_samples = list(set(pos_samples))  # 去重

    # 记录用户已交互物品
    user_seen = defaultdict(set)
    for u, i, _ in pos_samples:
        user_seen[u].add(i)

    # 负采样
    train_data = pos_samples.copy()
    for u in user_seen:
        candidate_items = [i for i in range(len(item_encoder)) if i not in user_seen[u]]
        if not candidate_items:
            continue

        probs = inv_freq[candidate_items]
        probs = probs / probs.sum()
        neg_sample_num = min(NEG_K, len(candidate_items))
        neg_items = np.random.choice(
            candidate_items, size=neg_sample_num, p=probs, replace=False
        ).tolist()
        train_data.extend([(u, j, 0) for j in neg_items])

    random.shuffle(train_data)
    print(f"训练样本总数（正负）：{len(train_data)}")
    return train_data


def save_processed_data(data_dict):
    with open(f"{PATH}/processed_data.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    print(f"预处理数据已保存至 {PATH}/processed_data.pkl")

if __name__ == "__main__":
    # 执行数据预处理流程
    inter_train, books, users, target_users, valid_uid2last, all_book_ids, gender_encoder, grade_encoder, dept_encoder, type_encoder = load_and_clean_data()
    user_enc, item_enc, uid2raw, idx2book, num_users, num_items = build_encoders(
        inter_train, books, target_users, all_book_ids
    )
    train_data = generate_pos_neg_samples(inter_train, user_enc, item_enc, idx2book, NEG_K)

    # 保存预处理数据
    processed_data = {
        "inter_train": inter_train,
        "books": books,
        "users": users,
        "user_encoder": user_enc,
        "uid2raw": uid2raw,
        "item_encoder": item_enc,
        "idx2book": idx2book,
        "target_users": target_users,
        "valid_uid2last": valid_uid2last,
        "train_data": train_data,
        "num_users": num_users,
        "num_items": num_items,
        "all_book_ids": all_book_ids,
        # 新增用户特征编码器
        "gender_encoder": gender_encoder,
        "grade_encoder": grade_encoder,
        "dept_encoder": dept_encoder,
        "type_encoder": type_encoder,
    }
    save_processed_data(processed_data)