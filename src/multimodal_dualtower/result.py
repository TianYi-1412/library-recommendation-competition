from model import *  # 复用模型里定义的 EMB_DIM 等常量

# -------------------------- 配置 --------------------------
TOP_K = 5
PATH = './datasets'

# -------------------------- 评估指标 --------------------------
def f1_at_k(pred_dict, valid_uid2last, k=TOP_K):
    precisions, recalls = [], []
    for raw_uid, pred_books in pred_dict.items():
        true_books = set(valid_uid2last.get(raw_uid, []))
        if not true_books:
            continue
        topk = set(pred_books[:k])
        hit = len(topk & true_books)
        precisions.append(hit / k)
        recalls.append(hit / len(true_books))
    avg_p = np.mean(precisions) if precisions else 0.0
    avg_r = np.mean(recalls) if recalls else 0.0
    f1 = 2 * avg_p * avg_r / (avg_p + avg_r + 1e-9)
    return f1, avg_p, avg_r


def calculate_diversity(pred_dict, books):
    book2cat = dict(zip(books['book_id'].astype(str), books['一级分类']))
    diversities = []
    for raw_uid, pred_books in pred_dict.items():
        cats = [book2cat.get(b, "未知") for b in pred_books]
        diversities.append(len(set(cats)) / len(cats))
    return np.mean(diversities) if diversities else 0.0


# -------------------------- 特征生成 --------------------------
def get_features(u_idx, raw_book_id, model_dict):
    try:
        user_vecs = model_dict["user_vecs"]
        item_vecs = model_dict["item_vecs"]
        item_text_emb = model_dict["item_text_emb"]
        raw2idx = model_dict["raw2idx"]
        books = model_dict["books"]
        inter_train = model_dict["inter_train"]
        EMB_DIM = model_dict["EMB_DIM"]
        uid2raw = {i: u for u, i in model_dict["user_encoder"].items()}

        if raw_book_id not in raw2idx:
            return np.zeros(EMB_DIM * 6 + 3 + 4)  # +4 用户特征占位

        i_idx = raw2idx[raw_book_id]
        u_vec = user_vecs[u_idx]
        i_vec = item_vecs[i_idx]
        text_vec = item_text_emb[i_idx]

        u_i_dot = u_vec * i_vec
        u_text_dot = u_vec * text_vec
        i_text_dot = i_vec * text_vec

        raw_uid = uid2raw[u_idx]
        item_pop = inter_train['book_id'].astype(str).value_counts().to_dict()
        book_pop = np.log1p(item_pop.get(raw_book_id, 1e-5))

        book_cat_row = books[books['book_id'].astype(str) == raw_book_id]
        book_cat = book_cat_row['一级分类'].iloc[0] if len(book_cat_row) else "未知"
        user_total = inter_train[inter_train['user_id'] == raw_uid]
        user_cat = inter_train[(inter_train['user_id'] == raw_uid) & (inter_train['一级分类'] == book_cat)]
        cat_sim = len(user_cat) / len(user_total) if len(user_total) else 0.0

        same_author = 0.0
        if len(book_cat_row) and not pd.isna(book_cat_row['作者'].iloc[0]):
            author = book_cat_row['作者'].iloc[0]
            same_author = 1.0 if len(inter_train[(inter_train['user_id'] == raw_uid) & (inter_train['作者'] == author)]) else 0.0

        # 用户自身特征
        user_feat = np.array([
            inter_train[inter_train['user_id'] == raw_uid]['gender_idx'].iloc[0],
            inter_train[inter_train['user_id'] == raw_uid]['grade_idx'].iloc[0],
            inter_train[inter_train['user_id'] == raw_uid]['dept_idx'].iloc[0],
            inter_train[inter_train['user_id'] == raw_uid]['type_idx'].iloc[0]
        ])

        feat = np.concatenate([u_vec, i_vec, text_vec, u_i_dot, u_text_dot, i_text_dot,
                               [book_pop, cat_sim, same_author], user_feat])

        # 标准化（使用训练时保存的均值方差）
        mean = model_dict.get('feature_mean', feat)
        std = model_dict.get('feature_std', np.ones_like(feat))
        return (feat - mean) / (std + 1e-8)
    except Exception as e:
        return np.zeros(EMB_DIM * 6 + 3 + 4)


# -------------------------- 推荐生成 --------------------------
def generate_recommendations(model_dict, valid_uid2last=None, is_submit=False):
    ranker = model_dict["ranker"]
    rec_candidates = model_dict["rec_candidates"]
    seen_by_user = model_dict["seen_by_user"]
    user_encoder = model_dict["user_encoder"]
    idx2book = model_dict["idx2book"]
    books = model_dict["books"]
    num_users = model_dict["num_users"]
    TOP_K = 5

    uid2raw = {i: u for u, i in user_encoder.items()}  # 编码→原始
    raw2uid = {u: i for i, u in uid2raw.items()}  # 原始→编码

    target_users = user_encoder.keys() if is_submit else valid_uid2last.keys()

    pred_dict = {}
    for raw_uid in tqdm(target_users, desc="提交推荐" if is_submit else "验证推荐"):
        u_idx = raw2uid[raw_uid]
        seen = seen_by_user.get(raw_uid, set())
        cands = [b for b in rec_candidates[u_idx] if b not in seen]

        if len(cands) < TOP_K:
            all_books = set(idx2book.values())
            unseen = list(all_books - seen - set(cands))
            cands += list(np.random.choice(unseen, min(TOP_K - len(cands), len(unseen)), replace=False))

        X = np.array([get_features(u_idx, b, model_dict) for b in cands])
        scores = ranker.predict(X)
        top_books = [cands[i] for i in np.argsort(-scores)[:TOP_K]]
        top_books = list(dict.fromkeys(top_books))  # 去重
        pred_dict[raw_uid] = top_books

    if not is_submit and valid_uid2last:
        f1, avg_p, avg_r = f1_at_k(pred_dict, valid_uid2last, k=TOP_K)
        diversity = calculate_diversity(pred_dict, books)
        print(f"验证 F1@{TOP_K}: {f1:.4f} | 精确率: {avg_p:.4f} | 召回率: {avg_r:.4f} | 多样性: {diversity:.4f}")

    return pred_dict


# -------------------------- 主函数 --------------------------
def main():
    with open(f"{PATH}/processed_data.pkl", "rb") as f:
        data_dict = pickle.load(f)
    with open(f"{PATH}/models.pkl", "rb") as f:
        model_dict = pickle.load(f)

    valid_uid2last = data_dict["valid_uid2last"]

    print("===== 验证集评估 =====")
    _ = generate_recommendations(model_dict, valid_uid2last, is_submit=False)

    print("\n===== 生成提交文件 =====")
    submit_pred = generate_recommendations(model_dict, is_submit=True)

    sub_df = pd.DataFrame([(u, b[0]) for u, b in submit_pred.items()],
                          columns=['user_id', 'book_id'])
    sub_df.to_csv(f"{PATH}/submission.csv", index=False, encoding='utf-8')
    print(f"提交文件已保存：{PATH}/submission.csv")
    print(f"多样性（不同书数/总用户数）：{sub_df['book_id'].nunique()}/{len(sub_df)}")


if __name__ == "__main__":
    main()