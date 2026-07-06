# 08_blend_inference.py
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import pickle

# ---------- 配置 ----------
OUT_DIR = '../output+hj'
DATA_DIR = '../data'
RESULT_PATH = '../result+hj/submission.csv'
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

# ---------- 加载全局表 ----------
inter = pd.read_csv(os.path.join(DATA_DIR, 'inter_preliminary.csv'))
inter['borrow_time'] = pd.to_datetime(inter['借阅时间'])
book = pd.read_csv(os.path.join(DATA_DIR, 'item.csv'))
user = pd.read_csv(os.path.join(DATA_DIR, 'user.csv'))
dept_pop = inter.merge(user[['借阅人', 'DEPT']], left_on='user_id', right_on='借阅人') \
                .groupby(['DEPT', 'book_id']).size().to_dict()
renew_mean = inter.groupby('book_id')['续借次数'].mean().to_dict()

# ---------- 加载模型 & categorical 列 ----------
lgb_model = joblib.load(os.path.join(OUT_DIR, 'lgb_rank.model'))
cat_feats = joblib.load(os.path.join(OUT_DIR, 'lgb_cat_features.pkl'))

deepfm_meta = joblib.load(os.path.join(OUT_DIR, 'deepfm_pytorch_meta.pkl'))
encoders = deepfm_meta['encoders']
sparse_cols = deepfm_meta['sparse_cols']
dense_cols = deepfm_meta['dense_cols']
field_dims = deepfm_meta['field_dims']

device = torch.device('cpu')

class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim=64, mlp_dims=(256, 128)):
        super(DeepFM, self).__init__()
        self.embedding = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in field_dims])
        self.fc = nn.Linear(len(field_dims) * embed_dim, mlp_dims[0])
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dims[0] + len(dense_cols), mlp_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[1], 1)
        )
    def forward(self, x_sparse, x_dense):
        embeds = [self.embedding[i](x_sparse[:, i]) for i in range(len(field_dims))]
        embed_vec = torch.cat(embeds, dim=1)
        fm_part = 0.5 * (embed_vec.sum(dim=1).unsqueeze(1) ** 2 - (embed_vec ** 2).sum(dim=1).unsqueeze(1))
        deep_part = self.fc(embed_vec)
        deep_part = torch.cat([deep_part, x_dense], dim=1)
        out = self.mlp(deep_part) + fm_part
        return torch.sigmoid(out).squeeze()

deepfm_model = DeepFM(field_dims).to(device)
deepfm_model.load_state_dict(torch.load(os.path.join(OUT_DIR, 'deepfm_pytorch.pt'), map_location=device))
deepfm_model.eval()

# ---------- 加载候选池 & 历史 ----------
cand_dict = pd.read_pickle(os.path.join(OUT_DIR, 'cand_dict_80.pkl'))
user_hist = pd.read_pickle(os.path.join(OUT_DIR, 'user_hist.pkl'))
user_dept = user.set_index('借阅人')['DEPT'].to_dict()

# ---------- Item2Vec 相似度 ----------
model_item2vec = pickle.load(open(os.path.join(OUT_DIR, 'item2vec_gensim.model'), 'rb'))
def sim(b1, b2):
    return model_item2vec.wv.similarity(str(b1), str(b2)) if str(b1) in model_item2vec.wv and str(b2) in model_item2vec.wv else 0.0

# ---------- 编码函数 ----------
def encode_df(df, encoders):
    for col in encoders:
        if col not in df.columns:
            continue
        le = encoders[col]
        df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else '<UNK>')
        if '<UNK>' not in le.classes_:
            le.classes_ = np.append(le.classes_, '<UNK>')
        df[col] = le.transform(df[col])
    return df

# ---------- 融合权重 ----------
w_lgb, w_dfm = 0.6, 0.4

# ---------- 逐用户推理 ----------
results = []
for uid in tqdm(cand_dict, desc='blend'):
    hist = user_hist[uid]
    cand_books = cand_dict[uid]
    if not cand_books:
        continue

    # 构造 80 行 DataFrame（保留所有稀疏+dense列）
    template = pd.read_csv(os.path.join(OUT_DIR, 'train_rank2.csv')).iloc[0]
    cand_df = pd.DataFrame([template] * len(cand_books))
    cand_df['cand_book_id'] = cand_books
    cand_df['user_id'] = uid

    # 最近 3 本历史
    last3 = hist[-3:] if len(hist) >= 3 else hist + [0] * (3 - len(hist))
    for i in range(1, 4):
        cand_df[f'last{i}_book_id'] = last3[-i] if i <= len(last3) else 0
        cand_df[f'last{i}_cate2'] = book.loc[book.book_id == last3[-i], '二级分类'].values[0] if i <= len(last3) else 'Unknown'
        cand_df[f'last{i}_renew'] = inter[(inter.user_id == uid) & (inter.book_id == last3[-i])]['续借次数'].values[-1] if i <= len(last3) else 0

    # 候选书动态特征
    for idx, b in enumerate(cand_books):
        cand_df.loc[idx, 'cand_cate2'] = book.loc[book.book_id == b, '二级分类'].values[0] if len(book.loc[book.book_id == b, '二级分类']) else 'Unknown'
        cand_df.loc[idx, 'cand_renew_mean'] = renew_mean.get(b, 0)
        cand_df.loc[idx, 'cand_sim_last1'] = sim(last3[-1], b)
        cand_df.loc[idx, 'cand_pop_dept'] = dept_pop.get((user_dept.get(uid, 'Unknown'), b), 0)

    # ---------- 关键修复：先编码全部稀疏+dense，再拆分 ----------
    cand_df_encoded = encode_df(cand_df.copy(), encoders)

    # LightGBM 只拿它要的列
    lgb_cols = lgb_model.feature_name()
    cand_df_lgb = cand_df_encoded[lgb_cols].copy()
    for col in cat_feats:
        cand_df_lgb[col] = cand_df_lgb[col].astype('category')

    # ---------- LightGBM 分数 ----------
    lgb_score = lgb_model.predict(cand_df_lgb, categorical_feature=cat_feats)

    # ---------- DeepFM 分数 ----------
    X_sparse = torch.tensor(cand_df_encoded[sparse_cols].values, dtype=torch.long).to(device)
    X_dense = torch.tensor(cand_df_encoded[dense_cols].values, dtype=torch.float32).to(device)
    with torch.no_grad():
        dfm_score = deepfm_model(X_sparse, X_dense).numpy()

    # ---------- 融合 ----------
    final_score = w_lgb * lgb_score + w_dfm * dfm_score
    top1 = cand_books[final_score.argmax()]
    results.append({'user_id': uid, 'book_id': int(top1)})

# ---------- 保存 ----------
pd.DataFrame(results).to_csv(RESULT_PATH, index=False, encoding='utf-8')
print(f'✅ 融合推理完成，已保存：{RESULT_PATH}  共 {len(results)} 用户')