# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from tqdm import tqdm
import lightgbm as lgb

# 路径
OUT_DIR = '../output'
DATA_DIR = '../data'
RESULT_PATH = '../result/result.csv'
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

# 1. 加载模型
model = joblib.load(os.path.join(OUT_DIR, 'lgb_rank.model'))

# 2. 读取三表
inter = pd.read_csv(os.path.join(DATA_DIR, 'inter_preliminary.csv'))
inter['borrow_time'] = pd.to_datetime(inter['借阅时间'])
inter = inter.sort_values(['user_id', 'borrow_time'])
book  = pd.read_csv(os.path.join(DATA_DIR, 'item.csv'))
user  = pd.read_csv(os.path.join(DATA_DIR, 'user.csv'))

# 3. 合并二级分类 & 续借
inter['续借次数'] = inter['续借次数'].fillna(0).astype(int)
inter = inter.merge(book[['book_id', '二级分类']], on='book_id', how='left').rename(columns={'二级分类':'cate2'})

# 4. 全局统计（复用训练脚本指标）
book_pop = inter['book_id'].value_counts().to_dict()
dept_pop = inter.merge(user[['借阅人','DEPT']], left_on='user_id', right_on='借阅人')\
                .groupby(['DEPT','book_id']).size().to_dict()
renew_mean = inter.groupby('book_id')['续借次数'].mean().to_dict()

# 5. 加载 item2vec 计算相似度
model_item2vec = pickle.load(open(os.path.join(OUT_DIR, 'item2vec_gensim.model'), 'rb'))
def sim(b1, b2):
    return model_item2vec.wv.similarity(str(b1), str(b2)) if str(b1) in model_item2vec.wv and str(b2) in model_item2vec.wv else 0.0

# 6. 构造候选池（同分类 + Item2Vec 相似 + 院系热门）
def build_candidates(uid, hist_books, dept):
    last_book = hist_books[-1]
    # 同分类
    last_cate = book.loc[book.book_id==last_book, '二级分类'].values
    last_cate = last_cate[0] if len(last_cate) else 'Unknown'
    same_cate_books = book.loc[book['二级分类']==last_cate, 'book_id'].tolist()
    # 去掉已借
    cand_pool = list(set(same_cate_books) - set(hist_books))
    if len(cand_pool) < 20:  # 不足再补
        extra_pool = set(book.sample(50, random_state=42)['book_id'].tolist()) - set(hist_books)
        cand_pool.extend(list(extra_pool))
    cand_pool = cand_pool[:50]
    # 计算特征
    feats = []
    for b in cand_pool:
        row = {
            'user_id': uid,
            'cand_book_id': b,
            'last1_book_id': hist_books[-1],
            'last1_cate2': last_cate,
            'last1_renew': inter[(inter.user_id==uid)&(inter.book_id==hist_books[-1])]['续借次数'].values[-1],
            'last2_book_id': hist_books[-2] if len(hist_books)>=2 else 0,
            'last2_cate2': book.loc[book.book_id==hist_books[-2],'二级分类'].values[0] if len(hist_books)>=2 else 'Unknown',
            'last2_renew': inter[(inter.user_id==uid)&(inter.book_id==hist_books[-2])]['续借次数'].values[-1] if len(hist_books)>=2 else 0,
            'last3_book_id': hist_books[-3] if len(hist_books)>=3 else 0,
            'last3_cate2': book.loc[book.book_id==hist_books[-3],'二级分类'].values[0] if len(hist_books)>=3 else 'Unknown',
            'last3_renew': inter[(inter.user_id==uid)&(inter.book_id==hist_books[-3])]['续借次数'].values[-1] if len(hist_books)>=3 else 0,
            'cand_cate2': book.loc[book.book_id==b,'二级分类'].values[0] if len(book.loc[book.book_id==b,'二级分类'].values) else 'Unknown',
            'cand_renew_mean': renew_mean.get(b, 0),
            'cand_sim_last1': sim(last_book, b),
            'cand_pop_dept': dept_pop.get((dept, b), 0)
        }
        feats.append(row)
    return pd.DataFrame(feats)

# 7. 逐用户推理
test_users = inter['user_id'].unique()
results = []
for uid in tqdm(test_users, desc='inference'):
    hist = inter[inter.user_id==uid]['book_id'].tolist()
    dept = user.loc[user['借阅人']==uid, 'DEPT'].values
    dept = dept[0] if len(dept) else 'Unknown'
    cand_df = build_candidates(uid, hist, dept)
    # 类别转 category 保持与训练一致
    for col in ['last1_cate2','last2_cate2','last3_cate2','cand_cate2']:
        cand_df[col] = cand_df[col].astype('category')
    X = cand_df[model.feature_name()]
    scores = model.predict(X)
    cand_df['score'] = scores
    top1 = cand_df.loc[cand_df['score'].idxmax(), 'cand_book_id']
    results.append({'user_id': uid, 'book_id': int(top1)})

# 8. 保存提交
pd.DataFrame(results).to_csv(RESULT_PATH, index=False, encoding='utf-8')
print(f'提交文件已生成: {RESULT_PATH}  共 {len(results)} 用户')