# -*- coding: utf-8 -*-
"""
构造 LightGBM 排序样本
依赖: pip install pandas numpy tqdm scikit-learn
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os, pickle

# 1. 路径
DATA_DIR  = '../data'
OUT_DIR   = '../output'
os.makedirs(OUT_DIR, exist_ok=True)

inter_path = os.path.join(DATA_DIR, 'inter_preliminary.csv')
book_path  = os.path.join(DATA_DIR, 'item.csv')
user_path  = os.path.join(DATA_DIR, 'user.csv')

# 2. 读取三表
inter = pd.read_csv(inter_path)
book  = pd.read_csv(book_path)
user  = pd.read_csv(user_path)

# 2.1 时间转换 + 排序
inter['borrow_time'] = pd.to_datetime(inter['借阅时间'])
inter = inter.sort_values(['user_id', 'borrow_time'])
inter = inter.reset_index(drop=True)

# 2.2 续借次数填 NA 为 0
inter['续借次数'] = inter['续借次数'].fillna(0).astype(int)

# 2.3 合并图书二级分类
inter = inter.merge(book[['book_id', '二级分类']], on='book_id', how='left')
inter.rename(columns={'二级分类': 'cate2'}, inplace=True)

# 3. 构造用户序列
user_seq = inter.groupby('user_id').apply(
    lambda x: x[['book_id', 'cate2', '续借次数']].to_numpy()
).to_dict()   # {uid: np.array([[book,cate,renew], ...]), ...}

# 4. 统计图书/院系热度
book_pop_global = inter['book_id'].value_counts().to_dict()
dept_book_pop   = inter.merge(user[['借阅人', 'DEPT']], left_on='user_id', right_on='借阅人')\
                        .groupby(['DEPT', 'book_id']).size().to_dict()

# 5. 加载 item2vec 相似度
model = pickle.load(open(os.path.join(OUT_DIR, 'item2vec_gensim.model'), 'rb'))
def sim(b1, b2):
    return model.wv.similarity(str(b1), str(b2)) if str(b1) in model.wv and str(b2) in model.wv else 0.0

# 6. 生成样本函数
SEQ_LEN = 3
NEG_SAMPLES = 4   # 负采样倍数，可调

def gen_samples(uid, seq):
    samples = []
    n = len(seq)
    if n < SEQ_LEN + 1:
        return samples
    # 滑动窗口
    for i in range(SEQ_LEN, n):
        hist = seq[i-SEQ_LEN:i]          # 最近 SEQ_LEN 本
        next_book, next_cate, next_renew = seq[i]
        # 负采样：随机抽 NEG_SAMPLES 本，且不在用户历史里
        neg_pool = list(book_pop_global.keys() - set(seq[:, 0]))
        neg_books = np.random.choice(neg_pool, NEG_SAMPLES, replace=False)
        cand_list = [(next_book, 1)] + [(int(nb), 0) for nb in neg_books]
        for cand_book, label in cand_list:
            cand_cate = book.loc[book.book_id==cand_book, '二级分类'].values
            cand_cate = cand_cate[0] if len(cand_cate) else 'Unknown'
            cand_renew_mean = inter[inter.book_id==cand_book]['续借次数'].mean()
            cand_sim = sim(hist[-1][0], cand_book)
            dept = user.loc[user['借阅人']==uid, 'DEPT'].values
            dept = dept[0] if len(dept) else 'Unknown'
            cand_pop_dept = dept_book_pop.get((dept, cand_book), 0)

            row = [uid, cand_book] + \
                  [int(hist[j][0]) for j in range(SEQ_LEN)] + \
                  [hist[j][1] for j in range(SEQ_LEN)] + \
                  [int(hist[j][2]) for j in range(SEQ_LEN)] + \
                  [cand_cate, cand_renew_mean, cand_sim, cand_pop_dept, label]
            samples.append(row)
    return samples

# 7. 多进程加速（可选）
all_samples = []
for uid, seq in tqdm(user_seq.items(), desc='build samples'):
    all_samples.extend(gen_samples(uid, seq))

# 8. 转 DataFrame + 划分训练/验证
cols = ['user_id','cand_book_id'] + \
       [f'last{i}_book_id' for i in range(1,SEQ_LEN+1)] + \
       [f'last{i}_cate2' for i in range(1,SEQ_LEN+1)] + \
       [f'last{i}_renew' for i in range(1,SEQ_LEN+1)] + \
       ['cand_cate2','cand_renew_mean','cand_sim_last1','cand_pop_dept','label']

df = pd.DataFrame(all_samples, columns=cols)
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])

train_df.to_csv(os.path.join(OUT_DIR, 'train_rank.csv'), index=False)
valid_df.to_csv(os.path.join(OUT_DIR, 'valid_rank.csv'), index=False)

print(f'rank 样本构造完成：train {len(train_df)}  valid {len(valid_df)}')