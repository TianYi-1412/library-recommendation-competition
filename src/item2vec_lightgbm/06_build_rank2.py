# 06_build_rank2.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

SEQ_LEN = 5
NEG_SAMPLES = 20
OUT_DIR = '../output+hj'

# 读取
inter = pd.read_csv('../data/inter_preliminary.csv')
inter['borrow_time'] = pd.to_datetime(inter['借阅时间'])
inter = inter.sort_values(['user_id', 'borrow_time'])
book = pd.read_csv('../data/item.csv')
user = pd.read_csv('../data/user.csv')
inter = inter.merge(book[['book_id', '二级分类']], on='book_id').rename(columns={'二级分类': 'cate2'})
inter['续借次数'] = inter['续借次数'].fillna(0).astype(int)

# 用户序列
user_seq = inter.groupby('user_id').apply(
    lambda x: x[['book_id', 'cate2', '续借次数']].to_numpy()
).to_dict()

# 全局负样本池
all_books = set(book['book_id'])
book_pop = inter['book_id'].value_counts().to_dict()

# 加载召回池
cand_dict = pd.read_pickle(os.path.join(OUT_DIR, 'cand_dict_80.pkl'))

# 构造样本
samples = []
for uid, seq in tqdm(user_seq.items(), desc='build rank2'):
    if len(seq) < SEQ_LEN + 1:
        continue
    for i in range(SEQ_LEN, len(seq)):
        hist = seq[i - SEQ_LEN:i]
        next_book, next_cate, next_renew = seq[i]
        # 正样本
        pos_row = [uid, next_book] + \
                  [int(hist[j][0]) for j in range(SEQ_LEN)] + \
                  [hist[j][1] for j in range(SEQ_LEN)] + \
                  [int(hist[j][2]) for j in range(SEQ_LEN)] + \
                  [next_cate, next_renew, 1]
        samples.append(pos_row)
        # 负样本（从召回池里采 20 本）
        neg_pool = list(set(cand_dict[uid]) - {next_book})
        if len(neg_pool) < NEG_SAMPLES:
            extra = list(all_books - set(seq[:, 0]) - {next_book})
            neg_pool.extend(extra)
        neg_books = np.random.choice(neg_pool, NEG_SAMPLES, replace=False)
        for nb in neg_books:
            neg_cate = book.loc[book.book_id == nb, '二级分类'].values
            neg_cate = neg_cate[0] if len(neg_cate) else 'Unknown'
            neg_renew = inter[inter.book_id == nb]['续借次数'].mean()
            neg_row = [uid, nb] + \
                      [int(hist[j][0]) for j in range(SEQ_LEN)] + \
                      [hist[j][1] for j in range(SEQ_LEN)] + \
                      [int(hist[j][2]) for j in range(SEQ_LEN)] + \
                      [neg_cate, neg_renew, 0]
            samples.append(neg_row)

# 保存
cols = ['user_id', 'cand_book_id'] + \
       [f'last{i}_book_id' for i in range(1, SEQ_LEN + 1)] + \
       [f'last{i}_cate2' for i in range(1, SEQ_LEN + 1)] + \
       [f'last{i}_renew' for i in range(1, SEQ_LEN + 1)] + \
       ['cand_cate2', 'cand_renew_mean', 'label']
df = pd.DataFrame(samples, columns=cols)
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
train_df.to_csv(os.path.join(OUT_DIR, 'train_rank2.csv'), index=False)
valid_df.to_csv(os.path.join(OUT_DIR, 'valid_rank2.csv'), index=False)
print('rank2 样本完成', len(train_df), len(valid_df))