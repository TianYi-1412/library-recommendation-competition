# -*- coding: utf-8 -*-
"""
Item2Vec 训练脚本
依赖: pip install pandas gensim numpy tqdm
"""
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from gensim.models import Word2Vec
import os

# 1. 路径配置
DATA_DIR = '../data'
INTER_PATH = os.path.join(DATA_DIR, 'inter_preliminary.csv')
BOOK_PATH  = os.path.join(DATA_DIR, 'item.csv')
OUT_DIR = '../output'
os.makedirs(OUT_DIR, exist_ok=True)

# 2. 读取交互数据
inter = pd.read_csv(INTER_PATH)
# 把时间列转成 datetime，再转 Unix 时间戳，用于排序
inter['borrow_time'] = pd.to_datetime(inter['借阅时间'])
inter = inter.sort_values(['user_id', 'borrow_time'])

# 3. 构造用户借阅序列（list of list）
user_seq = inter.groupby('user_id')['book_id'].apply(list).tolist()
print(f'共 {len(user_seq)} 个用户序列，总交互 {len(inter)} 条')

# 4. 训练 Item2Vec
emb_dim = 64
window  = 5          # 可调，建议 3~7
min_cnt = 1          # 冷门书也保留
sg      = 1          # 1=Skip-gram, 0=CBOW
workers = 8

model = Word2Vec(
    sentences=user_seq,
    vector_size=emb_dim,
    window=window,
    min_count=min_cnt,
    sg=sg,
    workers=workers,
    epochs=20,
    seed=42
)

# 5. 导出 embedding 矩阵
# 5.1 拿到所有 book_id 列表
books = pd.read_csv(BOOK_PATH)['book_id'].tolist()
max_id = max(books)                 # 假设 book_id 是 1~N 的整数
emb_matrix = np.zeros((max_id + 1, emb_dim), dtype=np.float32)

oov_cnt = 0
for bid in tqdm(books, desc='fill emb matrix'):
    if str(bid) in model.wv:        # gensim 默认 key 是 str
        emb_matrix[bid] = model.wv[str(bid)]
    else:
        oov_cnt += 1
print(f'oov 书籍数: {oov_cnt} / {len(books)}')

np.save(os.path.join(OUT_DIR, 'book_emb_64.npy'), emb_matrix)

# 6. 导出 {book_id: vector} 字典
book2vec = {}
for bid in books:
    key = str(bid)
    book2vec[bid] = model.wv[key] if key in model.wv else np.zeros(emb_dim)
with open(os.path.join(OUT_DIR, 'book2vec_dict.pkl'), 'wb') as f:
    pickle.dump(book2vec, f)

# 7. 保存 gensim 原生模型（可选）
model.save(os.path.join(OUT_DIR, 'item2vec_gensim.model'))

print('Item2Vec 训练完成，已输出至 output/ 目录')