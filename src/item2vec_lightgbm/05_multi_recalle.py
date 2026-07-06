# 05_multi_recalle.py
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from apyori import apriori

# 1. 读取
inter = pd.read_csv('../data/inter_preliminary.csv')
inter['borrow_time'] = pd.to_datetime(inter['借阅时间'])
book = pd.read_csv('../data/item.csv')
user = pd.read_csv('../data/user.csv')

# 2. 阶段划分
def time2phase(dt):
    month, day = dt.month, dt.day
    if (month == 9 and day >= 1) or month == 10:
        return 'autumn_start'
    if (month == 11 and day >= 15) or month == 12:
        return 'autumn_final'
    if (month == 3 and day <= 15):
        return 'spring_start'
    if month == 6:
        return 'spring_final'
    return 'normal'

inter['phase'] = inter['borrow_time'].apply(time2phase)

# 3. 环境热度 Top50
env_hot = pd.read_pickle('../output+hj/env_hot.pkl')

def env_top50(uid, dept):
    last_time = inter[inter.user_id == uid]['borrow_time'].iloc[-1]
    ph = time2phase(last_time)
    sub = env_hot[(env_hot.phase == ph) & (env_hot.DEPT == dept)]
    return sub.nlargest(50, 'env_hot')['book_id'].tolist()

# 4. Item2Vec Top50  （🔧 修复版）
model = pickle.load(open('../output+hj/item2vec_gensim.model', 'rb'))

def ivec_top50(uid, hist):
    last = hist[-1]
    # 防御：如果 last 不在词汇表，用全局热门前 50 代替
    if str(last) not in model.wv:
        global_top = inter['book_id'].value_counts().head(100).index.tolist()
        return [int(b) for b in global_top if int(b) not in hist][:50]
    return [int(b) for b, _ in model.wv.most_similar(str(last), topn=100)
            if int(b) not in hist][:50]

# 5. Apriori 期末共现 Top30
final_seq = (
    inter[inter.phase.isin(['autumn_final', 'spring_final'])]
    .sort_values(['user_id', 'borrow_time'])
    .groupby('user_id')['book_id']
    .apply(list)
    .tolist()
)
rules = apriori(final_seq, min_support=0.003, min_confidence=0.3)
apriori_books = set()
for r in rules:
    stats = r[1] if isinstance(r[1], (list, tuple)) else [r[1]]
    for stat in stats:
        if isinstance(stat, float):
            continue
        if getattr(stat, 'confidence', 0) >= 0.3:
            apriori_books.add(int(stat.items_add[0]))
apriori_top30 = list(apriori_books)[:30]
pd.Series(apriori_top30).to_pickle('../output+hj/apriori_final_top30.pkl')

# 6. 逐用户生成候选池
user_dept = user.set_index('借阅人')['DEPT'].to_dict()
user_hist = inter.groupby('user_id')['book_id'].apply(list).to_dict()  # 已有
cand_dict = {}
for uid in tqdm(user_hist, desc='multi-recall'):
    hist = user_hist[uid]
    dept = user_dept.get(uid, 'Unknown')
    c1 = env_top50(uid, dept)
    c2 = ivec_top50(uid, hist)
    c3 = apriori_top30
    pool = list(set(c1 + c2 + c3) - set(hist))[:80]
    cand_dict[uid] = pool

pd.to_pickle(cand_dict, '../output+hj/cand_dict_80.pkl')
# ✅ 新增：保存历史序列
pd.to_pickle(user_hist, '../output+hj/user_hist.pkl')
print('多路召回完成，已保存 cand_dict_80.pkl 和 user_hist.pkl')