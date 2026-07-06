# -*- coding: utf-8 -*-
import pandas as pd
import lightgbm as lgb
import joblib
import os

OUT_DIR = '../output'
MODEL_PATH = os.path.join(OUT_DIR, 'lgb_rank.model')

# 1. 读取样本
train_df = pd.read_csv(os.path.join(OUT_DIR, 'train_rank.csv'))
valid_df = pd.read_csv(os.path.join(OUT_DIR, 'valid_rank.csv'))

# 2. 特征列
feat_cols = [c for c in train_df.columns if c not in ['user_id','cand_book_id','label']]
cate_cols = [c for c in feat_cols if 'cate2' in c]   # 类别特征
for col in cate_cols:
    train_df[col] = train_df[col].astype('category')
    valid_df[col] = valid_df[col].astype('category')

# 3. 数据集
dtrain = lgb.Dataset(train_df[feat_cols], label=train_df['label'], free_raw_data=False)
dvalid = lgb.Dataset(valid_df[feat_cols], label=valid_df['label'], reference=dtrain, free_raw_data=False)

# 4. 参数
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'seed': 42
}

# 5. 训练
model = lgb.train(
    params,
    dtrain,
    num_boost_round=5000,
    valid_sets=[dvalid],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)
cat_feats = [col for col in train_df.columns if train_df[col].dtype.name == 'category']
joblib.dump(cat_feats, os.path.join(OUT_DIR, 'lgb_cat_features.pkl'))
print('已保存 categorical 列列表：', cat_feats)
# 6. 保存
joblib.dump(model, MODEL_PATH)
print('LightGBM 排序模型已保存至', MODEL_PATH)