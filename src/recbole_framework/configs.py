import torch
import os
from typing import Dict, Any

class FixedBaseConfig:
    """修复的基础配置类"""

    def __init__(self):
        # 数据路径
        self.data_path = os.path.abspath('./data')
        self.dataset_name = 'library'

        # 字段配置
        self.user_id_field = 'user_id'
        self.item_id_field = 'item_id'
        self.rating_field = 'rating'
        self.time_field = 'timestamp_unix'

        # 训练配置
        self.epochs = 30
        self.learning_rate = 0.001
        self.train_batch_size = 512  # 减少batch size避免内存问题
        self.eval_batch_size = 1024
        self.early_stop_patience = 3

        # 评估配置
        self.topk = [1, 5, 10]
        self.metrics = ['Recall', 'Precision', 'Hit', 'NDCG', 'MRR']
        self.valid_metric = 'Recall@10'

        # 设备配置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 随机种子
        self.seed = 2023

    def to_dict(self, model_name: str) -> Dict[str, Any]:
        """转换为RecBole配置字典"""
        config_dict = {
            'data_path': self.data_path,
            'dataset': self.dataset_name,
            'USER_ID_FIELD': self.user_id_field,
            'ITEM_ID_FIELD': self.item_id_field,
            'RATING_FIELD': self.rating_field,
            'TIME_FIELD': self.time_field,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'train_batch_size': self.train_batch_size,
            'eval_batch_size': self.eval_batch_size,
            'topk': self.topk,
            'metrics': self.metrics,
            'valid_metric': self.valid_metric,
            'device': self.device,
            'seed': self.seed,
            'reproducibility': True,
            'model': model_name,
            'early_stop_patience': self.early_stop_patience,
        }
        config_dict['load_col'] = {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp_unix']
        }
        config_dict['encoding'] = 'utf-8'
        # 修复的模型配置
        if model_name == 'BPR':
            config_dict.update({
                'embedding_size': 64,
            })
        elif model_name == 'NeuMF':
            config_dict.update({
                'embedding_size': 64,
                'mlp_hidden_size': [64, 32, 16],
                'dropout_prob': 0.2,
            })
        elif model_name == 'SASRec':
            config_dict.update({
                'embedding_size': 64,
                'n_layers': 2,
                'n_heads': 2,
                'hidden_size': 64,
                'inner_size': 128,
                'hidden_dropout_prob': 0.2,
                'attn_dropout_prob': 0.2,
                'hidden_act': 'gelu',
                'layer_norm_eps': 1e-12,
                'MAX_ITEM_LIST_LENGTH': 20,
                'train_neg_sample_args': None,
                'TIME_FIELD': self.time_field,  # ✅ 显式声明
            })
        elif model_name == 'ItemKNN':
            config_dict.update({
                'k': 100,
            })
        elif model_name == 'Pop':
            config_dict.update({})
        elif model_name == 'GRU4Rec':
            config_dict.update({
                'embedding_size': 64,
                'hidden_size': 128,
                'num_layers': 1,
                'dropout_prob': 0.2,
                'train_neg_sample_args': None,
                'TIME_FIELD': self.time_field,  # ✅ 显式声明
            })
        elif model_name == 'NeuMF':
            config_dict.update({
                'embedding_size': 64,
                'mlp_hidden_size': [128, 64, 32],
                'dropout_prob': 0.2,
                'train_neg_sample_args': {'distribution': 'uniform', 'sample_num': 4},
                'metrics': ['Recall'], 'topk': [1], 'valid_metric': 'Recall@1',
            })
        elif model_name == 'Caser':
            config_dict.update({
                'embedding_size': 64,
                'nh': 4,
                'nv': 2,
                'dropout_prob': 0.2,
                'MAX_ITEM_LIST_LENGTH': 20,
                'train_neg_sample_args': None,
                'TIME_FIELD': self.time_field,
            })
        elif model_name == 'BERT4Rec':
            config_dict.update({
                'n_layers': 3,
                'n_heads': 4,
                'hidden_size': 128,
                'embedding_size': 128,
                'hidden_dropout_prob': 0.2,
                'attn_dropout_prob': 0.2,
                'hidden_act': 'gelu',
                'layer_norm_eps': 1e-12,
                'MAX_ITEM_LIST_LENGTH': 80,  # 比 SASRec 长，效果更好
                'train_neg_sample_args': None,
                'TIME_FIELD': self.time_field,
                'mask_ratio': 0.15,  # BERT 式掩码比例
            })
            # 在 configs.txt 中更新 DeepFM 配置
        elif model_name == 'DeepFM':
            config_dict.update({
                'embedding_size': 64,
                'mlp_hidden_size': [400, 400, 400],
                'dropout_prob': 0.1,
                'train_batch_size': 512,  # 减小batch size
                'eval_batch_size': 1024,
                'learning_rate': 0.001,
                'early_stop_patience': 3,
                'load_col': {
                'inter': ['user_id', 'item_id', 'rating', 'timestamp_unix', 'label'],  # ✅ 加了 label
                'user': ['user_id:token', 'gender:token', 'dept:token', 'grade:token', 'user_type:token'],
                'item': ['item_id:token', 'title:token', 'author:token', 'publisher:token', 'category1:token',
                         'category2:token']
            },
                'eval_args': {
                    'split': {'RS': [0.8, 0.1, 0.1]},
                    'group_by': 'user',
                    'order': 'TO',  # 改为时间顺序
                    'mode': 'full',
                },
                'metrics': ['Recall', 'NDCG'],  # 简化评估指标
                'valid_metric': 'Recall@10',
            })





        return config_dict