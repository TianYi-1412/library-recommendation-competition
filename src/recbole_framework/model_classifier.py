#model_classifier.py
from typing import Dict, List, Set
import numpy as np


class ModelClassifier:
    """模型分类器 - 根据模型特性进行分类和组合"""

    def __init__(self):
        self.model_categories = {
            'collaborative_filtering': ['ItemKNN', 'BPR', 'NCF', 'NeuMF'],
            'matrix_factorization': ['BPR', 'NCF', 'FM', 'NeuMF'],
            'deep_learning': ['NeuMF', 'NCF', 'DeepFM', 'AutoInt', 'DSSM'],
            'graph_based': ['LightGCN', 'NGCF'],
            'sequential': ['SASRec', 'GRU4Rec', 'BERT4Rec'],
            'content_based': ['DSSM'],
            'hybrid': ['DeepFM', 'AutoInt'],
            'traditional': ['ItemKNN', 'Pop', 'BPR', 'FM']
        }

        self.model_descriptions = {
            'ItemKNN': '基于物品的协同过滤，适合发现相似物品',
            'BPR': '贝叶斯个性化排序，经典的矩阵分解方法',
            'NeuMF': '神经矩阵分解，结合MF和DNN',
            'SASRec': '自注意力序列推荐，适合序列模式',
            'Pop': '基于流行度的推荐，简单但有效',
            'NGCF': '图卷积网络，捕捉高阶连接',
            'DSSM': '双塔模型，适合内容和协同信号',
            'GRU4Rec': '基于GRU的序列推荐',
            'NCF': '神经协同过滤，广义的矩阵分解',
            'FM': '因子分解机，处理特征交互',
            'DeepFM': '深度因子分解机，结合FM和DNN',
            'AutoInt': '自动特征交互，自注意力机制',
            'BERT4Rec': '基于BERT的序列推荐，双向编码'
        }

    def get_model_categories(self, model_name: str) -> List[str]:
        """获取模型所属的类别"""
        categories = []
        for category, models in self.model_categories.items():
            if model_name in models:
                categories.append(category)
        return categories

    def get_diverse_model_set(self, available_models: List[str]) -> List[str]:
        """选择多样化的模型集合"""
        selected_models = set()
        category_coverage = set()

        # 按类别优先级选择
        priority_categories = ['collaborative_filtering', 'sequential', 'deep_learning', 'graph_based']

        for category in priority_categories:
            category_models = set(self.model_categories[category]) & set(available_models)
            if category_models and category not in category_coverage:
                # 选择该类别中性能预期最好的模型
                best_model = self._select_best_model_in_category(category_models, category)
                if best_model:
                    selected_models.add(best_model)
                    category_coverage.add(category)

        # 确保至少选择3个模型
        if len(selected_models) < 3:
            remaining_models = set(available_models) - selected_models
            needed = 3 - len(selected_models)
            additional_models = list(remaining_models)[:needed]
            selected_models.update(additional_models)

        return list(selected_models)

    def _select_best_model_in_category(self, models: Set[str], category: str) -> str:
        """在类别中选择预期性能最好的模型"""
        # 基于经验给出模型性能预期排序
        performance_ranking = {
            'sequential': ['BERT4Rec', 'SASRec', 'GRU4Rec'],
            'deep_learning': ['AutoInt', 'DeepFM', 'NeuMF', 'NCF'],
            'graph_based': ['LightGCN', 'NGCF'],
            'collaborative_filtering': ['NeuMF', 'NCF', 'BPR', 'ItemKNN'],
            'matrix_factorization': ['NeuMF', 'NCF', 'BPR', 'FM']
        }

        if category in performance_ranking:
            for model in performance_ranking[category]:
                if model in models:
                    return model

        # 如果没有匹配，返回第一个模型
        return list(models)[0] if models else None

    def analyze_model_complementarity(self, model_results: Dict) -> Dict:
        """分析模型的互补性"""
        complementarity_scores = {}

        model_names = [name for name, result in model_results.items() if result is not None]

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:
                    # 计算模型差异度（基于类别差异）
                    categories1 = set(self.get_model_categories(model1))
                    categories2 = set(self.get_model_categories(model2))

                    similarity = len(categories1 & categories2) / len(
                        categories1 | categories2) if categories1 | categories2 else 0
                    complementarity = 1 - similarity

                    pair_key = f"{model1}-{model2}"
                    complementarity_scores[pair_key] = {
                        'complementarity': complementarity,
                        'shared_categories': list(categories1 & categories2),
                        'unique_categories': {
                            model1: list(categories1 - categories2),
                            model2: list(categories2 - categories1)
                        }
                    }

        return complementarity_scores

    def get_recommended_ensembles(self, available_models: List[str]) -> List[List[str]]:
        """推荐模型集成组合"""
        ensembles = []

        # 组合1: 传统 + 深度学习 + 序列
        ensemble1 = []
        for category in ['traditional', 'deep_learning', 'sequential']:
            category_models = set(self.model_categories[category]) & set(available_models)
            if category_models:
                ensemble1.append(self._select_best_model_in_category(category_models, category))
        if ensemble1:
            ensembles.append(ensemble1)

        # 组合2: 协同过滤 + 图网络 + 混合模型
        ensemble2 = []
        for category in ['collaborative_filtering', 'graph_based', 'hybrid']:
            category_models = set(self.model_categories[category]) & set(available_models)
            if category_models:
                ensemble2.append(self._select_best_model_in_category(category_models, category))
        if ensemble2:
            ensembles.append(ensemble2)

        # 组合3: 所有类别各选一个
        ensemble3 = []
        for category in ['collaborative_filtering', 'sequential', 'deep_learning', 'graph_based']:
            category_models = set(self.model_categories[category]) & set(available_models)
            if category_models:
                ensemble3.append(self._select_best_model_in_category(category_models, category))
        if ensemble3:
            ensembles.append(ensemble3)

        return ensembles