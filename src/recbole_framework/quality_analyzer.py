# quality_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


class RecommendationQualityAnalyzer:
    """推荐质量分析器"""

    def __init__(self):
        self.analysis_results = {}

    def analyze_recommendation_quality(self, all_recommendations: Dict,
                                       user_df: pd.DataFrame, item_df: pd.DataFrame) -> Dict:
        """分析推荐质量"""
        print("分析推荐质量...")

        quality_report = {}

        for model_name, recommendations in all_recommendations.items():
            if not recommendations:
                continue

            analysis = self._analyze_single_model(recommendations, user_df, item_df)
            quality_report[model_name] = analysis

            print(f"  {model_name}:")
            print(f"    多样性: {analysis['diversity']:.4f}")
            print(f"    覆盖率: {analysis['coverage']:.4f}")
            print(f"    集中度: {analysis['concentration']:.4f}")
            print(f"    热门比例: {analysis['popular_ratio']:.4f}")

        self.analysis_results = quality_report
        return quality_report

    def _analyze_single_model(self, recommendations: Dict, user_df: pd.DataFrame,
                              item_df: pd.DataFrame) -> Dict:
        """分析单个模型的推荐质量"""
        # 收集所有推荐物品
        all_recommended_items = []
        user_coverage = len(recommendations)

        for user_items in recommendations.values():
            all_recommended_items.extend(user_items)

        if not all_recommended_items:
            return {
                'diversity': 0.0,
                'coverage': 0.0,
                'concentration': 1.0,
                'popular_ratio': 0.0,
                'user_coverage': 0
            }

        total_recommendations = len(all_recommended_items)
        item_counts = Counter(all_recommended_items)
        unique_items = len(item_counts)

        # 计算多样性
        diversity = unique_items / total_recommendations

        # 计算覆盖率
        total_items = len(item_df) if item_df is not None else unique_items
        coverage = unique_items / total_items if total_items > 0 else 0

        # 计算集中度（基尼系数）
        concentration = self._calculate_gini_coefficient(item_counts)

        # 计算热门物品比例（假设前10%为热门）
        popular_threshold = total_items * 0.1 if total_items > 0 else 10
        popular_items = set(item_df['item_id'].iloc[:popular_threshold]) if item_df is not None else set()
        popular_recommendations = sum(1 for item in all_recommended_items if item in popular_items)
        popular_ratio = popular_recommendations / total_recommendations

        return {
            'diversity': diversity,
            'coverage': coverage,
            'concentration': concentration,
            'popular_ratio': popular_ratio,
            'user_coverage': user_coverage,
            'total_recommendations': total_recommendations,
            'unique_items': unique_items
        }

    def _calculate_gini_coefficient(self, item_counts: Dict) -> float:
        """计算基尼系数（衡量集中度）"""
        if not item_counts:
            return 0.0

        counts = list(item_counts.values())
        counts.sort()
        n = len(counts)

        if n == 0:
            return 0.0

        # 计算基尼系数
        total = sum(counts)
        if total == 0:
            return 0.0

        gini_sum = 0
        for i, count in enumerate(counts):
            gini_sum += (2 * i - n + 1) * count

        gini = gini_sum / (n * total)
        return gini

    def get_model_quality_ranking(self) -> List[Tuple[str, float]]:
        """获取模型质量排名"""
        if not self.analysis_results:
            return []

        quality_scores = []
        for model_name, analysis in self.analysis_results.items():
            # 综合质量分数（多样性 + 覆盖率 - 集中度）
            quality_score = (analysis['diversity'] + analysis['coverage'] - analysis['concentration'])
            quality_scores.append((model_name, quality_score))

        # 按质量分数排序
        quality_scores.sort(key=lambda x: x[1], reverse=True)
        return quality_scores

    def detect_biased_models(self, threshold: float = 0.5) -> List[str]:
        """检测有偏见的模型（推荐过于集中）"""
        biased_models = []

        for model_name, analysis in self.analysis_results.items():
            if analysis['concentration'] > threshold or analysis['diversity'] < 0.1:
                biased_models.append(model_name)

        return biased_models