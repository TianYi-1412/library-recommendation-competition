# intelligent_ensemble.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter
from smart_weights import SmartWeightCalculator
from quality_analyzer import RecommendationQualityAnalyzer


class IntelligentEnsembler:
    """智能集成器"""

    def __init__(self):
        self.weight_calculator = SmartWeightCalculator()
        self.quality_analyzer = RecommendationQualityAnalyzer()
        self.final_weights = {}

    def intelligent_fusion(self, all_recommendations: Dict, model_results: Dict,
                           user_df: pd.DataFrame, item_df: pd.DataFrame,
                           top_k: int = 1) -> Dict:
        """智能融合策略"""
        print("开始智能融合...")

        # 分析推荐质量
        quality_report = self.quality_analyzer.analyze_recommendation_quality(
            all_recommendations, user_df, item_df
        )

        # 检测有偏见的模型
        biased_models = self.quality_analyzer.detect_biased_models()
        if biased_models:
            print(f"检测到有偏见的模型: {biased_models}")

        # 计算物品流行度（用于新颖性计算）
        item_popularity = self._calculate_item_popularity(all_recommendations)
        all_items = item_df['item_id'].tolist() if item_df is not None else []

        # 计算智能权重
        smart_weights = self.weight_calculator.calculate_smart_weights(
            model_results, all_recommendations, all_items, item_popularity
        )

        self.final_weights = smart_weights

        # 执行融合
        ensemble_recommendations = self._execute_fusion(all_recommendations, smart_weights, top_k)

        # 分析融合结果
        self._analyze_fusion_results(ensemble_recommendations, all_recommendations)

        return ensemble_recommendations

    def _calculate_item_popularity(self, all_recommendations: Dict) -> Dict:
        """计算物品流行度"""
        all_recommended_items = []
        for recommendations in all_recommendations.values():
            for user_items in recommendations.values():
                all_recommended_items.extend(user_items)

        if not all_recommended_items:
            return {}

        total_recommendations = len(all_recommended_items)
        item_counts = Counter(all_recommended_items)

        # 归一化流行度
        item_popularity = {}
        for item, count in item_counts.items():
            item_popularity[item] = count / total_recommendations

        return item_popularity

    def _execute_fusion(self, all_recommendations: Dict, weights: Dict, top_k: int) -> Dict:
        """执行融合"""
        ensemble_recommendations = {}
        common_users = self._get_common_users(all_recommendations)

        for user_id in common_users:
            item_scores = {}

            for model_name, recommendations in all_recommendations.items():
                if user_id in recommendations:
                    model_weight = weights.get(model_name, 0)
                    items = recommendations[user_id]

                    # 为每个物品分配分数
                    for rank, item in enumerate(items):
                        if item not in item_scores:
                            item_scores[item] = 0
                        # 使用指数衰减的排名分数
                        rank_score = np.exp(-rank * 0.5)  # 指数衰减
                        item_scores[item] += model_weight * rank_score

            if item_scores:
                # 按分数排序
                sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
                # 应用多样性提升
                final_items = self._apply_diversity_boost(sorted_items, top_k)
                ensemble_recommendations[user_id] = final_items
            else:
                # 使用最佳模型的推荐
                best_model = max(weights.items(), key=lambda x: x[1])[0]
                ensemble_recommendations[user_id] = all_recommendations[best_model].get(user_id, [])[:top_k]

        return ensemble_recommendations

    def _apply_diversity_boost(self, sorted_items: List[Tuple], top_k: int) -> List:
        """应用多样性提升"""
        if len(sorted_items) <= top_k:
            return [item for item, score in sorted_items]

        selected_items = []
        remaining_items = sorted_items.copy()

        while len(selected_items) < top_k and remaining_items:
            # 选择当前分数最高的物品
            best_item, best_score = remaining_items[0]
            selected_items.append(best_item)

            # 移除已选择的物品
            remaining_items = [(item, score) for item, score in remaining_items if item != best_item]

            # 如果还有剩余位置，可以考虑多样性
            if len(selected_items) < top_k and remaining_items:
                # 稍微偏向多样性，但不要完全牺牲质量
                if len(remaining_items) > 1:
                    # 选择下一个物品时，稍微考虑多样性
                    next_item = remaining_items[0]  # 仍然选择分数最高的
                    selected_items.append(next_item[0])
                    remaining_items = remaining_items[1:]

        return selected_items[:top_k]

    def _get_common_users(self, all_recommendations: Dict) -> List:
        """获取所有模型共有的用户"""
        user_sets = [set(recommendations.keys()) for recommendations in all_recommendations.values()]
        if user_sets:
            common_users = set.intersection(*user_sets)
            return list(common_users)
        return []

    def _analyze_fusion_results(self, ensemble_recommendations: Dict, all_recommendations: Dict):
        """分析融合结果"""
        print("\n融合结果分析:")

        # 分析最终推荐的多样性
        all_final_items = []
        for user_items in ensemble_recommendations.values():
            all_final_items.extend(user_items)

        if all_final_items:
            final_diversity = len(set(all_final_items)) / len(all_final_items)
            print(f"  最终推荐多样性: {final_diversity:.4f}")

            # 分析各模型的贡献
            model_contributions = Counter()
            for user_id, final_items in ensemble_recommendations.items():
                for final_item in final_items:
                    for model_name, recommendations in all_recommendations.items():
                        if user_id in recommendations and final_item in recommendations[user_id]:
                            model_contributions[model_name] += 1

            total_contributions = sum(model_contributions.values())
            if total_contributions > 0:
                print("  各模型贡献度:")
                for model, count in model_contributions.most_common():
                    percentage = count / total_contributions
                    weight = self.final_weights.get(model, 0)
                    print(f"    {model}: {percentage:.2%} (权重: {weight:.4f})")

    def get_fusion_analysis_report(self) -> Dict:
        """获取融合分析报告"""
        return {
            'final_weights': self.final_weights,
            'quality_analysis': self.quality_analyzer.analysis_results,
            'diversity_scores': self.weight_calculator.model_diversity_scores,
            'coverage_scores': self.weight_calculator.model_coverage_scores
        }