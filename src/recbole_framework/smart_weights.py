# smart_weights.py
import numpy as np
from typing import Dict, List
from collections import Counter


class SmartWeightCalculator:
    """智能权重计算器"""

    def __init__(self):
        self.model_diversity_scores = {}
        self.model_coverage_scores = {}
        self.model_novelty_scores = {}

    def calculate_diversity_score(self, recommendations: Dict) -> float:
        """计算推荐多样性分数"""
        if not recommendations:
            return 0.0

        all_recommended_items = []
        for user_items in recommendations.values():
            all_recommended_items.extend(user_items)

        if not all_recommended_items:
            return 0.0

        # 计算推荐物品的分布
        item_counts = Counter(all_recommended_items)
        total_recommendations = len(all_recommended_items)
        unique_items = len(item_counts)

        # 多样性 = 唯一物品数 / 总推荐数
        diversity = unique_items / total_recommendations if total_recommendations > 0 else 0

        # 惩罚过度集中的推荐
        max_count = max(item_counts.values()) if item_counts else 0
        concentration_penalty = max_count / total_recommendations if total_recommendations > 0 else 1.0

        # 最终多样性分数
        final_diversity = diversity * (1 - concentration_penalty * 0.5)

        return max(0.0, final_diversity)

    def calculate_coverage_score(self, recommendations: Dict, all_items: List) -> float:
        """计算推荐覆盖率分数"""
        if not recommendations or not all_items:
            return 0.0

        recommended_items = set()
        for user_items in recommendations.values():
            recommended_items.update(user_items)

        coverage = len(recommended_items) / len(all_items) if all_items else 0
        return coverage

    def calculate_novelty_score(self, recommendations: Dict, item_popularity: Dict) -> float:
        """计算推荐新颖性分数"""
        if not recommendations or not item_popularity:
            return 0.0

        novelty_scores = []
        for user_items in recommendations.values():
            for item in user_items:
                # 物品越不流行，新颖性越高
                popularity = item_popularity.get(item, 0)
                novelty = 1 - popularity
                novelty_scores.append(novelty)

        return np.mean(novelty_scores) if novelty_scores else 0.0

    def calculate_pop_penalty(self, recommendations: Dict, threshold: float = 0.3) -> float:
        """计算Pop模型惩罚因子"""
        if not recommendations:
            return 1.0  # 无惩罚

        all_recommended_items = []
        for user_items in recommendations.values():
            all_recommended_items.extend(user_items)

        if not all_recommended_items:
            return 1.0

        item_counts = Counter(all_recommended_items)
        total_recommendations = len(all_recommended_items)

        # 计算最热门物品的集中度
        if item_counts:
            max_count = max(item_counts.values())
            concentration = max_count / total_recommendations

            # 如果集中度超过阈值，应用惩罚
            if concentration > threshold:
                penalty = 1.0 - (concentration - threshold) / (1 - threshold)
                return max(0.1, penalty)  # 最小惩罚为0.1

        return 1.0

    def calculate_smart_weights(self, model_results: Dict, all_recommendations: Dict,
                                all_items: List, item_popularity: Dict) -> Dict:
        """计算智能权重"""
        print("计算智能模型权重...")

        # 第一阶段：基础性能权重
        performance_weights = self._calculate_performance_weights(model_results)

        # 第二阶段：多样性调整
        diversity_weights = self._calculate_diversity_weights(all_recommendations, performance_weights)

        # 第三阶段：覆盖率和新颖性调整
        quality_weights = self._calculate_quality_weights(all_recommendations, diversity_weights,
                                                          all_items, item_popularity)

        # 第四阶段：特殊模型处理（如Pop模型）
        final_weights = self._apply_special_rules(quality_weights, all_recommendations)

        # 归一化
        final_weights = self._normalize_weights(final_weights)

        print("智能权重计算结果:")
        for model, weight in final_weights.items():
            perf = performance_weights.get(model, 0)
            div = self.model_diversity_scores.get(model, 0)
            print(f"  {model}: 最终权重={weight:.4f}, 性能权重={perf:.4f}, 多样性={div:.4f}")

        return final_weights

    def _calculate_performance_weights(self, model_results: Dict) -> Dict:
        """计算基于性能的权重"""
        scores = {}
        for model_name, result in model_results.items():
            if result is not None and 'test_result' in result:
                # 使用多个指标的综合评分
                recall = result['test_result'].get('recall@10', 0)
                ndcg = result['test_result'].get('ndcg@10', 0)
                precision = result['test_result'].get('precision@10', 0)

                # 加权综合评分
                combined_score = 0.5 * recall + 0.3 * ndcg + 0.2 * precision
                scores[model_name] = combined_score

        return self._normalize_weights(scores)

    def _calculate_diversity_weights(self, all_recommendations: Dict, base_weights: Dict) -> Dict:
        """基于多样性调整权重"""
        diversity_weights = base_weights.copy()

        for model_name, recommendations in all_recommendations.items():
            diversity_score = self.calculate_diversity_score(recommendations)
            self.model_diversity_scores[model_name] = diversity_score

            # 多样性高的模型获得奖励
            diversity_bonus = 1.0 + diversity_score * 0.5
            if model_name in diversity_weights:
                diversity_weights[model_name] *= diversity_bonus

        return self._normalize_weights(diversity_weights)

    def _calculate_quality_weights(self, all_recommendations: Dict, diversity_weights: Dict,
                                   all_items: List, item_popularity: Dict) -> Dict:
        """基于覆盖率和新颖性调整权重"""
        quality_weights = diversity_weights.copy()

        for model_name, recommendations in all_recommendations.items():
            coverage_score = self.calculate_coverage_score(recommendations, all_items)
            novelty_score = self.calculate_novelty_score(recommendations, item_popularity)

            self.model_coverage_scores[model_name] = coverage_score
            self.model_novelty_scores[model_name] = novelty_score

            # 质量和新颖性奖励
            quality_bonus = 1.0 + (coverage_score * 0.3 + novelty_score * 0.2)
            if model_name in quality_weights:
                quality_weights[model_name] *= quality_bonus

        return self._normalize_weights(quality_weights)

    def _apply_special_rules(self, weights: Dict, all_recommendations: Dict) -> Dict:
        """应用特殊规则（如Pop模型惩罚）"""
        adjusted_weights = weights.copy()

        for model_name in weights.keys():
            if model_name.lower() == 'pop':
                # 对Pop模型应用集中度惩罚
                pop_penalty = self.calculate_pop_penalty(all_recommendations.get(model_name, {}))
                adjusted_weights[model_name] *= pop_penalty
                print(f"  Pop模型惩罚因子: {pop_penalty:.4f}")

            # 对其他单一推荐模式的模型也应用惩罚
            elif model_name in all_recommendations:
                diversity = self.model_diversity_scores.get(model_name, 1.0)
                if diversity < 0.1:  # 多样性极低
                    penalty = 0.5  # 严重惩罚
                    adjusted_weights[model_name] *= penalty
                    print(f"  模型 {model_name} 因低多样性被惩罚: {penalty}")

        return adjusted_weights

    def _normalize_weights(self, weights: Dict) -> Dict:
        """归一化权重"""
        if not weights:
            return {}

        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        else:
            # 如果所有权重都为0，使用平均权重
            avg_weight = 1.0 / len(weights)
            return {k: avg_weight for k in weights.keys()}