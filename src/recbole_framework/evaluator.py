from typing import Dict
import pandas as pd
import numpy as np

class EnhancedRenewAwareEvaluator:
    """增强的续借感知评估器"""

    def __init__(self):
        self.user_renew_rates = {}
        self.item_renew_rates = {}
        self.global_renew_rate = 0
        self.user_categories_pref = {}
        self.item_category_popularity = {}

    def analyze_enhanced_renew_behavior(self, inter_df: pd.DataFrame, book_df: pd.DataFrame) -> Dict:
        """分析增强的续借行为"""
        print("分析增强的用户续借行为...")

        # 合并图书分类信息
        inter_enriched = inter_df.merge(
            book_df[['item_id', 'category1']].drop_duplicates(),
            on='item_id',
            how='left'
        )

        # 填充缺失的分类
        inter_enriched['category1'] = inter_enriched['category1'].fillna('未知')

        # 用户续借分析
        user_renew_stats = inter_enriched.groupby('user_id').agg({
            'has_renewed': ['sum', 'count', 'mean'],
            'renew_count': 'sum'
        }).reset_index()
        user_renew_stats.columns = ['user_id', 'renew_count', 'total_count', 'renew_rate', 'total_renew_times']

        # 用户分类偏好
        user_category_pref = inter_enriched.groupby(['user_id', 'category1']).agg({
            'has_renewed': 'mean',
            'item_id': 'count'
        }).reset_index()
        user_category_pref = user_category_pref.rename(columns={'item_id': 'category_count'})

        # 转换为用户偏好的字典格式
        for user_id in user_category_pref['user_id'].unique():
            user_data = user_category_pref[user_category_pref['user_id'] == user_id]
            pref_dict = dict(zip(user_data['category1'], user_data['has_renewed']))
            self.user_categories_pref[user_id] = pref_dict

        # 物品续借分析
        item_renew_stats = inter_enriched.groupby('item_id').agg({
            'has_renewed': ['sum', 'count', 'mean'],
            'category1': 'first'
        }).reset_index()
        item_renew_stats.columns = ['item_id', 'renew_count', 'total_count', 'renew_rate', 'category1']

        # 分类流行度
        category_popularity = inter_enriched.groupby('category1').agg({
            'has_renewed': 'mean',
            'user_id': 'count'
        }).reset_index()
        self.item_category_popularity = dict(zip(category_popularity['category1'],
                                               category_popularity['has_renewed']))

        self.user_renew_rates = user_renew_stats.set_index('user_id')['renew_rate'].to_dict()
        self.item_renew_rates = item_renew_stats.set_index('item_id')['renew_rate'].to_dict()
        self.global_renew_rate = inter_df['has_renewed'].mean()

        print(f"全局续借率: {self.global_renew_rate:.4f}")
        print(f"分析完成: {len(self.user_renew_rates)} 用户, {len(self.item_renew_rates)} 图书")

        return {
            'user_renew_rates': self.user_renew_rates,
            'item_renew_rates': self.item_renew_rates,
            'global_renew_rate': self.global_renew_rate,
            'user_categories_pref': self.user_categories_pref
        }

    def enhanced_adjust_recommendations(self, recommendations: Dict, book_df: pd.DataFrame, top_k: int = 1) -> Dict:
        """增强的推荐结果调整"""
        print("使用增强策略调整推荐结果...")

        if not self.user_renew_rates:
            print("警告：未找到续借行为数据，返回原始推荐")
            return recommendations

        # 创建物品分类映射
        item_category_map = book_df.set_index('item_id')['category1'].to_dict()

        adjusted_recommendations = {}

        for user_id, items in recommendations.items():
            user_scores = []

            user_renew_rate = self.user_renew_rates.get(user_id, self.global_renew_rate)
            user_category_pref = self.user_categories_pref.get(user_id, {})

            for item_id in items:
                if item_id == '[PAD]' or not str(item_id).isdigit():
                    continue

                try:
                    item_id_int = int(item_id)
                except ValueError:
                    continue

                base_score = 1.0

                # 获取物品续借率和分类
                item_renew_rate = self.item_renew_rates.get(item_id_int, self.global_renew_rate)
                item_category = item_category_map.get(item_id_int, '未知')

                # 多维度调整策略
                adjustment = 1.0

                # 1. 续借率调整
                if user_renew_rate > 0.4:  # 高频续借用户
                    adjustment += item_renew_rate * 0.6
                elif user_renew_rate > 0.2:  # 中频续借用户
                    adjustment += item_renew_rate * 0.3
                else:  # 低频续借用户
                    adjustment += item_renew_rate * 0.1

                # 2. 分类偏好调整
                category_pref = user_category_pref.get(item_category, 0)
                adjustment += category_pref * 0.3

                # 3. 分类流行度调整
                category_pop = self.item_category_popularity.get(item_category, self.global_renew_rate)
                adjustment += category_pop * 0.1

                adjusted_score = base_score * adjustment
                user_scores.append((item_id, adjusted_score))

            # 排序并选择top-k
            if user_scores:
                user_scores.sort(key=lambda x: x[1], reverse=True)
                top_items = [item[0] for item in user_scores[:top_k]]
                adjusted_recommendations[user_id] = top_items
            else:
                # 如果没有有效物品，使用原始推荐
                adjusted_recommendations[user_id] = items[:top_k]

        print(f"成功为 {len(adjusted_recommendations)} 个用户调整推荐")
        return adjusted_recommendations