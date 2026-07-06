# fixed_diversity_main.py
import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import time
import traceback
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from configs import FixedBaseConfig
    from data_processor import EnhancedDataProcessor
    from model_trainer import FixedModelTrainer
    from evaluator import EnhancedRenewAwareEvaluator
    from utils import save_candidates
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有文件都在同一目录下")


def fixed_diversity_main():
    """修复的多样性增强主程序"""
    try:
        print("=" * 60)
        print("开始构建修复版多样性增强推荐系统...")
        print("=" * 60)
        start_time = time.time()

        # 步骤1: 配置
        base_config = FixedBaseConfig()
        print(f"使用设备: {base_config.device}")

        # 步骤2: 数据加载和预处理
        data_processor = EnhancedDataProcessor(base_config.data_path)
        book_df, user_df, inter_df = data_processor.load_data()
        book_df, user_df, inter_df = data_processor.enhanced_preprocess_data()

        # 步骤3: 数据划分
        train_df, valid_df, test_df = data_processor.time_aware_split()

        # 步骤4: 为RecBole准备数据
        full_inter_df = data_processor.save_data_for_recbole(train_df, valid_df, test_df)

        # 步骤5: 分析续借行为
        evaluator = EnhancedRenewAwareEvaluator()
        renew_stats = evaluator.analyze_enhanced_renew_behavior(full_inter_df, book_df)

        # 步骤6: 训练模型 - 修复版本
        stable_models = ['ItemKNN']
        model_results = {}
        all_recommendations = {}
        failed_models = []

        print(f"计划训练的模型: {stable_models}")

        for model_name in stable_models:
            print(f"\n--- 训练模型: {model_name} ---")

            try:
                # 获取模型配置
                config_dict = base_config.to_dict(model_name)

                # 训练和评估模型
                trainer = FixedModelTrainer(config_dict)

                # 准备数据
                if not trainer.prepare_data():
                    print(f"❌ 模型 {model_name} 数据准备失败")
                    failed_models.append(model_name)
                    continue

                # 训练模型
                print("开始训练模型...")
                valid_score, valid_result = trainer.train()
                test_result = trainer.evaluate()

                model_results[model_name] = {
                    'valid_score': valid_score,
                    'valid_result': valid_result,
                    'test_result': test_result,
                    'trainer': trainer
                }

                # 生成推荐结果 - 使用修复的方法
                test_users = test_df['user_id'].unique().tolist()
                print(f"为 {len(test_users)} 个测试用户生成推荐...")

                recommendations = fixed_generate_recommendations(trainer, test_users, top_k=10)

                if recommendations and len(recommendations) > 0:
                    all_recommendations[model_name] = recommendations
                    print(f"✅ 模型 {model_name} 训练成功，为 {len(recommendations)} 用户生成有效推荐")
                else:
                    print(f"⚠️ 模型 {model_name} 生成了空推荐")
                    failed_models.append(model_name)

            except Exception as e:
                print(f"❌ 模型 {model_name} 训练失败: {e}")
                failed_models.append(model_name)

        # 检查成功模型
        successful_models = list(all_recommendations.keys())
        print(f"\n📊 训练总结:")
        print(f"  成功模型: {successful_models} ({len(successful_models)}个)")
        print(f"  失败模型: {failed_models}")

        # 确保至少有一个成功模型
        if not successful_models:
            print("\n❌ 所有模型训练失败，启用紧急方案...")
            # 使用简单的Pop推荐
            final_recommendations = generate_emergency_recommendations(test_df, book_df)
        else:
            # 步骤7: 多样性增强集成
            print(f"\n{'=' * 60}")
            print("开始多样性增强集成...")
            print(f"{'=' * 60}")

            if len(successful_models) >= 2:
                print(f"使用 {len(successful_models)} 个模型进行集成: {successful_models}")

                # 使用修复的多样性融合
                ensemble_recommendations = fixed_diversity_fusion(
                    all_recommendations, model_results, book_df, top_k=5
                )

                if ensemble_recommendations:
                    # 应用续借感知调整
                    final_recommendations = evaluator.enhanced_adjust_recommendations(
                        ensemble_recommendations, book_df, top_k=1
                    )
                else:
                    print("❌ 集成失败，使用最佳模型")
                    best_model = select_best_model(model_results, successful_models)
                    final_recommendations = all_recommendations[best_model]
            else:
                # 单一模型
                best_model = successful_models[0]
                print(f"使用单一模型 {best_model}")
                final_recommendations = all_recommendations[best_model]

        # 分析最终推荐多样性
        analyze_final_diversity(final_recommendations)

        # 步骤8: 保存结果
        output_dir = './fixed_outputs'
        os.makedirs(output_dir, exist_ok=True)
        save_candidates(final_recommendations, './fixed_outputs/candidates_10.csv')


        # 步骤9: 计算性能
        print(f"\n{'=' * 60}")
        print("计算最终性能...")
        print(f"{'=' * 60}")

        fixed_calculate_f1(test_df, save_candidates, successful_models)

        # 计算总运行时间
        total_time = time.time() - start_time
        print(f"\n⏱️ 总运行时间: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")

        # 生成报告
        generate_fixed_diversity_report(model_results, all_recommendations, final_recommendations, output_dir)

        print("🎉 修复版多样性增强推荐系统构建完成！")

    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        traceback.print_exc()


def fixed_generate_recommendations(trainer, test_users, top_k=10):
    """修复的推荐生成方法"""
    try:
        recommendations = trainer.generate_recommendations(user_list=test_users, top_k=top_k)
        return fixed_process_recommendations(recommendations)
    except Exception as e:
        print(f"推荐生成失败: {e}")
        return {}


def fixed_process_recommendations(recommendations):
    """修复的推荐结果处理"""
    if recommendations is None:
        return {}

    valid_recommendations = {}

    for user_id, items in recommendations.items():
        # 修复：使用明确的类型检查
        if items is None:
            continue

        # 修复：检查是否为可迭代对象且有内容
        try:
            if hasattr(items, '__len__') and len(items) > 0:
                # 转换为列表
                if hasattr(items, 'tolist'):  # numpy数组
                    items_list = items.tolist()
                elif isinstance(items, (list, tuple)):
                    items_list = list(items)
                else:
                    items_list = [items]

                # 过滤有效物品
                valid_items = []
                for item in items_list:
                    try:
                        item_str = str(item).strip()
                        # 检查是否为有效物品ID
                        if (item_str and
                                item_str != '[PAD]' and
                                item_str != 'nan' and
                                item_str.isdigit()):
                            valid_items.append(item_str)
                    except (ValueError, TypeError):
                        continue

                if valid_items:
                    valid_recommendations[str(user_id)] = valid_items
        except Exception as e:
            print(f"处理用户 {user_id} 推荐时出错: {e}")
            continue

    return valid_recommendations


def fixed_diversity_fusion(all_recommendations, model_results, book_df, top_k=5):
    """修复的多样性融合"""
    print("使用修复的多样性融合...")

    if not all_recommendations:
        return {}

    # 获取所有用户
    all_users = set()
    for recommendations in all_recommendations.values():
        all_users.update(recommendations.keys())

    all_users = list(all_users)
    print(f"处理用户数: {len(all_users)}")

    # 计算物品流行度
    item_popularity = calculate_simple_popularity(all_recommendations)

    # 获取分类信息
    item_category_map = {}
    if book_df is not None and 'item_id' in book_df.columns and 'category1' in book_df.columns:
        item_category_map = book_df.set_index('item_id')['category1'].to_dict()

    ensemble_recommendations = {}

    for user_id in all_users:
        # 收集所有模型的推荐
        candidate_items = {}

        for model_name, recommendations in all_recommendations.items():
            if user_id in recommendations:
                items = recommendations[user_id]
                if items:
                    # 为每个物品计分
                    for rank, item in enumerate(items):
                        if item not in candidate_items:
                            candidate_items[item] = 0

                        # 基础分数：模型权重 + 排名衰减
                        model_weight = get_model_weight(model_name, model_results)
                        rank_score = 1.0 / (rank + 1)  # 排名越高分数越高
                        candidate_items[item] += model_weight * rank_score

        if candidate_items:
            # 多样性调整
            diversity_adjusted_items = apply_diversity_adjustment(
                candidate_items, item_popularity, item_category_map, top_k
            )
            ensemble_recommendations[user_id] = diversity_adjusted_items
        else:
            # 使用第一个模型的推荐
            for recommendations in all_recommendations.values():
                if user_id in recommendations and recommendations[user_id]:
                    ensemble_recommendations[user_id] = recommendations[user_id][:top_k]
                    break

    print(f"修复的多样性融合完成，为 {len(ensemble_recommendations)} 用户生成推荐")
    return ensemble_recommendations


def calculate_simple_popularity(all_recommendations):
    """计算简单的物品流行度"""
    all_items = []
    for recommendations in all_recommendations.values():
        for items in recommendations.values():
            all_items.extend(items)

    if not all_items:
        return {}

    total = len(all_items)
    item_counts = Counter(all_items)

    popularity = {}
    for item, count in item_counts.items():
        popularity[item] = count / total

    return popularity


def get_model_weight(model_name, model_results):
    """获取模型权重"""
    # 默认权重
    default_weights = {
        'Pop': 0.1,  # Pop模型权重最低
        'ItemKNN': 0.6,
        'BPR': 0.3,
    }

    if model_name in model_results and model_results[model_name] is not None:
        result = model_results[model_name].get('test_result', {})
        recall = result.get('recall@10', 0.1)
        # 基于性能调整权重
        perf_weight = min(1.0, recall * 2)  # 归一化到0-1
        return default_weights.get(model_name, 0.2) * (0.5 + 0.5 * perf_weight)

    return default_weights.get(model_name, 0.2)


def apply_diversity_adjustment(candidate_items, item_popularity, item_category_map, top_k):
    """应用多样性调整"""
    # 按原始分数排序
    sorted_items = sorted(candidate_items.items(), key=lambda x: x[1], reverse=True)

    selected_items = []
    selected_categories = set()

    # 第一轮：选择top物品，考虑多样性
    for item, score in sorted_items:
        if len(selected_items) >= top_k:
            break

        category = item_category_map.get(item, '未知')

        # 多样性检查
        if category not in selected_categories:
            selected_items.append(item)
            selected_categories.add(category)
        else:
            # 检查是否应该替换已有的同分类物品
            popularity = item_popularity.get(item, 0)
            if popularity < 0.05:  # 不流行的物品有优势
                # 找到同分类中分数最低的物品
                replace_index = -1
                min_score = float('inf')
                for i, sel_item in enumerate(selected_items):
                    sel_category = item_category_map.get(sel_item, '未知')
                    if sel_category == category and candidate_items[sel_item] < min_score:
                        min_score = candidate_items[sel_item]
                        replace_index = i

                # 如果当前物品分数足够高，替换
                if replace_index != -1 and score > min_score * 1.2:
                    selected_items[replace_index] = item

    # 如果还不够，按分数补足
    if len(selected_items) < top_k:
        for item, score in sorted_items:
            if item not in selected_items:
                selected_items.append(item)
                if len(selected_items) >= top_k:
                    break

    return selected_items[:top_k]


def select_best_model(model_results, successful_models):
    """选择最佳模型"""
    best_model = None
    best_score = -1

    for model_name in successful_models:
        if (model_name in model_results and
                model_results[model_name] is not None and
                'test_result' in model_results[model_name]):

            result = model_results[model_name]['test_result']
            recall = result.get('recall@1', 0)

            if recall > best_score:
                best_score = recall
                best_model = model_name

    return best_model if best_model else successful_models[0]


def analyze_final_diversity(recommendations):
    """分析最终推荐多样性"""
    if not recommendations:
        print("最终推荐: 无结果")
        return

    all_items = []
    for items in recommendations.values():
        if items:  # 修复：确保items不为空
            all_items.extend(items)

    if not all_items:
        print("最终推荐: 空列表")
        return

    total = len(all_items)
    unique = len(set(all_items))
    diversity = unique / total

    item_counts = Counter(all_items)
    top_items = item_counts.most_common(10)

    print(f"\n最终推荐多样性分析:")
    print(f"  总推荐数: {total}")
    print(f"  唯一物品数: {unique}")
    print(f"  多样性分数: {diversity:.4f}")
    print(f"  前10推荐分布:")

    for i, (item, count) in enumerate(top_items, 1):
        percentage = count / total * 100
        print(f"    {i:2d}. 物品 {item}: {count:3d}次 ({percentage:5.1f}%)")


def generate_emergency_recommendations(test_df, book_df):
    """生成紧急推荐（当所有模型失败时）"""
    print("使用紧急推荐方案...")

    test_users = test_df['user_id'].unique().tolist()
    recommendations = {}

    # 基于用户历史生成简单推荐
    user_history = test_df.groupby('user_id')['item_id'].apply(list).to_dict()

    for user_id in test_users:
        if user_id in user_history and user_history[user_id]:
            # 使用用户最近借阅的书籍
            recent_books = user_history[user_id][-3:]  # 最近3本
            recommendations[str(user_id)] = recent_books[:1]  # 取最近1本
        else:
            # 使用随机推荐
            if book_df is not None and len(book_df) > 0:
                random_book = str(book_df.sample(1)['item_id'].iloc[0])
                recommendations[str(user_id)] = [random_book]
            else:
                recommendations[str(user_id)] = ['1']  # 默认书籍

    print(f"紧急推荐为 {len(recommendations)} 用户生成推荐")
    return recommendations


def fixed_calculate_f1(test_df, pred_file, successful_models):
    """修复的F1计算"""
    print("\n计算修复的性能指标...")

    # 构造真实标签
    truth = test_df.groupby('user_id')['item_id'].first().to_dict()
    print(f"真实标签用户数: {len(truth)}")

    try:
        # 读取推荐结果
        pred_df = pd.read_csv(pred_file)
        print(f"推荐结果行数: {len(pred_df)}")

        # 检查列名
        print(f"推荐结果列名: {pred_df.columns.tolist()}")

        pred = dict(zip(pred_df['user_id'].astype(str), pred_df['book_id'].astype(str)))
        print(f"推荐结果用户数: {len(pred)}")

        # 对齐用户
        common = set(truth.keys()) & set(pred.keys())
        print(f"共同用户数: {len(common)}")

        if not common:
            print("❌ 没有共同用户")
            return 0.0

        y_true = [str(truth[u]) for u in common]
        y_pred = [str(pred[u]) for u in common]

        # 计算 F1@1
        f1 = f1_score(y_true, y_pred, average='micro')

        # 额外指标
        correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct_predictions / len(y_true) if y_true else 0

        print(f"📊 修复的性能指标:")
        print(f"   F1@1 分数: {f1:.4f}")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   覆盖用户数: {len(common)}/{len(truth)}")
        print(f"   正确预测数: {correct_predictions}/{len(y_true)}")

        return f1

    except Exception as e:
        print(f"❌ 计算F1分数失败: {e}")
        return 0.0


def generate_fixed_diversity_report(model_results, all_recommendations, final_recommendations, output_dir):
    """生成修复的多样性报告"""
    report_file = os.path.join(output_dir, 'fixed_diversity_report.txt')

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("修复版多样性增强推荐系统报告\n")
        f.write("=" * 50 + "\n\n")

        # 最终结果分析
        all_final_items = []
        for items in final_recommendations.values():
            if items:
                all_final_items.extend(items)

        if all_final_items:
            total = len(all_final_items)
            unique = len(set(all_final_items))
            diversity = unique / total

            f.write(f"最终推荐统计:\n")
            f.write(f"  总推荐数: {total}\n")
            f.write(f"  唯一物品数: {unique}\n")
            f.write(f"  多样性分数: {diversity:.4f}\n\n")

            # 热门推荐分布
            item_counts = Counter(all_final_items)
            top_20 = item_counts.most_common(20)

            f.write(f"前20热门推荐分布:\n")
            for item, count in top_20:
                percentage = count / total * 100
                f.write(f"  物品 {item}: {count}次 ({percentage:.1f}%)\n")

    print(f"📊 修复的多样性报告已保存: {report_file}")


if __name__ == "__main__":
    fixed_diversity_main()