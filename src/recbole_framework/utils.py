import os
import pandas as pd
from typing import Dict

def save_candidates(recommendations: Dict, output_file: str = 'candidates.csv'):
    """保存 10 本候选集，供后续重排"""
    print(f"保存 10 本候选集到 {output_file}...")
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('user_id,book_id\n')
        for user_id, items in recommendations.items():
            for item in items[:10]:          # 取前 10
                f.write(f'{user_id},{item}\n')
    print(f"✅ 候选集已保存：{output_file}，共 {sum(len(v) for v in recommendations.values())} 条")

def calculate_f1_score(test_results: Dict) -> float:
    """计算F1分数"""
    precision = test_results.get('precision@1', 0)
    recall = test_results.get('recall@1', 0)

    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score
    return 0.0

def compare_models(model_results: Dict, metric: str = 'Recall@1') -> str:
    """比较模型性能"""
    return compare_enhanced_models(model_results, metric)

def compare_enhanced_models(model_results: Dict, metric: str = 'Recall@1') -> str:
    """比较增强的模型性能"""
    best_model = None
    best_score = -1

    for model_name, result in model_results.items():
        if result is not None and 'test_result' in result:
            score = result['test_result'].get(metric, -1)
            if score > best_score:
                best_score = score
                best_model = model_name

    if best_model:
        print(f"🎯 最佳模型: {best_model}, {metric}: {best_score:.4f}")
    else:
        print("⚠️ 没有找到有效的模型")

    return best_model