import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings('ignore')


class SequenceSampleGenerator:
    def __init__(self, user_sequences, book_info_dict, user_df=None):
        """
        初始化序列样本生成器

        Args:
            user_sequences: 用户借阅序列字典
            book_info_dict: 图书信息字典
            user_df: 用户信息DataFrame（可选）
        """
        self.user_sequences = user_sequences
        self.book_info_dict = book_info_dict
        self.user_df = user_df

        # 统计信息
        self.stats = {}

        # 构建图书到类别的映射
        self.book_to_category = {}
        for book_id, info in book_info_dict.items():
            self.book_to_category[book_id] = {
                'category_1': info.get('category_1', '未知'),
                'category_2': info.get('category_2', '未知')
            }

        # 预计算图书流行度（优化性能）
        self._precompute_popularity()

    def _precompute_popularity(self):
        """预计算图书流行度，优化负采样性能"""
        print("预计算图书流行度...")
        self.book_popularity = Counter()
        for user_data in self.user_sequences.values():
            self.book_popularity.update(user_data['book_sequence'])

        # 构建流行度分桶
        self.popular_books = [book for book, count in self.book_popularity.most_common(1000)]
        self.mid_popular_books = [book for book, count in self.book_popularity.items()
                                  if 5 <= count <= 50 and book not in self.popular_books]
        self.low_popular_books = [book for book, count in self.book_popularity.items()
                                  if
                                  count < 5 and book not in self.popular_books and book not in self.mid_popular_books]

        print(
            f"流行度分桶完成: 热门{len(self.popular_books)}, 中等{len(self.mid_popular_books)}, 冷门{len(self.low_popular_books)}")

    def generate_fixed_length_samples(self, sequence_length=5, step_size=1, min_sequence_length=6):
        """
        生成固定长度的序列样本（基础方法）

        Args:
            sequence_length: 输入序列长度
            step_size: 滑动窗口步长
            min_sequence_length: 用户最小序列长度要求

        Returns:
            samples: 样本字典
        """
        print(f"生成固定长度样本 - 序列长度: {sequence_length}, 步长: {step_size}")

        sequences = []  # 输入序列
        targets = []  # 目标图书
        user_ids = []  # 用户ID
        sequence_info = []  # 序列附加信息
        valid_users = 0

        for user_id, user_data in self.user_sequences.items():
            book_sequence = user_data['book_sequence']
            time_sequence = user_data['time_sequence']

            # 检查序列长度是否足够
            if len(book_sequence) < min_sequence_length:
                continue

            valid_users += 1

            # 使用滑动窗口创建样本
            for i in range(sequence_length, len(book_sequence), step_size):
                # 输入序列：最近的sequence_length次借阅
                input_seq = book_sequence[i - sequence_length:i]
                input_times = time_sequence[i - sequence_length:i]

                # 目标：下一次借阅
                target_book = book_sequence[i]
                target_time = time_sequence[i]

                sequences.append(input_seq)
                targets.append(target_book)
                user_ids.append(user_id)

                # 保存序列的附加信息
                seq_info = {
                    'input_times': input_times,
                    'target_time': target_time,
                    'time_gap': (target_time - input_times[-1]).days if input_times else 0,
                    'sequence_position': i,  # 在用户序列中的位置
                    'user_sequence_length': len(book_sequence)
                }
                sequence_info.append(seq_info)

        # 统计信息
        self.stats['fixed_length'] = {
            'valid_users': valid_users,
            'total_samples': len(sequences),
            'avg_samples_per_user': len(sequences) / valid_users if valid_users > 0 else 0,
            'sequence_length': sequence_length
        }

        print(f"固定长度样本生成完成:")
        print(f"  - 有效用户数: {valid_users}")
        print(f"  - 总样本数: {len(sequences)}")
        print(f"  - 平均每用户样本数: {self.stats['fixed_length']['avg_samples_per_user']:.2f}")

        return {
            'sequences': sequences,
            'targets': targets,
            'user_ids': user_ids,
            'sequence_info': sequence_info,
            'type': 'fixed_length'
        }

    def generate_negative_samples_optimized(self, positive_samples, negative_ratio=4, strategy='popularity',
                                            batch_size=10000):
        """
        优化的负样本生成方法 - 解决性能问题

        Args:
            positive_samples: 正样本字典
            negative_ratio: 负样本与正样本的比例
            strategy: 负采样策略 ('popularity', 'random', 'category')
            batch_size: 批处理大小，避免内存溢出

        Returns:
            samples: 包含正负样本的完整数据集
        """
        print(f"生成负样本 (优化版) - 策略: {strategy}, 比例: 1:{negative_ratio}")

        sequences = positive_samples['sequences']
        targets = positive_samples['targets']
        user_ids = positive_samples['user_ids']
        sequence_info = positive_samples['sequence_info']

        num_positive = len(sequences)
        num_negative = num_positive * negative_ratio

        print(f"正样本数: {num_positive}, 需要负样本: {num_negative}")

        # 预计算所有可能用到的数据
        all_books = list(self.book_info_dict.keys())
        print(f"总图书数: {len(all_books)}")

        # 构建类别映射（优化性能）
        category_to_books = defaultdict(list)
        for book_id in all_books:
            category = self.book_to_category.get(book_id, {}).get('category_1', '未知')
            category_to_books[category].append(book_id)

        # 批量生成负样本
        negative_sequences = []
        negative_targets = []
        negative_user_ids = []
        negative_info = []

        # 分批处理避免内存问题
        for batch_start in range(0, num_negative, batch_size):
            batch_end = min(batch_start + batch_size, num_negative)
            batch_size_current = batch_end - batch_start

            if batch_start % 50000 == 0:
                print(f"处理负样本批次: {batch_start}-{batch_end}")

            # 随机选择基础样本索引
            base_indices = np.random.randint(0, num_positive, batch_size_current)

            # 批量生成负样本目标
            if strategy == 'popularity':
                # 优化的流行度采样：使用预计算的分桶
                negative_books_batch = self._sample_by_popularity_batch(batch_size_current)
            elif strategy == 'category':
                # 优化的类别采样
                base_categories = []
                for idx in base_indices:
                    base_seq = sequences[idx]
                    if base_seq:
                        last_book = base_seq[-1]
                        last_category = self.book_to_category.get(last_book, {}).get('category_1', '未知')
                    else:
                        last_category = '未知'
                    base_categories.append(last_category)

                negative_books_batch = self._sample_by_category_batch(base_categories, category_to_books)
            else:  # random
                negative_books_batch = random.choices(all_books, k=batch_size_current)

            # 添加到结果
            for i, base_idx in enumerate(base_indices):
                negative_sequences.append(sequences[base_idx])
                negative_targets.append(negative_books_batch[i])
                negative_user_ids.append(user_ids[base_idx])
                negative_info.append({
                    **sequence_info[base_idx],
                    'is_negative': True,
                    'sampling_strategy': strategy
                })

        # 合并正负样本
        all_sequences = sequences + negative_sequences
        all_targets = targets + negative_targets
        all_user_ids = user_ids + negative_user_ids
        all_info = sequence_info + negative_info

        # 创建标签（1为正样本，0为负样本）
        labels = [1] * len(sequences) + [0] * len(negative_sequences)

        self.stats['negative_sampling'] = {
            'positive_samples': len(sequences),
            'negative_samples': len(negative_sequences),
            'total_samples': len(all_sequences),
            'negative_ratio': negative_ratio,
            'strategy': strategy
        }

        print(f"负样本生成完成:")
        print(f"  - 正样本数: {len(sequences)}")
        print(f"  - 负样本数: {len(negative_sequences)}")
        print(f"  - 总样本数: {len(all_sequences)}")
        print(f"  - 正负比例: 1:{negative_ratio}")

        return {
            'sequences': all_sequences,
            'targets': all_targets,
            'user_ids': all_user_ids,
            'sequence_info': all_info,
            'labels': labels,
            'type': f"{positive_samples.get('type', 'unknown')}_with_negative"
        }

    def _sample_by_popularity_batch(self, batch_size):
        """批量流行度采样（优化性能）"""
        # 使用预计算的分桶进行采样
        weights = [0.6, 0.3, 0.1]  # 热门:中等:冷门的采样权重
        buckets = [self.popular_books, self.mid_popular_books, self.low_popular_books]

        # 选择桶
        bucket_choices = random.choices([0, 1, 2], weights=weights, k=batch_size)

        negative_books = []
        for bucket_idx in bucket_choices:
            bucket = buckets[bucket_idx]
            if bucket:
                negative_books.append(random.choice(bucket))
            else:
                # 如果桶为空，从所有图书中随机选择
                negative_books.append(
                    random.choice(self.popular_books + self.mid_popular_books + self.low_popular_books))

        return negative_books

    def _sample_by_category_batch(self, base_categories, category_to_books):
        """批量类别采样（优化性能）"""
        negative_books = []

        for base_category in base_categories:
            # 获取所有其他类别的图书
            other_categories = [cat for cat in category_to_books.keys() if cat != base_category]

            if other_categories:
                # 随机选择一个不同类别
                selected_category = random.choice(other_categories)
                books_in_category = category_to_books[selected_category]

                if books_in_category:
                    negative_books.append(random.choice(books_in_category))
                else:
                    # 如果类别为空，随机选择
                    all_books = [book for books in category_to_books.values() for book in books]
                    negative_books.append(random.choice(all_books))
            else:
                # 如果没有其他类别，随机选择
                all_books = [book for books in category_to_books.values() for book in books]
                negative_books.append(random.choice(all_books))

        return negative_books

    def generate_time_aware_samples(self, sequence_length=5, max_time_gap=180, min_time_gap=1):
        """
        生成时间感知的序列样本（考虑借阅时间间隔）
        """
        print(f"生成时间感知样本 - 序列长度: {sequence_length}")

        sequences = []
        targets = []
        user_ids = []
        sequence_info = []
        time_gaps = []
        valid_users = 0

        for user_id, user_data in self.user_sequences.items():
            book_sequence = user_data['book_sequence']
            time_sequence = user_data['time_sequence']

            if len(book_sequence) < sequence_length + 1:
                continue

            valid_users += 1

            for i in range(sequence_length, len(book_sequence)):
                input_seq = book_sequence[i - sequence_length:i]
                input_times = time_sequence[i - sequence_length:i]
                target_book = book_sequence[i]
                target_time = time_sequence[i]

                # 计算时间间隔
                time_gap = (target_time - input_times[-1]).days

                # 过滤异常时间间隔
                if time_gap < min_time_gap or time_gap > max_time_gap:
                    continue

                sequences.append(input_seq)
                targets.append(target_book)
                user_ids.append(user_id)
                time_gaps.append(time_gap)

                seq_info = {
                    'input_times': input_times,
                    'target_time': target_time,
                    'time_gap': time_gap,
                    'sequence_position': i,
                    'time_gap_category': self._categorize_time_gap(time_gap)
                }
                sequence_info.append(seq_info)

        self.stats['time_aware'] = {
            'valid_users': valid_users,
            'total_samples': len(sequences),
            'avg_time_gap': np.mean(time_gaps) if time_gaps else 0,
            'time_gap_distribution': Counter([info['time_gap_category'] for info in sequence_info])
        }

        print(f"时间感知样本生成完成:")
        print(f"  - 有效用户数: {valid_users}")
        print(f"  - 总样本数: {len(sequences)}")
        print(f"  - 平均时间间隔: {self.stats['time_aware']['avg_time_gap']:.2f} 天")
        print(f"  - 时间间隔分类: {dict(self.stats['time_aware']['time_gap_distribution'])}")

        return {
            'sequences': sequences,
            'targets': targets,
            'user_ids': user_ids,
            'sequence_info': sequence_info,
            'time_gaps': time_gaps,
            'type': 'time_aware'
        }

    def generate_category_aware_samples(self, sequence_length=5, category_level='category_1'):
        """
        生成类别感知的序列样本（考虑图书类别连续性）
        """
        print(f"生成类别感知样本 - 序列长度: {sequence_length}, 类别级别: {category_level}")

        sequences = []
        targets = []
        user_ids = []
        sequence_info = []
        category_sequences = []
        valid_users = 0

        for user_id, user_data in self.user_sequences.items():
            book_sequence = user_data['book_sequence']
            time_sequence = user_data['time_sequence']

            if len(book_sequence) < sequence_length + 1:
                continue

            valid_users += 1

            for i in range(sequence_length, len(book_sequence)):
                input_seq = book_sequence[i - sequence_length:i]
                input_times = time_sequence[i - sequence_length:i]
                target_book = book_sequence[i]
                target_time = time_sequence[i]

                # 获取类别序列
                input_categories = [self.book_to_category.get(book, {}).get(category_level, '未知')
                                    for book in input_seq]
                target_category = self.book_to_category.get(target_book, {}).get(category_level, '未知')

                sequences.append(input_seq)
                targets.append(target_book)
                user_ids.append(user_id)
                category_sequences.append(input_categories + [target_category])

                seq_info = {
                    'input_times': input_times,
                    'target_time': target_time,
                    'input_categories': input_categories,
                    'target_category': target_category,
                    'category_transition': f"{input_categories[-1]}->{target_category}",
                    'sequence_position': i
                }
                sequence_info.append(seq_info)

        # 分析类别模式
        category_transitions = [info['category_transition'] for info in sequence_info]
        common_transitions = Counter(category_transitions).most_common(10)

        self.stats['category_aware'] = {
            'valid_users': valid_users,
            'total_samples': len(sequences),
            'common_category_transitions': common_transitions,
            'category_level': category_level
        }

        print(f"类别感知样本生成完成:")
        print(f"  - 有效用户数: {valid_users}")
        print(f"  - 总样本数: {len(sequences)}")
        print(f"  - 最常见的类别转换: {common_transitions}")

        return {
            'sequences': sequences,
            'targets': targets,
            'user_ids': user_ids,
            'sequence_info': sequence_info,
            'category_sequences': category_sequences,
            'type': 'category_aware'
        }

    def split_samples_by_time(self, samples, split_ratio=0.8, time_column='target_time'):
        """
        按时间划分训练集和测试集
        """
        print(f"按时间划分样本集 - 训练集比例: {split_ratio}")

        # 提取所有样本的时间
        times = [info[time_column] for info in samples['sequence_info']]

        # 按时间排序
        sorted_indices = sorted(range(len(times)), key=lambda i: times[i])

        # 按时间划分
        split_idx = int(len(sorted_indices) * split_ratio)
        train_indices = sorted_indices[:split_idx]
        test_indices = sorted_indices[split_idx:]

        # 创建训练集和测试集
        train_samples = {
            'sequences': [samples['sequences'][i] for i in train_indices],
            'targets': [samples['targets'][i] for i in train_indices],
            'user_ids': [samples['user_ids'][i] for i in train_indices],
            'sequence_info': [samples['sequence_info'][i] for i in train_indices],
            'type': samples.get('type', 'unknown') + '_train'
        }

        test_samples = {
            'sequences': [samples['sequences'][i] for i in test_indices],
            'targets': [samples['targets'][i] for i in test_indices],
            'user_ids': [samples['user_ids'][i] for i in test_indices],
            'sequence_info': [samples['sequence_info'][i] for i in test_indices],
            'type': samples.get('type', 'unknown') + '_test'
        }

        # 添加额外的字段
        if 'labels' in samples:
            train_samples['labels'] = [samples['labels'][i] for i in train_indices]
            test_samples['labels'] = [samples['labels'][i] for i in test_indices]

        if 'sequence_lengths' in samples:
            train_samples['sequence_lengths'] = [samples['sequence_lengths'][i] for i in train_indices]
            test_samples['sequence_lengths'] = [samples['sequence_lengths'][i] for i in test_indices]

        print(f"时间划分完成:")
        print(f"  - 训练集样本数: {len(train_samples['sequences'])}")
        print(f"  - 测试集样本数: {len(test_samples['sequences'])}")
        print(
            f"  - 训练集时间范围: {train_samples['sequence_info'][0][time_column]} 到 {train_samples['sequence_info'][-1][time_column]}")
        print(
            f"  - 测试集时间范围: {test_samples['sequence_info'][0][time_column]} 到 {test_samples['sequence_info'][-1][time_column]}")

        return train_samples, test_samples

    def analyze_samples(self, samples):
        """
        分析样本集的统计特征
        """
        print(f"\n样本集分析 - 类型: {samples.get('type', 'unknown')}")
        print("=" * 50)

        sequences = samples['sequences']
        targets = samples['targets']
        user_ids = samples['user_ids']

        # 基本统计
        print(f"总样本数: {len(sequences)}")
        print(f"唯一用户数: {len(set(user_ids))}")
        print(f"唯一图书数: {len(set(targets))}")

        # 序列长度分析
        seq_lengths = [len(seq) for seq in sequences]
        print(f"序列长度 - 平均: {np.mean(seq_lengths):.2f}, 最小: {min(seq_lengths)}, 最大: {max(seq_lengths)}")

        # 目标图书分布
        target_counts = Counter(targets)
        print(f"最热门目标图书: {target_counts.most_common(5)}")

        # 用户活跃度
        user_sample_counts = Counter(user_ids)
        print(f"用户样本数 - 平均: {np.mean(list(user_sample_counts.values())):.2f}, "
              f"最多: {max(user_sample_counts.values())}, 最少: {min(user_sample_counts.values())}")

        # 时间间隔分析（如果有）
        if 'sequence_info' in samples and samples['sequence_info']:
            time_gaps = [info.get('time_gap', 0) for info in samples['sequence_info'] if 'time_gap' in info]
            if time_gaps:
                print(f"时间间隔(天) - 平均: {np.mean(time_gaps):.2f}, 最小: {min(time_gaps)}, 最大: {max(time_gaps)}")

        # 类别分析（如果有）
        if 'sequence_info' in samples and samples['sequence_info']:
            categories = [info.get('target_category', '未知') for info in samples['sequence_info'] if
                          'target_category' in info]
            if categories:
                common_categories = Counter(categories).most_common(5)
                print(f"最常见目标类别: {common_categories}")

        # 负样本分析（如果有）
        if 'labels' in samples:
            labels = samples['labels']
            positive_count = sum(labels)
            negative_count = len(labels) - positive_count
            print(
                f"正负样本分布 - 正样本: {positive_count}, 负样本: {negative_count}, 比例: 1:{negative_count / positive_count:.2f}")

        print("=" * 50)
        return {
            'total_samples': len(sequences),
            'unique_users': len(set(user_ids)),
            'unique_books': len(set(targets)),
            'avg_sequence_length': np.mean(seq_lengths),
            'target_distribution': target_counts
        }

    def _categorize_time_gap(self, time_gap):
        """将时间间隔分类"""
        if time_gap <= 7:
            return "一周内"
        elif time_gap <= 30:
            return "一月内"
        elif time_gap <= 90:
            return "三月内"
        else:
            return "三月以上"


# 使用示例
if __name__ == "__main__":
    # 假设已经加载了预处理数据
    import pickle

    # 加载预处理数据
    with open('processed_sequences.pkl', 'rb') as f:
        processed_data = pickle.load(f)

    user_sequences = processed_data['user_sequences']
    book_info_dict = processed_data['book_info_dict']
    user_df = processed_data['user_df']

    # 初始化样本生成器
    generator = SequenceSampleGenerator(user_sequences, book_info_dict, user_df)

    # 1. 生成固定长度样本
    print("生成固定长度样本...")
    fixed_samples = generator.generate_fixed_length_samples(sequence_length=5, step_size=1)
    generator.analyze_samples(fixed_samples)

    # 2. 生成时间感知样本
    print("\n生成时间感知样本...")
    time_aware_samples = generator.generate_time_aware_samples(sequence_length=5)
    generator.analyze_samples(time_aware_samples)

    # 3. 生成类别感知样本
    print("\n生成类别感知样本...")
    category_samples = generator.generate_category_aware_samples(sequence_length=5, category_level='category_1')
    generator.analyze_samples(category_samples)

    # 4. 使用优化方法生成负样本
    print("\n生成负样本 (优化版)...")
    samples_with_negative = generator.generate_negative_samples_optimized(
        fixed_samples, negative_ratio=4, strategy='popularity', batch_size=10000
    )
    generator.analyze_samples(samples_with_negative)

    # 5. 按时间划分数据集
    print("\n按时间划分数据集...")
    train_samples, test_samples = generator.split_samples_by_time(fixed_samples, split_ratio=0.8)

    print(f"\n训练集统计:")
    generator.analyze_samples(train_samples)

    print(f"\n测试集统计:")
    generator.analyze_samples(test_samples)

    # 6. 保存生成的样本
    all_samples = {
        'fixed_samples': fixed_samples,
        'time_aware_samples': time_aware_samples,
        'category_samples': category_samples,
        'samples_with_negative': samples_with_negative,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'stats': generator.stats
    }

    with open('generated_samples_optimized.pkl', 'wb') as f:
        pickle.dump(all_samples, f)

    print(f"\n所有样本已保存至: generated_samples_optimized.pkl")