import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self):
        self.book_df = None
        self.inter_df = None
        self.user_df = None
        self.user_sequences = {}
        self.book_info_dict = {}

    def load_data(self, inter_path, user_path, book_path):
        """
        加载三个数据文件并进行基本清洗
        """
        try:
            # 读取交互数据
            self.inter_df = pd.read_csv(inter_path)
            print(f"成功加载交互数据，共 {len(self.inter_df)} 条记录")

            # 读取用户数据
            self.user_df = pd.read_csv(user_path)
            print(f"成功加载用户数据，共 {len(self.user_df)} 个用户")

            # 读取图书数据
            self.book_df = pd.read_csv(book_path)
            print(f"成功加载图书数据，共 {len(self.book_df)} 本图书")

        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
        return True

    def clean_interaction_data(self):
        """
        清洗交互数据：处理时间格式、去重、异常值等
        """
        print("\n开始清洗交互数据...")

        # 备份原始数据
        original_count = len(self.inter_df)

        # 1. 去除重复记录
        self.inter_df = self.inter_df.drop_duplicates(subset=['inter_id'], keep='first')
        print(f"去重后记录数: {len(self.inter_df)} (移除 {original_count - len(self.inter_df)} 条重复记录)")

        # 2. 处理时间格式 - 统一格式并转换为datetime
        time_columns = ['借阅时间', '还书时间', '续借时间']

        for col in time_columns:
            if col in self.inter_df.columns:
                # 处理时间格式不一致问题（有的有空格，有的没有）
                self.inter_df[col] = self.inter_df[col].astype(str).str.replace(' ', '')

                # 尝试多种时间格式解析
                try:
                    self.inter_df[col] = pd.to_datetime(self.inter_df[col],
                                                        format='%Y-%m-%d%H:%M:%S',
                                                        errors='coerce')
                except:
                    try:
                        self.inter_df[col] = pd.to_datetime(self.inter_df[col],
                                                            errors='coerce')
                    except:
                        print(f"警告: 列 {col} 的时间格式解析失败")

        # 3. 处理缺失值
        # 借阅时间不能为空
        self.inter_df = self.inter_df.dropna(subset=['借阅时间'])
        print(f"移除借阅时间为空的记录后: {len(self.inter_df)} 条")

        # 4. 验证数据逻辑：还书时间不能早于借阅时间
        mask = (self.inter_df['还书时间'].notna()) & (self.inter_df['还书时间'] < self.inter_df['借阅时间'])
        if mask.any():
            print(f"发现 {mask.sum()} 条还书时间早于借阅时间的异常记录，已移除")
            self.inter_df = self.inter_df[~mask]

        # 5. 处理用户和图书ID
        self.inter_df['user_id'] = self.inter_df['user_id'].astype(str)
        self.inter_df['book_id'] = self.inter_df['book_id'].astype(str)

        print(f"数据清洗完成，最终有效记录: {len(self.inter_df)} 条")
        return True

    def build_user_sequences(self, min_sequence_length=2):
        """
        构建用户借阅序列，按时间排序
        """
        print("\n开始构建用户借阅序列...")

        # 按用户分组，并按借阅时间排序
        user_groups = self.inter_df.groupby('user_id')

        sequences_info = {}

        for user_id, group in user_groups:
            # 按借阅时间排序
            user_data = group.sort_values('借阅时间')

            # 获取该用户的借阅序列（book_id列表）
            book_sequence = user_data['book_id'].tolist()
            time_sequence = user_data['借阅时间'].tolist()

            # 只保留有足够历史记录的用户
            if len(book_sequence) >= min_sequence_length:
                sequences_info[user_id] = {
                    'book_sequence': book_sequence,
                    'time_sequence': time_sequence,
                    'sequence_length': len(book_sequence)
                }

        self.user_sequences = sequences_info

        # 统计信息
        total_users = len(sequences_info)
        avg_sequence_length = np.mean([info['sequence_length'] for info in sequences_info.values()])
        max_sequence_length = np.max([info['sequence_length'] for info in sequences_info.values()])
        min_sequence_length = np.min([info['sequence_length'] for info in sequences_info.values()])

        print(f"用户序列构建完成:")
        print(f"  - 有效用户数: {total_users}")
        print(f"  - 平均序列长度: {avg_sequence_length:.2f}")
        print(f"  - 最长序列: {max_sequence_length}")
        print(f"  - 最短序列: {min_sequence_length}")

        return sequences_info

    def build_book_info_dict(self):
        """
        构建图书信息字典，便于后续特征工程
        """
        print("\n开始构建图书信息字典...")

        # 确保book_id为字符串类型
        self.book_df['book_id'] = self.book_df['book_id'].astype(str)

        # 创建图书信息字典
        for _, row in self.book_df.iterrows():
            book_id = row['book_id']
            self.book_info_dict[book_id] = {
                'title': row.get('题名', ''),
                'author': row.get('作者', ''),
                'publisher': row.get('出版社', ''),
                'category_1': row.get('一级分类', ''),
                'category_2': row.get('二级分类', '')
            }

        print(f"图书信息字典构建完成，共 {len(self.book_info_dict)} 本图书")
        return self.book_info_dict

    def generate_sequence_samples(self, sequence_length=5, step_size=1):
        """
        生成训练样本：使用滑动窗口从用户序列中创建(输入序列, 目标图书)对
        """
        print(f"\n开始生成序列样本 (序列长度: {sequence_length}, 步长: {step_size})...")

        sequences = []  # 输入序列
        targets = []  # 目标图书
        user_ids = []  # 用户ID
        sequence_info = []  # 序列附加信息

        for user_id, user_data in self.user_sequences.items():
            book_sequence = user_data['book_sequence']
            time_sequence = user_data['time_sequence']

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

                # 保存序列的附加信息（用于后续分析）
                seq_info = {
                    'input_times': input_times,
                    'target_time': target_time,
                    'time_gap': (target_time - input_times[-1]).days if len(input_times) > 0 else 0
                }
                sequence_info.append(seq_info)

        print(f"样本生成完成:")
        print(f"  - 总样本数: {len(sequences)}")
        print(f"  - 输入序列形状: {len(sequences)} × {sequence_length}")
        print(f"  - 目标数量: {len(targets)}")

        return {
            'sequences': sequences,
            'targets': targets,
            'user_ids': user_ids,
            'sequence_info': sequence_info
        }

    def analyze_sequences(self):
        """
        分析序列数据的统计特征
        """
        print("\n开始序列数据分析...")

        if not self.user_sequences:
            print("请先构建用户序列")
            return

        # 序列长度分布
        sequence_lengths = [info['sequence_length'] for info in self.user_sequences.values()]

        # 时间间隔分析
        all_time_gaps = []
        for user_data in self.user_sequences.values():
            times = user_data['time_sequence']
            if len(times) > 1:
                for i in range(1, len(times)):
                    gap = (times[i] - times[i - 1]).days
                    all_time_gaps.append(gap)

        print("序列统计信息:")
        print(
            f"  - 序列长度分布: min={min(sequence_lengths)}, max={max(sequence_lengths)}, mean={np.mean(sequence_lengths):.2f}")
        if all_time_gaps:
            print(
                f"  - 借阅间隔(天): min={min(all_time_gaps)}, max={max(all_time_gaps)}, mean={np.mean(all_time_gaps):.2f}")

        # 图书流行度分析
        all_books = []
        for user_data in self.user_sequences.values():
            all_books.extend(user_data['book_sequence'])

        book_counts = pd.Series(all_books).value_counts()
        print(f"  - 总借阅次数: {len(all_books)}")
        print(f"  - 唯一图书数: {len(book_counts)}")
        print(f"  - 最热门图书: {book_counts.head(3).to_dict()}")

    def save_processed_data(self, output_path='processed_sequences.pkl'):
        """
        保存处理后的序列数据
        """
        import pickle

        processed_data = {
            'user_sequences': self.user_sequences,
            'book_info_dict': self.book_info_dict,
            'user_df': self.user_df,
            'book_df': self.book_df,
            'inter_df': self.inter_df
        }

        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)

        print(f"\n处理后的数据已保存至: {output_path}")

    def load_processed_data(self, input_path='processed_sequences.pkl'):
        """
        加载已处理的数据
        """
        import pickle

        try:
            with open(input_path, 'rb') as f:
                processed_data = pickle.load(f)

            self.user_sequences = processed_data['user_sequences']
            self.book_info_dict = processed_data['book_info_dict']
            self.user_df = processed_data['user_df']
            self.book_df = processed_data['book_df']
            self.inter_df = processed_data['inter_df']

            print(f"成功加载已处理的数据从: {input_path}")
            return True
        except Exception as e:
            print(f"加载处理数据失败: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    # 初始化预处理器
    preprocessor = DataPreprocessor()

    # 加载数据（请替换为实际文件路径）
    data_loaded = preprocessor.load_data(
        inter_path='./data/inter_preliminary.csv',  # 替换为实际路径
        user_path='./data/user.csv',  # 替换为实际路径
        book_path='./data/item.csv'  # 替换为实际路径
    )

    if data_loaded:
        # 数据清洗
        preprocessor.clean_interaction_data()

        # 构建用户序列
        preprocessor.build_user_sequences(min_sequence_length=2)

        # 构建图书信息字典
        preprocessor.build_book_info_dict()

        # 分析序列特征
        preprocessor.analyze_sequences()

        # 生成训练样本
        samples = preprocessor.generate_sequence_samples(sequence_length=5, step_size=1)

        # 保存处理后的数据
        preprocessor.save_processed_data('processed_sequences.pkl')

        # 打印样本示例
        print("\n前5个样本示例:")
        for i in range(min(5, len(samples['sequences']))):
            print(f"样本 {i + 1}:")
            print(f"  用户: {samples['user_ids'][i]}")
            print(f"  输入序列: {samples['sequences'][i]}")
            print(f"  目标图书: {samples['targets'][i]}")
            print(f"  时间间隔: {samples['sequence_info'][i]['time_gap']} 天")
            print()