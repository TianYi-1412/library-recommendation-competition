import pandas as pd
import numpy as np
import os

class EnhancedDataProcessor:
    """增强的数据处理类"""

    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        self.book_df = None
        self.user_df = None
        self.inter_df = None

    def load_data(self):
        """加载数据"""
        print("开始加载数据...")

        try:
            self.book_df = pd.read_csv(os.path.join(self.data_path, 'item.csv'))
            self.user_df = pd.read_csv(os.path.join(self.data_path, 'user.csv'))
            self.inter_df = pd.read_csv(os.path.join(self.data_path, 'inter_reevaluation.csv'))
            print("CSV格式数据加载成功")
        except FileNotFoundError as e:
            print(f"文件未找到: {e}")
            # 尝试其他可能的文件名
            try:
                self.book_df = pd.read_csv(os.path.join(self.data_path, 'book.csv'))
                self.user_df = pd.read_csv(os.path.join(self.data_path, 'user.csv'))
                self.inter_df = pd.read_csv(os.path.join(self.data_path, 'inter.csv'))
                print("使用备用文件名加载成功")
            except FileNotFoundError:
                raise FileNotFoundError("无法找到数据文件，请检查文件路径和名称")

        print(f"图书数据: {self.book_df.shape[0]} 行")
        print(f"用户数据: {self.user_df.shape[0]} 行")
        print(f"交互数据: {self.inter_df.shape[0]} 行")

        return self.book_df, self.user_df, self.inter_df

    def enhanced_preprocess_data(self):
        """增强的数据预处理"""
        print("开始增强数据预处理...")

        # 重命名列
        if 'book_id' in self.book_df.columns:
            self.book_df = self.book_df.rename(columns={
                'book_id': 'item_id', '题名': 'title', '作者': 'author',
                '出版社': 'publisher', '一级分类': 'category1', '二级分类': 'category2'
            })

        if '借阅人' in self.user_df.columns:
            self.user_df = self.user_df.rename(columns={
                '借阅人': 'user_id', '性别': 'gender', 'DEPT': 'dept',
                '年级': 'grade', '类型': 'user_type'
            })

        if 'inter_id' in self.inter_df.columns:
            self.inter_df = self.inter_df.rename(columns={
                'inter_id': 'interaction_id', 'user_id': 'user_id',
                'book_id': 'item_id', '借阅时间': 'timestamp', '续借次数': 'renew_count'
            })

        # 处理时间戳
        print("处理时间戳...")
        self.inter_df['timestamp'] = pd.to_datetime(self.inter_df['timestamp'], errors='coerce')
        self.inter_df = self.inter_df.dropna(subset=['timestamp'])
        self.inter_df['timestamp_unix'] = self.inter_df['timestamp'].astype('int64') // 10 ** 9

        # 提取时间特征
        self.inter_df['borrow_hour'] = self.inter_df['timestamp'].dt.hour
        self.inter_df['borrow_dayofweek'] = self.inter_df['timestamp'].dt.dayofweek
        self.inter_df['borrow_month'] = self.inter_df['timestamp'].dt.month

        # 增强续借行为处理
        print("处理续借行为...")
        self.inter_df['has_renewed'] = (self.inter_df['renew_count'] > 0).astype(int)
        # 动态评分策略
        self.inter_df['rating'] = 1.0 + self.inter_df['has_renewed'] * 0.8 + \
                                 (self.inter_df['renew_count'] > 1).astype(int) * 0.4

        # 数据清洗
        print("数据清洗...")
        self.inter_df = self.inter_df.drop_duplicates(subset=['user_id', 'item_id', 'timestamp'])

        valid_users = set(self.user_df['user_id'])
        valid_items = set(self.book_df['item_id'])

        original_size = len(self.inter_df)
        self.inter_df = self.inter_df[
            self.inter_df['user_id'].isin(valid_users) &
            self.inter_df['item_id'].isin(valid_items)
        ]
        filtered_size = len(self.inter_df)
        self.book_df['class'] = self.book_df.get('category1', '未知').fillna('未知')

        print(f"清洗后交互数据: {filtered_size} 行 (过滤了 {original_size - filtered_size} 行无效数据)")
        print("增强数据预处理完成")

        return self.book_df, self.user_df, self.inter_df

    def time_aware_split(self, test_ratio=0.2, valid_ratio=0.2):
        """按时间划分数据"""
        print("开始时间感知数据划分...")

        # 按时间排序
        inter_df_sorted = self.inter_df.sort_values(['user_id', 'timestamp'])

        # 找到每个用户最后借阅的图书作为测试集
        last_interactions = inter_df_sorted.groupby('user_id').last().reset_index()

        # 训练集：除最后一条外的所有记录
        train_df = inter_df_sorted.merge(
            last_interactions[['user_id', 'item_id', 'timestamp']],
            on=['user_id', 'item_id', 'timestamp'],
            how='left',
            indicator=True
        )
        train_df = train_df[train_df['_merge'] == 'left_only'].drop('_merge', axis=1)

        # 测试集：每个用户的最后一条记录
        test_df = last_interactions.copy()

        print(f"训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")

        # 验证集：从训练集中划分一部分
        train_users = train_df['user_id'].unique()
        np.random.seed(2023)
        valid_users = np.random.choice(
            train_users,
            size=int(valid_ratio * len(train_users)),
            replace=False
        )
        valid_df = train_df[train_df['user_id'].isin(valid_users)]
        train_final_df = train_df[~train_df['user_id'].isin(valid_users)]

        print(f"最终训练集大小: {len(train_final_df)}")
        print(f"验证集大小: {len(valid_df)}")

        return train_final_df, valid_df, test_df

    def save_data_for_recbole(self, train_df, valid_df, test_df):
        """为RecBole保存数据"""
        print("为RecBole准备数据文件...")

        dataset_dir = os.path.join(self.data_path, 'library')
        os.makedirs(dataset_dir, exist_ok=True)

        # 保存用户数据
        user_df_out = self.user_df.copy()
        user_df_out.columns = [f"{col}:token" for col in user_df_out.columns]
        user_df_out.to_csv(
            os.path.join(dataset_dir, 'library.user'),
            index=False,
            sep='\t'
        )

        # 保存物品数据
        book_df_out = self.book_df.copy()
        book_df_out.columns = [f"{col}:token" for col in book_df_out.columns]
        book_df_out.to_csv(
            os.path.join(dataset_dir, 'library.item'),
            index=False,
            sep='\t'
        )

        # 保存交互数据
        inter_df_out = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        inter_df_out = inter_df_out[['user_id', 'item_id', 'rating', 'timestamp_unix']]
        inter_df_out.columns = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp_unix:float']
        inter_df_out['label:float'] = inter_df_out['rating:float'].copy()
        inter_df_out.to_csv(
            os.path.join(dataset_dir, 'library.inter'),
            index=False,
            sep='\t'
        )


        # 返回用于评估的完整交互数据
        inter_df_for_eval = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        print("RecBole数据文件准备完成")
        return inter_df_for_eval