import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')


class BookRecommendationSystem:
    def __init__(self):
        self.user_df = None
        self.book_df = None
        self.inter_df = None
        self.user_item_matrix = None
        self.item_similarity = None
        self.user_similarity = None
        self.svd_model = None
        self.tfidf_matrix = None
        self.book_similarity_content = None

    def load_data(self, user_file, book_file, inter_file):
        """加载数据"""
        try:
            self.user_df = pd.read_csv(user_file)
            self.book_df = pd.read_csv(book_file)
            self.inter_df = pd.read_csv(inter_file)

            print(f"用户数据: {self.user_df.shape}")
            print(f"图书数据: {self.book_df.shape}")
            print(f"交互数据: {self.inter_df.shape}")

            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def preprocess_data(self):
        """数据预处理"""
        # 处理交互数据
        self.inter_df['借阅时间'] = pd.to_datetime(self.inter_df['借阅时间'], errors='coerce')
        self.inter_df = self.inter_df.dropna(subset=['user_id', 'book_id'])

        # 处理图书数据
        self.book_df = self.book_df.dropna(subset=['book_id'])

        # 创建图书特征
        self.book_df['combined_features'] = (
                self.book_df['题名'].fillna('') + ' ' +
                self.book_df['作者'].fillna('') + ' ' +
                self.book_df['出版社'].fillna('') + ' ' +
                self.book_df['一级分类'].fillna('') + ' ' +
                self.book_df['二级分类'].fillna('')
        )

        print("数据预处理完成")

    def create_user_item_matrix(self):
        """创建用户-物品矩阵"""
        # 为每个用户-图书对创建交互计数
        user_item_counts = self.inter_df.groupby(['user_id', 'book_id']).size().reset_index(name='count')

        # 创建稀疏矩阵
        users = user_item_counts['user_id'].astype('category')
        books = user_item_counts['book_id'].astype('category')

        self.user_item_matrix = csr_matrix(
            (user_item_counts['count'],
             (users.cat.codes, books.cat.codes)),
            shape=(len(users.cat.categories), len(books.cat.categories))
        )

        self.user_ids = users.cat.categories
        self.book_ids = books.cat.categories

        print(f"用户-物品矩阵形状: {self.user_item_matrix.shape}")

    def build_content_based_model(self):
        """构建基于内容的推荐模型"""
        # 使用TF-IDF处理图书特征
        tfidf = TfidfVectorizer(max_features=1000, stop_words=None)
        self.tfidf_matrix = tfidf.fit_transform(self.book_df['combined_features'])

        # 计算图书间的内容相似度
        self.book_similarity_content = cosine_similarity(self.tfidf_matrix)

        # 创建图书ID到索引的映射
        self.book_id_to_index = {book_id: idx for idx, book_id in enumerate(self.book_df['book_id'])}
        self.index_to_book_id = {idx: book_id for book_id, idx in self.book_id_to_index.items()}

        print("基于内容的模型构建完成")

    def build_collaborative_filtering(self):
        """构建协同过滤模型"""
        # 基于物品的协同过滤
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity = csr_matrix(item_similarity)

        # 基于用户的协同过滤
        user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity = csr_matrix(user_similarity)

        print("协同过滤模型构建完成")

    def build_latent_factor_model(self, n_components=50):
        """构建隐语义模型"""
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd_model.components_.T

        print(f"隐语义模型构建完成，成分数: {n_components}")

    def get_user_recent_books(self, user_id, n=5):
        """获取用户最近借阅的图书"""
        user_interactions = self.inter_df[self.inter_df['user_id'] == user_id]
        recent_books = user_interactions.sort_values('借阅时间', ascending=False).head(n)
        return recent_books['book_id'].tolist()

    def content_based_recommendation(self, user_id, n_recommendations=10):
        """基于内容的推荐"""
        if user_id not in self.inter_df['user_id'].values:
            return []

        # 获取用户最近借阅的图书
        recent_books = self.get_user_recent_books(user_id)

        if not recent_books:
            return []

        # 计算用户兴趣向量
        user_profile = None
        for book_id in recent_books:
            if book_id in self.book_id_to_index:
                book_idx = self.book_id_to_index[book_id]
                if user_profile is None:
                    user_profile = self.tfidf_matrix[book_idx]
                else:
                    user_profile += self.tfidf_matrix[book_idx]

        if user_profile is None:
            return []

        # 计算与所有图书的相似度
        similarities = cosine_similarity(user_profile, self.tfidf_matrix).flatten()

        # 获取推荐图书
        book_indices = similarities.argsort()[::-1]
        recommendations = []

        for idx in book_indices:
            book_id = self.index_to_book_id[idx]
            # 排除用户已经借阅过的图书
            if book_id not in recent_books and book_id not in recommendations:
                recommendations.append(book_id)
            if len(recommendations) >= n_recommendations:
                break

        return recommendations

    def item_based_recommendation(self, user_id, n_recommendations=10):
        """基于物品的协同过滤推荐"""
        if user_id not in self.user_ids:
            return []

        user_idx = np.where(self.user_ids == user_id)[0][0]

        # 获取用户借阅过的图书
        user_interactions = self.user_item_matrix[user_idx]
        interacted_items = user_interactions.indices

        if len(interacted_items) == 0:
            return []

        # 计算推荐分数
        scores = np.zeros(self.user_item_matrix.shape[1])

        for item_idx in interacted_items:
            similarities = self.item_similarity[item_idx].toarray().flatten()
            scores += similarities * user_interactions[0, item_idx]

        # 排除已经借阅过的图书
        scores[interacted_items] = -1

        # 获取推荐图书
        recommended_indices = scores.argsort()[::-1][:n_recommendations]
        recommendations = [self.book_ids[idx] for idx in recommended_indices if scores[idx] > 0]

        return recommendations

    def latent_factor_recommendation(self, user_id, n_recommendations=10):
        """基于隐语义模型的推荐"""
        if user_id not in self.user_ids:
            return []

        user_idx = np.where(self.user_ids == user_id)[0][0]

        # 预测用户对所有图书的评分
        user_prediction = self.user_factors[user_idx] @ self.item_factors.T

        # 获取用户已经借阅过的图书
        user_interactions = self.user_item_matrix[user_idx]
        interacted_items = user_interactions.indices

        # 排除已经借阅过的图书
        user_prediction[interacted_items] = -1

        # 获取推荐图书
        recommended_indices = user_prediction.argsort()[::-1][:n_recommendations]
        recommendations = [self.book_ids[idx] for idx in recommended_indices if user_prediction[idx] > 0]

        return recommendations

    def hybrid_recommendation(self, user_id, n_recommendations=1):
        """混合推荐 - 为主要方法"""
        all_recommendations = []

        # 尝试不同的推荐方法
        methods = [
            self.latent_factor_recommendation,
            self.item_based_recommendation,
            self.content_based_recommendation
        ]

        for method in methods:
            try:
                recs = method(user_id, n_recommendations * 3)  # 获取更多候选
                all_recommendations.extend(recs)
            except:
                continue

        # 去除重复并排序
        unique_recommendations = []
        seen = set()
        for rec in all_recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)

        # 如果推荐不足，添加热门图书
        if len(unique_recommendations) < n_recommendations:
            popular_books = self.inter_df['book_id'].value_counts().head(20).index.tolist()
            for book in popular_books:
                if book not in unique_recommendations and len(unique_recommendations) < n_recommendations:
                    unique_recommendations.append(book)

        return unique_recommendations[:n_recommendations]

    def train(self):
        """训练所有模型"""
        print("开始训练推荐模型...")

        self.preprocess_data()
        self.create_user_item_matrix()
        self.build_content_based_model()
        self.build_collaborative_filtering()
        self.build_latent_factor_model()

        print("所有模型训练完成!")

    def predict_for_users(self, user_list):
        """为指定用户列表生成预测"""
        predictions = []

        for user_id in user_list:
            recommendations = self.hybrid_recommendation(user_id, n_recommendations=1)

            if recommendations:
                predicted_book = recommendations[0]
            else:
                # 备用方案：推荐热门图书
                popular_books = self.inter_df['book_id'].value_counts().head(10)
                predicted_book = popular_books.index[0] if len(popular_books) > 0 else None

            if predicted_book:
                predictions.append({'user_id': user_id, 'book_id': predicted_book})

        return predictions


def main():
    """主函数"""
    # 初始化推荐系统
    recommender = BookRecommendationSystem()

    # 加载数据
    print("正在加载数据...")

    success = recommender.load_data('./data/user.csv', './data/item.csv', './data/inter_preliminary.csv')

    if not success:
        print("数据加载失败，请检查文件路径")
        return

    # 训练模型
    recommender.train()

    # 获取需要预测的用户列表（从交互数据中提取所有用户）
    all_users = recommender.inter_df['user_id'].unique()
    print(f"需要预测的用户数量: {len(all_users)}")

    # 生成预测
    print("正在生成预测...")
    predictions = recommender.predict_for_users(all_users)

    # 创建提交文件
    submission_df = pd.DataFrame(predictions)
    submission_df = submission_df[['user_id', 'book_id']]

    # 保存结果
    submission_df.to_csv('submission.csv', index=False)
    print(f"预测完成! 共生成 {len(submission_df)} 条预测")
    print("结果已保存至 submission.csv")

    # 显示部分预测结果
    print("\n前10条预测结果:")
    print(submission_df.head(10))


if __name__ == "__main__":
    main()