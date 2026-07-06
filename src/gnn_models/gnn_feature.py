import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import LGConv  # 修改：使用LGConv而不是LightGCN
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class EnhancedGNNEmbedder:
    def __init__(self, inter_df, book_df, user_df, embed_dim=64, num_layers=2):
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self._build_mapping(inter_df, book_df, user_df)
        self._build_multi_layer_graph(inter_df, book_df, user_df)
        self.model = self._build_enhanced_model()
        self._train()

    # ---------- 建立 ID 映射 ----------
    def _build_mapping(self, inter_df, book_df, user_df):
        self.user2idx = {uid: i for i, uid in enumerate(sorted(inter_df['user_id'].unique()))}
        self.book2idx = {bid: i for i, bid in enumerate(sorted(inter_df['book_id'].unique()))}
        self.idx2user = {i: uid for uid, i in self.user2idx.items()}
        self.idx2book = {i: bid for bid, i in self.book2idx.items()}

        # 建立院系映射
        self.dept2idx = {dept: i for i, dept in enumerate(user_df['DEPT'].unique())}
        self.idx2dept = {i: dept for dept, i in self.dept2idx.items()}

    # ---------- 构建多层图结构 ----------
    def _build_multi_layer_graph(self, inter_df, book_df, user_df):
        self.hetero_data = HeteroData()

        # 1. 用户-图书交互图 (主要关系)
        user_idx = inter_df['user_id'].map(self.user2idx).values
        book_idx = inter_df['book_id'].map(self.book2idx).values
        self.hetero_data['user', 'borrows', 'book'].edge_index = torch.tensor([user_idx, book_idx], dtype=torch.long)

        # 添加边权重（续借次数 + 时间衰减）
        edge_weights = self._compute_edge_weights(inter_df)
        self.hetero_data['user', 'borrows', 'book'].edge_weight = edge_weights

        # 2. 用户-用户相似图
        user_user_edges = self._build_user_similarity_graph(inter_df)
        if user_user_edges is not None:
            self.hetero_data['user', 'similar_to', 'user'].edge_index = user_user_edges

        # 3. 图书-图书相似图
        book_book_edges = self._build_book_similarity_graph(book_df)
        if book_book_edges is not None:
            self.hetero_data['book', 'similar_to', 'book'].edge_index = book_book_edges

        # 4. 用户-院系归属图
        user_dept_edges = self._build_user_dept_graph(user_df)
        if user_dept_edges is not None:
            self.hetero_data['user', 'belongs_to', 'dept'].edge_index = user_dept_edges

        print(f"✅ 多层图结构构建完成")
        print(f"   - 用户-图书边: {self.hetero_data['user', 'borrows', 'book'].edge_index.shape[1]}")
        if 'user' in self.hetero_data.edge_types and 'similar_to' in self.hetero_data.edge_types:
            print(f"   - 用户-用户边: {self.hetero_data['user', 'similar_to', 'user'].edge_index.shape[1]}")
        if 'book' in self.hetero_data.edge_types and 'similar_to' in self.hetero_data.edge_types:
            print(f"   - 图书-图书边: {self.hetero_data['book', 'similar_to', 'book'].edge_index.shape[1]}")

    def _compute_edge_weights(self, inter_df):
        """计算边权重：续借次数 + 时间衰减"""
        weights = []
        latest_time = inter_df['借阅时间'].max()

        for _, row in inter_df.iterrows():
            # 续借次数权重
            renew_weight = 1.0 + 0.5 * row['续借次数']

            # 时间衰减权重 (90天半衰期)
            days_diff = (latest_time - row['借阅时间']).days
            time_weight = np.exp(-days_diff / 90)

            weights.append(renew_weight * time_weight)

        return torch.tensor(weights, dtype=torch.float)

    def _build_user_similarity_graph(self, inter_df, similarity_threshold=0.3, top_k=10):
        """构建用户相似图 - 基于借阅历史相似度"""
        try:
            # 构建用户-图书交互矩阵
            user_book_matrix = np.zeros((len(self.user2idx), len(self.book2idx)))
            for _, row in inter_df.iterrows():
                u_idx = self.user2idx[row['user_id']]
                b_idx = self.book2idx[row['book_id']]
                user_book_matrix[u_idx, b_idx] = 1

            # 计算余弦相似度
            user_similarity = cosine_similarity(user_book_matrix)

            # 构建边 (只保留高相似度的边)
            edges = []
            for i in range(len(self.user2idx)):
                similarities = user_similarity[i]
                # 排除自己
                similarities[i] = 0
                # 选择top-k相似用户
                top_indices = np.argsort(similarities)[-top_k:]
                for j in top_indices:
                    if similarities[j] > similarity_threshold:
                        edges.append([i, j])

            if edges:
                return torch.tensor(edges, dtype=torch.long).t()
            return None
        except Exception as e:
            print(f"⚠️ 用户相似图构建失败: {e}，跳过")
            return None

    def _build_book_similarity_graph(self, book_df, similarity_threshold=0.4, top_k=15):
        """构建图书相似图 - 基于类别和关键词"""
        try:
            # 构建图书特征 (二级分类 + 分词题名)
            book_features = []
            book_ids = []

            for book_id, idx in self.book2idx.items():
                book_info = book_df[book_df['book_id'] == book_id].iloc[0]
                feature_str = f"{book_info['二级分类']} {book_info.get('分词题名', '')}"
                book_features.append(feature_str)
                book_ids.append(idx)

            # 使用TF-IDF计算相似度
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(book_features)
            book_similarity = cosine_similarity(tfidf_matrix)

            # 构建边
            edges = []
            for i, idx_i in enumerate(book_ids):
                similarities = book_similarity[i]
                # 选择top-k相似图书
                top_indices = np.argsort(similarities)[-top_k:]
                for j in top_indices:
                    if j != i and similarities[j] > similarity_threshold:
                        edges.append([idx_i, book_ids[j]])

            if edges:
                return torch.tensor(edges, dtype=torch.long).t()
            return None
        except Exception as e:
            print(f"⚠️ 图书相似图构建失败: {e}，跳过")
            return None

    def _build_user_dept_graph(self, user_df):
        """构建用户-院系归属图"""
        try:
            edges = []
            for _, row in user_df.iterrows():
                if row['user_id'] in self.user2idx:
                    u_idx = self.user2idx[row['user_id']]
                    dept = row['DEPT']
                    if dept in self.dept2idx:
                        d_idx = self.dept2idx[dept]
                        edges.append([u_idx, d_idx])

            if edges:
                return torch.tensor(edges, dtype=torch.long).t()
            return None
        except Exception as e:
            print(f"⚠️ 用户-院系图构建失败: {e}，跳过")
            return None

    # ---------- 增强的GNN模型 ----------
    def _build_enhanced_model(self):
        return MultiRelationGNN(
            num_users=len(self.user2idx),
            num_books=len(self.book2idx),
            num_depts=len(self.dept2idx),
            embed_dim=self.embed_dim,
            num_layers=self.num_layers
        )

    # ---------- 训练过程 ----------
    def _train(self, epochs=20, lr=5e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        print("⏳ 增强GNN训练中...")

        for epoch in range(epochs):
            optimizer.zero_grad()

            # 前向传播
            out = self.model(self.hetero_data)

            # 损失计算
            loss = self._compute_hetero_loss(out)

            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"   GNN epoch {epoch:02d} | loss {loss.item():.4f}")

        print("✅ 增强GNN训练完成")

    def _compute_hetero_loss(self, out):
        """计算异构图的损失 - 使用BPR损失"""
        # 正样本：实际存在的用户-图书交互
        pos_edge_index = self.hetero_data['user', 'borrows', 'book'].edge_index
        pos_users = out['user'][pos_edge_index[0]]
        pos_books = out['book'][pos_edge_index[1]]
        pos_scores = (pos_users * pos_books).sum(dim=1)

        # 负采样：为每个正样本生成一个负样本
        num_pos = pos_edge_index.shape[1]
        neg_books = torch.randint(0, len(self.book2idx), (num_pos,))
        neg_scores = (pos_users * out['book'][neg_books]).sum(dim=1)

        # BPR损失
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        return loss

    # ---------- 获取嵌入 ----------
    def get_embeddings(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.hetero_data)

        user_embed = out['user'].numpy()
        book_embed = out['book'].numpy()

        return user_embed, book_embed

    def export(self):
        user_emb, book_emb = self.get_embeddings()
        self.user_embed = {uid: user_emb[i] for uid, i in self.user2idx.items()}
        self.book_embed = {bid: book_emb[i] for bid, i in self.book2idx.items()}
        return self.user_embed, self.book_embed


class MultiRelationGNN(nn.Module):
    """处理多关系异构图的GNN模型"""

    def __init__(self, num_users, num_books, num_depts, embed_dim=64, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 节点嵌入
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.book_embed = nn.Embedding(num_books, embed_dim)
        self.dept_embed = nn.Embedding(num_depts, embed_dim // 2)

        # 修改：使用LGConv而不是LightGCN
        self.borrow_layers = nn.ModuleList([
            LGConv() for _ in range(num_layers)
        ])

        # 用户相似图卷积层
        self.user_sim_layers = nn.ModuleList([
            LGConv() for _ in range(num_layers)
        ])

        # 图书相似图卷积层
        self.book_sim_layers = nn.ModuleList([
            LGConv() for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, hetero_data):
        # 初始化节点特征
        user_x = self.user_embed.weight
        book_x = self.book_embed.weight

        # 多关系信息传播
        user_final = user_x.clone()
        book_final = book_x.clone()

        # 1. 用户-图书交互传播
        if 'borrows' in hetero_data.edge_types:
            edge_index = hetero_data['user', 'borrows', 'book'].edge_index
            for layer in self.borrow_layers:
                # 分别更新用户和图书嵌入
                user_borrow_out = layer(user_final, edge_index.flip([0]))  # 反转边方向：图书->用户
                book_borrow_out = layer(book_final, edge_index)  # 用户->图书

                user_final = user_final + self.dropout(user_borrow_out)
                book_final = book_final + self.dropout(book_borrow_out)

        # 2. 用户-用户相似传播
        if 'similar_to' in hetero_data.edge_types and 'user' in hetero_data.edge_types:
            user_edge_index = hetero_data['user', 'similar_to', 'user'].edge_index
            for layer in self.user_sim_layers:
                user_sim_out = layer(user_final, user_edge_index)
                user_final = user_final + self.dropout(user_sim_out)

        # 3. 图书-图书相似传播
        if 'similar_to' in hetero_data.edge_types and 'book' in hetero_data.edge_types:
            book_edge_index = hetero_data['book', 'similar_to', 'book'].edge_index
            for layer in self.book_sim_layers:
                book_sim_out = layer(book_final, book_edge_index)
                book_final = book_final + self.dropout(book_sim_out)

        return {
            'user': user_final,
            'book': book_final
        }