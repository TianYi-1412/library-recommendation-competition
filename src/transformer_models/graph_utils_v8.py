# graph_utils_v8.py
import numpy as np
from sklearn.decomposition import PCA

class BipartiteGraph:
    def __init__(self, edges, num_users, num_books, embed_dim=64):
        self.edges = np.array(edges)
        self.num_users = num_users
        self.num_books = num_books
        self.embed_dim = embed_dim
        self._build_adj()

    def _build_adj(self):
        # 构造邻接矩阵（用户×图书）
        adj = np.zeros((self.num_users, self.num_books))
        for u, b in self.edges:
            adj[u, b] = 1
        # 轻量 PCA 嵌入
        self.user_embed = PCA(n_components=self.embed_dim).fit_transform(adj)
        self.book_embed = PCA(n_components=self.embed_dim).fit_transform(adj.T)

    def get_user_vec(self, user_idx):
        return self.user_embed[user_idx]

    def get_book_vec(self, book_idx):
        return self.book_embed[book_idx]