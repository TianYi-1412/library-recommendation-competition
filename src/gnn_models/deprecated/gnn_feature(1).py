import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import LightGCN
import numpy as np
import time

class GNNEmbedder:
    def __init__(self, inter_df, book_df, user_df, embed_dim=64, num_layers=2):
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self._build_mapping(inter_df, book_df, user_df)
        self._build_edge_index(inter_df)
        self.model = self._build_model()
        self._train()

    # ---------- 建立 ID 映射 ----------
    def _build_mapping(self, inter_df, book_df, user_df):
        self.user2idx = {uid: i for i, uid in enumerate(sorted(inter_df['user_id'].unique()))}
        self.book2idx = {bid: i for i, bid in enumerate(sorted(inter_df['book_id'].unique()))}
        self.idx2user = {i: uid for uid, i in self.user2idx.items()}
        self.idx2book = {i: bid for bid, i in self.book2idx.items()}

    # ---------- 构建异构图边 ----------
    def _build_edge_index(self, inter_df):
        user_idx = inter_df['user_id'].map(self.user2idx).values
        book_idx = inter_df['book_id'].map(self.book2idx).values
        self.edge_index = torch.tensor([user_idx, book_idx], dtype=torch.long)

    # ---------- 模型定义 ----------
    def _build_model(self):
        return LightGCN(
            num_nodes=len(self.user2idx) + len(self.book2idx),
            embedding_dim=self.embed_dim,
            num_layers=self.num_layers,
            alpha=0.5
        )

    # ---------- 训练 ----------
    def _train(self, epochs=20, lr=5e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        print("⏳ demo 训练中...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(self.edge_index)
            loss = self.model.recommendation_loss(self.edge_index, out)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"   demo epoch {epoch:02d} | loss {loss.item():.4f}")
        print("✅ demo 训练完成")

    # ---------- 获取嵌入 ----------
    def get_embeddings(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.edge_index)
        user_embed = out[:len(self.user2idx)]
        book_embed = out[len(self.user2idx):]
        return user_embed.numpy(), book_embed.numpy()

    # ---------- 快速查询 ----------
    def lookup(self, user_id=None, book_id=None):
        if user_id is not None:
            return self.user_embed[self.user2idx[user_id]]
        if book_id is not None:
            return self.book_embed[self.book2idx[book_id]]
        return None

    # ---------- 一次性导出 ----------
    def export(self):
        user_emb, book_emb = self.get_embeddings()
        self.user_embed = {uid: user_emb[i] for uid, i in self.user2idx.items()}
        self.book_embed = {bid: book_emb[i] for bid, i in self.book2idx.items()}
        return self.user_embed, self.book_embed