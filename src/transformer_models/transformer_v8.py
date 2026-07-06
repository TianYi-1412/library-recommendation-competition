# transformer_v8.py
import torch
import torch.nn as nn

class LightTransformer(nn.Module):
    def __init__(self, num_nodes, embed_dim=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)  # ← 压缩到 1D

    def forward(self, x):
        emb = self.embedding(x)  # [B, N, 64]
        out = self.transformer(emb)  # [B, N, 64]
        out = out.mean(dim=1)  # [B, 64]
        return torch.sigmoid(self.fc(out).squeeze(1))  # [B]