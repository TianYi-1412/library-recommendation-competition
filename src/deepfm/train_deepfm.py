# 07_train_deepfm_pytorch.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import os

# 1. 读取并合并
train = pd.read_csv('../output+hj/train_rank2.csv')
valid = pd.read_csv('../output+hj/valid_rank2.csv')
df = pd.concat([train, valid], ignore_index=True)

# 2. 特征列
sparse_cols = ['user_id', 'cand_book_id'] + [f'last{i}_book_id' for i in range(1,6)] + [f'last{i}_cate2' for i in range(1,6)] + ['cand_cate2']
dense_cols = ['cand_renew_mean']

# 3. 统一编码（关键）
encoders = {}
for col in sparse_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# 4. 重新切回训练/验证
train = df.iloc[:len(train)]
valid = df.iloc[len(train):]

# 5. 输入
X_sparse_train = train[sparse_cols].values
X_dense_train  = train[dense_cols].values
y_train = train['label'].values

X_sparse_valid = valid[sparse_cols].values
X_dense_valid  = valid[dense_cols].values
y_valid = valid['label'].values

# 6. 超参
field_dims = [X_sparse_train[:, i].max() + 1 for i in range(X_sparse_train.shape[1])]
embed_dim = 64
batch_size = 4096
epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 7. Dataset & DataLoader
class DeepFMDataset(Dataset):
    def __init__(self, X_sparse, X_dense, y):
        self.X_sparse = torch.tensor(X_sparse, dtype=torch.long)
        self.X_dense  = torch.tensor(X_dense, dtype=torch.float32)
        self.y        = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X_sparse[idx], self.X_dense[idx], self.y[idx]

train_loader = DataLoader(DeepFMDataset(X_sparse_train, X_dense_train, y_train), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(DeepFMDataset(X_sparse_valid, X_dense_valid, y_valid), batch_size=batch_size, shuffle=False)

# 8. DeepFM 模型（PyTorch）
class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims=(256, 128)):
        super(DeepFM, self).__init__()
        self.embedding = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in field_dims])
        self.fc = nn.Linear(len(field_dims) * embed_dim, mlp_dims[0])
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dims[0] + len(dense_cols), mlp_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[1], 1)
        )
    def forward(self, x_sparse, x_dense):
        embeds = [self.embedding[i](x_sparse[:, i]) for i in range(len(field_dims))]
        embed_vec = torch.cat(embeds, dim=1)  # [B, F*K]
        fm_part = 0.5 * (embed_vec.sum(dim=1).unsqueeze(1) ** 2 - (embed_vec ** 2).sum(dim=1).unsqueeze(1))
        deep_part = self.fc(embed_vec)
        deep_part = torch.cat([deep_part, x_dense], dim=1)
        out = self.mlp(deep_part) + fm_part
        return torch.sigmoid(out).squeeze()

# 9. 训练
model = DeepFM(field_dims, embed_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_sparse, x_dense, y in train_loader:
        x_sparse, x_dense, y = x_sparse.to(device), x_dense.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x_sparse, x_dense)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}  loss: {total_loss/len(train_loader):.4f}')

# 10. 保存
torch.save(model.state_dict(), '../output+hj/deepfm_pytorch.pt')
joblib.dump({'encoders': encoders, 'sparse_cols': sparse_cols, 'dense_cols': dense_cols, 'field_dims': field_dims},
            '../output+hj/deepfm_pytorch_meta.pkl')
print('PyTorch DeepFM 训练完成')