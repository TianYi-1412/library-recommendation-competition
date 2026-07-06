
import torch
from torch_geometric.nn import LightGCN
from torch_geometric.data import Data
import numpy as np

class GNNV5:
    def __init__(self, edges, num_nodes, embed_dim=64, num_layers=2):
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.data = Data(edge_index=edge_index, num_nodes=num_nodes)
        self.model = LightGCN(num_nodes=num_nodes, embedding_dim=embed_dim, num_layers=num_layers)
        self._train()

    def _train(self, epochs=20, lr=5e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(self.data.edge_index)
            loss = self.model.recommendation_loss(self.data.edge_index, out)
            loss.backward()
            optimizer.step()

    def get_embed_dict(self, idx2entity):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.edge_index)
        return {idx2entity[i]: out[i].numpy() for i in idx2entity}