import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv

class model(nn.Module):
    def __init__(self, embed_size, head_num):
        super().__init__()
        self.GAT_layer = GATConv(embed_size, embed_size, head_num)
        #self.Linear_layer = nn.Linear(embed_size, embed_size)

    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())

    def forward(self, graph, features_in, item_index):
        features = self.GAT_layer(graph, features_in)
        features = features.mean(1)
        n = features.shape[0]
        features = features.reshape(n, -1)
        user_embedding = features[0, :]
        #user_embedding = self.Linear_layer(user_embedding)
        return user_embedding


