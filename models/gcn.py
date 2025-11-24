import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
        self.conv2 = SAGEConv(nhid, 20, normalize=True)
        self.linear = nn.Linear(in_features=20, out_features=nclass)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(x)
        return x
