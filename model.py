import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()

        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return x


class PairingClassifier(torch.nn.Module):
    def forward(self, x_ing1: Tensor, x_ing2: Tensor, edge_label_index: Tensor):
        edge_feat_ing1 = x_ing1[edge_label_index[0]]
        edge_feat_ing2 = x_ing2[edge_label_index[1]]
        return (edge_feat_ing1 * edge_feat_ing2).sum(dim=-1)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward():
        pass
