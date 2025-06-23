import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.nn import Dropout
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import negative_sampling


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = Dropout(p=dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
