from typing import Dict

import torch
from torch import Tensor
from torch.nn import Dropout, Embedding, Module
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero


class GNNEncoder(Module):
    def __init__(self, hidden_channels: int, out_channels: int, dropout: int = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = Dropout(p=dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class PairClassifier(Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        # Remove the self.classifier wrapper
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x_dict, edge_index):
        src = x_dict["ingredient"][edge_index[0]]
        dst = x_dict["ingredient"][edge_index[1]]
        return self.layers(
            torch.cat([src, dst], dim=-1)
        )  # Changed from self.classifier to self.layers


class CompoundClassifier(Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        # Remove the self.classifier wrapper
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x_dict, edge_index):
        src = x_dict["ingredient"][edge_index[0]]
        dst = x_dict["compound"][edge_index[1]]
        return self.layers(
            torch.cat([src, dst], dim=-1)
        )  # Changed from self.compound_classifier to self.layers


class HeteroGNN(Module):
    def __init__(self, hidden_channels: int, data: HeteroData, dropout: int = 0.3):
        super().__init__()

        self.ingredient_emb = Embedding(data["ingredient"].num_nodes, hidden_channels)
        self.compound_emb = Embedding(data["compound"].num_nodes, hidden_channels)

        self.gnn = to_hetero(
            GNNEncoder(hidden_channels, hidden_channels, dropout),
            data.metadata(),
            aggr="mean",
        )

        self.pair_classifier = PairClassifier(hidden_channels)
        self.compound_classifier = CompoundClassifier(hidden_channels)

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        x_dict = {
            "ingredient": self.ingredient_emb.weight,
            "compound": self.compound_emb.weight,
        }
        x = self.gnn(x_dict, data.edge_index_dict)
        return x

    def predict_pair_links(self, x_dict, edge_index):
        return self.pair_classifier(x_dict, edge_index)

    def predict_compound_links(self, x_dict, edge_index):
        return self.compound_classifier(x_dict, edge_index)
