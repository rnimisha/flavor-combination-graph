from typing import Dict

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


class IngredientCompoundGraph:
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self.data = HeteroData()

    def create_nodes_mapping(self) -> Dict[str, Dict[int, int]]:
        node_types = self.nodes_df["node_type"].unique()
        mappings = {}

        for node_type in node_types:
            filtered_node_data = self.nodes_df[self.nodes_df["node_type"] == node_type]
            mappings[node_type] = {
                original_id: id
                for id, original_id in enumerate(filtered_node_data["node_id"].unique())
            }
        return mappings

    def create_edges(self, mappings: Dict[str, Dict[int, int]]):
        edge_types = self.edges_df["edge_type"].unique()

        for edge_type in edge_types:
            filtered_edges = self.edges_df[self.edges_df["edge_type"] == edge_type]

            src_node_type = dest_node_type = "ingredient"
            relation = "paired_with"

            if edge_type == "ingr-fcomp":
                dest_node_type = "compound"
                relation = "associated_with"

            src_nodes = [mappings[src_node_type][id] for id in filtered_edges["id_1"]]
            dest_nodes = [mappings[dest_node_type][id] for id in filtered_edges["id_2"]]
            edge_index = torch.tensor([src_nodes, dest_nodes], dtype=torch.long)

            self.data[src_node_type, relation, dest_node_type].edge_index = edge_index

            self.data[src_node_type, relation, dest_node_type].score = torch.tensor(
                filtered_edges["score"].values, dtype=torch.float
            )

        self.data = ToUndirected()(self.data)

        return self.data
