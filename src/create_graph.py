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
        self.id_to_name = {}
        self.original_to_index = {}

    def create_nodes_mapping(self) -> Dict[str, Dict[int, int]]:
        node_types = self.nodes_df["node_type"].unique()
        mappings = {}

        for node_type in node_types:
            filtered_node_data = self.nodes_df[self.nodes_df["node_type"] == node_type]
            mappings[node_type] = {
                original_id: id
                for id, original_id in enumerate(filtered_node_data["node_id"].unique())
            }
            self.id_to_name[node_type] = {
                original_id: name
                for original_id, name in zip(
                    filtered_node_data["node_id"], filtered_node_data["name"]
                )
            }
            self.original_to_index[node_type] = mappings[node_type]
        return mappings

    def add_nodes(self, mappings: Dict[str, Dict[int, int]]):
        for node_type, node_mapping in mappings.items():
            filtered_nodes = self.nodes_df[self.nodes_df["node_type"] == node_type]
            num_nodes = len(filtered_nodes)
            self.data[node_type].num_nodes = num_nodes
            self.data[node_type].node_id = torch.arange(num_nodes)
            self.data[node_type].x = torch.eye(num_nodes, dtype=torch.float)

    def add_edges(self, mappings: Dict[str, Dict[int, int]]) -> None:
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

            self.data[src_node_type, relation, dest_node_type].edge_weight = (
                torch.tensor(filtered_edges["score"].values, dtype=torch.float)
            )

        self.data = ToUndirected()(self.data)

    def create(self) -> HeteroData:
        mappings = self.create_nodes_mapping()
        self.add_nodes(mappings)
        self.add_edges(mappings)
        return self.data

    def get_name_mapping(self, node_type="ingredient"):
        return {
            self.original_to_index[node_type][original_id]: name
            for original_id, name in self.id_to_name[node_type].items()
        }
