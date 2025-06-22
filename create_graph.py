from typing import Dict

import pandas as pd


class IngredientCompoundGraph:
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
        self.nodes_df = nodes_df
        self.edges_df = edges_df

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
