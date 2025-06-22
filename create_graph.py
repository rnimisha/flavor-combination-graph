from typing import Dict

import pandas as pd


def create_nodes_mapping(nodes_df: pd.DataFrame) -> Dict[str, Dict[int, int]]:
    node_types = nodes_df["node_type"].unique()
    mappings = {}

    for node_type in node_types:
        filtered_node_data = nodes_df[nodes_df["node_type"] == node_type]
        mappings[node_type] = {
            original_id: id
            for id, original_id in enumerate(filtered_node_data["node_id"].unique())
        }

    return mappings
