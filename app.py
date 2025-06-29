import os

import streamlit as st
import torch

from src.create_graph import IngredientCompoundGraph
from src.load_data import load_data
from src.model import PairingModel


@st.cache_resource
def load_models():
    with st.spinner("Loading....."):
        nodes_df, edges_df = load_data()
        ingredientCompoundGraph = IngredientCompoundGraph(nodes_df, edges_df)
        data = ingredientCompoundGraph.create()
        idx_to_name = ingredientCompoundGraph.get_name_mapping(node_type="ingredient")
        name_to_idx = {v: k for k, v in idx_to_name.items()}

        metapath = [
            ("ingredient", "paired_with", "ingredient"),
            ("ingredient", "associated_with", "compound"),
            ("compound", "rev_associated_with", "ingredient"),
        ]

        model = PairingModel(data)
        recommender = model.create_model(metapath, 128)
        recommender.load_state_dict(
            torch.load(os.path.join("model", "combined_model.pt"))
        )

        return {
            "data": data,
            "idx_to_name": idx_to_name,
            "name_to_idx": name_to_idx,
            "recommender": recommender,
            "all_ingredients": list(name_to_idx.keys()),
        }
