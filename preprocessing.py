# %% imports
import os

import torch

from model import PairingModel
from src.create_graph import IngredientCompoundGraph
from src.load_data import load_data
from src.similarity import recommend_pairs

# %%
nodes_df, edges_df = load_data()

# %%
ingredientCompoundGraph = IngredientCompoundGraph(nodes_df, edges_df)
data = ingredientCompoundGraph.create()
idx_to_name = ingredientCompoundGraph.get_name_mapping(node_type="ingredient")
# %%
metapath_recipe = [("ingredient", "paired_with", "ingredient")]
metapath_chemical = [
    ("ingredient", "associated_with", "compound"),
    ("compound", "rev_associated_with", "ingredient"),
]
# %%
model = PairingModel(data)

# %%
metapaths = [("ingredient", "paired_with", "ingredient")]
recipe_model = model.create_model(metapaths, embedding_dim=128)
# %%

save_path = os.path.join("model", "recipe_model.pt")
# %%
model.train_model(recipe_model, save_path)

# %%

recipe_model.load_state_dict(torch.load(save_path))
recommender = recipe_model

# %%
a = recommend_pairs(recommender, "tomato", idx_to_name)
# %%
a
# %%
