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
recipe_model = model.create_model(metapath_recipe, embedding_dim=128)
comp_ingr_model = model.create_model(metapath_chemical, embedding_dim=128)
# %%

save_path_recipe = os.path.join("model", "recipe_model.pt")
save_path_substitution = os.path.join("model", "comp_ingr_model.pt")
# %%
model.train_model(recipe_model, save_path_recipe)

# %%
model.train_model(comp_ingr_model, save_path_substitution)

# %%

recipe_model.load_state_dict(torch.load(save_path_recipe))
recipe_recommender = recipe_model

# %%
a = recommend_pairs(recipe_recommender, "tomato", idx_to_name)
# %%
a
# %%
comp_ingr_model.load_state_dict(torch.load(save_path_substitution))
subsitution_recommender = comp_ingr_model

# %%
a = recommend_pairs(comp_ingr_model, "onion", idx_to_name)
# %%
a
# %%
data

# %%
