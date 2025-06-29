# %% imports
import os

import torch

from src.create_graph import IngredientCompoundGraph
from src.load_data import load_data
from src.model import PairingModel
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
metapath = [
    ("ingredient", "paired_with", "ingredient"),
    ("ingredient", "associated_with", "compound"),
    ("compound", "rev_associated_with", "ingredient"),
]
# %%
model = PairingModel(data)

# %%
recipe_model = model.create_model(metapath_recipe, embedding_dim=128)
comp_ingr_model = model.create_model(metapath_chemical, embedding_dim=128)
combined_model = model.create_model(metapath, 128)
# %%

save_path_recipe = os.path.join("model", "recipe_model.pt")
save_path_substitution = os.path.join("model", "comp_ingr_model.pt")
save_path_combined = os.path.join("model", "combined_model.pt")
# %%
model.train_model(recipe_model, save_path_recipe)

# %%
model.train_model(comp_ingr_model, save_path_substitution)


# %%
model.train_model(combined_model, save_path_combined)
# # %%

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
combined_model.load_state_dict(torch.load(save_path_combined))
combined_recommender = combined_model

# %%
a = recommend_pairs(combined_model, "onion", idx_to_name)

# %%
a
# %%
