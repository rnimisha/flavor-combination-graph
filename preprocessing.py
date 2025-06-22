# %% imports
from create_graph import IngredientCompoundGraph
from load_data import load_data

# %% load data
nodes_df, edges_df = load_data()

# %%
print(nodes_df.head())
# %%
print(edges_df.head())
# %%
print(nodes_df.node_type.value_counts())
# %%
print(edges_df.edge_type.value_counts())


# %%
ingredientCompoundGraph = IngredientCompoundGraph(nodes_df, edges_df)
mapped = ingredientCompoundGraph.create_nodes_mapping()
print(mapped.keys())

# %%
print(mapped["ingredient"])
# %%
print(nodes_df.node_type.unique())
# %%
ingredients = nodes_df[nodes_df["node_type"] == "ingredient"]
ingredients["node_id"].unique().shape

compounds = nodes_df[nodes_df["node_type"] == "compound"]
compounds["node_id"].unique().shape
# %%
print(len(mapped["ingredient"]))
# %%
print(ingredients["node_id"].unique().shape[0] == len(mapped["ingredient"]))

# %%
# %%
print(compounds["node_id"].unique().shape[0] == len(mapped["compound"]))

# %%
edges_df.head()
# %%
edges_types = edges_df["edge_type"].unique()
print(edges_types)
# %%
filtered_ing_ing = edges_df[edges_df["edge_type"] == "ingr-ingr"]
# %%
filtered_ing_ing
# %%
import numpy as np

mapped["ingredient"][np.int64(5063)]
mapped["ingredient"][np.int64(6083)]
# %%
edge_indexes = ingredientCompoundGraph.create_edges(mapped)
edge_indexes
# %%
edge_indexes.size()
# %%
