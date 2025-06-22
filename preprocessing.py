# %% imports
from create_graph import create_nodes_mapping
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
mapped = create_nodes_mapping(nodes_df)
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
