# %% imports
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
