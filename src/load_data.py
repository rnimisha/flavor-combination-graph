import pandas as pd


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv("./data/nodes_191120.csv")
    edges = pd.read_csv("./data/edges_191120.csv")

    edges = edges[edges["edge_type"] != "ingr-dcomp"]
    return nodes, edges
