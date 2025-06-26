from typing import Tuple

from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit


def split_data(data: HeteroData) -> Tuple[HeteroData, HeteroData, HeteroData]:

    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        edge_types=[("ingredient", "paired_with", "ingredient")],
    )

    return transform(data)
