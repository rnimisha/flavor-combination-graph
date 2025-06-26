from typing import List, Tuple

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec
from tqdm import tqdm


def generate_embeddings(
    data: HeteroData,
    metapath: List[Tuple[str, str, str]],
    save_path: str,
    embedding_dim: int = 300,
    epochs: int = 20,
    lr: float = 0.005,
) -> None:
    model = MetaPath2Vec(
        edge_index_dict=data.edge_index_dict,
        embedding_dim=embedding_dim,
        metapath=metapath,
        walk_length=60,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=10,
        sparse=True,
    )

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)

    model.train()

    best_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, save_path)
