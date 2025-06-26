import torch
import torch.nn.functional as F
from torch_geometric.nn import MetaPath2Vec


def recommend_pairs(model: MetaPath2Vec, query: str, idx_to_name: dict, top_k=5):
    name_to_idx = {v: k for k, v in idx_to_name.items()}
    if query not in name_to_idx:
        print(f"Ingredient '{query}' not found in vocabulary")
        return

    query_id = name_to_idx[query]

    query_embed = model("ingredient", torch.tensor([query_id]))
    all_ingredients = torch.arange(len(idx_to_name))
    all_embeds = model("ingredient", all_ingredients)

    sims = F.cosine_similarity(query_embed, all_embeds)
    top_indices = (-sims).argsort()[1 : top_k + 1]
    top_scores = sims[top_indices]

    results = []
    for idx, score in zip(top_indices, top_scores):
        ingredient_name = idx_to_name[idx.item()]
        results.append((ingredient_name, score.item(), idx.item()))

    return results
