from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec


def recommend_pairs(
    model: MetaPath2Vec, query: str, idx_to_name: dict, name_to_idx: dict, top_k=5
):

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


def recommend_substitue(
    data: HeteroData,
    query: str,
    name_to_idx: dict,
    idx_to_name: dict,
    top_k=5,
) -> list:
    if query not in name_to_idx:
        print(f"Ingredient '{query}' not found in vocabulary")
        return

    edge_index = data[("ingredient", "associated_with", "compound")].edge_index
    ingredient_to_compounds = defaultdict(set)

    for ingr_idx, comp_idx in edge_index.t():
        ingredient_to_compounds[ingr_idx.item()].add(comp_idx.item())

    query_idx = name_to_idx[query]
    query_compounds = ingredient_to_compounds.get(query_idx, set())

    if not query_compounds:
        return []

    results = []
    for ingr_idx, compounds in ingredient_to_compounds.items():
        if ingr_idx == query_idx:
            continue

        shared = len(query_compounds & compounds)
        union = len(query_compounds | compounds)

        min_shared = 1
        if shared >= min_shared:
            jaccard = shared / union
            ingr_name = idx_to_name[ingr_idx]
            results.append((ingr_name, jaccard, shared))

    results.sort(key=lambda x: -x[1])
    return results[:top_k]
