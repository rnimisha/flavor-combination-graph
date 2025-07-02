# 𓌉◯𓇋 Ingredient Pairing and Substitution Recommender

The project provides a recommendation system for food ingredients, offering both pairing suggestions (what ingredients go well together) and substitution recommendations (similar ingredients that can replace each other in recipes).

## ☘︎ Features

### Ingredient Pairing Recommendations

- Suggests ingredients that commonly pair well with a given ingredient, based on learned graph embeddings.

### Ingredient Substitution Recommendations

- Suggests substitute ingredients based on shared chemical compounds using Jaccard similarity.

## ☘︎ Technical Approach

### Graph Architecture

The system models culinary relationships as a **heterogeneous graph** with:

**Node Types:**

- `ingredient`: Culinary ingredients (e.g., tomato, basil, garlic)
- `compound`: Chemical flavor compounds (e.g., limonene, vanillin)

**Edge Types:**

- `ingredient-ingredient` (`paired_with`): Direct pairing relationships from recipe co-occurrence
- `ingredient-compound` (`associated_with`): Chemical composition relationships

## ☘︎ Method

1. **Graph Construction**: Build heterogeneous graph from ingredient pairing data and chemical composition
2. **Embedding Learning**: Use **MetaPath2Vec** to learn dense vector representations that capture:
   - Direct ingredient pairing patterns
   - Indirect relationships through shared flavor compounds
   - Complex multi-hop culinary associations
3. **Recommendation Generation**:
   - **Pairings**: Find ingredients with similar embeddings in the learned space
   - **Substitutions**: Calculate Jaccard similarity based on shared chemical compounds

## ☘︎ Data Source

This project utilizes data from the comprehensive **FlavorGraph** dataset:

> Park, D.M., Kim, J., Park, J. et al. FlavorGraph: a large-scale food-chemical graph for generating food representations and recommending food pairings. _Sci Rep_ 11, 931 (2021). https://doi.org/10.1038/s41598-020-79422-8

## ☘︎ UI Screenshots

![UI](https://raw.githubusercontent.com/rnimisha/flavor-combination-graph/main/images/pairing.png)

## ☘︎ Installation

```bash
  git clone https://github.com/rnimisha/flavor-combination-graph.git
```

```bash
  cd flavor-combination-graph
  pip install -r requirements.txt
```

```bash
  streamlit run app.py
```
