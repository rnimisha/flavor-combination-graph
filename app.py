import os

import streamlit as st
import torch

from src.create_graph import IngredientCompoundGraph
from src.load_data import load_data
from src.model import PairingModel
from src.similarity import recommend_pairs

st.markdown(
    """
        <style>
            .main {
                background-color: #f8f9fa;
            }
            .stTextInput>div>div>input {
                background-color: #ffffff;
            }
            .recommendation-card {
                background-color: #1E1E1E;
                color: #FAFAFA;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 4px solid #4A8BDF;
            }
            .highlight {
                background-color: #4A8BDF33;
                color: #4A8BDF;
                padding: 2px 5px;
                border-radius: 3px;
                font-weight: bold;
            }

        </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    with st.spinner("Loading....."):
        nodes_df, edges_df = load_data()
        ingredientCompoundGraph = IngredientCompoundGraph(nodes_df, edges_df)
        data = ingredientCompoundGraph.create()
        idx_to_name = ingredientCompoundGraph.get_name_mapping(node_type="ingredient")
        name_to_idx = {v: k for k, v in idx_to_name.items()}

        metapath = [
            ("ingredient", "paired_with", "ingredient"),
            ("ingredient", "associated_with", "compound"),
            ("compound", "rev_associated_with", "ingredient"),
        ]

        model = PairingModel(data)
        recommender = model.create_model(metapath, 128)
        recommender.load_state_dict(
            torch.load(os.path.join("model", "combined_model.pt"))
        )

        return {
            "data": data,
            "idx_to_name": idx_to_name,
            "name_to_idx": name_to_idx,
            "recommender": recommender,
            "all_ingredients": list(name_to_idx.keys()),
        }


models_data = load_models()

st.title("Ingredient Pairing and Substitution")
st.markdown("Discover new pairings and substitutions for recipes.")


ingredient = st.selectbox(
    "Search Ingredient:",
    options=models_data["all_ingredients"],
)


model_type = st.radio(
    "Recommendation Type:",
    options=[
        "Pairing Recommendation",
        "Substitution Recommendation",
    ],
    index=0,
    horizontal=True,
)

if st.button("Get Recommendations", type="primary"):
    if not ingredient:
        st.warning("Please enter an ingredient")

    else:
        with st.spinner(f"Finding recommendation for {ingredient}..."):
            if model_type == "Pairing Recommendation":
                result = recommend_pairs(
                    models_data["recommender"],
                    ingredient,
                    models_data["idx_to_name"],
                    models_data["name_to_idx"],
                    5,
                )

                if result:
                    st.subheader(
                        f"Ingredients that pair well with {ingredient.lower()}"
                    )

                    for name, score, _ in result:
                        st.markdown(
                            f"""
                                <div class="recommendation-card">
                                    <h4>{name}</h4>
                                    <p>Pairing score: <span class="highlight">{score:.3f}</span></p>
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning(
                        f"No pairing recommendations found for {ingredient.lower()}"
                    )

            elif model_type == "Substitution Recommendation":
                pass
