import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import requests
import os

# --------------------------------------------------
# Configuration
# --------------------------------------------------

EMBEDDINGS_URL = (
    "https://huggingface.co/personalML/movie-embeddings/"
    "resolve/main/movie_embeddings.pkl"
)
LOCAL_PATH = "movie_embeddings.pkl"

# --------------------------------------------------
# Data loading (cached)
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_embeddings():
    if not os.path.exists(LOCAL_PATH):
        with st.spinner("Downloading embeddings (first run only)..."):
            response = requests.get(EMBEDDINGS_URL)
            response.raise_for_status()
            with open(LOCAL_PATH, "wb") as f:
                f.write(response.content)

    with open(LOCAL_PATH, "rb") as f:
        return pickle.load(f)

df = load_embeddings()
embedding_matrix = np.vstack(df["embedding"].values)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def find_movie_index(title):
    matches = df[df["title"].str.lower().str.contains(title.lower(), na=False)]
    if matches.empty:
        return None
    return matches.index[0]

def recommend_movies(movie_title, top_n, method):
    idx = find_movie_index(movie_title)
    if idx is None:
        return None

    query_vector = embedding_matrix[idx].reshape(1, -1)

    if method == "Cosine Similarity":
        scores = cosine_similarity(query_vector, embedding_matrix)[0]
        higher_is_better = True

    elif method == "Dot Product (raw)":
        scores = (embedding_matrix @ query_vector.T).flatten()
        higher_is_better = True

    elif method == "Euclidean Distance":
        scores = euclidean_distances(query_vector, embedding_matrix)[0]
        higher_is_better = False

    # Sort results correctly
    if higher_is_better:
        indices = scores.argsort()[::-1][1 : top_n + 1]
    else:
        indices = scores.argsort()[1 : top_n + 1]

    results = df.iloc[indices][["title", "overview"]].copy()
    results["score"] = scores[indices]

    return results

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🎬 Movie Recommendation System")
st.write(
    "Explore movie recommendations using different vector similarity methods."
)

movie_name = st.text_input("Enter a movie name:", "")
top_n = st.slider("Number of recommendations", 1, 10, 5)

search_method = st.selectbox(
    "Choose similarity method",
    [
        "Cosine Similarity",
        "Dot Product (raw)",
        "Euclidean Distance",
    ],
)

method_explanations = {
    "Cosine Similarity": "Compares direction of embeddings (semantic similarity).",
    "Dot Product (raw)": "Considers direction and magnitude (recommender-style).",
    "Euclidean Distance": "Measures absolute distance in embedding space.",
}

st.info(method_explanations[search_method])

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        recommendations = recommend_movies(
            movie_name,
            top_n,
            search_method,
        )

        if recommendations is None:
            st.error("Movie not found. Try another title.")
        else:
            for _, row in recommendations.iterrows():
                st.subheader(row["title"])
                st.write(row["overview"])

                if search_method == "Euclidean Distance":
                    st.caption(f"Distance: {row['score']:.3f}")
                else:
                    st.caption(f"Similarity score: {row['score']:.3f}")
