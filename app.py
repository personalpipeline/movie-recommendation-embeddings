import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

EMBEDDINGS_URL = "https://huggingface.co/personalML/movie-embeddings/resolve/main/movie_embeddings.pkl"
LOCAL_PATH = "movie_embeddings.pkl"

@st.cache_data
def load_embeddings():
    if not os.path.exists(LOCAL_PATH):
        with st.spinner("Downloading embeddings..."):
            response = requests.get(EMBEDDINGS_URL)
            with open(LOCAL_PATH, "wb") as f:
                f.write(response.content)

    with open(LOCAL_PATH, "rb") as f:
        return pickle.load(f)

df = load_embeddings()
embedding_matrix = np.vstack(df["embedding"].values)

def find_movie_index(title):
    matches = df[df["title"].str.lower().str.contains(title.lower(), na=False)]
    if matches.empty:
        return None
    return matches.index[0]

def recommend_movies(movie_title, top_n):
    idx = find_movie_index(movie_title)
    if idx is None:
        return None

    movie_vector = embedding_matrix[idx].reshape(1, -1)
    similarities = cosine_similarity(movie_vector, embedding_matrix)[0]
    similar_indices = similarities.argsort()[::-1][1:top_n + 1]

    results = df.iloc[similar_indices][["title", "overview"]].copy()
    results["similarity"] = similarities[similar_indices]
    return results

# ---------- UI ----------

st.title("🎬 Movie Recommendation System")
st.write("Type a movie name and get similar movie recommendations using AI embeddings.")

movie_name = st.text_input("Enter a movie name:", "")
top_n = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        recommendations = recommend_movies(movie_name, top_n)
        if recommendations is None:
            st.error("Movie not found. Try another title.")
        else:
            for _, row in recommendations.iterrows():
                st.subheader(row["title"])
                st.write(row["overview"])
                st.caption(f"Similarity score: {row['similarity']:.2f}")
