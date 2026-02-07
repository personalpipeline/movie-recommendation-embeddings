import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
with open("embeddings/movie_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

print("Data loaded:", df.shape)

# Create embedding matrix
embedding_matrix = np.vstack(df["embedding"].values)

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

    # Sort correctly
    if higher_is_better:
        similar_indices = scores.argsort()[::-1][1 : top_n + 1]
    else:
        similar_indices = scores.argsort()[1 : top_n + 1]

    results = df.iloc[similar_indices][["title", "overview"]].copy()
    results["score"] = scores[similar_indices]

    return results
