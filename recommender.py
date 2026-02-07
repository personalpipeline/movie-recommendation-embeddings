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

def recommend_movies(movie_title, top_n=5):
    idx = find_movie_index(movie_title)

    if idx is None:
        print("Movie not found")
        return None

    movie_vector = embedding_matrix[idx].reshape(1, -1)
    similarities = cosine_similarity(movie_vector, embedding_matrix)[0]

    similar_indices = similarities.argsort()[::-1][1:top_n + 1]

    results = df.iloc[similar_indices][["title", "overview"]].copy()
    results["similarity"] = similarities[similar_indices]

    return results

if __name__ == "__main__":
    recs = recommend_movies("Inception", top_n=5)
    print(recs)
