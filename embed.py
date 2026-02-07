
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# Load cleaned movie data
df = pd.read_csv("data/movies_clean.csv")
print("Loaded data:", df.shape)

# Load embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# Extract text
texts = df["embedding_text"].tolist()

# Generate embeddings
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

# Store embeddings
df["embedding"] = embeddings.tolist()

# Save to disk
with open("embeddings/movie_embeddings.pkl", "wb") as f:
    pickle.dump(df, f)

print("Embeddings saved successfully.")
