import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os


def main():

    # --------------------------------------------------
    # Paths (Safe for Render)
    # --------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "shl_catalog.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found. Run scraper first.")

    df = pd.read_csv(DATA_PATH)

    print("Total products loaded:", len(df))

    # --------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------
    df.drop_duplicates(subset=["url"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("After removing duplicates:", len(df))

    # --------------------------------------------------
    # Clean metadata
    # --------------------------------------------------
    df["name"] = df["name"].fillna("")
    df["description"] = df["description"].fillna("")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype(int)
    df["adaptive_support"] = df["adaptive_support"].fillna("No")
    df["remote_support"] = df["remote_support"].fillna("No")
    df["test_type"] = df["test_type"].fillna("Not specified")

    if "combined_text" not in df.columns:
        df["combined_text"] = df["name"] + ". " + df["description"]

    # --------------------------------------------------
    # Load embedding model
    # --------------------------------------------------
    print("Loading embedding model...")
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    texts = df["combined_text"].astype(str).tolist()

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    # --------------------------------------------------
    # Normalize for COSINE similarity
    # --------------------------------------------------
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("Total vectors indexed:", index.ntotal)

    # --------------------------------------------------
    # Save files
    # --------------------------------------------------
    index_path = os.path.join(MODEL_DIR, "faiss_index.index")
    metadata_path = os.path.join(MODEL_DIR, "metadata.pkl")

    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(df, f)

    print("✅ FAISS index built successfully using COSINE similarity!")


# --------------------------------------------------
# Allow standalone execution
# --------------------------------------------------
if __name__ == "__main__":
    main()