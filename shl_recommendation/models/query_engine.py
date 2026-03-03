import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Load FAISS index
index = faiss.read_index("shl_recommendation/models/faiss_index.index")

# Load metadata
with open("shl_recommendation/models/metadata.pkl", "rb") as f:
    df = pickle.load(f)


def search(query, top_k=5):

    # Encode query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search top 20 for better balancing
    distances, indices = index.search(query_embedding, 20)
    results = df.iloc[indices[0]].to_dict(orient="records")

    technical = []
    behavioral = []

    for r in results:
        test_types = r.get("test_type", "")

        # Convert to list safely
        if isinstance(test_types, str):
            types = [t.strip() for t in test_types.split(",")]
        else:
            types = test_types

        if any("Knowledge & Skills" in t for t in types):
            technical.append(r)

        if any("Competencies" in t or "Personality & Behaviour" in t for t in types):
            behavioral.append(r)

    balanced_results = []

    # Add 1 behavioral if available
    if behavioral:
        balanced_results.append(behavioral[0])

    # Add 1 technical if available
    if technical:
        if technical[0] not in balanced_results:
            balanced_results.append(technical[0])
    else:
        # 🔥 FORCE technical if none found in top 20
        tech_df = df[df["test_type"].str.contains("Knowledge & Skills", na=False)]
        if not tech_df.empty:
            forced_tech = tech_df.iloc[0].to_dict()
            balanced_results.append(forced_tech)

    # Fill remaining slots
    for r in results:
        if r not in balanced_results:
            balanced_results.append(r)

    return balanced_results[:top_k]


if __name__ == "__main__":
    user_query = input("Enter job description: ")
    results = search(user_query)

    print("\nTop Recommendations:\n")
    for r in results:
        print(r["name"], "->", r["url"])