import pandas as pd
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------
# URL NORMALIZATION FUNCTION
# --------------------------------------
def normalize_url(url):
    if pd.isna(url):
        return ""

    url = str(url).lower().strip()
    url = url.replace("/solutions", "")
    url = url.split("?")[0]
    url = url.rstrip("/")

    return url


# --------------------------------------
# LOAD FAISS INDEX
# --------------------------------------
index = faiss.read_index("shl_recommendation/models/faiss_index.index")

# --------------------------------------
# LOAD METADATA
# --------------------------------------
with open("shl_recommendation/models/metadata.pkl", "rb") as f:
    df = pickle.load(f)

df["normalized_url"] = df["url"].apply(normalize_url)

# --------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------
model = SentenceTransformer("all-mpnet-base-v2")

# --------------------------------------
# LOAD TRAIN DATASET
# --------------------------------------
train_df = pd.read_excel("Gen_AI Dataset.xlsx", sheet_name="Train-Set")


# --------------------------------------
# IMPROVED RECALL@K FUNCTION
# --------------------------------------
def recall_at_k(query, relevant_urls, k=10):

    # Encode query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    # Retrieve top 25 candidates
    scores, indices = index.search(query_embedding, 25)

    candidates = df.iloc[indices[0]].copy()

    # Keyword overlap boost
    query_words = set(query.lower().split())

    def keyword_score(text):
        words = set(str(text).lower().split())
        return len(query_words & words)

    candidates["boost"] = candidates["combined_text"].apply(keyword_score)

    # Combine semantic score + keyword boost
    candidates["semantic_score"] = scores[0]
    candidates["final_score"] = candidates["semantic_score"] + 0.05 * candidates["boost"]

    # Re-rank
    candidates = candidates.sort_values(by="final_score", ascending=False)

    # Take top K
    top_k_urls = candidates.head(k)["normalized_url"].tolist()

    relevant_set = set(normalize_url(u) for u in relevant_urls)
    retrieved_set = set(top_k_urls)

    if len(relevant_set) == 0:
        return 0

    return len(relevant_set & retrieved_set) / len(relevant_set)


# --------------------------------------
# COMPUTE MEAN RECALL
# --------------------------------------
recalls = []

for _, row in train_df.iterrows():

    query = row.iloc[0]

    relevant_urls = []
    for col in row.index[1:]:
        if pd.notna(row[col]):
            relevant_urls.append(row[col])

    if len(relevant_urls) > 0:
        score = recall_at_k(query, relevant_urls, k=10)
        recalls.append(score)

mean_recall = sum(recalls) / len(recalls)

print("\nMean Recall@10:", round(mean_recall, 4))