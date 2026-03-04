from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pickle
import os
import faiss
import torch
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
metadata_path = os.path.join(MODEL_DIR, "metadata.pkl")
index_path = os.path.join(MODEL_DIR, "faiss_index.index")

# Global variables for lazy loading
_model = None
_index = None
_df = None

class QueryRequest(BaseModel):
    query: str

def get_resources():
    """Lazy load resources to keep startup memory low."""
    global _model, _index, _df
    if _model is None:
        # Limit threads to prevent memory/CPU spikes on free tier
        torch.set_num_threads(1)
        # Use a very small model (~33MB) to fit in RAM
        _model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    
    if _df is None:
        with open(metadata_path, "rb") as f:
            _df = pickle.load(f)
            
    if _index is None:
        _index = faiss.read_index(index_path)
    
    return _model, _index, _df

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Access resources only when needed
    model, index, df = get_resources()

    # Inference
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, 15)

    results = []
    used_types = set()

    for idx in indices[0]:
        if idx >= len(df) or idx < 0: continue
        
        row = df.iloc[idx]
        test_type_raw = str(row.get("test_type", "")).strip()
        test_type_list = [t.strip() for t in test_type_raw.split(",") if t.strip()]

        if any(t not in used_types for t in test_type_list) or len(results) < 5:
            results.append({
                "url": str(row.get("url", "")),
                "name": str(row.get("name", "")),
                "adaptive_support": "Yes" if str(row.get("adaptive_support", "")).lower() == "yes" else "No",
                "description": str(row.get("description", "")),
                "duration": int(float(row.get("duration", 0))),
                "remote_support": "Yes" if str(row.get("remote_support", "")).lower() == "yes" else "No",
                "test_type": test_type_list or ["Not specified"]
            })
            used_types.update(test_type_list)
        if len(results) == 10: break

    return {"recommended_assessments": results}

@app.get("/", response_class=HTMLResponse)
def frontend():
    # ... (Keep your existing HTML string here) ...
    return "HTML_STRING_FROM_YOUR_CODE"
