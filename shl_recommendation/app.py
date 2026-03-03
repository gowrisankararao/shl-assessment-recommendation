from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pickle
import os
import faiss
import torch
import uvicorn
from sentence_transformers import SentenceTransformer

app = FastAPI()

# -----------------------------
# Configuration & Paths
# -----------------------------
# Updated to correctly find 'models' relative to app.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
metadata_path = os.path.join(MODEL_DIR, "metadata.pkl")
index_path = os.path.join(MODEL_DIR, "faiss_index.index")

# Global variables for lazy loading (Crucial for 512MB RAM limit)
_model = None
_index = None
_df = None

class QueryRequest(BaseModel):
    query: str

def get_resources():
    """Lazy load resources to keep startup memory low."""
    global _model, _index, _df
    if _model is None:
        torch.set_num_threads(1) # Prevents CPU/RAM spikes on Render
        _model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    
    if _df is None:
        with open(metadata_path, "rb") as f:
            _df = pickle.load(f)
            
    if _index is None:
        _index = faiss.read_index(index_path)
    
    return _model, _index, _df

# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    model, index, df = get_resources()

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

# -----------------------------
# Fixed Frontend UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def frontend():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SHL Smart Recommender</title>
        <style>
            :root { --bg: #0f172a; --card: #1e293b; --accent: #38bdf8; --text: #f8fafc; }
            body { font-family: 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); padding: 40px; margin: 0; }
            .container { max-width: 850px; margin: auto; }
            h2 { text-align: center; color: var(--accent); font-size: 2rem; }
            textarea { width: 100%; padding: 18px; border-radius: 12px; border: 1px solid #334155; background: #1e293b; color: white; font-size: 16px; margin-bottom: 20px; box-sizing: border-box; }
            button { width: 100%; background: var(--accent); color: #0f172a; border: none; padding: 16px; border-radius: 10px; font-weight: bold; font-size: 1rem; cursor: pointer; transition: 0.3s; }
            button:hover { opacity: 0.9; transform: scale(1.01); }
            #loading { display: none; text-align: center; margin: 20px; color: var(--accent); font-weight: bold; }
            .card { background: var(--card); padding: 25px; border-radius: 15px; margin-top: 25px; border: 1px solid #334155; }
            .meta { font-size: 14px; color: #94a3b8; margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap; }
            .badge { background: #0f172a; padding: 5px 12px; border-radius: 6px; border: 1px solid #334155; }
            a { color: var(--accent); text-decoration: none; font-weight: bold; display: inline-block; margin-top: 20px; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>SHL Assessment Recommendation</h2>
            <textarea id="query" rows="5" placeholder="Paste Job Description here (e.g., 'Looking for a Senior Python Developer with Leadership skills')..."></textarea>
            <button onclick="search()">Find Best Assessments</button>
            <div id="loading">Analyzing... (This can take 15s on first load)</div>
            <div id="results"></div>
        </div>

        <script>
            async function search() {
                const query = document.getElementById("query").value;
                const resultsDiv = document.getElementById("results");
                const loader = document.getElementById("loading");
                if(!query.trim()) return alert("Please enter a query.");

                resultsDiv.innerHTML = "";
                loader.style.display = "block";

                try {
                    const response = await fetch("/recommend", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({query: query})
                    });
                    const data = await response.json();
                    loader.style.display = "none";

                    data.recommended_assessments.forEach(item => {
                        resultsDiv.innerHTML += `
                            <div class="card">
                                <h3>${item.name}</h3>
                                <p>${item.description}</p>
                                <div class="meta">
                                    <span class="badge">⏱ ${item.duration} min</span>
                                    <span class="badge">⚙ Adaptive: ${item.adaptive_support}</span>
                                    <span class="badge">🌐 Remote: ${item.remote_support}</span>
                                    <span class="badge">🏷 ${item.test_type.join(", ")}</span>
                                </div>
                                <a href="${item.url}" target="_blank">Explore Catalog Details →</a>
                            </div>`;
                    });
                } catch (err) {
                    loader.style.display = "none";
                    resultsDiv.innerHTML = "<p style='color:red; text-align:center;'>Instance is waking up. Please refresh and try again in 15 seconds.</p>";
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
