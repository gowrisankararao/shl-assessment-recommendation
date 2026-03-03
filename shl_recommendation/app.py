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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
metadata_path = os.path.join(MODEL_DIR, "metadata.pkl")
index_path = os.path.join(MODEL_DIR, "faiss_index.index")

# Global variables for lazy loading (Saves RAM on startup)
_model = None
_index = None
_df = None

class QueryRequest(BaseModel):
    query: str

def get_resources():
    """Load model and data only when the first request hits the API."""
    global _model, _index, _df
    
    if _model is None:
        # Crucial for Render: Prevent CPU/RAM spikes
        torch.set_num_threads(1)
        # Use a small, efficient model to fit in RAM
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

    # Load resources only when needed
    model, index, df = get_resources()

    # Generate Embeddings
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Search top 15 candidates
    scores, indices = index.search(query_embedding, 15)

    results = []
    used_types = set()

    for idx in indices[0]:
        if idx >= len(df) or idx < 0:
            continue
        
        row = df.iloc[idx]
        
        # Parse test types
        test_type_raw = str(row.get("test_type", "")).strip()
        test_type_list = [t.strip() for t in test_type_raw.split(",") if t.strip()]

        # Balance variety or fill up to minimum requirement
        if any(t not in used_types for t in test_type_list) or len(results) < 5:
            
            try:
                duration = int(float(row.get("duration", 0)))
            except:
                duration = 0

            results.append({
                "url": str(row.get("url", "")),
                "name": str(row.get("name", "")),
                "adaptive_support": "Yes" if str(row.get("adaptive_support", "")).lower() == "yes" else "No",
                "description": str(row.get("description", "")),
                "duration": duration,
                "remote_support": "Yes" if str(row.get("remote_support", "")).lower() == "yes" else "No",
                "test_type": test_type_list if test_type_list else ["Not specified"]
            })
            used_types.update(test_type_list)

        if len(results) == 10:
            break

    return {"recommended_assessments": results}

# -----------------------------
# Frontend UI
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def frontend():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SHL Assessment Recommendation</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg,#0f172a,#1e293b); color:white; padding:40px; min-height: 100vh;}
        .container{max-width:900px; margin:auto;}
        h2 { text-align: center; color: #38bdf8; margin-bottom: 30px; }
        textarea{width:100%; padding:15px; border-radius:8px; border:none; font-size:16px; margin-bottom:20px; background: #334155; color: white; box-sizing: border-box;}
        button{background:#3b82f6; border:none; padding:12px 25px; border-radius:6px; color:white; font-size:16px; cursor:pointer; width: 100%; transition: 0.3s;}
        button:hover{background:#2563eb; transform: translateY(-2px);}
        .card{background:#1e293b; padding:25px; border-radius:12px; margin-top:20px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);}
        .card h3 { color: #38bdf8; margin-top: 0; }
        .meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px; }
        .meta{font-size:14px; color:#cbd5e1; background: #0f172a; padding: 5px 10px; border-radius: 4px;}
        a{display: inline-block; margin-top: 15px; color:#38bdf8; text-decoration: none; font-weight: bold;}
        a:hover { text-decoration: underline; }
        #loading { display: none; text-align: center; margin-top: 20px; color: #38bdf8; font-weight: bold;}
    </style>
</head>
<body>
<div class="container">
    <h2>SHL Assessment Recommendation System</h2>
    <textarea id="query" rows="5" placeholder="Paste a Job Description or query..."></textarea>
    <button onclick="search()">Get Recommendations</button>
    <div id="loading">Connecting to model... (This may take 15s on the first search)</div>
    <div id="results"></div>
</div>

<script>
async function search() {
    const query = document.getElementById("query").value;
    const resultsDiv = document.getElementById("results");
    const loading = document.getElementById("loading");
    
    if(!query.trim()) return alert("Please enter a query");

    resultsDiv.innerHTML = "";
    loading.style.display = "block";

    try {
        const response = await fetch("/recommend", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({query: query})
        });

        const data = await response.json();
        loading.style.display = "none";

        if(!data.recommended_assessments || data.recommended_assessments.length === 0){
            resultsDiv.innerHTML = "<p style='text-align:center;'>No relevant assessments found.</p>";
            return;
        }

        data.recommended_assessments.forEach(item => {
            const card = `
            <div class="card">
                <h3>${item.name}</h3>
                <p>${item.description}</p>
                <div class="meta-grid">
                    <div class="meta">⏱ ${item.duration} mins</div>
                    <div class="meta">⚙ Adaptive: ${item.adaptive_support}</div>
                    <div class="meta">🌐 Remote: ${item.remote_support}</div>
                    <div class="meta">🏷 ${item.test_type.join(", ")}</div>
                </div>
                <a href="${item.url}" target="_blank">View Assessment →</a>
            </div>`;
            resultsDiv.innerHTML += card;
        });
    } catch (err) {
        loading.style.display = "none";
        resultsDiv.innerHTML = "<p style='color:red;'>Error connecting to server. Please try again in a moment.</p>";
    }
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
