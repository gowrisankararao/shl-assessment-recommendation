from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import uvicorn

app = FastAPI()

# -----------------------------
# Request Model
# -----------------------------
class QueryRequest(BaseModel):
    query: str


# -----------------------------
# Load Model + Index + Metadata
# -----------------------------
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(BASE_DIR, "models", "faiss_index.index")
    metadata_path = os.path.join(BASE_DIR, "models", "metadata.pkl")

    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        df = pickle.load(f)

except Exception as e:
    print("Error loading model or data:", e)
    raise e


# -----------------------------
# HEALTH ENDPOINT
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# -----------------------------
# RECOMMEND ENDPOINT
# -----------------------------
@app.post("/recommend")
def recommend(request: QueryRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Encode query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    # Retrieve top 10
    scores, indices = index.search(query_embedding, 10)

    results = []

    for idx in indices[0]:

        if idx >= len(df):
            continue

        row = df.iloc[idx]

        # Clean duration
        duration = 0
        if pd.notna(row.get("duration")):
            try:
                duration = int(float(row.get("duration")))
            except:
                duration = 0

        # Clean adaptive_support
        adaptive = str(row.get("adaptive_support", "No")).strip()
        adaptive = "Yes" if adaptive.lower() == "yes" else "No"

        # Clean remote_support
        remote = str(row.get("remote_support", "No")).strip()
        remote = "Yes" if remote.lower() == "yes" else "No"

        # Clean test_type
        test_type_raw = str(row.get("test_type", "")).strip()
        test_type_list = [t.strip() for t in test_type_raw.split(",") if t.strip()]

        if not test_type_list:
            test_type_list = ["Not specified"]

        results.append({
            "url": str(row.get("url", "")),
            "name": str(row.get("name", "")),
            "adaptive_support": adaptive,
            "description": str(row.get("description", "")),
            "duration": duration,
            "remote_support": remote,
            "test_type": test_type_list
        })

    if len(results) < 5:
        raise HTTPException(status_code=500, detail="Less than 5 recommendations generated")

    return {
        "recommended_assessments": results[:10]
    }


# -----------------------------
# FRONTEND UI
# -----------------------------
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def frontend():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>SHL Assessment Recommendation</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1f2937, #111827);
            color: #f9fafb;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 900px;
            margin: 50px auto;
            padding: 20px;
        }

        h1 {
            text-align: center;
            font-size: 32px;
            margin-bottom: 30px;
            color: #60a5fa;
        }

        textarea {
            width: 100%;
            padding: 15px;
            border-radius: 10px;
            border: none;
            font-size: 16px;
            resize: none;
            margin-bottom: 20px;
        }

        button {
            background: #3b82f6;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: 0.3s;
        }

        button:hover {
            background: #2563eb;
        }

        .results {
            margin-top: 30px;
        }

        .card {
            background: #1f2937;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            transition: 0.3s;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .card h3 {
            margin-top: 0;
            color: #93c5fd;
        }

        .meta {
            font-size: 14px;
            margin: 8px 0;
            color: #d1d5db;
        }

        .view-btn {
            display: inline-block;
            margin-top: 10px;
            background: #10b981;
            padding: 8px 15px;
            border-radius: 6px;
            text-decoration: none;
            color: white;
            font-size: 14px;
        }

        .view-btn:hover {
            background: #059669;
        }

        .loading {
            margin-top: 15px;
            font-style: italic;
            color: #9ca3af;
        }
    </style>
</head>

<body>

<div class="container">
    <h1>SHL Assessment Recommendation System</h1>

    <textarea id="query" rows="4" placeholder="Enter job description or query..."></textarea>
    <button onclick="getRecommendations()">Get Recommendations</button>

    <div id="loading" class="loading"></div>
    <div class="results" id="results"></div>
</div>

<script>
async function getRecommendations() {

    const query = document.getElementById("query").value;
    const resultsDiv = document.getElementById("results");
    const loadingDiv = document.getElementById("loading");

    resultsDiv.innerHTML = "";
    loadingDiv.innerText = "Searching best assessments...";

    const response = await fetch("/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query })
    });

    const data = await response.json();
    loadingDiv.innerText = "";

    if (!data.recommended_assessments) {
        resultsDiv.innerHTML = "<p>No recommendations found.</p>";
        return;
    }

    data.recommended_assessments.forEach(item => {

        const card = `
            <div class="card">
                <h3>${item.name}</h3>
                <p>${item.description}</p>
                <div class="meta"><strong>Duration:</strong> ${item.duration} mins</div>
                <div class="meta"><strong>Adaptive:</strong> ${item.adaptive_support}</div>
                <div class="meta"><strong>Remote:</strong> ${item.remote_support}</div>
                <div class="meta"><strong>Test Type:</strong> ${item.test_type.join(", ")}</div>
                <a href="${item.url}" target="_blank" class="view-btn">View Assessment</a>
            </div>
        `;

        resultsDiv.innerHTML += card;
    });
}
</script>

</body>
</html>
"""

# -----------------------------
# DEPLOYMENT FIX (Render/Railway)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)