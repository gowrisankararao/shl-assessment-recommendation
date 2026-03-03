from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
import faiss
import uvicorn

app = FastAPI()

# -----------------------------
# Request Model
# -----------------------------
class QueryRequest(BaseModel):
    query: str


# -----------------------------
# Load Model + FAISS + Metadata
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

metadata_path = os.path.join(BASE_DIR, "models", "metadata.pkl")
index_path = os.path.join(BASE_DIR, "models", "faiss_index.index")

with open(metadata_path, "rb") as f:
    df = pickle.load(f)

index = faiss.read_index(index_path)

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")


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

    # Normalize (important for cosine similarity in FAISS)
    faiss.normalize_L2(query_embedding)

    # Search top 15
    scores, indices = index.search(query_embedding, 15)

    results = []
    used_types = set()

    for idx in indices[0]:

        if idx >= len(df):
            continue

        row = df.iloc[idx]

        test_type_raw = str(row.get("test_type", "")).strip()
        test_type_list = [t.strip() for t in test_type_raw.split(",") if t.strip()]

        if not test_type_list:
            test_type_list = ["Not specified"]

        # Balance logic
        if any(t not in used_types for t in test_type_list) or len(results) < 5:

            try:
                duration = int(float(row.get("duration", 0)))
            except:
                duration = 0

            adaptive = "Yes" if str(row.get("adaptive_support", "")).lower() == "yes" else "No"
            remote = "Yes" if str(row.get("remote_support", "")).lower() == "yes" else "No"

            results.append({
                "url": str(row.get("url", "")),
                "name": str(row.get("name", "")),
                "adaptive_support": adaptive,
                "description": str(row.get("description", "")),
                "duration": duration,
                "remote_support": remote,
                "test_type": test_type_list
            })

            used_types.update(test_type_list)

        if len(results) == 10:
            break

    if len(results) < 5:
        raise HTTPException(status_code=500, detail="Less than 5 recommendations generated")

    return {"recommended_assessments": results}


# -----------------------------
# FRONTEND UI
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
body { font-family: Arial; background: linear-gradient(135deg,#0f172a,#1e293b); color:white; padding:40px;}
.container{max-width:900px;margin:auto;}
textarea{width:100%;padding:15px;border-radius:8px;border:none;font-size:16px;margin-bottom:20px;}
button{background:#3b82f6;border:none;padding:12px 20px;border-radius:6px;color:white;font-size:16px;cursor:pointer;}
button:hover{background:#2563eb;}
.card{background:#1e293b;padding:20px;border-radius:10px;margin-top:20px;}
.meta{font-size:14px;color:#cbd5e1;}
a{color:#38bdf8;}
</style>
</head>
<body>
<div class="container">
<h2>SHL Assessment Recommendation System</h2>
<textarea id="query" rows="4" placeholder="Enter job description or query..."></textarea>
<button onclick="search()">Get Recommendations</button>
<div id="results"></div>
</div>

<script>
async function search() {
const query=document.getElementById("query").value;
const resultsDiv=document.getElementById("results");
resultsDiv.innerHTML="<p>Loading...</p>";

const response=await fetch("/recommend",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({query:query})
});

const data=await response.json();
resultsDiv.innerHTML="";

if(!data.recommended_assessments){
resultsDiv.innerHTML="<p>No recommendations found.</p>";
return;
}

data.recommended_assessments.forEach(item=>{
const card=`
<div class="card">
<h3>${item.name}</h3>
<p>${item.description}</p>
<div class="meta">Duration: ${item.duration} mins</div>
<div class="meta">Adaptive: ${item.adaptive_support}</div>
<div class="meta">Remote: ${item.remote_support}</div>
<div class="meta">Test Type: ${item.test_type.join(", ")}</div>
<br>
<a href="${item.url}" target="_blank">View Assessment</a>
</div>`;
resultsDiv.innerHTML+=card;
});
}
</script>
</body>
</html>
"""


# -----------------------------
# LOCAL RUN FIX
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)