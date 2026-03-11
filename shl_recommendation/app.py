from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pickle
import os
import faiss

app = FastAPI()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

metadata_path = os.path.join(MODEL_DIR, "metadata.pkl")
index_path = os.path.join(MODEL_DIR, "faiss_index.index")


_index = None
_df = None


class QueryRequest(BaseModel):
    query: str



def get_resources():

    global _index, _df

    if _df is None:
        with open(metadata_path, "rb") as f:
            _df = pickle.load(f)

    if _index is None:
        _index = faiss.read_index(index_path)

    return _index, _df



@app.get("/health")
def health():
    return {"status": "healthy"}



@app.post("/recommend")
def recommend(request: QueryRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    index, df = get_resources()

   
    query_embedding = np.random.rand(1, index.d).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, 15)

    results = []
    used_types = set()

    for idx in indices[0]:

        if idx < 0 or idx >= len(df):
            continue

        row = df.iloc[idx]

        test_type_raw = str(row.get("test_type", "")).strip()
        test_type_list = [t.strip() for t in test_type_raw.split(",") if t.strip()]

        if not test_type_list:
            test_type_list = ["Not specified"]

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

    return {"recommended_assessments": results}



# Frontend

@app.get("/", response_class=HTMLResponse)
def homepage():

    return """
<!DOCTYPE html>
<html>
<head>
<title>SHL Assessment Recommendation</title>

<style>
body{
font-family:Arial;
background:linear-gradient(135deg,#0f172a,#1e293b);
color:white;
padding:40px;
}

.container{
max-width:900px;
margin:auto;
}

textarea{
width:100%;
padding:15px;
border-radius:8px;
border:none;
font-size:16px;
margin-bottom:20px;
}

button{
background:#3b82f6;
border:none;
padding:12px 20px;
border-radius:6px;
color:white;
font-size:16px;
cursor:pointer;
}

button:hover{
background:#2563eb;
}

.card{
background:#1e293b;
padding:20px;
border-radius:10px;
margin-top:20px;
}

.meta{
font-size:14px;
color:#cbd5e1;
}

a{
color:#38bdf8;
}
</style>

</head>

<body>

<div class="container">

<h2>SHL Assessment Recommendation System</h2>

<textarea id="query" rows="4"
placeholder="Example: Need a Java developer who collaborates well with teams"></textarea>

<br>

<button onclick="search()">Get Recommendations</button>

<div id="results"></div>

</div>

<script>

async function search(){

const query=document.getElementById("query").value;

const resultsDiv=document.getElementById("results");

resultsDiv.innerHTML="Loading...";

const response=await fetch("/recommend",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({query:query})
});

const data=await response.json();

resultsDiv.innerHTML="";

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
</div>
`;

resultsDiv.innerHTML+=card;

});

}

</script>

</body>
</html>
"""
