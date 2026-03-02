# SHL Assessment Recommendation System

## Overview

This project implements an intelligent assessment recommendation system for SHL's product catalog.

The system:
- Accepts a natural language query or job description
- Retrieves relevant SHL assessments
- Returns a balanced set of recommendations
- Uses embedding-based retrieval (SentenceTransformers + FAISS)

---

## Architecture

1. Data Ingestion
   - SHL product catalogue scraped
   - Cleaned and structured into CSV format

2. Embedding & Indexing
   - SentenceTransformer model used for embeddings
   - FAISS used for similarity search

3. Recommendation Engine
   - Query converted into embedding
   - Top K results retrieved using FAISS
   - Balance logic ensures mix of test types

4. API Layer
   - Built using FastAPI
   - JSON response format as per assignment specification

---

## Tech Stack

- Python
- FastAPI
- SentenceTransformers
- FAISS
- Pandas
- NumPy

---

## API Endpoints

### 1. Health Check

GET /health

Response:
{
  "status": "healthy"
}

---

### 2. Recommendation Endpoint

POST /recommend

Request Body:
{
  "query": "Need a Java developer who collaborates well with teams"
}

Response Format:
{
  "recommended_assessments": [
    {
      "url": "valid_shl_url",
      "name": "assessment_name",
      "adaptive_support": "Yes/No",
      "description": "assessment_description",
      "duration": 60,
      "remote_support": "Yes/No",
      "test_type": ["Knowledge & Skills", "Competencies"]
    }
  ]
}

---

## Evaluation

- Mean Recall@K used for performance measurement
- Validated using provided labeled train dataset
- Submission predictions generated for unlabeled test dataset

---

## How to Run Locally

1. Create virtual environment:
   python -m venv venv

2. Activate environment:
   venv\Scripts\activate  (Windows)

3. Install dependencies:
   pip install -r requirements.txt

4. Run server:
   uvicorn app:app --reload

Server will start at:
http://127.0.0.1:8000

---

## Submission Files

- API endpoint
- GitHub repository
- submission_test_predictions.csv
- Documentation (2-page PDF)