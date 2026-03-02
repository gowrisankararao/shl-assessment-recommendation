import pandas as pd
import requests
import os

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "http://127.0.0.1:8000/recommend"
INPUT_PATH = "shl_recommendation/evaluation/unlabeled_test.csv"
OUTPUT_PATH = "submission_test_predictions.csv"

# -----------------------------
# LOAD TEST DATA (fix encoding)
# -----------------------------
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"File not found: {INPUT_PATH}")

test_df = pd.read_csv(INPUT_PATH, encoding="cp1252")

if "Query" not in test_df.columns:
    raise ValueError("CSV must contain a column named 'Query'")

rows = []

print("🚀 Generating predictions...\n")

# -----------------------------
# CALL API FOR EACH QUERY
# -----------------------------
for idx, row in test_df.iterrows():
    query = str(row["Query"]).strip()

    print(f"{idx+1}/{len(test_df)} Processing query...")

    try:
        response = requests.post(
            API_URL,
            json={"query": query},
            timeout=30
        )

        if response.status_code == 200:
            results = response.json().get("recommended_assessments", [])

            for item in results:
                rows.append({
                    "Query": query,
                    "Assessment_url": item.get("url", "")
                })
        else:
            print(f"❌ API error ({response.status_code}) for query: {query}")

    except Exception as e:
        print(f"❌ Exception for query: {query}")
        print(e)

# -----------------------------
# SAVE IN REQUIRED FORMAT
# -----------------------------
if not rows:
    raise ValueError("No predictions generated. Check API.")

output_df = pd.DataFrame(rows)

# Ensure exactly 2 columns
output_df = output_df[["Query", "Assessment_url"]]

output_df.to_csv(OUTPUT_PATH, index=False)

print("\n✅ submission_test_predictions.csv generated successfully!")
print(f"Total rows generated: {len(output_df)}")