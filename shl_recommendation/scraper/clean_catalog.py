import pandas as pd

INPUT_PATH = "shl_recommendation/data/shl_catalog.csv"
OUTPUT_PATH = "shl_recommendation/data/shl_catalog_cleaned.csv"

df = pd.read_csv(INPUT_PATH)

print("Original count:", len(df))

# Remove duplicates only
df = df.drop_duplicates(subset=["url"])

# Remove rows where URL is missing
df = df[df["url"].notna()]
df = df[df["url"].str.strip() != ""]

# Remove rows where name is missing
df = df[df["name"].notna()]
df = df[df["name"].str.strip() != ""]

df = df.reset_index(drop=True)

print("Cleaned count:", len(df))

df.to_csv(OUTPUT_PATH, index=False)

print("Cleaned file saved successfully!")