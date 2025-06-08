import pandas as pd

# Load the CSV
df = pd.read_csv("data/ITSM_cleaned.csv")

# Save as Parquet
df.to_parquet("src/itsm_analysis/feature_repo/data/itsm_cleaned.parquet", engine="pyarrow", index=False)

print("CSV converted to Parquet successfully.")