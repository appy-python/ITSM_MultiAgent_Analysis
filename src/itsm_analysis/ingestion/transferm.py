# mongo_to_feast_ingest.py
import os
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder


from dotenv import load_dotenv
load_dotenv()


MONGO_DB_URI = os.getenv("MONGO_URI")
DATABASE = os.getenv("DATABASE")
COLLECTION = os.getenv("COLLECTION")
print(MONGO_DB_URI)


def load_from_mongo():
    client = MongoClient(MONGO_DB_URI)
    db = client[DATABASE]
    collection = db[COLLECTION]
    df = pd.DataFrame(list(collection.find()))
    print("Rows:", df.shape[0], "| Unique Incident_IDs:", df['Incident_ID'].nunique())
    return df


def preprocess(df):
    # Date parsing
    for col in ['Open_Time', 'Resolved_Time', 'Close_Time', 'Reopen_Time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')




    # Compute handling time
    df['Handle_Time_hrs'] = (df['Resolved_Time'] - df['Open_Time']).dt.total_seconds() / 3600


    # SLA Breach logic (assuming 48 hrs as example SLA)
    df['Resolution_SLA_Breach'] = df['Handle_Time_hrs'] > 48


    # Fill missing
    cat_cols = ['CI_Name', 'CI_Cat', 'CI_Subcat', 'Closure_Code', 'Priority', 'Impact', 'Urgency']
    df[cat_cols] = df[cat_cols].fillna('Unknown').astype(str)


    # Optional: Handle KB presence as binary
    if 'KB_number' in df.columns:
        df['Has_KB'] = df['KB_number'].notnull().astype(int)


    # Encode categorical
    #for col in cat_cols:
        #le = LabelEncoder()
        #df[f'{col}_enc'] = le.fit_transform(df[col])
   
    category_mappings = {}
    # Encode categorical variables (LabelEncoder not ideal for Feast due to lack of inverse mapping)
    for col in cat_cols:
        df[col] = df[col].astype('category')
        df[f'{col}_enc'] = df[col].cat.codes
        category_mappings[col] = dict(enumerate(df[col].cat.categories))




    # Select Feast-friendly features
    feast_cols = [
        'Incident_ID', 'Open_Time', 'Handle_Time_hrs',
        'CI_Name_enc', 'CI_Cat_enc', 'CI_Subcat_enc',
        'Closure_Code_enc', 'Priority_enc', 'Impact_enc', 'Urgency_enc',
        'Resolution_SLA_Breach'
    ]
   
    if 'Has_KB' in df.columns:
        feast_cols.append('Has_KB')


    # Keep only useful columns for Feast
    df_feast = df[feast_cols].copy()
    df_feast = df_feast.drop_duplicates(subset=['Incident_ID'])
   
    return df_feast, category_mappings


def save_to_csv(df, output_path):
    df.to_csv(output_path, index=False)


if __name__ == "__main__":


    out_file = "data/ITSM_cleaned.csv"
    map_file = "data/category_mappings.json"


    df_raw = load_from_mongo()
    df_cleaned, mappings = preprocess(df_raw)


    print("Final row count:", df_cleaned.shape[0])
    print("Unique Incident_IDs:", df_cleaned['Incident_ID'].nunique())


    save_to_csv(df_cleaned, out_file)
   
    # Save mappings
    import json
    with open(map_file, "w") as f:
        json.dump(mappings, f)


    print("Preprocessed data saved to:", out_file)
    print("Category mappings saved to:", map_file)