from pymongo import MongoClient

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "hr_dxc_database"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client[DB_NAME]
    candidates_collection = db["candidates"]

    print("--- MongoDB Inspection ---")
    candidate_count = candidates_collection.count_documents({})

    print(f"RESULT: Found {candidate_count} candidates in the '{DB_NAME}' database.")

    if candidate_count > 0:
        print("\nDIAGNOSIS: Data exists. The problem is in the search logic of your app.py.")
    else:
        print("\nDIAGNOSIS: No data found. The problem is in your ingestion script (cv_processor.py).")

except Exception as e:
    print(f"An error occurred: {e}")