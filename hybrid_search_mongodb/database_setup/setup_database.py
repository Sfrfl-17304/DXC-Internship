# setup_database.py
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()


MONGO_USER = os.getenv("MONGO_USERNAME")
MONGO_PASS = os.getenv("MONGO_PASSWORD")
MONGO_HOST = "localhost"
MONGO_PORT = "27017"
MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/?authSource=admin"
DB_NAME = "hr_dxc_database" # Use a distinct name for the new version

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info() # Will raise an exception if connection fails
    print("MongoDB connection successful.")
    db = client[DB_NAME]

    if "candidates" not in db.list_collection_names():
        db.create_collection("candidates")
        print("Collection 'candidates' created.")

    if "skills_dictionary" not in db.list_collection_names():
        db.create_collection("skills_dictionary")
        initial_skills = [
            {"skill_name": "Python"}, {"skill_name": "Java"},
            {"skill_name": "SQL"}, {"skill_name": "Project Management"}
        ]
        db.skills_dictionary.insert_many(initial_skills)
        print("Collection 'skills_dictionary' created and populated.")

    print("\nDatabase setup is complete.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your MongoDB Docker container is running via 'docker start mongo-hr'.")