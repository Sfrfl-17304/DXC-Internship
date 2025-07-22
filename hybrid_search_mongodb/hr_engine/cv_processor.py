import os
import json
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from dxc_rag_pipeline.loaders import get_document_loader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
MONGO_USER = os.getenv("MONGO_USERNAME")
MONGO_PASS = os.getenv("MONGO_PASSWORD")
MONGO_HOST = "localhost"
MONGO_PORT = "27017" # Use "27018" for the Protonow project

MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/?authSource=admin"
DB_NAME = "hr_dxc_database"

# --- Initialize Clients ---
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # The server_info() call is what actually checks the connection.
    client.server_info()
    db = client[DB_NAME]
    candidates_collection = db["candidates"]
    skills_collection = db["skills_dictionary"]
    print("✅ MongoDB connection successful.")
except Exception as e:
    # If this block runs, the app will fail here, which is the correct behavior.
    print(f"❌ MongoDB connection error: {e}")
    db = None

llm_groq = ChatGroq(model_name="llama3-70b-8192")
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

def process_cv_directory(directory_path: str):
    """
    Processes all supported documents in a given directory, extracts data,
    and stores it in MongoDB.
    """
    

    print(f"Starting to process documents in: {directory_path}")

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if not os.path.isfile(file_path):
            continue

        print(f"\n--- Processing file: {filename} ---")
        
        try:
            loader = get_document_loader(file_path)
            documents = loader.load()
            raw_text = " ".join([doc.page_content for doc in documents])

            extraction_prompt = f"""
            You are a precise HR data extraction system. Analyze the following CV text and output ONLY a valid JSON object with these exact keys: "candidate_name" (string), "experience_years" (integer), and "skills" (a list of strings). Do not invent information. If a value is not present, return null for that key. Output ONLY the JSON object and nothing else.

            CV Text:
            ---
            {raw_text}
            ---
            """
            response = llm_groq.invoke(extraction_prompt)
            structured_data = json.loads(response.content)
            print(f"✅ Structured data extracted for {structured_data.get('candidate_name')}.")

            extracted_skills = structured_data.get("skills", [])
            if extracted_skills:
                for skill in extracted_skills:
                    skills_collection.update_one(
                        {"skill_name": skill},
                        {"$setOnInsert": {"skill_name": skill}},
                        upsert=True
                    )
            
            print("⏳ Creating vector embedding...")
            text_embedding = embedding_model.embed_query(raw_text)
            
            candidate_document = {
                "source_file": filename,
                "candidate_name": structured_data.get("candidate_name"),
                "experience_years": structured_data.get("experience_years"),
                "skills": extracted_skills,
                "full_text_content": raw_text,
                "text_embedding_cv": text_embedding
            }
            candidates_collection.insert_one(candidate_document)
            print(f"✅ Candidate '{structured_data.get('candidate_name')}' successfully added to the database.")

        except ValueError as ve:
            print(f"⚠️ Skipping unsupported file: {filename} ({ve})")
        except Exception as e:
            print(f"❌ An error occurred while processing {filename}: {e}")

    print("\n--- Batch processing complete. ---")

if __name__ == '__main__':
    cv_directory = "./cv_samples"
    if not os.path.isdir(cv_directory):
        os.makedirs(cv_directory)
        print(f"Created sample directory: {cv_directory}")
        print("Please add your CV PDFs to this folder and run the script again.")
    elif db is not None: # We check the connection status before proceeding
        process_cv_directory(cv_directory)
    else:
        print("Cannot start processing, database connection is not valid.")