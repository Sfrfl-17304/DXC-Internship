import streamlit as st
from pymongo import MongoClient
import os
import re # Import the regular expression module
from hybrid_search_mongodb.hr_engine.cv_processor import process_cv_directory
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()


# --- Page Configuration ---
st.set_page_config(
    page_title="HR Hybrid Search Engine",
    page_icon="logo.png",
    layout="wide"
)

# --- Database Connection ---
MONGO_USER = os.getenv("MONGO_USERNAME")
MONGO_PASS = os.getenv("MONGO_PASSWORD")
MONGO_HOST = "localhost"
MONGO_PORT = "27017" # Use "27018" for the Protonow project

MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/?authSource=admin"
DB_NAME = "hr_dxc_database"
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client[DB_NAME]
    candidates_collection = db["candidates"]
    st.sidebar.success("MongoDB Connected")
except Exception as e:
    st.sidebar.error(f"MongoDB Connection Failed: {e}")
    st.stop()

# --- Embedding Model ---
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# --- UI ---
st.title("HR Hybrid Search Engine")
st.write("This tool helps HR find the best candidates by combining structured filtering and semantic search.")

# --- Sidebar for Actions ---
st.sidebar.title("Admin Panel")
st.sidebar.markdown("---")
cv_directory = st.sidebar.text_input("CVs Folder Path", "./cv_samples")
if st.sidebar.button("Process All CVs in Folder"):
    if not os.path.isdir(cv_directory):
        st.sidebar.error(f"Directory not found: {cv_directory}")
    else:
        with st.spinner(f"Processing all documents in {cv_directory}..."):
            process_cv_directory(cv_directory)
        st.sidebar.success("Batch processing complete!")

# --- Search Section ---
st.header("Find Candidates")
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Structured Filtering")
    required_skills = st.text_input("Required Skills (comma-separated)", "Python, LangChain")
    min_experience = st.slider("Minimum Years of Experience", 0, 20, 3)

with col2:
    st.subheader("2. Semantic Search (Optional)")
    semantic_query = st.text_input("Describe the ideal candidate profile", "A candidate with experience building AI agents")

if st.button("Search Candidates", type="primary"):
    st.subheader("Search Results")

    with st.spinner("Executing structured filter..."):
        skills_list = [skill.strip() for skill in required_skills.split(',') if skill.strip()]
        
        # --- THIS IS THE CORRECTED LOGIC ---
        filter_query = {
            "experience_years": {"$gte": min_experience}
        }
        if skills_list:
            # Create a case-insensitive regex for each skill
            regex_skills = [re.compile(f"^{re.escape(skill)}$", re.IGNORECASE) for skill in skills_list]
            filter_query["skills"] = {"$all": regex_skills}
        # ------------------------------------

        results = list(candidates_collection.find(filter_query))

    if not results:
        st.warning("No candidates found matching the specified criteria.")
    else:
        # The rest of the logic for semantic ranking and display remains the same...
        st.success(f"Found {len(results)} candidates matching the filters.")
        
        if semantic_query:
            with st.spinner("Reranking results with semantic search..."):
                query_embedding = embedding_model.embed_query(semantic_query)
                
                def get_similarity(candidate_embedding):
                    import numpy as np
                    return np.dot(query_embedding, candidate_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding))

                for candidate in results:
                    candidate['similarity'] = get_similarity(candidate['text_embedding_cv'])
                
                ranked_candidates = sorted(results, key=lambda x: x['similarity'], reverse=True)
                st.info("Results have been semantically reranked.")
        else:
            ranked_candidates = results

        for candidate in ranked_candidates:
            with st.expander(f"**{candidate.get('candidate_name')}** - {candidate.get('experience_years')} years experience"):
                st.write("**Skills:**", ", ".join(candidate.get('skills', [])))
                if 'similarity' in candidate:
                    st.write(f"**Relevance Score:** {candidate['similarity']:.2f}")
                st.write("**Source File:**", candidate.get('source_file'))
                st.text_area("Full CV Text", candidate.get('full_text_content'), height=200, key=str(candidate['_id']))