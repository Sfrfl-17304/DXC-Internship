import streamlit as st
from pymongo import MongoClient
import os
import re
import logging

# Get the logger for the pymongo driver
pymongo_logger = logging.getLogger('pymongo.command')
# Set the logging level to DEBUG to see all commands
pymongo_logger.setLevel(logging.DEBUG)

# Add a handler to print the logs to your terminal
handler = logging.StreamHandler()
formatter = logging.Formatter('MongoDB Query: %(message)s')
handler.setFormatter(formatter)
pymongo_logger.addHandler(handler)

# Also log at the root level for better visibility
logging.basicConfig(level=logging.DEBUG)
print("MongoDB query logging enabled - you'll see database operations in the terminal")
# ------------------------------------
# Import with error handling for cv_processor
try:
    from hybrid_search_mongodb.hr_engine.cv_processor import process_cv_directory
    CV_PROCESSOR_AVAILABLE = True
except ImportError as e:
    st.warning(f"CV Processor module not available: {e}")
    CV_PROCESSOR_AVAILABLE = False
    def process_cv_directory(directory):
        st.error("CV processing functionality is currently unavailable due to import issues.")
        return False

from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="HR Hybrid Search Engine",
    page_icon="/images/logo.png",
    layout="wide"
)

# --- Database Connection ---
MONGO_USER = os.getenv("MONGO_USERNAME")
MONGO_PASS = os.getenv("MONGO_PASSWORD")
MONGO_HOST = "localhost"
MONGO_PORT = "27017"

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

# --- Models ---
embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

# Initialize Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables!")
    st.stop()

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",  # Llama3 70B model
    temperature=0.1,  # Low temperature for consistent extraction
    max_tokens=1000
)

def extract_skills_from_prompt(prompt):
    """Extract skills and logical operators from natural language prompt using Groq LLM with regex fallback"""
    
    # First, try regex-based logic detection as it's more reliable
    def detect_logic_with_regex(text):
        text_lower = text.lower()
        
        # Strong OR indicators
        or_patterns = [
            r'\bor\b',
            r'\beither\b.*\bor\b',
            r'\balternatively\b',
            r'\bany of\b',
            r'\bone of\b'
        ]
        
        # Strong AND indicators
        and_patterns = [
            r'\band\b',
            r'\bwith\b',
            r'\bplus\b',
            r'\balso\b',
            r'\bincluding\b',
            r'\balong with\b'
        ]
        
        or_matches = sum(1 for pattern in or_patterns if re.search(pattern, text_lower))
        and_matches = sum(1 for pattern in and_patterns if re.search(pattern, text_lower))
        
        # If we find OR patterns and no strong AND patterns, use OR
        if or_matches > 0 and and_matches == 0:
            return "OR"
        # If we find more OR patterns than AND patterns, use OR
        elif or_matches > and_matches:
            return "OR"
        else:
            return "AND"
    
    # Detect logic using regex first
    regex_logic = detect_logic_with_regex(prompt)
    
    skill_extraction_prompt = f"""
    Extract technical skills from this text and determine the logical relationships.
    
    Text: "{prompt}"
    
    Return a valid JSON object with this format:
    {{
        "skills": ["skill1", "skill2"],
        "logic": "OR"
    }}
    
    IMPORTANT LOGIC RULES:
    - Look for "or", "either", "alternatively", "any of" → use "logic": "OR"
    - Look for "and", "with", "plus", "also", "including" → use "logic": "AND"
    - If you see "React or Vue" → use "logic": "OR"
    - If you see "Python and Django" → use "logic": "AND"
    - Pay close attention to OR indicators like "or", "either"
    
    Based on the text, the logic should be: {regex_logic}
    
    Extract only technical skills (programming languages, frameworks, tools, technologies).
    
    Examples:
    - "Python developer" → {{"skills": ["Python"], "logic": "AND"}}
    - "React or Vue developer" → {{"skills": ["React", "Vue"], "logic": "OR"}}
    - "Java with Spring Boot" → {{"skills": ["Java", "Spring Boot"], "logic": "AND"}}
    - "Docker or Kubernetes experience" → {{"skills": ["Docker", "Kubernetes"], "logic": "OR"}}
    
    JSON response:"""
    
    try:
        # Get skills from LLM with logic override from regex
        response = llm.invoke(skill_extraction_prompt)
        
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse JSON response with multiple fallback strategies
        skills = []
        ai_logic = 'AND'  # Default
        
        try:
            # Strategy 1: Try to extract complete JSON
            json_patterns = [
                r'\{[^{}]*"skills"[^{}]*"logic"[^{}]*\}',
                r'\{[^{}]*"logic"[^{}]*"skills"[^{}]*\}',
                r'\{(?:[^{}]|{[^{}]*})*\}'
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    json_str = json_str.replace("'", '"')
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        if 'skills' in result:
                            skills = [s.strip() for s in result['skills'] if s.strip()]
                        if 'logic' in result:
                            ai_logic = result['logic']
                        break
                        
        except json.JSONDecodeError:
            # Strategy 2: Extract skills array only
            try:
                skills_match = re.search(r'"skills":\s*\[(.*?)\]', response_text, re.DOTALL)
                if skills_match:
                    skills_str = skills_match.group(1)
                    skills = [s.strip().strip('"\'') for s in skills_str.split(',') if s.strip().strip('"\'')]
            except:
                # Strategy 3: Fallback to regex extraction
                skill_words = re.findall(r'\b[A-Z][A-Za-z0-9+#.]*(?:\s+[A-Z][A-Za-z0-9+#.]*)*\b', response_text)
                skills = [s for s in skill_words if len(s) > 1 and s not in ['JSON', 'Text', 'Examples', 'Extract', 'Return']]
        
        # Override AI logic with regex logic if regex detected OR
        final_logic = regex_logic if regex_logic == 'OR' else ai_logic
        
        return {
            'skills': skills,
            'logic': final_logic,
            'is_complex': False
        }
        
    except Exception as e:
        st.error(f"Error in skill extraction: {e}")
        return {'skills': [], 'logic': 'AND', 'is_complex': False}
    
    try:
        # Use invoke for ChatGroq (newer LangChain versions)
        response = llm.invoke(skill_extraction_prompt)
        
        # Extract content from response (ChatGroq returns AIMessage)
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Clean the response and try to parse as JSON
        cleaned_response = response_text.strip()
        
        # Try multiple JSON extraction strategies
        json_result = None
        
        # Strategy 1: Find JSON object with regex (most permissive)
        json_patterns = [
            r'\{[^{}]*"skills"[^{}]*\}',  # Simple object
            r'\{(?:[^{}]|{[^{}]*})*\}',   # Nested object
            r'\{.*?\}',                   # Any object
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, cleaned_response, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group()
                    # Fix common JSON issues
                    json_str = json_str.replace("'", '"')  # Replace single quotes
                    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                    json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                    
                    json_result = json.loads(json_str)
                    break
                except json.JSONDecodeError:
                    continue
        
        # Strategy 2: Try parsing the entire response
        if not json_result:
            try:
                # Clean up common issues
                cleaned_response = cleaned_response.replace("'", '"')
                cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
                cleaned_response = re.sub(r',\s*]', ']', cleaned_response)
                json_result = json.loads(cleaned_response)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Extract from code blocks
        if not json_result:
            code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned_response, re.DOTALL)
            if code_match:
                try:
                    json_str = code_match.group(1)
                    json_str = json_str.replace("'", '"')
                    json_result = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # Process the result if we found valid JSON
        if json_result and isinstance(json_result, dict) and 'skills' in json_result:
            skills = [skill.strip() for skill in json_result['skills'] if skill.strip()]
            
            # Handle logic field, but prioritize regex detection
            llm_logic = json_result.get('logic', 'AND') if json_result.get('logic') in ['AND', 'OR'] else 'AND'
            
            # Use regex logic if it detected OR, otherwise trust LLM
            final_logic = regex_logic if regex_logic == 'OR' else llm_logic
            
            return {
                'skills': skills, 
                'logic': final_logic,
                'is_complex': False
            }
        
        # Strategy 4: Fallback to regex skill extraction
        if not json_result:
            st.warning("Could not parse JSON response, attempting regex extraction...")
            
            # Extract skills using common patterns
            skill_patterns = [
                r'"([^"]+)"',  # Quoted strings
                r"'([^']+)'",  # Single quoted strings
                r'\b([A-Za-z][A-Za-z0-9+#.]*(?:\s+[A-Za-z][A-Za-z0-9+#.]*)*)\b'  # Tech terms
            ]
            
            extracted_skills = []
            for pattern in skill_patterns:
                matches = re.findall(pattern, response_text)
                for match in matches:
                    if len(match) > 1 and match.lower() not in ['skills', 'logic', 'and', 'or']:
                        extracted_skills.append(match)
            
            # Remove duplicates and common words
            tech_skills = []
            common_words = {'the', 'and', 'or', 'with', 'for', 'in', 'on', 'at', 'to', 'a', 'an'}
            for skill in extracted_skills:
                if skill.lower() not in common_words and len(skill) > 2:
                    tech_skills.append(skill)
            
            if tech_skills:
                return {'skills': list(set(tech_skills)), 'logic': regex_logic, 'is_complex': False}
        
        # Final fallback
        return {'skills': [], 'logic': regex_logic, 'is_complex': False}
    
    except Exception as e:
        st.error(f"Unexpected error in skill extraction: {e}")
        st.write("**Raw LLM Response for debugging:**")
        st.code(response_text if 'response_text' in locals() else "No response captured")
        return {'skills': [], 'logic': 'AND', 'is_complex': False}

def build_complex_skills_query(skills_data):
    """Build MongoDB query from extracted skills data"""
    if not skills_data or not skills_data.get('skills'):
        return {}
    
    skills = skills_data['skills']
    logic = skills_data.get('logic', 'AND')
    
    # Simple logic handling
    if logic == 'OR':
        return {"$or": [{"skills": {"$in": [skill.lower()]}} for skill in skills]}
    else:  # Default to AND
        return {"$and": [{"skills": {"$in": [skill.lower()]}} for skill in skills]}

def extract_experience_from_prompt(prompt):
    """Extract minimum experience requirement from prompt using Groq LLM"""
    experience_prompt = f"""
    From the following text, extract the minimum years of experience required.
    Look for:
    1. Explicit numbers: "3+ years", "minimum 5 years", "at least 2 years experience"
    2. Seniority levels:
       - "junior" or "entry-level" → 1 year minimum
       - "mid-level" or "intermediate" → 3 years minimum  
       - "senior" → 5 years minimum
       - "lead" or "principal" → 7 years minimum
       - "expert" or "architect" → 10 years minimum
    
    "{prompt}"
    
    If no specific experience requirement is mentioned, return 0.
    Return only a single number (the minimum years).
    
    Examples:
    - "senior developer" → 5
    - "junior Python programmer" → 1
    - "mid-level engineer with 4+ years" → 4 (take the higher value)
    - "lead architect" → 7
    - "entry-level position" → 1
    
    Minimum years of experience:"""
    
    try:
        # Use invoke for ChatGroq
        response = llm.invoke(experience_prompt)
        
        # Extract content from response
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Extract number from response
        numbers = re.findall(r'\d+', response_text.strip())
        if numbers:
            llm_years = int(numbers[0])
        else:
            llm_years = 0
        
        # Fallback: Use regex patterns to detect seniority levels
        prompt_lower = prompt.lower()
        regex_years = 0
        
        # Define seniority mappings
        seniority_mapping = {
            r'\b(junior|entry-level|entry level|fresher|graduate)\b': 1,
            r'\b(mid-level|mid level|intermediate|middle)\b': 3,
            r'\b(senior|sr\.?)\b': 5,
            r'\b(lead|principal|team lead|tech lead)\b': 7,
            r'\b(expert|architect|chief|director)\b': 10
        }
        
        # Check for seniority patterns
        for pattern, years in seniority_mapping.items():
            if re.search(pattern, prompt_lower):
                regex_years = max(regex_years, years)
        
        # Also check for explicit number patterns
        explicit_patterns = [
            r'(\d+)\+?\s*years?',
            r'minimum\s+(\d+)\s*years?',
            r'at least\s+(\d+)\s*years?',
            r'(\d+)\s*to\s*\d+\s*years?',
            r'(\d+)\s*-\s*\d+\s*years?'
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, prompt_lower)
            if matches:
                regex_years = max(regex_years, int(matches[0]))
        
        # Return the higher value between LLM and regex extraction
        return max(llm_years, regex_years)
        
    except Exception as e:
        st.warning(f"Could not parse experience requirement: {e}")
        return 0

# --- UI ---
st.title("HR Hybrid Search Engine")
st.write("This tool helps HR find the best candidates by combining structured filtering and semantic search.")

# --- Sidebar for Actions ---
st.sidebar.image("logo.png", width=100)
st.sidebar.title("Admin Panel")
st.sidebar.markdown("---")

if CV_PROCESSOR_AVAILABLE:
    cv_directory = st.sidebar.text_input("CVs Folder Path", "./cv_samples")
    if st.sidebar.button("Process All CVs in Folder"):
        if not os.path.isdir(cv_directory):
            st.sidebar.error(f"Directory not found: {cv_directory}")
        else:
            with st.spinner(f"Processing all documents in {cv_directory}..."):
                result = process_cv_directory(cv_directory)
                if result:
                    st.sidebar.success("Batch processing complete!")
                else:
                    st.sidebar.error("Processing failed!")
else:
    st.sidebar.error("CV Processing is currently unavailable")
    st.sidebar.info("Fix the circular import in cv_processor.py to enable this feature")

# --- Search Section ---
st.header("Find Candidates")
st.markdown("---")

# Single prompt input that will be processed for both structured and semantic search
st.subheader("Describe Your Ideal Candidate")
user_prompt = st.text_area(
    "Enter your requirements in natural language:",
    placeholder="Looking for a senior Python developer with 5+ years experience in machine learning, familiar with TensorFlow and AWS cloud services. Should have experience building scalable web applications.",
    height=100
)

def get_all_skills_from_db():
    """Fetch all unique skills from the database"""
    try:
        pipeline = [
            {"$unwind": "$skills"},
            {"$group": {"_id": "$skills"}},
            {"$sort": {"_id": 1}}
        ]
        skills_cursor = candidates_collection.aggregate(pipeline)
        skills = [doc["_id"] for doc in skills_cursor if doc["_id"]]
        return sorted(skills, key=str.lower)
    except Exception as e:
        st.error(f"Error fetching skills from database: {e}")
        return []

# Manual override options (collapsible)
with st.expander("Manual Override Options", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Skills Selection")
        
        # Fetch all available skills
        with st.spinner("Loading available skills..."):
            all_skills = get_all_skills_from_db()
        
        if all_skills:
            st.write(f"**{len(all_skills)} skills available in database**")
            
            # Multi-select for skills
            selected_skills = st.multiselect(
                "Select Skills:",
                options=all_skills,
                placeholder="Choose skills from database",
                help="Select multiple skills that candidates must have"
            )
            
            # Option to require ALL selected skills or ANY selected skills
            skill_match_type = st.radio(
                "Skill Matching:",
                options=["Must have ALL selected skills", "Must have ANY selected skills"],
                index=0,
                help="Choose whether candidates must have all skills or just any of the selected skills"
            )
            
            # Text input as fallback/additional option
            additional_skills = st.text_input(
                "Additional Skills (comma-separated):",
                placeholder="Add skills not in database",
                help="Add custom skills that might not be in the database yet"
            )
            
        else:
            st.warning("No skills found in database")
            manual_skills = st.text_input(
                "Manual Skills (comma-separated)", 
                placeholder="Enter skills manually",
                help="Enter skills separated by commas"
            )
    
    with col2:
        st.subheader("Experience & Filters")
        manual_experience = st.number_input(
            "Minimum Experience (years)", 
            min_value=0, 
            max_value=20, 
            value=0,
            help="Set to 0 to use AI-extracted experience requirement"
        )
        
        # Additional filters
        st.write("**Additional Filters:**")
        max_experience = st.number_input(
            "Maximum Experience (years)", 
            min_value=0, 
            max_value=30, 
            value=0,
            help="Set to 0 for no maximum limit"
        )
        
        # Show skill statistics
        if all_skills:
            with st.expander("Skill Statistics"):
                # Get skill frequency
                try:
                    pipeline = [
                        {"$unwind": "$skills"},
                        {"$group": {"_id": "$skills", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}},
                        {"$limit": 10}
                    ]
                    top_skills = list(candidates_collection.aggregate(pipeline))
                    
                    st.write("**Top 10 Most Common Skills:**")
                    for skill_doc in top_skills:
                        st.write(f"• {skill_doc['_id']}: {skill_doc['count']} candidates")
                        
                except Exception as e:
                    st.error(f"Error getting skill statistics: {e}")

if st.button("Search Candidates", type="primary"):
    if not user_prompt.strip():
        st.error("Please enter your candidate requirements.")
    else:
        st.subheader("Search Results")
        
        # Extract structured criteria from prompt
        with st.spinner("Analyzing your requirements..."):
            # Determine which skills to use
            final_skills_list = []
            
            # Check if manual skills are selected from database
            if 'selected_skills' in locals() and selected_skills:
                final_skills_list.extend(selected_skills)
                st.info(f"Using selected database skills: {', '.join(selected_skills)}")
            
            # Add any additional manual skills
            if 'additional_skills' in locals() and additional_skills.strip():
                additional_skills_list = [skill.strip() for skill in additional_skills.split(',') if skill.strip()]
                final_skills_list.extend(additional_skills_list)
                st.info(f"Added custom skills: {', '.join(additional_skills_list)}")
            
            # Fall back to manual text input if no database skills available
            if 'manual_skills' in locals() and manual_skills and manual_skills.strip() and not final_skills_list:
                final_skills_list = [skill.strip() for skill in manual_skills.split(',') if skill.strip()]
                st.info(f"Using manual text skills: {', '.join(final_skills_list)}")
            
            # If no manual skills selected, extract from prompt
            if not final_skills_list:
                skills_data = extract_skills_from_prompt(user_prompt)
                final_skills_list = skills_data['skills']
                extracted_logic = skills_data.get('logic', 'AND')
                
                if final_skills_list:
                    # Show logic detection details
                    logic_source = "AI + Regex" if extracted_logic == 'OR' else "AI"
                    st.info(f"AI-extracted skills: {', '.join(final_skills_list)} (Logic: {extracted_logic} - {logic_source})")
                    
                    # Show debug info for OR logic
                    if extracted_logic == 'OR':
                        or_words_found = []
                        text_lower = user_prompt.lower()
                        if 'or' in text_lower:
                            or_words_found.append("'or'")
                        if 'either' in text_lower:
                            or_words_found.append("'either'")
                        if 'alternatively' in text_lower:
                            or_words_found.append("'alternatively'")
                        if or_words_found:
                            st.success(f"OR logic detected from: {', '.join(or_words_found)}")
                else:
                    st.info("No specific technical skills detected in prompt")
                    extracted_logic = 'AND'
            else:
                # Use manual skill match type if skills were manually selected
                extracted_logic = 'AND' if 'skill_match_type' not in locals() or skill_match_type == "Must have ALL selected skills" else 'OR'
                skills_data = {'logic': extracted_logic, 'skills': final_skills_list}
            
            skills_list = final_skills_list
            
            if manual_experience > 0:
                min_experience = manual_experience
                st.info(f"Using manual experience requirement: {min_experience} years")
            else:
                min_experience = extract_experience_from_prompt(user_prompt)
                if min_experience > 0:
                    st.info(f"Extracted experience requirement: {min_experience} years")
                else:
                    st.info("No specific experience requirement detected")
            
            # Show AI extraction details in expandable section
            with st.expander("View AI Extraction Details", expanded=False):
                st.write("**Original Prompt:**")
                st.code(user_prompt, language="text")
                
                st.write("**AI Analysis Results:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Skills Extracted:**")
                    if skills_list:
                        for skill in skills_list:
                            st.write(f"• {skill}")
                        st.write(f"**Logic Detected:** {extracted_logic if 'extracted_logic' in locals() else 'AND'}")
                    else:
                        st.write("• No technical skills detected")
                
                with col2:
                    st.write("**Experience Extracted:**")
                    if min_experience > 0:
                        st.write(f"• Minimum: {min_experience} years")
                        
                        # Show seniority level interpretation
                        prompt_lower = user_prompt.lower()
                        detected_seniority = []
                        if re.search(r'\b(junior|entry-level|entry level|fresher|graduate)\b', prompt_lower):
                            detected_seniority.append("Junior (1+ years)")
                        if re.search(r'\b(mid-level|mid level|intermediate|middle)\b', prompt_lower):
                            detected_seniority.append("Mid-level (3+ years)")
                        if re.search(r'\b(senior|sr\.?)\b', prompt_lower):
                            detected_seniority.append("Senior (5+ years)")
                        if re.search(r'\b(lead|principal|team lead|tech lead)\b', prompt_lower):
                            detected_seniority.append("Lead (7+ years)")
                        if re.search(r'\b(expert|architect|chief|director)\b', prompt_lower):
                            detected_seniority.append("Expert (10+ years)")
                        
                        if detected_seniority:
                            st.write("**Seniority Detected:**")
                            for seniority in detected_seniority:
                                st.write(f"• {seniority}")
                    else:
                        st.write("• No specific experience requirement")
                
                st.write("**LLM Models Used:**")
                st.write("• **Skill Extraction:** Llama3-70B-8192 (Groq)")
                st.write("• **Experience Extraction:** Llama3-70B-8192 (Groq) + Regex patterns")
                st.write("• **Embeddings:** mxbai-embed-large (Ollama)")
                
                st.write("**Seniority Level Mappings:**")
                st.write("• **Junior/Entry-level:** 1+ years")
                st.write("• **Mid-level/Intermediate:** 3+ years")
                st.write("• **Senior:** 5+ years")
                st.write("• **Lead/Principal:** 7+ years")
                st.write("• **Expert/Architect:** 10+ years")

        # Build and execute structured query
        with st.spinner("Searching database with structured criteria..."):
            filter_query = {}
            
            # Add experience filters
            if manual_experience > 0:
                filter_query["experience_years"] = {"$gte": manual_experience}
                st.info(f"Using manual minimum experience: {manual_experience} years")
            else:
                min_experience = extract_experience_from_prompt(user_prompt)
                if min_experience > 0:
                    filter_query["experience_years"] = {"$gte": min_experience}
                    st.info(f"AI-extracted experience requirement: {min_experience} years")
                else:
                    st.info("No specific experience requirement detected")
            
            # Add maximum experience filter if specified
            if 'max_experience' in locals() and max_experience > 0:
                if "experience_years" in filter_query:
                    filter_query["experience_years"]["$lte"] = max_experience
                else:
                    filter_query["experience_years"] = {"$lte": max_experience}
                st.info(f"Maximum experience limit: {max_experience} years")
            
            # Add skills filter based on matching type or extracted logic
            if skills_list:
                regex_skills = [re.compile(f"^{re.escape(skill)}$", re.IGNORECASE) for skill in skills_list]
                
                # Use extracted logic or manual preference
                use_or_logic = False
                if 'extracted_logic' in locals() and extracted_logic == 'OR':
                    use_or_logic = True
                    st.info(f"AI detected OR logic - searching for candidates with ANY of these skills: {', '.join(skills_list)}")
                elif 'skill_match_type' in locals() and skill_match_type == "Must have ANY selected skills":
                    use_or_logic = True
                    st.info(f"Manual selection - searching for candidates with ANY of these skills: {', '.join(skills_list)}")
                else:
                    st.info(f"Using AND logic - searching for candidates with ALL of these skills: {', '.join(skills_list)}")
                
                if use_or_logic:
                    filter_query["skills"] = {"$in": regex_skills}
                else:
                    filter_query["skills"] = {"$all": regex_skills}
            
            # Display the generated MongoDB query in an expandable section
            with st.expander("View Generated MongoDB Query", expanded=False):
                if filter_query:
                    st.code(f"db.candidates.find({json.dumps(filter_query, indent=2, default=str)})", language="javascript")
                    st.json(filter_query)
                else:
                    st.code("db.candidates.find({})", language="javascript")
                    st.info("No filters applied - retrieving all candidates for semantic ranking")
            
            # If no structured criteria, get all candidates for semantic ranking
            if not filter_query:
                st.info("No structured criteria found - will use semantic search only")
                results = list(candidates_collection.find())
            else:
                results = list(candidates_collection.find(filter_query))

        if not results:
            st.warning("No candidates found matching the specified criteria.")
        else:
            st.success(f"Found {len(results)} candidates matching the filters.")
            
            # Always perform semantic ranking with the original prompt
            with st.spinner("Ranking candidates by relevance..."):
                query_embedding = embedding_model.embed_query(user_prompt)
                
                # Show semantic search details in expandable section
                with st.expander("View Semantic Search Details", expanded=False):
                    st.write("**Original Prompt for Semantic Ranking:**")
                    st.code(user_prompt, language="text")
                    st.write("**Semantic Search Process:**")
                    st.write("1. Convert prompt to vector embedding using mxbai-embed-large model")
                    st.write("2. Calculate cosine similarity with each candidate's CV embedding")
                    st.write("3. Rank candidates by similarity score (0.0 to 1.0)")
                    st.write(f"**Query Embedding Dimensions:** {len(query_embedding)}")
                
                def get_similarity(candidate_embedding):
                    import numpy as np
                    return np.dot(query_embedding, candidate_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
                    )

                for candidate in results:
                    candidate['similarity'] = get_similarity(candidate['text_embedding_cv'])
                
                ranked_candidates = sorted(results, key=lambda x: x['similarity'], reverse=True)
                st.success("Results have been ranked by semantic relevance.")

            # Display results
            for i, candidate in enumerate(ranked_candidates, 1):
                with st.expander(
                    f"#{i} **{candidate.get('candidate_name')}** - {candidate.get('experience_years')} years experience (Score: {candidate['similarity']:.3f})"
                ):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Skills:**", ", ".join(candidate.get('skills', [])))
                        st.write(f"**Relevance Score:** {candidate['similarity']:.3f}")
                        st.write("**Source File:**", candidate.get('source_file'))
                    
                    with col2:
                        # Highlight matching skills and show logic used
                        candidate_skills = set([skill.lower() for skill in candidate.get('skills', [])])
                        extracted_skills_set = set([skill.lower() for skill in skills_list])
                        matching_skills = candidate_skills.intersection(extracted_skills_set)
                        
                        if matching_skills:
                            st.write("**Matching Skills:**")
                            for skill in matching_skills:
                                st.write(f"• {skill.title()}")
                            
                            # Show logic information
                            if 'extracted_logic' in locals() and extracted_logic == 'OR':
                                st.write("**Logic:** OR (any skill match)")
                            elif 'skill_match_type' in locals() and skill_match_type == "Must have ANY selected skills":
                                st.write("**Logic:** OR (any skill match)")
                            else:
                                st.write("**Logic:** AND (all skills required)")
                                
                            # Show match percentage for AND logic
                            if len(skills_list) > 1 and len(matching_skills) < len(skills_list):
                                match_percentage = (len(matching_skills) / len(skills_list)) * 100
                                st.write(f"**Skill Match:** {len(matching_skills)}/{len(skills_list)} ({match_percentage:.1f}%)")
                    
                    st.text_area(
                        "Full CV Text", 
                        candidate.get('full_text_content', ''), 
                        height=150, 
                        key=f"cv_text_{candidate['_id']}"
                    )

# --- Help Section ---
with st.sidebar:
    st.markdown("---")
    st.subheader("Tips")
    st.markdown("""
    **How to use:**
    - Describe your ideal candidate in natural language
    - Use specific technical skills and experience levels
    - The system will automatically extract requirements
    - Results are ranked by relevance
    
    **Example:**
    "Senior Python developer with 5+ years experience"
    """)
    
    st.markdown("**The system will:**")
    st.markdown("- Extract technical skills automatically")
    st.markdown("- Detect experience requirements")
    st.markdown("- Filter candidates by criteria")
    st.markdown("- Rank results by relevance")