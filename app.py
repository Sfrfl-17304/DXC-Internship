import os
import streamlit as st
from pathlib import Path
import sys
import re
from dotenv import load_dotenv
from streamlit_pdf_viewer import pdf_viewer

# --- Load environment variables ---
load_dotenv()

# --- Streamlit page config ---
st.set_page_config(
    page_title="DXC Chatbot Demo",
    page_icon="logo.png",
    layout="wide"
)

# --- Project imports setup ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dxc_rag_pipeline.rag_chain import create_rag_chain
from dxc_rag_pipeline.database_manager import build_database, DB_PATH

# --- Custom CSS to hide Streamlit header/buttons ---
st.markdown("""
<style>
    header, .stActionButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("logo.png", width=75)
    st.title("DXC RAG Engine")
    st.markdown("---")
    st.info("This demo answers questions based on the `source_documents` directory.")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main header ---
st.image("banner.png")
st.subheader("AI Knowledge Engine Demo")

# --- Initialize RAG system ---
@st.cache_resource
def initialize_system():
    if not Path(DB_PATH).exists():
        with st.spinner("Database not found. Building..."):
            build_database()
    return create_rag_chain()

try:
    rag_chain = initialize_system()
    st.success("RAG Engine is loaded and ready.")
except Exception as e:
    st.error(f"Failed to load the RAG engine. Ensure Ollama is running. Error: {e}")
    st.stop()

# --- Initialize session state for chat messages ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am DXC bot. How can I help you today?"}]

# --- Display all past chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input handling ---
if prompt := st.chat_input("Ask DXC bot..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("DXC Bot is thinking..."):
            # Prepare formatted chat history for model input
            chat_history = ""
            for msg in st.session_state.messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                chat_history += f"{role}: {content}\n"

            # Invoke RAG chain
            result = rag_chain.invoke({
                "question": prompt,
                "chat_history": chat_history
            })

            response_text = result.get("answer", "No answer found.")
            sources = result.get("context", [])

            # --- Display AI response ---
            st.markdown(response_text)

            # --- Extract PDF filename from response using regex (assuming AI outputs Filename: xyz.pdf) ---
            pdf_files = re.findall(r'Filename:\s*(\S+\.pdf)', response_text, re.IGNORECASE)
            current_dir = os.getcwd()  # or set to your known absolute path to 'cv' folder

            if pdf_files:
                for pdf_file in pdf_files:
                    selected_pdf_path = os.path.join(current_dir, "cv", pdf_file)
                    if os.path.exists(selected_pdf_path):
                        st.markdown(f"### 📄 Aperçu du CV sélectionné : `{pdf_file}`")
                        pdf_viewer(selected_pdf_path, width=700, height=1000)
                    else:
                        st.warning(f"Impossible d'afficher le CV sélectionné : {pdf_file}")

            # --- Display source documents if any ---
            if sources:
                with st.expander("Show Sources"):
                    for i, doc in enumerate(sources):
                        st.write(f"**Source {i+1} (ID: {doc.metadata.get('source', 'N/A')})**")
                        st.code(doc.page_content, language=None)

    # Append assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})
