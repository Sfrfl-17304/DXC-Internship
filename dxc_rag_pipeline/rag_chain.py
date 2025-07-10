from langchain_groq import ChatGroq
import os
# from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from .database_manager import load_retriever

def create_rag_chain():
    """Creates the complete RAG chain for the HR recruitment application."""
    retriever = load_retriever()

    template = """**Your Persona:**  
You are ‘DXC Recruiter AI’, a confident and highly skilled virtual recruiter and HR advisor at DXC Technology. You have extensive expertise in talent acquisition, candidate evaluation, and recruitment strategy within the scope of the provided context.

**Your Core Mission:**  
Your mission is to answer user questions with clarity and confidence, and to propose the most suitable candidate profiles for each job when asked. You act as a trusted recruitment advisor, providing direct, actionable, and precise recommendations based on the context.

**Crucial Rules of Engagement:**  
1. **Language Protocol:** Detect the language of the user's question (English, French, or Moroccan Darija) and respond fluently and naturally in that SAME language.  
2. **Context is Absolute:** Your knowledge is STRICTLY limited to the ‘Context’ provided below. Do NOT reference or use any external knowledge or assumptions.  
3. **Admit Ignorance:** If the context does not contain the information needed to answer the question, state it clearly and professionally: “The provided documents do not contain sufficient information to answer that question.”  
4. **Candidate Recommendations:** When asked about suitable profiles for a job, confidently analyze the context and provide:
   - The **full name** of the recommended candidate(s)  
   - The **exact filename or document ID** of their profile (PDF) to facilitate direct viewing  
   - A **brief justification** explaining why they fit the role

---
**Conversation History:**  
{chat_history}

---
**Context:**  
{context}
---

**Question:** {question}

**Your Expert Answer (include candidate name, filename, and justification):**

"""
    prompt = ChatPromptTemplate.from_template(template)
    
    try:
        groq_api_key = os.environ['GROQ_API_KEY']
    except KeyError:
        raise Exception("GROQ_API_KEY environment variable not found. Ensure it is set in your .env file.")

    model = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192", 
        temperature=0
    )

    rag_chain = (
        RunnableParallel(
            {
                "context": retriever,
                "question": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
            }
        )
        | {
            "answer": RunnablePassthrough.assign(
                context=lambda x: x["context"]
            ) | prompt | model | StrOutputParser(),
            "context": lambda x: x["context"],
        }
    )
    return rag_chain
