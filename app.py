import streamlit as st
import os
import numpy as np
from PyPDF2 import PdfReader

try:
    import chromadb
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.memory import ConversationBufferMemory
    from sentence_transformers import SentenceTransformer, util

    chromadb_installed = True
except ImportError as e:
    st.error(f"ChromaDB or its dependencies are not installed: {e}")
    chromadb_installed = False

# ... (rest of your functions: load_pdf, chunk_text, store_embeddings_in_chromadb, etc.) ...

# Initialize global variables (only if ChromaDB is installed)
if chromadb_installed:
    pdf_file_path = "Ritesh Chougule AI chatbot.pdf"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="YOUR_GROQ_API_KEY") #Replace with your groq api key
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db_4")
        collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        chromadb_installed = False # disable all chromadb functionality.

# ... (rest of your functions: get_recent_chat_history, retrieve_context, query_llama3, main) ...

def main():
    st.title("Ritesh Chougule AI Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about Ritesh Chougule"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if chromadb_installed: # only run chromadb code if it is installed.
            pdf_text = load_pdf(pdf_file_path)
            if pdf_text.strip():
                chunks = chunk_text(pdf_text)
                store_embeddings_in_chromadb(chunks, collection, embedding_model)

            response = query_llama3(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            st.write("ChromaDB is not available. Please deploy with Docker to use the full functionality.")

if __name__ == "__main__":
    main()
