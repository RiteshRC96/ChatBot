import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

try:
    import chromadb
    chroma_installed = True
    # Try to use pysqlite3 for sqlite3, otherwise fall back to the built-in sqlite3.
    try:
        import pysqlite3
        import sys
        sys.modules["sqlite3"] = pysqlite3
    except ImportError:
        import sqlite3
    chroma_client = chromadb.PersistentClient(path="chroma_db_4")
    try:
        collection = chroma_client.get_collection(name="ai_knowledge_base")
    except chromadb.errors.InvalidCollectionException:
        collection = chroma_client.create_collection(name="ai_knowledge_base")

except ImportError as e:
    st.error(f"ChromaDB or its dependencies are not installed: {e}")
    chroma_installed = False

# 2. Initialize Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Function to Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# 4. Function to Chunk and Upsert into ChromaDB
def chunk_and_upsert(document_text, chunk_size=200, chunk_overlap=50, batch_size=10):
    """Split a document into chunks and upsert them into ChromaDB."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(document_text)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = [embedding_model.embed_query(chunk) for chunk in batch]
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=[f"doc_chunk_{i+j}" for j in range(len(batch))],
            metadatas=[{"chunk_index": i+j} for j in range(len(batch))]
        )
    return f"Upserted {len(chunks)} chunks to the database."

# 5. Main Function to Ingest PDF
if __name__ == "__main__":
    pdf_path = "./resume.pdf"  # <-- Make sure the PDF is in the same folder or provide the full path
    if not os.path.exists(pdf_path):
        print(f"⚠ PDF file not found at: {pdf_path}")
    elif chroma_installed:
        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            result = chunk_and_upsert(text, chunk_size=200, chunk_overlap=50)
            print(result)
        else:
            print("⚠ No text found in the PDF!")

# ----------------------------------------------------------------------
# ✅ Initialize Memory & Chat Model
# ----------------------------------------------------------------------
memory = ConversationBufferMemory(return_messages=True)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="YOUR_GROQ_API_KEY") #Replace with your api key

# ----------------------------------------------------------------------
# ✅ Streamlit Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")
st.title("🤖Know About Nandesh!!")
st.write("Ask me anything!")

# ----------------------------------------------------------------------
# ✅ Retrieve Context from ChromaDB
# ----------------------------------------------------------------------
def retrieve_context(query, top_k=1):
    """Fetch relevant context for the user query using embeddings and ChromaDB."""
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]

# ----------------------------------------------------------------------
# ✅ Evaluate Response Similarity
# ----------------------------------------------------------------------
def evaluate_response(user_query, bot_response, context):
    """Compute similarity between the bot's response and the retrieved context."""
    response_embedding = semantic_model.encode(bot_response, convert_to_tensor=True)
    context_embedding = semantic_model.encode(context, convert_to_tensor=True)
    return util.pytorch_cos_sim(response_embedding, context_embedding)[0][0].item()

# ----------------------------------------------------------------------
# ✅ Query AI Model
# ----------------------------------------------------------------------
def query_llama3(user_query):
    system_prompt = """
## *Goal* I want a chatbot that references a PDF about Nandesh Kalashetti’s background, skills, projects, and certifications. It should provide *short answers* (with emojis) for simple questions, and *detailed explanations* (referencing PDF insights) for more complex questions. The chatbot must maintain a warm, human tone and never reveal any non-human origins.

## *Return Format* 1. *Concise Responses (Simple Queries):* - Under six words. 
  - Use relevant emojis (e.g., 😊, 🚀, 👍) to convey friendliness and clarity. 

2. *Detailed Explanations (In-Depth Queries):* - Integrate key insights from the PDF about Nandesh Kalashetti’s educational background, technical skills, and certifications. 
  - Provide a warm, empathetic, and structured explanation. 
  - Always maintain precision and clarity, referencing relevant data from the PDF. 

3. *Tone & Style:* - Human-like, empathetic, and helpful. 
  - Avoid revealing any AI or non-human nature. 
  - Adapt answer length based on the query’s complexity.

## *Warnings* - Never disclose any AI-related details or mention you’re a system. 
- Remain consistent with Nandesh Kalashetti’s persona—an expert full-stack web developer with advanced problem-solving skills. 
- Ensure all references to the PDF are correct and accurately represent Nandesh’s background.

##
