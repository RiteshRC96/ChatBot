import os
import streamlit as st
import chromadb
from chromadb.config import Settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader

PDF_FILE_PATH = "resume.pdf"  # or your PDF name
GROQ_API_KEY = "YOUR_GROQ_API_KEY"

# ✅ In-memory Chroma
chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=None
    )
)
try:
    collection = chroma_client.get_collection("ai_knowledge_base")
except chromadb.errors.InvalidCollectionException:
    collection = chroma_client.create_collection("ai_knowledge_base")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
memory = ConversationBufferMemory(return_messages=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
    except Exception as e:
        st.error(f"⚠️ Error reading PDF: {str(e)}")
    return text

def chunk_and_upsert(document_text, chunk_size=200, chunk_overlap=50, batch_size=10):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(document_text)
    existing_docs = set(collection.get().get("documents", []))
    new_chunks = [c for c in chunks if c not in existing_docs]

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        embeddings = [embedding_model.embed_query(chunk) for chunk in batch]
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=[f"doc_chunk_{i+j}" for j in range(len(batch))],
            metadatas=[{"chunk_index": i+j} for j in range(len(batch))]
        )
    return len(new_chunks)

def retrieve_context(query, top_k=1):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]

def evaluate_response(user_query, bot_response, context):
    resp_embed = semantic_model.encode(bot_response, convert_to_tensor=True)
    ctx_embed  = semantic_model.encode(context, convert_to_tensor=True)
    return util.pytorch_cos_sim(resp_embed, ctx_embed)[0][0].item()

def query_llama3(user_query):
    system_prompt = """
System Prompt: You are a helpful AI assistant with knowledge from a PDF. 
Respond as a human, never reveal AI nature, and use short or detailed answers as needed.
"""
    retrieved_context = retrieve_context(user_query)
    combined_context = f"Context from PDF: {retrieved_context}"

    messages = memory.chat_memory.messages + [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{combined_context}\n\nUser Query: {user_query}")
    ]

    try:
        response = chat.invoke(messages)
        return response.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("🤖 AI Chatbot")

if "db_initialized" not in st.session_state:
    st.session_state["db_initialized"] = False

if not st.session_state["db_initialized"]:
    if os.path.exists(PDF_FILE_PATH):
        pdf_text = extract_text_from_pdf(PDF_FILE_PATH)
        if pdf_text.strip():
            num_chunks = chunk_and_upsert(pdf_text)
            st.info(f"✅ {num_chunks} chunks added!")
        else:
            st.warning("⚠ No text found in the PDF!")
    else:
        st.warning(f"⚠ PDF not found: {PDF_FILE_PATH}")
    st.session_state["db_initialized"] = True

for msg in memory.chat_memory.messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("Ask me anything...")

if user_input:
    memory.chat_memory.add_user_message(user_input)
    st.chat_message("user").write(user_input)

    ai_response = query_llama3(user_input)
    memory.chat_memory.add_ai_message(ai_response)
    st.chat_message("assistant").write(ai_response)
