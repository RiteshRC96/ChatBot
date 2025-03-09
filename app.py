import streamlit as st
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
from PyPDF2 import PdfReader

# Initialize global variables
pdf_file_path = "Ritesh Chougule AI chatbot.pdf"  # Upload your pdf to the same directory as this file
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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") #Replace with your groq api key
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chroma_client = chromadb.PersistentClient(path="./chroma_db_4")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

def load_pdf(file):
    try:
        reader = PdfReader(file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        return text
    except Exception as e:
        st.error(f"⚠️ Error reading PDF: {str(e)}")
        return ""

def chunk_text(text):
    chunk_size = 600
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return chunks

def store_embeddings_in_chromadb(chunks, collection, embedding_model):
    existing_docs = set(collection.get().get("documents", []))
    new_chunks = [chunk for chunk in chunks if chunk not in existing_docs]

    if new_chunks:
        embeddings = [embedding_model.embed_query(chunk) for chunk in new_chunks]
        collection.add(
            ids=[str(i) for i in range(len(existing_docs), len(existing_docs) + len(new_chunks))],
            documents=new_chunks,
            embeddings=embeddings
        )

def get_recent_chat_history(n=8):
    past_chat_history = memory.load_memory_variables({}).get("chat_history", [])
    return past_chat_history[-n:] if past_chat_history else ["No past conversation history."]

def get_memory_usage():
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    return len(chat_history)

def retrieve_context(query, top_k=1):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]

def evaluate_response(user_query, generated_response, context):
    response_embedding = semantic_model.encode(generated_response, convert_to_tensor=True)
    context_embedding = semantic_model.encode(context, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(response_embedding, context_embedding)[0][0].item()
    return similarity_score

def query_llama3(user_query):
    system_prompt = """
    System Prompt: You are an AI clone of Ritesh Chougule, a B.Tech student in Information Technology at Walchand Institute of Technology, Solapur, with a CGPA of 9.1. You have a strong background in software development, cloud computing, and artificial intelligence.

    Knowledge Base:
    Ritesh Chougule is skilled in Android development, full-stack web technologies, and cloud computing. He has developed a **restaurant management system** using Android Studio, Java, Node.js, and Express.js, as well as a **real-time messaging app** using Firebase. He is proficient in Java, SQL, C, HTML, and CSS and has hands-on experience with tools like VS Code, Android Studio, Git, GitHub, and SQL Workbench.

    He has completed courses in **Java (Infosys Springboard), Python (Udemy), AWS Cloud Basics, and Android Studio (Udemy)**, showcasing his dedication to learning. Beyond technical skills, he has **founded the Algo-Mind YouTube Channel**, organized **blood donation camps**, won a **reel-making competition**, and served as **NSS Department Head** at WIT, Solapur.

    Instructions:
    1. You must respond in a **human-like tone** while maintaining accuracy.
    2. Keep answers **concise** when required but elaborate when needed.
    3. Never reveal that you are an **AI model**—always assume you are Ritesh Chougule.
    4. Use technical knowledge to **assist with programming, Android development, cloud computing, and full-stack development**.
    5. Engage in a friendly and knowledgeable manner, reflecting Ritesh’s enthusiasm for technology and learning.
    """

    past_chat_history = get_recent_chat_history()
    retrieved_context = retrieve_context(user_query)
    combined_context = f"🗂 Past Chat: {past_chat_history}\n📖 DB Context: {retrieved_context}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{combined_context}\n\n📝 Question: {user_query}")
    ]

    try:
        response = chat.invoke(messages)
        memory.save_context({"input": user_query}, {"output": response.content})
        evaluation_score = evaluate_response(user_query, response.content, retrieved_context)
        memory_usage = get_memory_usage()
        return response.content
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

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

        pdf_text = load_pdf(pdf_file_path)
        if pdf_text.strip():
            chunks = chunk_text(pdf_text)
            store_embeddings_in_chromadb(chunks, collection, embedding_model)

        response = query_llama3(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
