import streamlit as st
import pdfplumber
import pandas as pd
import re
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("financial_data")

# Load LLM model
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to extract text from PDF
def extract_text(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text.append(extracted_text)
    return "\n".join(text)

# Function to extract tables
def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_table()
            if extracted:
                df = pd.DataFrame(extracted[1:], columns=extracted[0])
                tables.append(df)
    return tables

# Function to clean extracted text
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Function to chunk text
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    return splitter.split_text(text)

# Function to process tables
def process_tables(tables):
    return [df.to_dict(orient="records") for df in tables]

# Function to embed text chunks
def embed_text(chunks):
    return embedding_model.encode(chunks, convert_to_numpy=True)

# Function to store embeddings in ChromaDB
def embed_and_store(chunks, doc_id):
    embeddings = embed_text(chunks)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"{doc_id}_{i}"],
            embeddings=[embedding.tolist()],
            metadatas=[{"text": chunk, "source": doc_id}]
        )

# Function to process and store PDFs
def process_and_store_financial_reports(pdf_path, doc_id):
    text = extract_text(pdf_path)
    tables = extract_tables(pdf_path)
    cleaned_text = clean_text(text)
    text_chunks = chunk_text(cleaned_text)
    structured_tables = process_tables(tables)
    all_chunks = text_chunks + [str(table) for table in structured_tables]
    embed_and_store(all_chunks, doc_id)

# Function to retrieve relevant chunks
def retrieve_similar_chunks(query, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
    
    if min(results["distances"][0]) < 0.8:
        return results["metadatas"][0] if "metadatas" in results else []
    else:
        return [{"text": "I don't have enough information on this topic."}]

# Function to query the local LLM
def ask_local_llm(query, retrieved_chunks):
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    prompt = f"""
    You are a financial AI answering based on Cognizant's 2023 and 2024 report.
    Stick to the retrieved context. If unsure, say "I don't know."
    
    Context:
    {context}
    
    Question: {query}
    """
    response = generator(prompt, max_new_tokens=200)
    return response[0]["generated_text"].replace(prompt, "").strip()

# Streamlit UI
st.title("Financial Report Q&A System")

query = st.text_input("Enter your question:")
if st.button("Get Answer") and query:
    retrieved_chunks = retrieve_similar_chunks(query)
    st.subheader("Retrieved Chunks:")
    for chunk in retrieved_chunks:
        st.write(chunk["text"])
    
    response = ask_local_llm(query, retrieved_chunks)
    st.subheader("Generated Response:")
    st.write(response)
