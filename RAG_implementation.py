import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pickle
import chromadb  # Vector Database
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
from scipy.special import softmax
from sentence_transformers import util
import os


# Function to extract text from PDF
def extract_text(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text())
    return "\n".join(filter(None, text))  # Join pages, remove None values

# Function to extract tables
def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_table()
            if extracted:
                df = pd.DataFrame(extracted[1:], columns=extracted[0])  # Use first row as header
                tables.append(df)
    return tables

# Function to clean extracted text
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)  # Normalize multiple newlines
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Function to chunk text into smaller pieces
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    return splitter.split_text(text)

# Function to convert tables into structured format
def process_tables(tables):
    structured_data = []
    for df in tables:
        structured_data.append(df.to_dict(orient="records"))  # Convert DataFrame to list of dicts
    return structured_data

# Function to embed text chunks
def embed_text(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    return embeddings

# Function to embed and store in ChromaDB
def embed_and_store(chunks, doc_id):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"{doc_id}_{i}"],
            embeddings=[embedding.tolist()],
            metadatas=[{"text": chunk, "source": doc_id}]
        )

# Function to process PDFs and store in ChromaDB
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
      return [{"text":"I don't have enough information on this topic."}]


def ask_local_llm(query, retrieved_chunks):
    # Extract text from retrieved chunks
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

pdf_2023 = "2024-02-06-COGNIZANT-REPORTS-FOURTH-QUARTER-AND-FULL-YEAR-2023-RESULTS.pdf"
pdf_2024 = "2025-02-05-Cognizant-Reports-Fourth-Quarter-and-Full-Year-2024-Results.pdf"

# Process reports
doc_2023 = "2023_Financial_Report"
doc_2024 = "2024_Financial_Report"

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("financial_data")

# Log in to your Hugging Face account
#login(token=os.getenv("HUGGING_FACE_TOKEN"))
login(token='hf_KWGCDOKcYOUrYUUJZurryXKJrwxqqNgnNK')

# Download and cache the model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create the pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

process_and_store_financial_reports("2024-02-06-COGNIZANT-REPORTS-FOURTH-QUARTER-AND-FULL-YEAR-2023-RESULTS.pdf", doc_2023)
process_and_store_financial_reports("2025-02-05-Cognizant-Reports-Fourth-Quarter-and-Full-Year-2024-Results.pdf", doc_2024)

# Example Query
query_DB = "How much did Cognizant return to shareholders through share repurchases and dividends in 2023 and 2024?"

retrieved_chunks_DB = retrieve_similar_chunks(query_DB)
print(retrieved_chunks_DB)

# Print retrieved results
print("Top Retrieved Chunks:")
for chunk in retrieved_chunks_DB:
  print(chunk["text"])

# Generate response using local LLM
response = ask_local_llm(query_DB, retrieved_chunks_DB)
print(response)







