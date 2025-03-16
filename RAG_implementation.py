import pdfplumber
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
from sentence_transformers import util
import math

embeding_model = None
chroma_client = None
collection = None
model = None
tokenizer = None
generator = None


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


def process_and_store():
    # Process reports
    doc_2023 = "2023_Financial_Report"
    doc_2024 = "2024_Financial_Report"
    process_and_store_financial_reports("2024-02-06-COGNIZANT-REPORTS-FOURTH-QUARTER-AND-FULL-YEAR-2023-RESULTS.pdf",
                                        doc_2023)
    process_and_store_financial_reports("2025-02-05-Cognizant-Reports-Fourth-Quarter-and-Full-Year-2024-Results.pdf",
                                        doc_2024)


# Function to retrieve relevant chunks
def retrieve_similar_chunks(query, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
    print(results)
    if min(results["distances"][0]) < 0.8:
        return results["metadatas"][0] if "metadatas" in results else []
    else:
        return [{"text": "I don't have enough information on this topic."}]


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


def prep():
    global embedding_model, chroma_client, collection, model, tokenizer, generator

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("financial_data")

    login(token='hf_KWGCDOKcYOUrYUUJZurryXKJrwxqqNgnNK')
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def scale_confidence(score):
    if score > 0.6:
        return round(score * 0.9 + 0.3, 2)  # Strong relevance
    elif score > 0.4:
        return round(score * 0.7 + 0.2, 2)  # Moderate relevance
    else:
        return round(score * 0.5 + 0.1, 2)  # Weak relevance, but not 0

# Multi-Stage Retrieval: BM25 + Embeddings + Re-Ranking
def retrieve_similar_chunks_Advanced_Rag(query, top_k=3,similarity_threshold=0.2):

    # BM25 retrieval
    all_metadatas = collection.get()["metadatas"]
    all_texts = [doc["text"] for doc in all_metadatas]
    bm25 = BM25Okapi([doc.split() for doc in all_texts])
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
    bm25_top_texts = [all_texts[i] for i in bm25_top_indices]

    # Embedding-based retrieval
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
    embedding_texts = [res["text"] for res in results["metadatas"][0]]

    # Combine BM25 and embedding results
    combined_texts = list(set(bm25_top_texts + embedding_texts))

    # Re-ranking based on embedding similarity
    chunk_embeddings = embedding_model.encode(combined_texts, convert_to_numpy=True)
    query_embedding = query_embedding.reshape(1, -1)
    similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()


    confidence_scores = [scale_confidence(sim) for sim in similarities]

    # Sort results by confidence score
    ranked_indices = np.argsort(similarities)[-top_k:][::-1]
    ranked_texts = [combined_texts[i] for i in ranked_indices]
    ranked_scores = [confidence_scores[i] for i in ranked_indices]

    return [{"text": text, "confidence": round(score, 2)} for text, score in zip(ranked_texts, ranked_scores)]


def ask_local_llm_adv(query, retrieved_chunks,confidence_threshold = 0.2):
    # Extract text from retrieved chunks
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    avg_confidence = sum(chunk["confidence"] for chunk in retrieved_chunks) / len(retrieved_chunks)
    # If confidence is low, return "I don't know."
    if avg_confidence < confidence_threshold:
        return  "I'm currently specialized in financial reports and related topics. Could you please ask a financial-related question?", avg_confidence


    prompt = f"""
    You are a financial AI answering based on Cognizant's 2023 and 2024 report.
    Stick to the retrieved context. If unsure, say "I don't know."

    Context:
    {context}

    Question: {query}
    """

    response = generator(prompt, max_new_tokens=200)

    return response[0]["generated_text"].replace(prompt, "").strip(), avg_confidence

import re

# List of restricted words (harmful or irrelevant)
BLOCKED_PATTERNS = [
    r"(hack|attack|exploit|cheat|steal|fraud|bomb|kill)",  # Malicious intent
    r"drop\s+table|delete\s+from|select\s+\*|insert\s+into",  # SQL injection
    r"http[s]?://|www\.",  # External URL injection
    r"\bshutdown\b|\breboot\b|;\s*rm\s+-rf\s+/",  # Command injection
    r"<script>|</script>|<img\s+src|onerror=",  # XSS
]

# Unwanted irrelevant topics
UNWANTED_TOPICS = [
    r"\b(weather|rain|temperature|climate|forecast|sunny|snow)\b",
    r"\b(politics|election|government|president|minister|senate|parliament)\b",
    r"\b(movies|celebrity|music|hollywood|bollywood|actor|actress)\b",
    r"\b(sports|football|cricket|basketball|soccer|tennis|olympics)\b",
    r"\b(religion|god|spiritual|church|temple|prayer)\b",
    r"\b(gossip|rumor|scandal)\b",
]

def validate_query(query):

  query_lower = query.lower().strip()
  for pattern in BLOCKED_PATTERNS+UNWANTED_TOPICS:
    match = re.search(pattern, query_lower)
    if match:
      keyword = match.group(0)
      return False, f"Your query seems to contain the term '{keyword}', which is outside the scope of financial topics. Please refine your query to focus on financial reports."

  return True, "Query is valid"


def GuardRail(query):
  is_valid, reason = validate_query(query)
  if is_valid:
    retrieved_chunks_Advanced_Rag = retrieve_similar_chunks_Advanced_Rag(query)
    # Generate response using local LLM
    #response,confidence = ask_local_llm(query, retrieved_chunks_Advanced_Rag)
    return retrieved_chunks_Advanced_Rag

  else:
    return reason





