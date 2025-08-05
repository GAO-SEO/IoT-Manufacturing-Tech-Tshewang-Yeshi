import os
import re
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
embedding_model = genai.embed_content

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text, chunk_size=500):
    text = re.sub(r'\s+', ' ', text)
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def embed_chunks(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        result = embedding_model(
            model="models/embedding-001",
            content=chunk,
            task_type="retrieval_document"
        )
        vector = np.array(result["embedding"], dtype="float32")
        embeddings.append(vector)
    return embeddings

def save_embeddings(chunks):
    vectors = embed_chunks(chunks)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors))
    faiss.write_index(index, "embeddings/vector.index")

    with open("data/texts.pkl", "wb") as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    raw_text = load_text("data/chunks.txt")
    chunks = split_text(raw_text)
    save_embeddings(chunks)
    print(f"Saved {len(chunks)} chunks to FAISS index.")
