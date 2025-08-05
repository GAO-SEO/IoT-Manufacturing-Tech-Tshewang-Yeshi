# build_faiss.py
import faiss
import numpy as np
import json
import pickle

with open("embeddings/embeddings.json", "r") as f:
    data = json.load(f)

dim = len(data[0]["embedding"])
index = faiss.IndexFlatL2(dim)

texts = []
vectors = []

for item in data:
    vectors.append(item["embedding"])
    texts.append(item["text"])

vectors = np.array(vectors).astype("float32")
index.add(vectors)

faiss.write_index(index, "vector.index")

with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)
