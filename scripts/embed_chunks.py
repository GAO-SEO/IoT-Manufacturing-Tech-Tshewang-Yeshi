# scripts/embed_chunks.py
import os, json, numpy as np, faiss
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

IN_JSONL = "data/chunks.jsonl"
IDX_PATH = "embeddings/vector.index"
META_JSONL = "embeddings/meta.jsonl"

os.makedirs("embeddings", exist_ok=True)

records = [json.loads(l) for l in open(IN_JSONL, encoding="utf-8")]
texts = [r["text"] for r in records]

def embed_many(strs, batch=32):
    vecs = []
    for i in range(0, len(strs), batch):
        batch_text = strs[i:i+batch]
        res = genai.embed_content(
            model="models/embedding-001",
            content=batch_text,
            task_type="retrieval_document",
        )
        # Gemini returns a dict per item; normalize to NxD
        if isinstance(res, dict) and "embedding" in res:
            vecs.append(np.asarray(res["embedding"], dtype="float32"))
        else:
            # new API returns {"data":[{"embedding":[...]}]}
            data = res.get("data", [])
            vecs.extend([np.asarray(d["embedding"], dtype="float32") for d in data])
    return np.vstack(vecs)

emb = embed_many(texts)
d = emb.shape[1]
index = faiss.IndexFlatIP(d)
# normalize vectors for IP similarity
faiss.normalize_L2(emb)
index.add(emb)
faiss.write_index(index, IDX_PATH)

# save metadata aligned by row
with open(META_JSONL, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
