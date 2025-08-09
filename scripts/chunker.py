# scripts/chunker.py
import json, re, os

IN_JSONL  = "data/site_content/pages.jsonl"
OUT_JSONL = "data/chunks.jsonl"

MAX_TOKENS = 450  # rough target; you can approximate by words/characters

def split_text(txt, max_len=1500):
    # simple paragraph-based splitter
    paras = [p.strip() for p in txt.split("\n") if p.strip()]
    chunks, cur = [], []
    size = 0
    for p in paras:
        size += len(p)
        cur.append(p)
        if size >= max_len:
            chunks.append("\n\n".join(cur))
            cur, size = [], 0
    if cur: chunks.append("\n\n".join(cur))
    return chunks

os.makedirs("data", exist_ok=True)
with open(IN_JSONL, "r", encoding="utf-8") as fin, open(OUT_JSONL, "w", encoding="utf-8") as fout:
    for line in fin:
        page = json.loads(line)
        url, title, text = page["url"], page["title"], page["text"]
        for i, chunk in enumerate(split_text(text)):
            rec = {"id": f"{url}#chunk{i}", "url": url, "title": title, "text": chunk}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
