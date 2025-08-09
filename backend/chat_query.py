# backend/chat_query.py
import os
import re
import json
import faiss
import pickle
import numpy as np
from typing import List, Dict

from dotenv import load_dotenv
import google.generativeai as genai

# -----------------------
# Config & model init
# -----------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create once per process to avoid re-instantiation overhead
MODEL = genai.GenerativeModel("gemini-2.5-pro")

INDEX_PATH = "embeddings/vector.index"
META_PATH  = "embeddings/meta.jsonl"

HOMEPAGE_URL   = "https://iotmanufacturingtech.com"
CONTACT_US_URL = "https://iotmanufacturingtech.com/contact-us/"

# -----------------------
# Embedding helpers
# -----------------------
def embed_query(query: str) -> np.ndarray:
    """Return a float32 embedding vector for the query (L2 normalized)."""
    res = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    # handle both old/new shapes
    if isinstance(res, dict) and "embedding" in res:
        vec = np.asarray(res["embedding"], dtype="float32")
    else:
        # newer format: {"data":[{"embedding":[...]}]}
        vec = np.asarray(res["data"][0]["embedding"], dtype="float32")
    # faiss IndexFlatIP works best with normalized vectors
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec


def _load_meta() -> List[Dict]:
    """Read aligned metadata rows: [{id,url,title,text}, ...]."""
    with open(META_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def retrieve(query: str, k: int = 5) -> List[Dict]:
    """Return top-k metadata rows (each has text,url,title,id)."""
    index = faiss.read_index(INDEX_PATH)
    meta  = _load_meta()

    q = embed_query(query).reshape(1, -1)
    # already normalized above, but harmless to call again
    faiss.normalize_L2(q)
    _, I = index.search(q, k)

    results = []
    for i in I[0]:
        if 0 <= i < len(meta):
            results.append(meta[i])
    return results

# -----------------------
# Link/HTML sanitizers
# -----------------------
def _sanitize_url(url: str) -> str:
    """Keep only iotmanufacturingtech.com URLs; fallback to homepage."""
    if not isinstance(url, str) or not url:
        return HOMEPAGE_URL
    url = url.strip()
    # block javascript/mailto/etc.
    if re.match(r"^(javascript|data|vbscript|mailto):", url, re.I):
        return HOMEPAGE_URL
    # allow only our domain
    if "iotmanufacturingtech.com" not in url:
        return HOMEPAGE_URL
    # drop fragments & query params
    url = re.sub(r"#.*$", "", url)
    url = re.sub(r"\?.*$", "", url)
    return url


def _sanitize_anchor_html(html: str) -> str:
    """Rewrite <a> tags to safe absolute URLs, open in new tab, nofollow."""
    if not html:
        return html

    def repl(m):
        href = m.group("href")
        text = m.group("text")
        safe = _sanitize_url(href)
        return f'<a href="{safe}" target="_blank" rel="noopener noreferrer">{text}</a>'

    # Match <a href="...">...</a>
    pattern = re.compile(r'<a\s+href="(?P<href>[^"]+)"[^>]*>(?P<text>.*?)</a>', re.I | re.S)
    return pattern.sub(repl, html)


def _format_context(retrieved: List[Dict]) -> str:
    """Join retrieved chunks with TITLE and URL headers."""
    blocks = []
    for r in retrieved:
        title = (r.get("title") or "").strip()
        url   = _sanitize_url(r.get("url", ""))
        text  = (r.get("text") or "").strip()
        blocks.append(f"TITLE: {title}\nURL: {url}\n\n{text}")
    return "\n\n---\n\n".join(blocks)


def _allowed_links(retrieved: List[Dict]) -> List[str]:
    """Unique list of allowed, sanitized URLs derived from retrieval."""
    seen, urls = set(), []
    for r in retrieved:
        u = _sanitize_url(r.get("url", ""))
        if u not in seen:
            seen.add(u)
            urls.append(u)
    # Always allow homepage + contact as safe fallbacks
    if HOMEPAGE_URL not in seen:
        urls.append(HOMEPAGE_URL)
    if CONTACT_US_URL not in seen:
        urls.append(CONTACT_US_URL)
    return urls

# -----------------------
# Main answer function
# -----------------------
def ask_gemini(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Build a constrained prompt that:
      - Uses only provided context
      - Restricts links to an allowed list
      - Produces clean, natural answers
    """
    normalized = query.strip().lower()

    # Friendly greeting only for greeting messages
    if normalized in {"hi", "hello", "hey", "halo"}:
        return (
            "Hi there! 👋 I’m here to help with IoT, sensors, asset tracking, and SOP questions.\n\n"
            "You can ask things like:\n"
            "1️⃣ Which IoT devices fit predictive maintenance?\n"
            "2️⃣ How do I set up a BLE gateway?\n"
            "3️⃣ Can you help create a system diagram or BOM?\n"
            "4️⃣ What are real-world examples of smart factory deployments?"
        )

    # Menu shortcuts (optional)
    mapped_prompts = {
        "1": "Which IoT devices are suitable for predictive maintenance or asset tracking?",
        "2": "Which platforms or protocols are supported and how do I set up a BLE gateway?",
        "3": "Help me design a system diagram and bill of materials (BOM) for an IoT solution.",
        "4": "How are IoT Manufacturing Tech solutions used in smart factories and automation?",
        "5": "I have another IoT or asset tracking question. Please assist.",
    }
    if normalized in mapped_prompts:
        query = mapped_prompts[normalized]
        retrieved_chunks = retrieve(query)

    # If caller passed raw chunks list of strings, wrap them
    if retrieved_chunks and isinstance(retrieved_chunks[0], str):
        retrieved_chunks = [{"url": HOMEPAGE_URL, "title": "", "text": t} for t in retrieved_chunks]

    # Build prompt pieces
    context = _format_context(retrieved_chunks)
    allowed = _allowed_links(retrieved_chunks)
    allowed_list = "\n".join(f"- {u}" for u in allowed)

    if not context.strip():
        # Graceful fallback with topic boundary
        return (
            "I’m here to help with IoT and asset-tracking topics like RFID, BLE gateways, "
            "smart-factory use cases, and related setups. Could you try rephrasing or ask about a "
            "specific product or scenario? You can also visit "
            f'<a href="{CONTACT_US_URL}" target="_blank" rel="noopener noreferrer">Contact Us</a> for direct help.'
        )

    prompt = f"""
You are a helpful assistant for IoT Manufacturing Tech. Answer clearly and naturally.
Do not use filler phrases like “Of course” or “Based on the context”.
Use ONLY the information in the context and the **Allowed Links** list below.
Never invent URLs. If a precise page isn’t listed, link to the homepage or Contact Us.

Format links as HTML anchors exactly like:
<a href="URL" target="_blank">Link text</a>
(Do NOT escape the HTML.)

Allowed Links:
{allowed_list}

Context:
{context}

User question: {query}

Your answer:
"""

    try:
        resp = MODEL.generate_content(prompt)
        # Gemini can return an empty candidate or no parts if it refused/finished early
        text = ""
        if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text.strip():
            text = resp.text.strip()
        elif getattr(resp, "candidates", None):
            # Try to stitch text from parts (rare case)
            parts = []
            for c in resp.candidates:
                if getattr(c, "content", None) and getattr(c.content, "parts", None):
                    for p in c.content.parts:
                        if hasattr(p, "text"):
                            parts.append(p.text)
            text = "\n".join(parts).strip()

        if not text:
            return (
                "I couldn’t find a specific answer in our knowledge base. "
                f'Please try another query or reach us via <a href="{CONTACT_US_URL}" target="_blank" rel="noopener noreferrer">Contact Us</a>.'
            )

        # Sanitize anchors to keep them on our domain & open in new tab
        text = _sanitize_anchor_html(text)
        return text

    except Exception as e:
        # Safe fallback on API issues
        return (
            "Sorry — I hit a snag processing that. Please try again in a moment, or visit "
            f'<a href="{CONTACT_US_URL}" target="_blank" rel="noopener noreferrer">Contact Us</a>.'
        )
