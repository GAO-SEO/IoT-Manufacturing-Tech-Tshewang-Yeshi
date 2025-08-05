import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model once globally (avoids re-instantiating every time)
model = genai.GenerativeModel("gemini-2.5-pro")

def embed_query(query):
    result = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return np.array(result["embedding"], dtype="float32")

def retrieve(query, k=5):
    index = faiss.read_index("embeddings/vector.index")
    with open("embeddings/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    q_vec = embed_query(query).reshape(1, -1)
    _, I = index.search(q_vec, k)
    return [texts[i] for i in I[0]]
def ask_gemini(query, retrieved_chunks):
    normalized = query.strip().lower()

    if normalized in ["hi", "hello", "hey", "halo"]:
        return (
            "Hi there! 👋 I'm here to help with IoT, sensors, asset tracking, and SOP-related questions.\n\n"
            "You can ask me things like:\n"
            "1️⃣ Which IoT devices are good for predictive maintenance?\n"
            "2️⃣ How do I set up a BLE gateway?\n"
            "3️⃣ Can you help create a system diagram?\n"
            "4️⃣ What are real-world examples of smart factory deployments?"
        )

    mapped_prompts = {
        "1": "What IoT devices are suitable for applications like predictive maintenance or asset tracking?",
        "2": "What platforms or protocols are supported by IoT Manufacturing Tech products, and how do I set them up?",
        "3": "Can you help me create a system diagram and bill of materials (BOM) for an IoT solution?",
        "4": "In what ways are IoT Manufacturing Tech solutions used in smart factories or automation?",
        "5": "I have another question related to IoT or asset tracking. Please assist."
    }

    if normalized in mapped_prompts:
        query = mapped_prompts[normalized]
        retrieved_chunks = retrieve(query)

    context = "\n\n".join(retrieved_chunks)

    if not context.strip():
        return (
            "I'm here to help with IoT and asset tracking topics like RFID, BLE gateways, smart factory use cases, and more. "
            "Could you try rephrasing your question or asking about a specific product or setup?"
        )

    prompt = f"""
You are a helpful, knowledgeable assistant working for IoT Manufacturing Tech who only answers whats there in your knowledge base that is about the IoT MAunfacturing Tech. Your job is to answer the user's question clearly and naturally using only the context below.

🚫 Do not say things like “Based on the provided context” or “According to the text.”  
✅ Just write the answer as if you're directly helping the user. Keep the flow like a natural conversation.  
✅ Avoid robotic phrases like “Of course” or “Certainly.”. Act like you know only about Gao Tek IoT manufacturing Tech. If anything beyond your knowledge base which is at least 20% not related to the knowledgebase and topics like RFID, BLE gateways, IoT, its use cases,  then just say "I'm here to help with IoT and asset tracking topics like RFID, BLE gateways, smart factory use cases, and more. Could you try rephrasing your question or asking about a specific product or setup?"

Context:
{context}

User question: {query}

Your reply:
"""

    try:
        response = model.generate_content(prompt)

        if response.candidates and response.candidates[0].content.parts:
            return response.text.strip()
        else:
            return (
                "Hmm, I couldn't find anything specific to answer that. Can you try asking in a different way?"
            )

    except Exception as e:
        return f"⚠️ Sorry, something went wrong while processing your request: {str(e)}"
