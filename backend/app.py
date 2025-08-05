from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.chat_query import retrieve, ask_gemini
from fastapi.responses import FileResponse

class Query(BaseModel):
    query: str

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve the frontend folder as static files
import os
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def root():
    return FileResponse("frontend/chatbot.html")

@app.post("/api/ask")
def ask(query: Query):
    chunks = retrieve(query.query)
    answer = ask_gemini(query.query, chunks)
    return {"answer": answer}
