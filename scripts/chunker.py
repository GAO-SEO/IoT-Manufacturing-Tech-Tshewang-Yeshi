# chunker.py
import os
import tiktoken

def chunk_text(text, max_tokens=500):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    words = text.split("\n")
    chunks, current_chunk = [], ""
    for line in words:
        if len(tokenizer.encode(current_chunk + line)) < max_tokens:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

input_dir = "site_content"
output_file = "chunks.txt"
all_chunks = []

for filename in os.listdir(input_dir):
    with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
        text = f.read()
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

with open(output_file, "w", encoding="utf-8") as f:
    for chunk in all_chunks:
        f.write(chunk + "\n---\n")
