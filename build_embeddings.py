import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

SOURCES = [
    "corpus_translated.json",   # published: posts + articles
    "transcripts_corpus.json",  # interviews
    "notes_corpus.json",        # unpublished notes (future)
]

corpus = []
for path in SOURCES:
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        continue
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} documents from {path}")
    corpus.extend(docs)

print(f"\nTotal: {len(corpus)} documents")

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection("boris")
except:
    pass

collection = client.create_collection("boris")

print(f"Embedding {len(corpus)} documents...")
batch_size = 50

for i in range(0, len(corpus), batch_size):
    batch = corpus[i:i+batch_size]
    texts = [doc["text"][:1000] for doc in batch]
    ids = [str(i + j) for j in range(len(batch))]
    metadatas = [{
        "source": doc.get("source", ""),
        "title": doc.get("title", ""),
        "date": doc.get("date", "") or "",
        "url": doc.get("url", "") or "",
    } for doc in batch]

    embeddings = model.encode(texts).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"  {min(i + batch_size, len(corpus))}/{len(corpus)}")

print(f"\nDone. {collection.count()} documents in ChromaDB.")
