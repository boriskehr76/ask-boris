import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load corpus
with open("corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)

# Load embedding model (downloads ~90MB on first run)
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up ChromaDB (saves to disk)
client = chromadb.PersistentClient(path="./chroma_db")

# Delete collection if rebuilding
try:
    client.delete_collection("boris")
except:
    pass

collection = client.create_collection("boris")

# Embed and store in batches
print(f"Embedding {len(corpus)} documents...")
batch_size = 50

for i in range(0, len(corpus), batch_size):
    batch = corpus[i:i+batch_size]
    texts = [doc["text"][:1000] for doc in batch]
    ids = [str(i+j) for j in range(len(batch))]
    metadatas = [{"source": doc["source"], "title": doc["title"],
                  "date": doc["date"], "url": doc["url"]} for doc in batch]

    embeddings = model.encode(texts).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"  {min(i+batch_size, len(corpus))}/{len(corpus)}")

print("Done. Vector database saved to ./chroma_db")