# add_document.py
collection.add(
    ids=["420"],  # next available ID
    embeddings=model.encode([new_text]).tolist(),
    documents=[new_text],
    metadatas=[{"source": "article", "title": title, "date": date, "url": url}]
)