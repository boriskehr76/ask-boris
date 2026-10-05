"""Embed the Ask Boris corpus with bge-m3 and load it into Vectorize.

Replaces build_embeddings.py + add_document.py for the Cloudflare version.
Re-run it any time the corpus changes: ids are stable, so it upserts in place.

    cd worker
    python3 scripts/ingest.py
"""
import json
import os
import time

from cf import embed, upsert

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SOURCES = [
    "corpus_translated.json",   # published: posts + articles
    "transcripts_corpus.json",  # interviews
    "notes_corpus.json",        # unpublished notes
]
BATCH = 20
MAX_CHARS = 1000  # same truncation as the Chroma version

corpus = []
for name in SOURCES:
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        print(f"Skipping {name} (not found)")
        continue
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} documents from {name}")
    corpus.extend(docs)

print(f"Total: {len(corpus)} documents\n")

for i in range(0, len(corpus), BATCH):
    batch = corpus[i : i + BATCH]
    texts = [doc["text"][:MAX_CHARS] for doc in batch]
    vectors = embed(texts)
    upsert([
        {
            "id": str(i + j),
            "values": vec,
            "metadata": {
                "source": doc.get("source", "") or "",
                "title": doc.get("title", "") or "",
                "date": doc.get("date", "") or "",
                "url": doc.get("url", "") or "",
                "text": text,
            },
        }
        for j, (doc, text, vec) in enumerate(zip(batch, texts, vectors))
    ])
    print(f"  {min(i + BATCH, len(corpus))}/{len(corpus)}")
    time.sleep(0.2)

print("\nDone. Vectorize indexes asynchronously; give it a minute before querying.")
