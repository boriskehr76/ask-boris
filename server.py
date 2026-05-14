from flask import Flask, request, jsonify, send_from_directory
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
import os

app = Flask(__name__)

print("Loading resources...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("boris")
model = SentenceTransformer("all-MiniLM-L6-v2")
claude = anthropic.Anthropic()
print("Ready.")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    vector = model.encode([question]).tolist()
    results = collection.query(query_embeddings=vector, n_results=5)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = ""
    sources = []
    for doc, meta in zip(docs, metas):
        context += f"\n---\n{doc}\n"
        if meta.get("url"):
            title = meta.get("title", "").strip()
            if not title:
                first_sentence = doc.split(".")[0].strip()[:80]
                title = first_sentence
            date = meta.get("date", "")[:10]
            sources.append({
                "title": title,
                "date": date,
                "url": meta["url"]
            })

    system = """You are Boris Kehr — a designer and ML student based in Stockholm with 15+ years of experience in UX, product design, and AI/ML.
Answer questions based only on the context provided. Be direct and opinionated, in Boris's voice.
If the context doesn't contain enough information, say so honestly.
Answer in the same language the question was asked in."""

    user_prompt = f"""Context from Boris's writing:
{context}

Question: {question}"""

    response = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user_prompt}]
    )

    return jsonify({
        "answer": response.content[0].text,
        "sources": sources
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port, host="0.0.0.0")