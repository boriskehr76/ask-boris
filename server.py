from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
import os
import json

app = Flask(__name__)

print("Loading resources...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("boris")
model = SentenceTransformer("all-MiniLM-L6-v2")
claude = anthropic.Anthropic()
print("Ready.")

SYSTEM_PROMPT = """You are Boris Kehr — a designer and ML student based in Stockholm with 15+ years of experience in UX, product design, and AI/ML. You write and think in a direct, opinionated way. You believe design is about solving real problems, not aesthetics. You are skeptical of hype, including AI hype.

Use the retrieval confidence signal to frame your answer:

- If confidence is HIGH: answer directly and confidently from the context, in Boris's voice. No disclaimers needed.
- If confidence is MEDIUM: start your answer with "Based on related topics I've written about..." and build from the context plus reasonable extrapolation.
- If confidence is LOW: you will NOT receive this — the frontend handles it separately.

Never break character. Never say you are an AI or a language model. Always answer as Boris.
Answer in the same language the question was asked in.
Keep answers focused and direct — Boris doesn't ramble."""


def get_context_and_sources(question):
    vector = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=vector,
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    avg_distance = sum(distances) / len(distances)
    if avg_distance < 0.5:
        confidence = "high"
    elif avg_distance < 0.8:
        confidence = "medium"
    else:
        confidence = "low"

    context = ""
    sources = []
    seen_source_types = set()

    for doc, meta in zip(docs, metas):
        context += f"\n---\n{doc}\n"
        source_type = meta.get("source", "")

        if meta.get("url"):
            title = meta.get("title", "").strip()
            if not title:
                title = doc.split(".")[0].strip()[:80]
            date = meta.get("date", "")[:10]
            sources.append({
                "title": title,
                "date": date,
                "url": meta["url"]
            })
        elif source_type == "interview" and "interview" not in seen_source_types:
            sources.append({
                "title": "Based on my conversations",
                "date": "",
                "url": None
            })
            seen_source_types.add("interview")
        elif source_type == "notes" and "notes" not in seen_source_types:
            sources.append({
                "title": "Based on my unpublished notes",
                "date": "",
                "url": None
            })
            seen_source_types.add("notes")

    return context, sources, confidence


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/chat")
def chat():
    return send_from_directory(".", "chat.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    confirmed = data.get("confirmed", False)

    if not question:
        return jsonify({"error": "No question provided"}), 400

    context, sources, confidence = get_context_and_sources(question)

    # Low confidence and not yet confirmed — ask the user first
    if confidence == "low" and not confirmed:
        return jsonify({"confirm_needed": True})

    # Stream the response
    user_prompt = f"""Retrieval confidence: {"low" if confirmed else confidence}
Context from Boris's writing:
{context}

Question: {question}"""

    def generate():
        # First chunk: send sources so frontend can render them early
        yield f"data: {json.dumps({'sources': sources[:3]})}\n\n"

        with claude.messages.stream(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'token': text})}\n\n"

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port, host="0.0.0.0")