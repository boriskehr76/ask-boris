from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# ── Model registry ─────────────────────────────────────────────────────────
# input_cost / output_cost are USD per 1M tokens
MODELS = {
    "claude-haiku-4-5-20251001": {
        "name": "Claude Haiku 4.5",
        "provider": "anthropic",
        "input_cost": 1.00,
        "output_cost": 5.00,
    },
    "claude-sonnet-4-6": {
        "name": "Claude Sonnet 4.6",
        "provider": "anthropic",
        "input_cost": 3.00,
        "output_cost": 15.00,
    },
    "gpt-4o-mini": {
        "name": "GPT-4o mini",
        "provider": "openai",
        "input_cost": 0.15,
        "output_cost": 0.60,
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "provider": "openai",
        "input_cost": 2.50,
        "output_cost": 10.00,
    },
    "gemini-1.5-flash": {
        "name": "Gemini 1.5 Flash",
        "provider": "google",
        "input_cost": 0.075,
        "output_cost": 0.30,
    },
    "gemini-1.5-pro": {
        "name": "Gemini 1.5 Pro",
        "provider": "google",
        "input_cost": 1.25,
        "output_cost": 5.00,
    },
    "mistral-small-latest": {
        "name": "Mistral Small",
        "provider": "mistral",
        "input_cost": 0.20,
        "output_cost": 0.60,
    },
    "mistral-large-latest": {
        "name": "Mistral Large",
        "provider": "mistral",
        "input_cost": 2.00,
        "output_cost": 6.00,
    },
}

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# ── Provider clients ────────────────────────────────────────────────────────
print("Loading resources...")
chroma_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
collection = chroma_client.get_collection("boris")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

claude_client = anthropic.Anthropic()

openai_client = None
if os.environ.get("OPENAI_API_KEY"):
    from openai import OpenAI
    openai_client = OpenAI()
    print("OpenAI client ready.")

google_ready = False
if os.environ.get("GOOGLE_API_KEY"):
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    google_ready = True
    print("Google client ready.")

mistral_client = None
if os.environ.get("MISTRAL_API_KEY"):
    try:
        from mistralai import Mistral
        mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        print("Mistral client ready.")
    except Exception as e:
        print(f"Mistral unavailable: {e}")

print("Ready.")

SYSTEM_PROMPT = """You are Boris Kehr — a designer and ML student based in Stockholm with 15+ years of experience in UX, product design, and AI/ML. You write and think in a direct, opinionated way. You believe design is about solving real problems, not aesthetics. You are skeptical of hype, including AI hype.

Use the retrieval confidence signal to frame your answer:

- If confidence is HIGH: answer directly and confidently from the context, in Boris's voice. No disclaimers needed.
- If confidence is MEDIUM: start your answer with "Based on related topics I've written about..." and build from the context plus reasonable extrapolation.
- If confidence is LOW: you will NOT receive this — the frontend handles it separately.

Never break character. Never say you are an AI or a language model. Always answer as Boris.
Answer in the same language the question was asked in.
Keep answers focused and direct — Boris doesn't ramble."""


def stream_tokens(provider, model_id, user_prompt):
    if provider == "anthropic":
        with claude_client.messages.stream(
            model=model_id,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text

    elif provider == "openai":
        stream = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_prompt}],
            stream=True,
            max_tokens=1000
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    elif provider == "google":
        gmodel = genai.GenerativeModel(model_name=model_id, system_instruction=SYSTEM_PROMPT)
        response = gmodel.generate_content(user_prompt, stream=True)
        for chunk in response:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text

    elif provider == "mistral":
        stream = mistral_client.chat.stream(
            model=model_id,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_prompt}],
            max_tokens=1000
        )
        for chunk in stream:
            delta = chunk.data.choices[0].delta.content
            if delta:
                yield delta


def get_context_and_sources(question):
    vector = embed_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=vector,
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    avg_distance = sum(distances) / len(distances)
    if avg_distance < 1.0:
        confidence = "high"
    elif avg_distance < 1.5:
        confidence = "medium"
    else:
        confidence = "low"

    context = ""
    sources = []
    seen_source_types = set()
    seen_urls = set()

    for doc, meta in zip(docs, metas):
        context += f"\n---\n{doc}\n"
        source_type = meta.get("source", "")

        if meta.get("url"):
            url = meta["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            title = meta.get("title", "").strip()
            if not title:
                title = doc.split(".")[0].strip()[:80]
            date = meta.get("date", "")[:10]
            sources.append({
                "title": title,
                "date": date,
                "url": url,
                "type": "post"
            })
        elif source_type == "interview" and "interview" not in seen_source_types:
            sources.append({
                "title": "",
                "date": "",
                "url": None,
                "type": "conversation"
            })
            seen_source_types.add("interview")
        elif source_type == "notes" and "notes" not in seen_source_types:
            sources.append({
                "title": "",
                "date": "",
                "url": None,
                "type": "note"
            })
            seen_source_types.add("notes")

    return context, sources, confidence


@app.route("/")
def chat():
    return send_from_directory(BASE_DIR, "chat.html")


@app.route("/report")
def report():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/models")
def get_models():
    available = {}
    for model_id, info in MODELS.items():
        provider = info["provider"]
        is_available = (
            (provider == "anthropic") or
            (provider == "openai" and openai_client is not None) or
            (provider == "google" and google_ready) or
            (provider == "mistral" and mistral_client is not None)
        )
        available[model_id] = {**info, "available": is_available}
    return jsonify(available)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    confirmed = data.get("confirmed", False)
    model_id = data.get("model", DEFAULT_MODEL)

    if model_id not in MODELS:
        model_id = DEFAULT_MODEL

    model_info = MODELS[model_id]
    provider = model_info["provider"]

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if provider == "openai" and not openai_client:
        return jsonify({"error": "OpenAI API key not configured"}), 400
    if provider == "google" and not google_ready:
        return jsonify({"error": "Google API key not configured"}), 400
    if provider == "mistral" and not mistral_client:
        return jsonify({"error": "Mistral API key not configured"}), 400

    context, sources, confidence = get_context_and_sources(question)

    if confidence == "low" and not confirmed:
        return jsonify({"confirm_needed": True})

    user_prompt = f"""Retrieval confidence: {"medium" if confirmed else confidence}
Context from Boris's writing:
{context}

Question: {question}"""

    def generate():
        yield f"data: {json.dumps({'sources': sources[:3]})}\n\n"
        try:
            for token in stream_tokens(provider, model_id, user_prompt):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port, host="0.0.0.0", threaded=True)