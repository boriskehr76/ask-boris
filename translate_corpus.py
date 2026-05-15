import json
import anthropic

client = anthropic.Anthropic()

with open("corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)

def is_swedish(text):
    swedish_words = ["och", "att", "det", "som", "för", "med", "på", "av", "är", "en", "ett", "har"]
    words = text.lower().split()[:50]
    hits = sum(1 for w in words if w in swedish_words)
    return hits >= 3

translated = 0
skipped = 0

for i, doc in enumerate(corpus):
    text = doc.get("text", "")
    if not text.strip():
        skipped += 1
        continue

    if not is_swedish(text):
        doc["lang"] = "en"
        continue

    print(f"[{i+1}/{len(corpus)}] Translating: {text[:60].strip()}...")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"""Translate this text from Swedish to English.
Keep the author's voice, tone and opinions intact.
Return only the translated text, no preamble.

{text}"""
        }]
    )

    doc["text"] = response.content[0].text
    doc["lang"] = "en"
    doc["translated"] = True
    translated += 1

with open("corpus_translated.json", "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)

print(f"\nDone. Translated {translated} documents. Skipped {skipped} empty. Saved to corpus_translated.json")
