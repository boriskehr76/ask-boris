import json
import anthropic

client = anthropic.Anthropic()

with open("articles_clean.json", encoding="utf-8") as f:
    articles = json.load(f)

def is_swedish(text):
    swedish_words = ["och", "att", "det", "som", "för", "med", "på", "av", "är", "en", "ett", "har"]
    words = text.lower().split()[:50]
    hits = sum(1 for w in words if w in swedish_words)
    return hits >= 3

translated = 0
for i, article in enumerate(articles):
    if not is_swedish(article["text"]):
        print(f"  [{i+1}/{len(articles)}] Already English: {article['title'][:50]}")
        continue

    print(f"  [{i+1}/{len(articles)}] Translating: {article['title'][:50]}")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"""Translate this article from Swedish to English. 
Keep the author's voice, tone and opinions intact. 
Return only the translated text, no preamble.

{article['text']}"""
        }]
    )

    article["text"] = response.content[0].text
    article["translated"] = True
    translated += 1

with open("articles_clean.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"\nDone. Translated {translated} articles.")