import json
import os
import anthropic

client = anthropic.Anthropic()

FOLDER = "JSON interviews"
BORIS = "Boris Kehr"
SWEDISH_WORDS = {"och", "att", "det", "som", "för", "med", "på", "av", "är", "en", "ett", "har"}


def is_swedish(text):
    words = text.lower().split()[:50]
    return sum(1 for w in words if w in SWEDISH_WORDS) >= 3


def group_turns(texts):
    if not texts:
        return []
    grouped = []
    cur_speaker = texts[0]["speaker"]
    cur_text = texts[0]["text"]
    for t in texts[1:]:
        if t["speaker"] == cur_speaker:
            cur_text += " " + t["text"]
        else:
            grouped.append({"speaker": cur_speaker, "text": cur_text.strip()})
            cur_speaker = t["speaker"]
            cur_text = t["text"]
    grouped.append({"speaker": cur_speaker, "text": cur_text.strip()})
    return grouped


def translate(text):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": (
                "Translate this interview excerpt from Swedish to English.\n"
                "Keep Boris's voice, tone and opinions intact. Preserve the Q&A format exactly.\n"
                "Return only the translated text, no preamble.\n\n"
                + text
            )
        }]
    )
    return response.content[0].text


all_docs = []

for fname in sorted(os.listdir(FOLDER)):
    if not fname.endswith(".json"):
        continue

    with open(os.path.join(FOLDER, fname), encoding="utf-8") as f:
        data = json.load(f)

    # Group consecutive turns, then anonymize non-Boris speakers
    grouped = group_turns(data.get("texts", []))
    for turn in grouped:
        if turn["speaker"] != BORIS:
            turn["speaker"] = "Interviewer"

    # Re-group after anonymization — consecutive Interviewer turns now merge
    turns = group_turns(grouped)

    # Build Q&A pairs
    pairs = []
    for i, turn in enumerate(turns):
        if turn["speaker"] != BORIS:
            continue
        if len(turn["text"].split()) < 30:
            continue
        if i == 0 or turns[i - 1]["speaker"] != "Interviewer":
            continue
        q_text = turns[i - 1]["text"]
        if len(q_text.split()) < 15:
            continue
        pairs.append((q_text, turn["text"]))

    print(f"{fname}: {len(pairs)} pairs", end="", flush=True)

    translated = 0
    for q, a in pairs:
        combined = f"Interviewer: {q}\nBoris: {a}"
        if is_swedish(combined):
            combined = translate(combined)
            translated += 1
        all_docs.append({
            "text": combined,
            "source": "interview",
            "draft": False,
            "url": None,
            "date": "",
            "title": "Interview excerpt"
        })

    if translated:
        print(f", {translated} translated")
    else:
        print()

with open("transcripts_corpus.json", "w", encoding="utf-8") as f:
    json.dump(all_docs, f, ensure_ascii=False, indent=2)

print(f"\nDone. {len(all_docs)} documents saved to transcripts_corpus.json")
