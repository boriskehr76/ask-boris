"""Find confidence thresholds for bge-m3 + Vectorize.

The old thresholds (avg L2 distance < 1.0 / < 1.5) belonged to all-MiniLM-L6-v2
and don't transfer to a new embedding model. This script measures the average
top-10 cosine similarity for questions Boris HAS written about and questions
he hasn't, then suggests CONF_HIGH and CONF_MEDIUM for wrangler.toml.

    cd worker
    python3 scripts/calibrate.py

Edit the two lists below to match what you actually want the bot to answer.
"""
from statistics import mean

from cf import embed, query

ON_TOPIC = [
    "What do you think about AI in design?",
    "How has the UX designer role changed?",
    "What makes someone a great designer?",
    "What's broken about how companies hire designers?",
    "How do you approach learning something new?",
    "Why do design teams need strategy and not just pixels?",
    "What should a junior designer focus on?",
    "How did teaching UX change how you think about design?",
    "Vad tycker du om AI i designarbetet?",
    "Varför är det så svårt att få jobb som UX-designer?",
]

OFF_TOPIC = [
    "What is a good recipe for lasagna?",
    "Who won the 2018 football World Cup?",
    "How do I change the oil in a Volvo V70?",
    "Explain quantum chromodynamics.",
    "What is the capital of Australia?",
    "Hur långt är det från Stockholm till Göteborg?",
]


def avg_scores(questions):
    vectors = embed(questions)
    out = []
    for q, v in zip(questions, vectors):
        matches = query(v, top_k=10)
        s = mean(m["score"] for m in matches) if matches else 0.0
        out.append((q, s))
    return out


on = avg_scores(ON_TOPIC)
off = avg_scores(OFF_TOPIC)

print("ON TOPIC (should be high/medium)")
for q, s in sorted(on, key=lambda x: -x[1]):
    print(f"  {s:.3f}  {q}")
print("\nOFF TOPIC (should be low)")
for q, s in sorted(off, key=lambda x: -x[1]):
    print(f"  {s:.3f}  {q}")

on_scores = sorted(s for _, s in on)
off_max = max(s for _, s in off)
on_min = on_scores[0]
on_median = on_scores[len(on_scores) // 2]

medium = round((off_max + on_min) / 2, 3)
high = round(max(on_median, medium + 0.02), 3)

print("\nSuggested thresholds:")
print(f'  CONF_HIGH = "{high}"')
print(f'  CONF_MEDIUM = "{medium}"')
if on_min <= off_max:
    print(
        "\nWarning: some on-topic questions score below the best off-topic one, so"
        " no threshold separates them cleanly. The confirm step will fire on some"
        " real questions; adjust by hand."
    )
