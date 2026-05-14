import json
import pandas as pd
from pathlib import Path

# Load articles
with open("articles_clean.json", encoding="utf-8") as f:
    articles = json.load(f)

# Load posts
df = pd.read_csv("Shares.csv")
df = df.dropna(subset=["ShareCommentary"])
df = df[df["ShareCommentary"].str.strip().str.len() > 50]

posts = []
for _, row in df.iterrows():
    posts.append({
        "source": "post",
        "title": "",
        "date": str(row["Date"])[:10],
        "url": row.get("ShareLink", ""),
        "text": row["ShareCommentary"],
        "filename": ""
    })

# Merge
corpus = articles + posts
corpus.sort(key=lambda x: x["date"])

with open("corpus.json", "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)

print(f"Articles:  {len(articles)}")
print(f"Posts:     {len(posts)}")
print(f"Total:     {len(corpus)}")
print(f"Date range: {corpus[0]['date']} → {corpus[-1]['date']}")