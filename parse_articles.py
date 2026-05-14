# parse_articles.py
import os
import json
from bs4 import BeautifulSoup
from pathlib import Path

articles_dir = Path("Articles/") 
output = []

for html_file in articles_dir.glob("*.html"):
    with open(html_file, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    title = soup.find("h1").get_text(strip=True) if soup.find("h1") else html_file.stem
    
    created = soup.find(class_="created")
    published = soup.find(class_="published")
    date = (published or created)
    date_str = date.get_text(strip=True).replace("Published on ", "").replace("Created on ", "") if date else ""
    
    url_tag = soup.find("h1").find("a") if soup.find("h1") else None
    url = url_tag["href"] if url_tag else ""
    
    # Remove style, script, img tags — keep text
    for tag in soup(["style", "script", "img"]):
        tag.decompose()
    
    text = soup.get_text(separator="\n", strip=True)
    
    output.append({
        "source": "article",
        "title": title,
        "date": date_str,
        "url": url,
        "text": text,
        "filename": html_file.name
    })

with open("articles_clean.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Parsed {len(output)} articles → articles_clean.json")