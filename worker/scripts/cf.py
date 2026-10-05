"""Small Cloudflare REST helpers shared by ingest.py and calibrate.py.

Needs two environment variables:
  CLOUDFLARE_ACCOUNT_ID  - shown in the Cloudflare dashboard sidebar
  CLOUDFLARE_API_TOKEN   - a token with "Workers AI: Read" and "Vectorize: Edit"
"""
import json
import os
import sys
import urllib.error
import urllib.request

EMBED_MODEL = "@cf/baai/bge-m3"
INDEX = os.environ.get("VECTORIZE_INDEX", "ask-boris")

ACCOUNT = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
if not ACCOUNT or not TOKEN:
    sys.exit("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN first.")

BASE = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}"


def _post(path, body, content_type="application/json"):
    data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method="POST",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            out = json.loads(res.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"Cloudflare API error {e.code} on {path}: {e.read().decode()[:500]}")
    if not out.get("success", True):
        sys.exit(f"Cloudflare API error on {path}: {out.get('errors')}")
    return out["result"]


def embed(texts):
    """Return one 1024-dim vector per text."""
    return _post(f"/ai/run/{EMBED_MODEL}", {"text": texts})["data"]


def upsert(vectors):
    """vectors: list of {"id", "values", "metadata"}."""
    ndjson = "\n".join(json.dumps(v, ensure_ascii=False) for v in vectors).encode("utf-8")
    return _post(f"/vectorize/v2/indexes/{INDEX}/upsert", ndjson, "application/x-ndjson")


def query(vector, top_k=10):
    return _post(
        f"/vectorize/v2/indexes/{INDEX}/query",
        {"vector": vector, "topK": top_k, "returnMetadata": "none"},
    )["matches"]
