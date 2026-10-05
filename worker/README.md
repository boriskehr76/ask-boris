# Ask Boris on Cloudflare (free tier)

Replaces the Railway deployment (`server.py`). Same API and the same chat page,
so `chat.html` works unchanged.

| | Railway (old) | Cloudflare (new) |
|---|---|---|
| Server | Flask | Worker (`src/index.js`) |
| Embeddings | all-MiniLM-L6-v2, English-only, loaded into RAM | Workers AI `bge-m3`, multilingual |
| Vector DB | ChromaDB on disk | Vectorize, cosine |
| Confidence | avg L2 distance, lower = better | avg cosine similarity, higher = better |
| Cost | Railway plan | Free tier (only LLM API calls cost money) |

Not ported: the Gemini 1.5 models (retired by Google). Claude, OpenAI and Mistral
are supported; each shows up in the model picker only when its key is set.

## One-time setup

From this `worker/` folder:

```bash
npm install
npx wrangler login                     # opens the browser, sign in to Cloudflare
npx wrangler vectorize create ask-boris --dimensions=1024 --metric=cosine
npx wrangler secret put ANTHROPIC_API_KEY
```

Load the corpus. Create an API token in the Cloudflare dashboard
(My Profile → API Tokens → Create Token → Custom) with **Workers AI: Read** and
**Vectorize: Edit**, then:

```bash
export CLOUDFLARE_ACCOUNT_ID=...       # dashboard sidebar → Account ID
export CLOUDFLARE_API_TOKEN=...
python3 scripts/ingest.py              # embeds 701 docs, about a minute
python3 scripts/calibrate.py           # wait a minute after ingest first
```

Paste the two numbers `calibrate.py` suggests into `CONF_HIGH` and `CONF_MEDIUM`
in `wrangler.toml`. **Don't skip this**: the current values are guesses, and the
old 1.0 / 1.5 thresholds belonged to the old embedding model.

Deploy:

```bash
npm run deploy                         # prints https://ask-boris.<you>.workers.dev
```

## Cut over

1. Open the workers.dev URL, ask the suggested questions, try one in Swedish and one off-topic.
2. In the portfolio, swap the Railway URL for the new one.
3. Optional: put it on `ask.boriskehr.se` (uncomment `routes` in `wrangler.toml`; needs the domain on Cloudflare DNS).
4. When it has run fine for a few days, delete the Railway service.

## Adding documents

Add them to the corpus JSON files and run `python3 scripts/ingest.py` again. It upserts by id.

## Tests

`npm test` runs offline with fake Workers AI, Vectorize and Anthropic streams.
