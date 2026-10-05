// Ask Boris — Cloudflare Worker version of server.py.
//
// Same API as the Flask app, so chat.html works unchanged:
//   GET  /models  -> model registry with availability
//   POST /ask     -> {confirm_needed:true} as JSON, or an SSE stream of
//                    {sources, confidence} then {token}... then [DONE]
// Static pages (chat at /, Ghosted report at /report) are served from ./public.
//
// Retrieval: Workers AI bge-m3 embeddings + Vectorize (cosine).
// bge-m3 is multilingual, so Swedish questions can match the translated
// English corpus directly.

const EMBED_MODEL = "@cf/baai/bge-m3";
const TOP_K = 10;

// input_cost / output_cost are USD per 1M tokens
const MODELS = {
  "claude-haiku-4-5-20251001": {
    name: "Claude Haiku 4.5",
    provider: "anthropic",
    input_cost: 1.0,
    output_cost: 5.0,
  },
  "gpt-4o-mini": {
    name: "GPT-4o mini",
    provider: "openai",
    input_cost: 0.15,
    output_cost: 0.6,
  },
  "mistral-small-latest": {
    name: "Mistral Small",
    provider: "mistral",
    input_cost: 0.2,
    output_cost: 0.6,
  },
  "mistral-large-latest": {
    name: "Mistral Large",
    provider: "mistral",
    input_cost: 2.0,
    output_cost: 6.0,
  },
};

const DEFAULT_MODEL = "claude-haiku-4-5-20251001";

const SYSTEM_PROMPT = `You are Boris Kehr — a designer and ML student based in Stockholm with 15+ years of experience in UX, product design, and AI/ML. You write and think in a direct, opinionated way. You believe design is about solving real problems, not aesthetics. You are skeptical of hype, including AI hype.

Use the retrieval confidence signal to frame your answer:

- If confidence is HIGH: answer directly and confidently from the context, in Boris's voice. No disclaimers needed.
- If confidence is MEDIUM: start your answer with "Based on related topics I've written about..." (or, only if the question is in Swedish, "Baserat på relaterade ämnen jag skrivit om..."). Build from the context plus reasonable extrapolation.
- If confidence is LOW: you will NOT receive this — the frontend handles it separately.

Never break character. Never say you are an AI or a language model. Always answer as Boris.

Language: default to English for your entire answer. The ONLY exception is when the user's question is itself written in Swedish — in that case, answer entirely in Swedish instead. Every other language (French, German, etc.) still gets an English answer. The retrieved context below is in English regardless of the question's language — that has no bearing on which language you answer in, which depends only on the question.
Keep answers focused and direct — Boris doesn't ramble.`;

// ── CORS ───────────────────────────────────────────────────────────────────
// Portfolio origins allowed to call /ask and /models from the browser.
const ALLOWED_ORIGINS = [
  "https://boriskehr.com",
  "https://www.boriskehr.com",
  "https://boriskehr.se",
  "https://www.boriskehr.se",
  "http://localhost:8080",
  "http://localhost:5173",
  /^https:\/\/.*\.lovable\.app$/,
  /^https:\/\/.*\.lovableproject\.com$/,
  /^https:\/\/.*\.pages\.dev$/,
];

function corsHeaders(request) {
  const origin = request.headers.get("Origin");
  if (!origin) return {};
  const ok = ALLOWED_ORIGINS.some((o) =>
    typeof o === "string" ? o === origin : o.test(origin)
  );
  if (!ok) return {};
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    Vary: "Origin",
  };
}

function json(data, status, extra) {
  return new Response(JSON.stringify(data), {
    status: status || 200,
    headers: { "Content-Type": "application/json", ...extra },
  });
}

// ── Providers ──────────────────────────────────────────────────────────────
function providerAvailable(provider, env) {
  if (provider === "anthropic") return Boolean(env.ANTHROPIC_API_KEY);
  if (provider === "openai") return Boolean(env.OPENAI_API_KEY);
  if (provider === "mistral") return Boolean(env.MISTRAL_API_KEY);
  return false;
}

// Reads an SSE response body and yields the parsed JSON of each data line.
async function* sseEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const raw = line.slice(5).trim();
      if (!raw || raw === "[DONE]") continue;
      try {
        yield JSON.parse(raw);
      } catch {
        // ignore partial or non-JSON lines
      }
    }
  }
}

async function upstreamError(res) {
  let detail = "";
  try {
    detail = (await res.text()).slice(0, 300);
  } catch {}
  return new Error(`Upstream ${res.status}: ${detail}`);
}

async function* streamTokens(provider, modelId, userPrompt, env) {
  if (provider === "anthropic") {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": env.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: modelId,
        max_tokens: 1000,
        system: SYSTEM_PROMPT,
        messages: [{ role: "user", content: userPrompt }],
        stream: true,
      }),
    });
    if (!res.ok) throw await upstreamError(res);
    for await (const ev of sseEvents(res.body)) {
      if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") {
        yield ev.delta.text;
      } else if (ev.type === "error") {
        throw new Error(ev.error?.message || "Anthropic stream error");
      }
    }
    return;
  }

  // OpenAI and Mistral share the same chat-completions streaming format.
  const endpoint =
    provider === "openai"
      ? "https://api.openai.com/v1/chat/completions"
      : "https://api.mistral.ai/v1/chat/completions";
  const key = provider === "openai" ? env.OPENAI_API_KEY : env.MISTRAL_API_KEY;
  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${key}`,
    },
    body: JSON.stringify({
      model: modelId,
      max_tokens: 1000,
      stream: true,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt },
      ],
    }),
  });
  if (!res.ok) throw await upstreamError(res);
  for await (const ev of sseEvents(res.body)) {
    const content = ev.choices?.[0]?.delta?.content;
    if (content) yield content;
  }
}

// ── Retrieval ──────────────────────────────────────────────────────────────
// Confidence is the average cosine similarity of the top 10 matches.
// Higher is better (the old Chroma version used L2 distance, lower is better).
// Calibrate these with scripts/calibrate.py and set them in wrangler.toml.
export function confidenceFor(avgScore, env) {
  const high = parseFloat(env.CONF_HIGH ?? "0.55");
  const medium = parseFloat(env.CONF_MEDIUM ?? "0.45");
  if (avgScore >= high) return "high";
  if (avgScore >= medium) return "medium";
  return "low";
}

export function buildContextAndSources(matches) {
  let context = "";
  const sources = [];
  const seenSourceTypes = new Set();
  const seenUrls = new Set();
  const seenTitles = new Set();

  for (const m of matches) {
    const meta = m.metadata || {};
    const doc = meta.text || "";
    context += `\n---\n${doc}\n`;
    const sourceType = meta.source || "";

    if (meta.url) {
      const url = meta.url;
      let title = (meta.title || "").trim();
      if (!title) title = doc.split(".")[0].trim().slice(0, 80);
      if (seenUrls.has(url) || seenTitles.has(title)) continue;
      seenUrls.add(url);
      seenTitles.add(title);
      sources.push({
        title,
        date: (meta.date || "").slice(0, 10),
        url,
        type: "post",
      });
    } else if (sourceType === "interview" && !seenSourceTypes.has("interview")) {
      sources.push({ title: "", date: "", url: null, type: "conversation" });
      seenSourceTypes.add("interview");
    } else if (sourceType === "notes" && !seenSourceTypes.has("notes")) {
      sources.push({ title: "", date: "", url: null, type: "note" });
      seenSourceTypes.add("notes");
    }
  }
  return { context, sources };
}

async function retrieve(question, env) {
  const emb = await env.AI.run(EMBED_MODEL, { text: [question] });
  const vector = emb.data[0];
  const result = await env.VECTORIZE.query(vector, {
    topK: TOP_K,
    returnMetadata: "all",
  });
  const matches = result.matches || [];
  const avgScore = matches.length
    ? matches.reduce((s, m) => s + m.score, 0) / matches.length
    : 0;
  const { context, sources } = buildContextAndSources(matches);
  return { context, sources, confidence: confidenceFor(avgScore, env), avgScore };
}

// ── Routes ─────────────────────────────────────────────────────────────────
function handleModels(env, cors) {
  const available = {};
  for (const [id, info] of Object.entries(MODELS)) {
    available[id] = { ...info, available: providerAvailable(info.provider, env) };
  }
  return json(available, 200, cors);
}

async function handleAsk(request, env, cors) {
  let data;
  try {
    data = await request.json();
  } catch {
    return json({ error: "Invalid JSON" }, 400, cors);
  }

  const question = String(data.question || "").trim().slice(0, 2000);
  const confirmed = Boolean(data.confirmed);
  let modelId = data.model || DEFAULT_MODEL;
  if (!MODELS[modelId]) modelId = DEFAULT_MODEL;
  const provider = MODELS[modelId].provider;

  if (!question) return json({ error: "No question provided" }, 400, cors);
  if (!providerAvailable(provider, env)) {
    return json({ error: `${MODELS[modelId].name} is not configured` }, 400, cors);
  }

  const { context, sources, confidence } = await retrieve(question, env);

  if (confidence === "low" && !confirmed) {
    return json({ confirm_needed: true }, 200, cors);
  }

  const reported = confirmed ? "medium" : confidence;
  const userPrompt = `Retrieval confidence: ${reported}
Context from Boris's writing:
${context}

Question: ${question}`;

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const enc = new TextEncoder();
  const send = (obj) => writer.write(enc.encode(`data: ${JSON.stringify(obj)}\n\n`));

  (async () => {
    try {
      await send({ sources: sources.slice(0, 3), confidence: reported });
      for await (const token of streamTokens(provider, modelId, userPrompt, env)) {
        await send({ token });
      }
    } catch (e) {
      await send({ error: String(e.message || e) });
    }
    await writer.write(enc.encode("data: [DONE]\n\n"));
    await writer.close();
  })();

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      ...cors,
    },
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const cors = corsHeaders(request);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors });
    }
    if (url.pathname === "/models" && request.method === "GET") {
      return handleModels(env, cors);
    }
    if (url.pathname === "/ask" && request.method === "POST") {
      try {
        return await handleAsk(request, env, cors);
      } catch (e) {
        return json({ error: "Something went wrong." }, 500, cors);
      }
    }
    // Everything else: static pages (chat at /, report at /report).
    return env.ASSETS.fetch(request);
  },
};
