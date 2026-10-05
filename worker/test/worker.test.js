// Offline test: fakes Workers AI, Vectorize, assets and the Anthropic stream.
import assert from "node:assert/strict";
import worker, { confidenceFor } from "../src/index.js";

const enc = new TextEncoder();
function sseBody(text, chunkSize) {
  return new ReadableStream({
    start(c) {
      for (let i = 0; i < text.length; i += chunkSize) c.enqueue(enc.encode(text.slice(i, i + chunkSize)));
      c.close();
    },
  });
}
const anthropicSSE = [
  'event: message_start\ndata: {"type":"message_start"}\n\n',
  'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hej "}}\n\n',
  'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"världen åäö"}}\n\n',
  'event: message_stop\ndata: {"type":"message_stop"}\n\n',
].join("");

let sentBody;
globalThis.fetch = async (url, init) => {
  assert.equal(url, "https://api.anthropic.com/v1/messages");
  assert.equal(init.headers["x-api-key"], "test-key");
  sentBody = JSON.parse(init.body);
  return new Response(sseBody(anthropicSSE, 7), { status: 200 });
};

function env(score) {
  return {
    ANTHROPIC_API_KEY: "test-key",
    CONF_HIGH: "0.55",
    CONF_MEDIUM: "0.45",
    AI: { run: async (m, { text }) => { assert.equal(m, "@cf/baai/bge-m3"); return { data: text.map(() => [0.1, 0.2]) }; } },
    VECTORIZE: { query: async () => ({ matches: [
      { score, metadata: { text: "Post A. Body", source: "post", title: "A", url: "https://x/a", date: "2024-01-02T00:00" } },
      { score, metadata: { text: "Dup", source: "post", title: "A", url: "https://x/a", date: "" } },
      { score, metadata: { text: "Interview text", source: "interview", url: "" } },
      { score, metadata: { text: "Interview 2", source: "interview", url: "" } },
    ] }) },
    ASSETS: { fetch: async () => new Response("asset") },
  };
}
const ask = (body, e, origin) => worker.fetch(new Request("https://w/ask", {
  method: "POST", headers: { "Content-Type": "application/json", ...(origin ? { Origin: origin } : {}) }, body: JSON.stringify(body),
}), e);

// thresholds
assert.equal(confidenceFor(0.6, {}), "high");
assert.equal(confidenceFor(0.5, {}), "medium");
assert.equal(confidenceFor(0.3, {}), "low");

// low confidence -> confirm
let r = await ask({ question: "lasagna?" }, env(0.2));
assert.deepEqual(await r.json(), { confirm_needed: true });

// high confidence -> SSE stream matching what chat.html parses
r = await ask({ question: "AI in design?" }, env(0.7), "https://boriskehr.se");
assert.match(r.headers.get("content-type"), /text\/event-stream/);
assert.equal(r.headers.get("access-control-allow-origin"), "https://boriskehr.se");
const text = await r.text();
const events = text.split("\n\n").filter(Boolean).map(l => l.slice(6));
assert.equal(events.at(-1), "[DONE]");
const first = JSON.parse(events[0]);
assert.equal(first.confidence, "high");
assert.deepEqual(first.sources, [
  { title: "A", date: "2024-01-02", url: "https://x/a", type: "post" },
  { title: "", date: "", url: null, type: "conversation" },
]);
const tokens = events.slice(1, -1).map(e => JSON.parse(e).token).join("");
assert.equal(tokens, "Hej världen åäö");
assert.match(sentBody.messages[0].content, /Retrieval confidence: high/);
assert.equal(sentBody.model, "claude-haiku-4-5-20251001");

// confirmed low -> medium
r = await ask({ question: "lasagna?", confirmed: true }, env(0.2));
assert.equal(JSON.parse((await r.text()).split("\n\n")[0].slice(6)).confidence, "medium");

// unknown origin gets no CORS header; models lists availability
r = await ask({ question: "x" }, env(0.2), "https://evil.example");
assert.equal(r.headers.get("access-control-allow-origin"), null);
r = await worker.fetch(new Request("https://w/models"), env(0.5));
const models = await r.json();
assert.equal(models["claude-haiku-4-5-20251001"].available, true);
assert.equal(models["gpt-4o-mini"].available, false);

// empty question, static fallthrough
r = await ask({ question: "  " }, env(0.7));
assert.equal(r.status, 400);
r = await worker.fetch(new Request("https://w/report"), env(0.5));
assert.equal(await r.text(), "asset");

console.log("All worker tests passed.");
