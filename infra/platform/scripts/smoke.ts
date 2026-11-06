// Exercises a running platform end to end over its public API: registers a
// throwaway lead, asks it a question over MCP, waits for the turn, and reads
// back the reply, the doc it rewrote, the turn's usage and cost, and its
// transcript. Pauses the lead at the end. Needs PLATFORM_URL and
// PLATFORM_API_TOKEN.
//
//   smoke.ts [--lane smoke] [--harness fake|claude-code|codex] [--model M] [--wait-minutes 10]
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const url = process.env.PLATFORM_URL?.replace(/\/$/, "");
const token = process.env.PLATFORM_API_TOKEN;
if (!url || !token) {
  console.error("set PLATFORM_URL and PLATFORM_API_TOKEN");
  process.exit(2);
}
const args = process.argv.slice(2);
const flag = (name: string, dflt: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1]! : dflt;
};
const lane = flag("lane", "smoke");
const harness = flag("harness", "fake");
const model = flag("model", harness === "fake" ? "fake/fake" : harness === "codex" ? "openai/gpt-5.3-codex" : "anthropic/claude-sonnet-5.5");
const waitMinutes = Number(flag("wait-minutes", "10"));
const auth = { authorization: `Bearer ${token}`, "content-type": "application/json" };

async function api(method: "GET" | "POST", path: string, body?: unknown) {
  const r = await fetch(`${url}${path}`, { method, headers: auth, body: body === undefined ? undefined : JSON.stringify(body) });
  const text = await r.text();
  if (!r.ok) throw new Error(`${method} ${path}: ${r.status} ${text.slice(0, 500)}`);
  return JSON.parse(text);
}

const log = (step: string, v: unknown) => console.log(`${new Date().toISOString()} ${step} ${typeof v === "string" ? v : JSON.stringify(v)}`);

const t0 = Date.now();
const reg = await api("POST", "/api/leads", {
  lane,
  harness,
  model,
  status: "active",
  roll: true,
  charter: "You are a smoke-test lead. Keep every reply to one line.",
  doc: "# smoke workstream\n\nNothing yet.\n",
});
log("registered", { lane: reg.lead.lane, harness: reg.lead.harness, model: reg.lead.model, runId: reg.runId });

const client = new Client({ name: "platform-smoke", version: "0" });
await client.connect(new StreamableHTTPClientTransport(new URL(`${url}/api/mcp`), { requestInit: { headers: auth } }));
const call = async (name: string, a: Record<string, unknown>) => {
  const r = (await client.callTool({ name, arguments: a })) as { content: { text: string }[]; isError?: boolean };
  if (r.isError) throw new Error(`${name}: ${r.content[0]?.text}`);
  return JSON.parse(r.content[0]!.text);
};
log("mcp tools", (await client.listTools()).tools.map((x) => x.name));

const question = "Reply with exactly one line: PONG followed by the number of lines now in WORKSTREAM.md. Before replying, append the line `- smoke ok` to WORKSTREAM.md.";
let answer = await call("ask_lead", { lane, question, from: "smoke", wait_seconds: 45 });
log("asked", { eventId: answer.eventId, status: answer.status });
while (answer.status !== "finished" && answer.status !== "failed" && Date.now() - t0 < waitMinutes * 60_000) {
  await new Promise((r) => setTimeout(r, 5000));
  answer = await call("get_answer", { event_id: answer.eventId });
}
log("answer", { status: answer.status, turnId: answer.turnId, reply: answer.reply });

let failed = answer.status !== "finished";
if (answer.turnId) {
  const tr = await api("GET", `/api/transcripts?turn=${encodeURIComponent(answer.turnId)}`);
  const types: Record<string, number> = {};
  for (const p of tr.parts as { type: string }[]) types[p.type] = (types[p.type] ?? 0) + 1;
  log("turn", { status: tr.turn.status, usage: tr.turn.usage, costUsd: tr.turn.costUsd, error: tr.turn.error, parts: types });
  const hits = await call("search_transcripts", { query: "PONG", lane, limit: 5 });
  log("search PONG", { hits: hits.length });
  failed ||= hits.length === 0;
}
const doc = await api("GET", `/api/leads?lane=${encodeURIComponent(lane)}&doc=1`);
log("doc", { docRev: doc.docRev, hasSmokeLine: doc.doc.includes("smoke ok") });
const st = await call("status", { lane });
log("spend", st.leads[0]?.monthToDate);
await api("POST", "/api/leads", { lane, status: "paused" });
await client.close();
log("done", { ok: !failed, seconds: Math.round((Date.now() - t0) / 1000) });
process.exit(failed ? 1 : 0);
