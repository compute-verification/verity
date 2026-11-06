// Exercises a running platform end to end over its public API: registers a
// throwaway lead, asks it a question over MCP, waits for the turn, and reads
// back the reply, the doc it rewrote, the turn's usage and cost, and its
// transcript. Pauses the lead at the end. Needs PLATFORM_URL and
// PLATFORM_API_TOKEN.
//
// With --signed it also signs the requests Slack and Linear would send (with
// SLACK_SIGNING_SECRET and LINEAR_WEBHOOK_SECRET): a top-level post and the
// lead's own post in its inbox wake nothing, a doorbell wakes it and its reply
// is addressed to the doorbell's thread, and an issue of its team wakes it.
// The channel is made up, so a deployed platform's reply is refused by Slack
// (a post-error part) rather than posted anywhere.
//
//   smoke.ts [--lane smoke] [--harness fake|claude-code|codex] [--model M] [--wait-minutes 10] [--signed]
import { createHmac } from "node:crypto";
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
const signed = args.includes("--signed");
const slackSecret = process.env.SLACK_SIGNING_SECRET;
const linearSecret = process.env.LINEAR_WEBHOOK_SECRET;
if (signed && (!slackSecret || !linearSecret)) {
  console.error("--signed needs SLACK_SIGNING_SECRET and LINEAR_WEBHOOK_SECRET");
  process.exit(2);
}
const CHANNEL = "C0SMOKE";
const TEAM = "SMOKE";
const unix = () => Math.floor(Date.now() / 1000);
const inbox = `${unix()}.000001`;
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
  ...(signed && { slackChannel: CHANNEL, slackInbox: inbox, linearTeam: TEAM }),
});
log("registered", { lane: reg.lead.lane, agentId: reg.lead.agentId, harness: reg.lead.harness, model: reg.lead.model, runId: reg.runId });

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

if (signed) {
  const slack = async (event: Record<string, unknown>) => {
    const ts = String(unix());
    const body = JSON.stringify({ type: "event_callback", event_id: `Ev${Math.random().toString(36).slice(2)}`, event: { type: "message", channel: CHANNEL, ts: `${ts}.${String(Math.random()).slice(2, 8)}`, ...event } });
    const sig = `v0=${createHmac("sha256", slackSecret!).update(`v0:${ts}:${body}`).digest("hex")}`;
    const r = await fetch(`${url}/api/slack/events`, { method: "POST", headers: { "x-slack-request-timestamp": ts, "x-slack-signature": sig, "content-type": "application/json" }, body });
    return ((await r.json()) as { routed: string[] }).routed;
  };
  // The turn whose prompt holds `word`, once it has finished, with its parts.
  type Part = { type: string; content: { threadTs?: string } | null; text: string };
  const turnWith = async (word: string) => {
    while (Date.now() - t0 < waitMinutes * 60_000) {
      const hits = (await api("GET", `/api/transcripts?lane=${encodeURIComponent(lane)}&q=${word}&limit=5`)).hits as { turnId: string; type: string }[];
      const id = hits.find((h) => h.type === "prompt")?.turnId;
      const tr = id ? await api("GET", `/api/transcripts?turn=${encodeURIComponent(id)}`) : null;
      if (tr && tr.turn.status !== "running") return { id: id!, reply: tr.turn.finalText as string, parts: tr.parts as Part[] };
      await new Promise((r) => setTimeout(r, 2000));
    }
    return null;
  };
  const word = (what: string) => `${what}${Date.now().toString(36)}`;
  const ignored = [
    await slack({ text: `@${lane} a top-level post`, username: "smoke" }),
    await slack({ text: "the lead's own post", thread_ts: inbox, username: lane }),
  ];
  log("ignored", ignored);
  failed ||= ignored.some((r) => r.length > 0);

  const thread = `${unix()}.000042`;
  const ding = word("ding");
  const rang = await slack({ text: `from smoke, thread ${thread}: @${lane} reply with one line: ${ding}`, thread_ts: inbox, username: "smoke" });
  const bell = await turnWith(ding);
  const addressed = (bell?.parts ?? []).filter((p) => (p.type === "post" || p.type === "post-error") && p.content?.threadTs === thread);
  log("doorbell", { routed: rang, turnId: bell?.id, reply: bell?.reply, addressed: addressed.map((p) => ({ type: p.type, text: p.text.slice(0, 120) })) });
  failed ||= rang[0] !== lane || addressed.length !== 1;

  const title = word("issue");
  const lbody = JSON.stringify({ type: "Issue", action: "create", webhookTimestamp: Date.now(), url: `https://linear.app/x/issue/${TEAM}-1`, data: { identifier: `${TEAM}-1`, title, team: { key: TEAM }, state: { name: "Todo" } } });
  const lres = await fetch(`${url}/api/linear/webhook`, { method: "POST", headers: { "linear-signature": createHmac("sha256", linearSecret!).update(lbody).digest("hex"), "linear-delivery": `smoke-${Date.now()}` }, body: lbody });
  const lrouted = ((await lres.json()) as { routed: string[] }).routed;
  const issue = await turnWith(title);
  log("linear", { routed: lrouted, turnId: issue?.id });
  failed ||= lrouted[0] !== lane || !issue;
}
await api("POST", "/api/leads", { lane, status: "paused" });
await client.close();
log("done", { ok: !failed, seconds: Math.round((Date.now() - t0) / 1000) });
process.exit(failed ? 1 : 0);
