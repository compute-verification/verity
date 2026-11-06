import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { beforeAll, describe, expect, it } from "vitest";
import { getRun, start } from "workflow/api";
import { tick } from "../src/cron.ts";
import { jobCallback, slackEvents, status } from "../src/http.ts";
import { mcpHandler } from "../src/mcp.ts";
import { type LeadResult, leadWorkflow } from "../workflows/lead.ts";
import { NOW, signedSlackRequest, slackBody, slackEnv, testRuntime } from "./helpers.ts";

// The whole platform in one process: real Workflow DevKit runs (the local world),
// the real HarnessAgent driving a scripted harness in directory sandboxes,
// PGlite for the record, and in-memory Slack and job queue.

let t: Awaited<ReturnType<typeof testRuntime>>;
const API = { PLATFORM_API_TOKEN: "test-api-token" };

async function say(channel: string, text: string, threadTs?: string) {
  const { ts, body } = slackBody({ channel, text, threadTs });
  const res = await slackEvents(signedSlackRequest(body), slackEnv, NOW);
  return { ts, thread: threadTs ?? ts, body, res: (await res.json()) as { ok: boolean; routed: string[] } };
}

const reply = (thread: string, has: string) => t.slack.waitFor((p) => p.threadTs === thread && p.text.includes(has));

async function filesUnder(dir: string): Promise<string[]> {
  const out: string[] = [];
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...(await filesUnder(p)));
    else out.push(p);
  }
  return out;
}

beforeAll(async () => {
  t = await testRuntime();
  await t.store.upsertLead({ lane: "infra", model: "fake/fake", harness: "fake", slackChannel: "C_INFRA", doc: "# infra workstream\n", monthlyBudgetUsd: 400 });
  await t.store.upsertLead({ lane: "proofs", model: "fake/fake", harness: "fake", slackChannel: "C_PROOFS", doc: "# proofs workstream\n" });
});

describe("a lead on the platform", () => {
  it("wakes on a Slack message, runs a harness turn, updates its doc and replies in the thread", async () => {
    const m = await say("C_INFRA", "status of the queue?");
    expect(m.res).toEqual({ ok: true, routed: ["infra"] });
    const post = await reply(m.thread, "slack/message");
    expect(post.channel).toBe("C_INFRA");
    const lead = (await t.store.getLead("infra"))!;
    expect(lead.docRev).toBe(1);
    expect(lead.doc).toContain("handled slack/message");
    expect(lead.runId).toBeTruthy();
    expect((await t.store.docRevisions("infra")).map((r) => r.rev)).toEqual([1]);
  });

  it("routes by mention, then by thread", async () => {
    const m = await say("C_GENERAL", "@proofs is the lock current?");
    expect(m.res.routed).toEqual(["proofs"]);
    await reply(m.thread, "slack/message");
    const followUp = await say("C_GENERAL", "and the audit?", m.thread);
    expect(followUp.res.routed).toEqual(["proofs"]);
    await t.slack.waitFor(() => t.slack.posts.filter((q) => q.threadTs === m.thread).length >= 2);
    expect((await t.store.getLead("proofs"))!.docRev).toBe(2);
  });

  it("refuses an unsigned request and records a redelivered event once", async () => {
    const bad = await slackEvents(signedSlackRequest(slackBody({ channel: "C_INFRA", text: "x" }).body, "wrong-secret"), slackEnv, NOW);
    expect(bad.status).toBe(401);
    const m = await say("C_INFRA", "once only");
    await slackEvents(signedSlackRequest(m.body), slackEnv, NOW);
    const { rows } = await t.store.db.query(`select count(*)::int as n from events where payload->>'text' = 'once only'`);
    expect(rows[0]!.n).toBe(1);
    await reply(m.thread, "slack/message");
  });

  it("wakes on a cron tick and posts in its channel", async () => {
    await t.store.db.query(`update leads set cron_every_minutes = 60, next_tick_at = now() - interval '1 minute' where lane = 'infra'`);
    const r = await tick();
    expect(r.ticked).toEqual(["infra"]);
    const post = await t.slack.waitFor((p) => p.channel === "C_INFRA" && p.threadTs === undefined && p.text.includes("cron/tick"));
    expect(post.text).toContain("Handled 1 event(s)");
    expect((await tick()).ticked).toEqual([]);
  });

  it("submits a job and is woken in the same thread by its callback", async () => {
    const m = await say("C_INFRA", "run the bench [[job: uv run bench.py --quick]]");
    await reply(m.thread, "submit_job");
    const sub = t.jobs.submitted.at(-1)!;
    expect(sub.spec.command).toBe("uv run bench.py --quick");
    const token = sub.callbackUrl.split("/").pop()!;
    const cb = () => jobCallback(new Request(sub.callbackUrl, { method: "POST", body: JSON.stringify({ status: "done", summary: "finished" }) }), token);
    expect(await (await cb()).json()).toMatchObject({ ok: true, duplicate: false });
    expect(await (await cb()).json()).toMatchObject({ ok: true, duplicate: true });
    expect((await jobCallback(new Request("http://platform.test/x", { method: "POST", body: "{}" }), "not-a-token")).status).toBe(404);
    await reply(m.thread, "job/job-done");
    expect((await t.store.getJob(sub.id))!.status).toBe("done");
  });

  it("dispatches a worker and is woken in the same thread when it finishes", async () => {
    const m = await say("C_INFRA", "[[dispatch: fix the flaky test in tests/test_clock.py]]");
    await reply(m.thread, "dispatch_worker");
    await reply(m.thread, "worker/worker-done");
    const [w] = await t.store.listWorkers({ lane: "infra" });
    expect(w).toMatchObject({ status: "finished", task: "fix the flaky test in tests/test_clock.py" });
    expect(w!.branch).toMatch(/^cursor\/infra-[0-9a-f]{6}$/);
    const turns = await t.store.listTurns({ lane: "infra" });
    expect(turns.some((x) => x.kind === "worker" && x.status === "finished")).toBe(true);
  });

  it("asks another lead and gets the answer back in the thread it asked from", async () => {
    const m = await say("C_INFRA", "[[ask proofs: which guarantees moved this week?]]");
    await reply(m.thread, "ask_lead");
    const answer = await reply(m.thread, "lead/answer");
    expect(answer.channel).toBe("C_INFRA");
    expect(t.slack.posts.some((p) => p.channel === "C_PROOFS" && p.text.includes("lead/ask"))).toBe(false);
  });

  it("serves ask_lead, assign, status and search over MCP", async () => {
    expect((await mcpHandler(new Request("http://platform.test/api/mcp", { method: "POST", body: "{}" }), API)).status).toBe(401);
    const client = new Client({ name: "e2e", version: "0" });
    await client.connect(
      new StreamableHTTPClientTransport(new URL("http://platform.test/api/mcp"), {
        fetch: (url, init) => mcpHandler(new Request(url, init), API),
        requestInit: { headers: { authorization: `Bearer ${API.PLATFORM_API_TOKEN}` } },
      }),
    );
    const names = (await client.listTools()).tools.map((x) => x.name).sort();
    expect(names).toEqual(["ask_lead", "assign", "get_answer", "search_transcripts", "status"]);
    const call = async (name: string, args: Record<string, unknown>) => {
      const r = (await client.callTool({ name, arguments: args })) as { content: { text: string }[] };
      return JSON.parse(r.content[0]!.text);
    };

    const asked = await call("ask_lead", { lane: "proofs", question: "what is open?", from: "daniel", wait_seconds: 30 });
    expect(asked).toMatchObject({ lane: "proofs", status: "finished" });
    expect(asked.reply).toContain("mcp/ask");
    expect(await call("get_answer", { event_id: asked.eventId })).toMatchObject({ status: "finished", reply: asked.reply });

    await call("assign", { lane: "proofs", task: "tidy the ledger" });
    await t.slack.waitFor((p) => p.channel === "C_PROOFS" && p.threadTs === undefined && p.text.includes("mcp/assign"));

    const st = await call("status", { lane: "infra" });
    const infra = st.leads[0];
    expect(infra.lane).toBe("infra");
    expect(infra.monthToDate.turns).toBeGreaterThan(5);
    expect(infra.monthToDate.budgetUsd).toBe(400);
    expect(infra.monthToDate.byKind.map((k: { kind: string }) => k.kind).sort()).toEqual(["lead", "worker"]);

    const hits = await call("search_transcripts", { query: "flaky", lane: "infra" });
    expect(hits.length).toBeGreaterThan(0);
    expect(hits.every((h: { lane: string }) => h.lane === "infra")).toBe(true);
    await client.close();
  });

  it("stores every turn's transcript and usage, queryable by lead, time and text", async () => {
    const all = await t.store.listTurns({});
    expect(all.length).toBeGreaterThan(8);
    expect(all.every((x) => x.status === "finished" && x.usage.inputTokens > 0 && x.usage.outputTokens > 0)).toBe(true);
    const dispatch = (await t.store.searchTranscripts({ text: "dispatch_worker", lane: "infra" })).find((h) => h.type === "tool-call")!;
    const parts = await t.store.transcript(dispatch.turnId);
    expect(parts.map((p) => p.type)).toEqual(expect.arrayContaining(["instructions", "prompt", "tool-call", "tool-result", "text", "finish", "post"]));
    expect(await t.store.searchTranscripts({ text: "flaky", lane: "proofs" })).toEqual([]);
    expect(await t.store.searchTranscripts({ text: "flaky", since: new Date(Date.now() + 3600_000).toISOString() })).toEqual([]);
    const res = await status(new Request("http://platform.test/api/leads", { headers: { authorization: `Bearer ${API.PLATFORM_API_TOKEN}` } }), API);
    const body = (await res.json()) as { leads: { lane: string; monthToDate: { turns: number } }[] };
    expect(body.leads.map((l) => l.lane)).toEqual(["infra", "proofs"]);
  });

  it("issues credentials centrally and never writes them into a sandbox or a transcript", async () => {
    const grants = await t.store.listGrants("infra");
    expect(grants.length).toBeGreaterThan(0);
    expect(grants.every((g) => g.kind === "github-app-token" && !g.ref.includes("local-github-token"))).toBe(true);
    const workerGrants = (await t.store.listGrants()).filter((g) => g.subject.startsWith("infra-w-"));
    expect(workerGrants.length).toBe(1);
    for (const f of await filesUnder(t.root)) expect(await readFile(f, "utf8")).not.toContain("local-github-token");
    const { rows } = await t.store.db.query(`select count(*)::int as n from transcript_parts where text like '%local-github-token%'`);
    expect(rows[0]!.n).toBe(0);
  });

  it("hands the lead to a newer run, which keeps answering", async () => {
    const old = (await t.store.getLead("infra"))!.runId!;
    const run = await start(leadWorkflow, [{ lane: "infra", epoch: 0 }]);
    const ended = (await getRun<LeadResult>(old).returnValue) as LeadResult;
    expect(ended).toMatchObject({ ended: "superseded", by: run.runId });
    const m = await say("C_INFRA", "still there?");
    await reply(m.thread, "slack/message");
    expect((await t.store.getLead("infra"))!.runId).toBe(run.runId);
  });
});
