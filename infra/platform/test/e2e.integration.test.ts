import { execFileSync } from "node:child_process";
import { mkdtemp, readdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { beforeAll, describe, expect, it } from "vitest";
import { getRun, start } from "workflow/api";
import { tick } from "../src/cron.ts";
import { changeLead, jobCallback, slackEvents, status } from "../src/http.ts";
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
    expect(await readFile(join(t.root, "lead-infra", ".gitconfig"), "utf8")).toContain("password=platform-github-placeholder");
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

  it("registers, rings, rebudgets, replaces the doc of and pauses a lead through the admin endpoint", async () => {
    const admin = (body: object, token = API.PLATFORM_API_TOKEN) =>
      changeLead(
        new Request("http://platform.test/api/leads", { method: "POST", headers: { authorization: `Bearer ${token}` }, body: JSON.stringify(body) }),
        API,
      );
    const get = async (q: string) =>
      (await status(new Request(`http://platform.test/api/leads?${q}`, { headers: { authorization: `Bearer ${API.PLATFORM_API_TOKEN}` } }), API)).json();
    expect((await admin({ lane: "comms" }, "wrong")).status).toBe(401);
    expect((await admin({ lane: "Not A Lane" })).status).toBe(400);

    const reg = await (
      await admin({
        lane: "comms",
        slackChannel: "C_COMMS",
        harness: "fake",
        model: "fake/fake",
        charter: "Keep the public story true.",
        doc: "# comms workstream\n",
        monthlyBudgetUsd: 200,
        status: "active",
        roll: true,
        kick: "You are on the platform now.",
      })
    ).json();
    expect(reg.runId).toBeTruthy();
    expect(reg.lead).toMatchObject({ lane: "comms", gatewayKeyId: "local-key-comms", docBytes: 19 });
    await t.slack.waitFor((p) => p.channel === "C_COMMS" && !p.threadTs && p.text.includes("mcp/cutover"));
    expect((await t.store.getLead("comms"))!.runId).toBe(reg.runId);

    await admin({ lane: "comms", monthlyBudgetUsd: 300, doc: "# comms, from the old side\n" });
    const keys = (await t.store.listGrants("comms")).filter((g) => g.kind === "ai-gateway-key");
    expect(keys.map((g) => g.scope).sort()).toEqual(["monthly $200", "monthly $300"]);
    const now = await get("lane=comms&doc=1");
    expect(now.doc).toBe("# comms, from the old side\n");
    expect(now.revisions.map((r: { turnId: string }) => r.turnId)).toContain("admin");
    expect((await get("lane=comms&doc=1&rev=0")).doc).toBe("# comms workstream\n");

    expect((await (await admin({ lane: "comms", status: "paused" })).json()).lead.status).toBe("paused");
    expect(await t.store.leadByChannel("C_COMMS")).toBeNull();
  });

  it("gives a lead a checkout that lasts between turns and a worker one on its own branch", async () => {
    const env = { ...process.env, GIT_CONFIG_GLOBAL: "/dev/null", GIT_CONFIG_NOSYSTEM: "1" };
    const git = (cwd: string, ...args: string[]) => execFileSync("git", ["-c", "user.name=t", "-c", "user.email=t@t", ...args], { cwd, env, encoding: "utf8" }).trim();
    const seed = await mkdtemp(join(tmpdir(), "platform-seed-"));
    git(seed, "init", "-q", "-b", "main");
    await writeFile(join(seed, "README.md"), "seed\n");
    git(seed, "add", "README.md");
    git(seed, "commit", "-q", "-m", "seed");
    t.rt.repoUrl = `file://${seed}`;
    try {
      await t.store.upsertLead({ lane: "circuits", model: "fake/fake", harness: "fake", slackChannel: "C_CIRCUITS", doc: "# circuits\n" });
      const first = await say("C_CIRCUITS", "[[dispatch: add a pin]]");
      await reply(first.thread, "worker/worker-done");
      const lead = join(t.root, "lead-circuits", "verity");
      expect(await readFile(join(lead, "README.md"), "utf8")).toBe("seed\n");
      await writeFile(join(lead, "scratch.txt"), "kept\n");
      git(seed, "commit", "-q", "--allow-empty", "-m", "second");

      const second = await say("C_CIRCUITS", "anything new?");
      await reply(second.thread, "slack/message");
      expect(await readFile(join(lead, "scratch.txt"), "utf8")).toBe("kept\n");
      expect(git(lead, "log", "-1", "--format=%s", "origin/main")).toBe("second");

      const [w] = await t.store.listWorkers({ lane: "circuits" });
      expect(git(join(t.root, `worker-${w!.id}`, "verity"), "branch", "--show-current")).toBe(w!.branch);
      const [turn] = (await t.store.listTurns({ lane: "circuits" })).filter((x) => x.kind === "lead");
      const parts = await t.store.transcript(turn!.id);
      expect(parts.find((p) => p.type === "instructions")!.text).toContain("verity/ is your checkout");
    } finally {
      t.rt.repoUrl = null;
    }
  });
});
