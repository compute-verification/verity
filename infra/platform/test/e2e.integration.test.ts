import { execFileSync } from "node:child_process";
import { mkdtemp, readdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { beforeAll, describe, expect, it } from "vitest";
import { getRun, start } from "workflow/api";
import { tick } from "../src/cron.ts";
import { changeLead, jobCallback, linearEvents, slackEvents, status } from "../src/http.ts";
import { signLinearBody } from "../src/linear.ts";
import { mcpHandler } from "../src/mcp.ts";
import { leadAgentId } from "../src/slack.ts";
import { type LeadResult, leadWorkflow } from "../workflows/lead.ts";
import { NOW, nextTs, signedSlackRequest, slackBody, slackEnv, testRuntime } from "./helpers.ts";

// The whole platform in one process: real Workflow DevKit runs (the local world),
// the real HarnessAgent driving a scripted harness in directory sandboxes,
// PGlite for the record, and in-memory Slack and job queue.

let t: Awaited<ReturnType<typeof testRuntime>>;
const API = { PLATFORM_API_TOKEN: "test-api-token" };
const COORD = "C_COORD";
const INBOX: Record<string, string> = {
  infra: "1700000000.000001",
  proofs: "1700000000.000002",
  comms: "1700000000.000003",
  circuits: "1700000000.000004",
};

async function event(o: { text: string; threadTs?: string; channel?: string; username?: string }) {
  const { ts, body } = slackBody({ channel: o.channel ?? COORD, text: o.text, threadTs: o.threadTs, botId: "BVERITY", username: o.username });
  const res = await slackEvents(signedSlackRequest(body), slackEnv, NOW);
  return { ts, body, res: (await res.json()) as { ok: boolean; routed: string[] } };
}

// What `research msg` posts in `lane`'s inbox thread for a post addressed to it
// in the front door (or in `channel`): a doorbell naming the post's thread.
async function ring(lane: string, text: string, o: { thread?: string; channel?: string; from?: string } = {}) {
  const thread = o.thread ?? nextTs();
  const where = `${o.channel ? `#${o.channel} ` : ""}thread ${thread}`;
  const e = await event({ text: `from ${o.from ?? "daniel"}, ${where}: @${lane} ${text}`, threadTs: INBOX[lane], username: "router" });
  return { ...e, thread };
}

const reply = (thread: string, has: string) => t.slack.waitFor((p) => p.threadTs === thread && p.text.includes(has));
const turnOf = (lane: string, has: string) => t.turnEnded((x) => x.lane === lane && x.kind === "lead" && (x.finalText ?? "").includes(has));

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
  for (const lane of ["infra", "proofs"]) {
    await t.store.upsertLead({ lane, model: "fake/fake", harness: "fake", slackChannel: COORD, slackInbox: INBOX[lane], doc: `# ${lane} workstream\n` });
  }
  await t.store.upsertLead({ lane: "infra", model: "fake/fake", monthlyBudgetUsd: 400 });
});

describe("a lead on the platform", () => {
  it("wakes on a doorbell in its inbox, runs a harness turn, updates its doc and replies in the post's thread", async () => {
    const m = await ring("infra", "status of the queue?");
    expect(m.res).toEqual({ ok: true, routed: ["infra"] });
    const post = await reply(m.thread, "slack/doorbell");
    expect(post).toMatchObject({ channel: COORD, username: "infra" });
    const lead = (await t.store.getLead("infra"))!;
    expect(lead.docRev).toBe(1);
    expect(lead.doc).toContain("handled slack/doorbell");
    expect(lead.runId).toBeTruthy();
    expect((await t.store.docRevisions("infra")).map((r) => r.rev)).toEqual([1]);
  });

  it("wakes only on its inbox: research msg decides who a post is for, and nothing the lead or the platform posts rings it", async () => {
    const elsewhere = await ring("proofs", "is the lock current?", { channel: "ask-the-team" });
    expect(elsewhere.res.routed).toEqual(["proofs"]);
    expect(await reply(elsewhere.thread, "slack/doorbell")).toMatchObject({ channel: "ask-the-team", username: "proofs" });

    expect((await event({ text: "<!subteam^SPROOFS|@proofs> a top-level post, rung by its sender" })).res.routed).toEqual([]);
    expect((await event({ text: "@proofs a reply in some other thread", threadTs: "1700000999.000001" })).res.routed).toEqual([]);
    expect((await event({ text: "from proofs, thread 1.2: x", threadTs: INBOX.proofs, username: "proofs" })).res.routed).toEqual([]);
    expect((await event({ text: "This turn failed", threadTs: INBOX.proofs, username: "platform" })).res.routed).toEqual([]);

    const typed = await event({ text: "a line someone typed in the inbox", threadTs: INBOX.proofs });
    expect(typed.res.routed).toEqual(["proofs"]);
    const turn = await turnOf("proofs", "slack/message");
    expect((await t.store.transcript(turn.id)).some((p) => p.type === "post")).toBe(false);
  });

  it("wakes the lead of an issue's team when the issue is created or changes state, and on nothing else", async () => {
    const env = { LINEAR_WEBHOOK_SECRET: "test-linear-secret" };
    const nowMs = NOW * 1000;
    const send = async (o: Record<string, unknown>, sign = env.LINEAR_WEBHOOK_SECRET, stamp = nowMs) => {
      const body = JSON.stringify({ type: "Issue", url: "https://linear.app/x/issue/INF-7", webhookTimestamp: stamp, ...o });
      const req = new Request("http://platform.test/api/linear/webhook", { method: "POST", headers: { "linear-signature": signLinearBody(sign, body), "linear-delivery": `d-${nextTs()}` }, body });
      const res = await linearEvents(req, env, nowMs);
      return { status: res.status, routed: ((await res.json()) as { routed?: string[] }).routed };
    };
    await t.store.upsertLead({ lane: "infra", model: "fake/fake", linearTeam: "INF" });
    const issue = { identifier: "INF-7", title: "node 2 is out of disk", team: { key: "INF" }, state: { name: "Todo" } };
    expect(await send({ action: "create", data: issue }, "wrong")).toMatchObject({ status: 401 });
    expect(await send({ action: "create", data: issue }, env.LINEAR_WEBHOOK_SECRET, nowMs - 120_000)).toMatchObject({ status: 401 });
    expect(await send({ action: "create", data: issue })).toEqual({ status: 200, routed: ["infra"] });
    expect((await turnOf("infra", "linear/issue-created")).finalText).toContain("Handled");
    expect(await send({ action: "update", data: { ...issue, state: { name: "Done" } }, updatedFrom: { stateId: "s1" } })).toEqual({ status: 200, routed: ["infra"] });
    expect(await send({ action: "update", data: { ...issue, title: "renamed" }, updatedFrom: { title: "old" } })).toEqual({ status: 200, routed: [] });
    expect(await send({ type: "Comment", action: "create", data: { body: "hi" } })).toEqual({ status: 200, routed: [] });
    expect(await send({ action: "create", data: { ...issue, identifier: "PRF-1", team: { key: "PRF" } } })).toEqual({ status: 200, routed: [] });
    await turnOf("infra", "linear/issue-state");
  });

  it("refuses an unsigned request and records a redelivered event once", async () => {
    const bad = await slackEvents(signedSlackRequest(slackBody({ channel: COORD, text: "x", threadTs: INBOX.infra }).body, "wrong-secret"), slackEnv, NOW);
    expect(bad.status).toBe(401);
    const m = await ring("infra", "once only");
    await slackEvents(signedSlackRequest(m.body), slackEnv, NOW);
    const { rows } = await t.store.db.query(`select count(*)::int as n from events where payload->>'text' like '%once only'`);
    expect(rows[0]!.n).toBe(1);
    await reply(m.thread, "slack/doorbell");
  });

  it("hands a reply that can't be posted back to the lead, once", async () => {
    const real = t.rt.reply;
    t.rt.reply = async () => {
      throw new Error("research msg reply: exit 1: refused: 130 words");
    };
    try {
      const m = await ring("infra", "a long answer, please");
      await turnOf("infra", "slack/reply-failed");
      const turns = await t.store.listTurns({ lane: "infra" });
      const errors = (await Promise.all(turns.map((x) => t.store.transcript(x.id)))).flat().filter((p) => p.type === "post-error");
      expect(errors.map((p) => p.content)).toEqual([{ channel: COORD, threadTs: m.thread }]);
      expect(errors[0]!.text).toContain("130 words");
      const { rows } = await t.store.db.query(`select count(*)::int as n from events where kind = 'reply-failed'`);
      expect(rows[0]!.n).toBe(1);
    } finally {
      t.rt.reply = real;
    }
  });

  it("wakes on a cron tick and posts nothing", async () => {
    await t.store.db.query(`update leads set cron_every_minutes = 60, next_tick_at = now() - interval '1 minute' where lane = 'infra'`);
    const r = await tick();
    expect(r.ticked).toEqual(["infra"]);
    const turn = await turnOf("infra", "cron/tick");
    expect(turn.finalText).toContain("Handled 1 event(s)");
    expect(t.slack.posts.some((p) => p.text.includes("cron/tick"))).toBe(false);
    expect((await tick()).ticked).toEqual([]);
  });

  it("submits a job and is woken in the same thread by its callback", async () => {
    const m = await ring("infra", "run the bench [[job: uv run bench.py --quick]]");
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

  it("learns of a job that ends without calling back from the queue, once", async () => {
    const m = await ring("infra", "[[job: lake build Security]]");
    await reply(m.thread, "submit_job");
    const sub = t.jobs.submitted.at(-1)!;
    const { externalId } = (await t.store.getJob(sub.id))!;
    expect((await tick()).jobsEnded).toEqual([]);
    t.jobs.end(externalId!, "dead", { outcome: "error", summary: "lease expired 3 times" });
    expect((await tick()).jobsEnded).toEqual([sub.id]);
    const post = await reply(m.thread, "job/job-done");
    expect(post.text).toContain("Handled 1 event(s)");
    expect((await t.store.getJob(sub.id))!.status).toBe("dead");
    expect((await tick()).jobsEnded).toEqual([]);
    const { rows } = await t.store.db.query(`select payload->>'text' as text from events where kind = 'job-done' and payload->>'jobId' = $1`, [sub.id]);
    expect(rows.map((r) => r.text)).toEqual([`Job ${sub.id} dead: lease expired 3 times`]);
  });

  it("dispatches a worker and is woken in the same thread when it finishes", async () => {
    const m = await ring("infra", "[[dispatch: fix the flaky test in tests/test_clock.py]]");
    await reply(m.thread, "dispatch_worker");
    await reply(m.thread, "worker/worker-done");
    const [w] = await t.store.listWorkers({ lane: "infra" });
    expect(w).toMatchObject({ status: "finished", task: "fix the flaky test in tests/test_clock.py" });
    expect(w!.branch).toMatch(/^cursor\/infra-[0-9a-f]{6}$/);
    const turns = await t.store.listTurns({ lane: "infra" });
    expect(turns.some((x) => x.kind === "worker" && x.status === "finished")).toBe(true);
  });

  it("asks another lead and gets the answer back in the thread it asked from", async () => {
    const m = await ring("infra", "[[ask proofs: which guarantees moved this week?]]");
    await reply(m.thread, "ask_lead");
    const answer = await reply(m.thread, "lead/answer");
    expect(answer).toMatchObject({ channel: COORD, username: "infra" });
    expect(t.slack.posts.some((p) => p.username === "proofs" && p.text.includes("lead/ask"))).toBe(false);
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

    const thread = nextTs();
    await call("assign", { lane: "proofs", task: "tidy the ledger", slack_channel: COORD, thread_ts: thread });
    expect(await reply(thread, "mcp/assign")).toMatchObject({ channel: COORD, username: "proofs" });

    const st = await call("status", { lane: "infra" });
    const infra = st.leads[0];
    expect(infra).toMatchObject({ lane: "infra", agentId: leadAgentId("infra"), inbox: { channel: COORD, threadTs: INBOX.infra } });
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
    expect(new Set(grants.map((g) => g.kind))).toEqual(new Set(["github-app-token", "api:slack", "api:linear"]));
    expect(grants.every((g) => !g.ref.includes("local-"))).toBe(true);
    const workerGrants = (await t.store.listGrants()).filter((g) => g.subject.startsWith("infra-w-"));
    expect(workerGrants.map((g) => g.kind)).toEqual(["github-app-token"]);
    for (const f of await filesUnder(t.root)) expect(await readFile(f, "utf8")).not.toMatch(/local-[a-z]+-token/);
    expect(await readFile(join(t.root, "lead-infra", ".gitconfig"), "utf8")).toContain("password=platform-github-placeholder");
    expect(await readFile(join(t.root, "lead-infra", ".verity-env"), "utf8")).toBe(
      [
        "export SLACK_BOT_TOKEN='platform-slack-placeholder'",
        "export LINEAR_API_KEY='platform-linear-placeholder'",
        "export RESEARCH_NOTES_TOKEN='platform-github-placeholder'",
        `export RESEARCH_AGENT_URL='http://platform.test/api/leads?lane=infra&agent=${leadAgentId("infra")}'`,
        "",
      ].join("\n"),
    );
    expect(await readFile(join(t.root, "lead-infra", ".bashrc"), "utf8")).toContain(". \"$HOME/.verity-env\"");
    const { rows } = await t.store.db.query(`select count(*)::int as n from transcript_parts where text ~ 'local-[a-z]+-token'`);
    expect(rows[0]!.n).toBe(0);
  });

  it("hands the lead to a newer run, which keeps answering", async () => {
    const old = (await t.store.getLead("infra"))!.runId!;
    const run = await start(leadWorkflow, [{ lane: "infra", epoch: 0 }]);
    const ended = (await getRun<LeadResult>(old).returnValue) as LeadResult;
    expect(ended).toMatchObject({ ended: "superseded", by: run.runId });
    const m = await ring("infra", "still there?");
    await reply(m.thread, "slack/doorbell");
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
    expect((await admin({ lane: "comms", apis: ["slack", "jira"] })).status).toBe(400);
    const taken = await admin({ lane: "comms", slackChannel: COORD, slackInbox: INBOX.infra });
    expect(taken.status).toBe(409);
    expect((await taken.json()).error).toContain("belongs to lead infra");
    expect(await t.store.getLead("comms")).toBeNull();

    const reg = await (
      await admin({
        lane: "comms",
        slackChannel: COORD,
        slackInbox: INBOX.comms,
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
    expect(reg.lead).toMatchObject({ lane: "comms", agentId: leadAgentId("comms"), gatewayKeyId: "local-key-comms", docBytes: 19 });
    await turnOf("comms", "mcp/cutover");
    expect((await t.store.getLead("comms"))!.runId).toBe(reg.runId);
    const m = await ring("comms", "is the post up?");
    await reply(m.thread, "slack/doorbell");

    await admin({ lane: "comms", monthlyBudgetUsd: 300, doc: "# comms, from the old side\n" });
    const keys = (await t.store.listGrants("comms")).filter((g) => g.kind === "ai-gateway-key");
    expect(keys.map((g) => g.scope).sort()).toEqual(["monthly $200", "monthly $300"]);
    const now = await get("lane=comms&doc=1");
    expect(now.doc).toBe("# comms, from the old side\n");
    expect(now.revisions.map((r: { turnId: string }) => r.turnId)).toContain("admin");
    expect((await get("lane=comms&doc=1&rev=0")).doc).toBe("# comms workstream\n");

    expect((await (await admin({ lane: "comms", status: "paused" })).json()).lead.status).toBe("paused");
    expect((await ring("comms", "anyone there?")).res.routed).toEqual([]);
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
      await t.store.upsertLead({ lane: "circuits", model: "fake/fake", harness: "fake", slackChannel: COORD, slackInbox: INBOX.circuits, doc: "# circuits\n" });
      const first = await ring("circuits", "[[dispatch: add a pin]]");
      await reply(first.thread, "worker/worker-done");
      const lead = join(t.root, "lead-circuits", "verity");
      expect(await readFile(join(lead, "README.md"), "utf8")).toBe("seed\n");
      await writeFile(join(lead, "scratch.txt"), "kept\n");
      git(seed, "commit", "-q", "--allow-empty", "-m", "second");

      const second = await ring("circuits", "anything new?");
      await reply(second.thread, "slack/doorbell");
      expect(await readFile(join(lead, "scratch.txt"), "utf8")).toBe("kept\n");
      expect(git(lead, "log", "-1", "--format=%s", "origin/main")).toBe("second");

      const [w] = await t.store.listWorkers({ lane: "circuits" });
      expect(git(join(t.root, `worker-${w!.id}`, "verity"), "branch", "--show-current")).toBe(w!.branch);
      const [turn] = (await t.store.listTurns({ lane: "circuits" })).filter((x) => x.kind === "lead");
      const parts = await t.store.transcript(turn!.id);
      const instructions = parts.find((p) => p.type === "instructions")!.text;
      expect(instructions).toContain("verity/ is your checkout");
      expect(instructions).toContain("research msg reply --as circuits");
    } finally {
      t.rt.repoUrl = null;
    }
  });
});
