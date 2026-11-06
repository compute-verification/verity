import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeEach, describe, expect, it } from "vitest";
import { localCredentials, seal, unseal } from "../src/credentials.ts";
import { openDb } from "../src/db.ts";
import { costUsd, priceFor } from "../src/prices.ts";
import { renderEvent, turnPrompt } from "../src/prompt.ts";
import { localSandboxSession } from "../src/sandbox/local.ts";
import { mentionedLanes, parseSlackEvent, signSlackBody, verifySlackSignature } from "../src/slack.ts";
import { type Store, createStore } from "../src/store.ts";
import { finalText, foldChunks, fromSdkUsage } from "../src/transcript.ts";

const usage = (o: Partial<Record<"inputTokens" | "outputTokens" | "cacheReadTokens" | "cacheWriteTokens" | "reasoningTokens", number>>) => ({
  inputTokens: 0,
  outputTokens: 0,
  cacheReadTokens: 0,
  cacheWriteTokens: 0,
  reasoningTokens: 0,
  ...o,
});

describe("prices", () => {
  it("prices uncached input, cache reads and output at their own rates", () => {
    // 1M input of which 800k cache reads, 100k output, Opus 5.5 at $4 / $0.2 / $20.
    expect(costUsd("anthropic/claude-opus-5.5", usage({ inputTokens: 1_000_000, cacheReadTokens: 800_000, outputTokens: 100_000 }))).toBeCloseTo(0.8 + 0.16 + 2, 6);
  });
  it("matches a model id with or without its provider and with dots or dashes", () => {
    expect(priceFor("claude-opus-5-5")).toEqual(priceFor("anthropic/claude-opus-5.5"));
    expect(priceFor("openai/gpt-5.3-codex")?.output).toBe(14);
    expect(costUsd("nobody/unknown", usage({ inputTokens: 10 }))).toBe(0);
  });
});

describe("slack", () => {
  const secret = "s3cret";
  const body = '{"type":"event_callback"}';
  it("accepts a fresh, correctly signed request and nothing else", () => {
    const sig = signSlackBody(secret, "1000", body);
    expect(verifySlackSignature({ secret, timestamp: "1000", signature: sig, body, nowSeconds: 1100 })).toBe(true);
    expect(verifySlackSignature({ secret, timestamp: "1000", signature: sig, body, nowSeconds: 1400 })).toBe(false);
    expect(verifySlackSignature({ secret: "other", timestamp: "1000", signature: sig, body, nowSeconds: 1100 })).toBe(false);
    expect(verifySlackSignature({ secret, timestamp: "1000", signature: sig, body: `${body} `, nowSeconds: 1100 })).toBe(false);
    expect(verifySlackSignature({ secret, timestamp: null, signature: sig, body, nowSeconds: 1100 })).toBe(false);
  });
  it("parses messages and bots' posts, and drops edits and other subtypes", () => {
    const ev = (e: Record<string, unknown>) => parseSlackEvent({ type: "event_callback", event_id: "Ev1", event: { type: "message", channel: "C1", text: "hi", ts: "1.1", ...e } });
    expect(ev({ user: "U1", thread_ts: "0.9" })).toEqual({ message: { eventId: "Ev1", channel: "C1", user: "U1", text: "hi", ts: "1.1", threadTs: "0.9" } });
    expect(ev({ bot_id: "B1" })).toEqual({ message: { eventId: "Ev1", channel: "C1", user: "B1", text: "hi", ts: "1.1", botId: "B1" } });
    expect(ev({ user: "U9", bot_id: "B1", subtype: "bot_message" })).toMatchObject({ message: { user: "U9", botId: "B1" } });
    expect(ev({ subtype: "message_changed" })).toBeNull();
    expect(parseSlackEvent({ type: "url_verification", challenge: "abc" })).toEqual({ challenge: "abc" });
  });
  it("finds lanes named by @lane or by a user-group mention", () => {
    const lanes = [{ lane: "infra" }, { lane: "proofs", slackUsergroup: "S123" }, { lane: "kernels", slackUsergroup: "S456" }];
    expect(mentionedLanes("@infra and <!subteam^S999|@proofs> please", lanes)).toEqual(["infra", "proofs"]);
    expect(mentionedLanes("<!subteam^S456> and <!subteam^S123|@renamed>", lanes)).toEqual(["kernels", "proofs"]);
    expect(mentionedLanes("<!subteam^S000>", lanes)).toEqual([]);
    expect(mentionedLanes("mail me at a@infra.example", lanes)).toEqual([]);
    expect(mentionedLanes("(@kernels) and @nobody", lanes)).toEqual(["kernels"]);
  });
});

describe("transcripts", () => {
  const chunks = [
    { type: "reasoning-start", id: "r" },
    { type: "reasoning-delta", id: "r", delta: "think" },
    { type: "text-start", id: "a" },
    { type: "text-delta", id: "a", delta: "Let me " },
    { type: "text-delta", id: "a", delta: "check." },
    { type: "tool-input-available", toolCallId: "c1", toolName: "submit_job", input: { command: "x" } },
    { type: "tool-output-available", toolCallId: "c1", output: { jobId: "j" } },
    { type: "text-start", id: "b" },
    { type: "text-delta", id: "b", delta: "Submitted." },
    { type: "finish", finishReason: "stop" },
  ];
  it("folds deltas into blocks and keeps every tool call and result", () => {
    const parts = foldChunks(chunks);
    expect(parts.map((p) => [p.type, p.text])).toEqual([
      ["reasoning", "think"],
      ["text", "Let me check."],
      ["tool-call", 'submit_job {"command":"x"}'],
      ["tool-result", '{"jobId":"j"}'],
      ["text", "Submitted."],
      ["finish", ""],
    ]);
  });
  it("takes the reply from after the last tool activity", () => {
    expect(finalText(foldChunks(chunks))).toBe("Submitted.");
    expect(finalText(foldChunks(chunks.slice(0, 5)))).toBe("Let me check.");
  });
  it("reads AI SDK usage", () => {
    expect(fromSdkUsage({ inputTokens: 10, outputTokens: 4, inputTokenDetails: { cacheReadTokens: 6 }, outputTokenDetails: { reasoningTokens: 2 } })).toEqual(
      usage({ inputTokens: 10, outputTokens: 4, cacheReadTokens: 6, reasoningTokens: 2 }),
    );
    expect(fromSdkUsage(undefined)).toEqual(usage({}));
  });
});

describe("credentials", () => {
  it("seals and unseals, and refuses a tampered or wrongly keyed value", () => {
    const sealed = seal("k1", "vck_example_value");
    expect(sealed).not.toContain("vck_example_value");
    expect(unseal("k1", sealed)).toBe("vck_example_value");
    expect(() => unseal("k2", sealed)).toThrow();
    const [iv, tag, body] = sealed.split(".");
    expect(() => unseal("k1", [iv, tag, `${body!.slice(0, -2)}AA`].join("."))).toThrow();
  });
});

describe("store", () => {
  let s: Store;
  beforeEach(async () => {
    s = createStore(await openDb("pglite:memory"));
    await s.migrate();
    await s.migrate();
    await s.upsertLead({ lane: "infra", model: "fake/fake", harness: "fake" });
  });
  const ev = (dedupeKey?: string) => s.addEvent({ lane: "infra", source: "mcp", kind: "assign", payload: { text: "t" }, replyTo: null, dedupeKey });

  it("records a deduplicated event once", async () => {
    expect(await ev("k")).not.toBeNull();
    expect(await ev("k")).toBeNull();
    expect(await ev()).not.toBeNull();
    expect(await ev()).not.toBeNull();
    expect(await s.pendingCount("infra")).toBe(3);
  });
  it("claims pending events once per turn, and a retried claim gets the same events", async () => {
    await ev();
    await ev();
    const first = await s.claimEvents("infra", "t1");
    await ev();
    expect((await s.claimEvents("infra", "t1")).map((e) => e.id)).toEqual(first.map((e) => e.id));
    expect(await s.claimEvents("infra", "t2")).toHaveLength(1);
    expect(await s.claimEvents("infra", "t3")).toHaveLength(0);
  });
  it("lets one turn hold the lease until it releases or the lease expires", async () => {
    expect(await s.acquireLease("infra", "t1", 60)).toBe(true);
    expect(await s.acquireLease("infra", "t1", 60)).toBe(true);
    expect(await s.acquireLease("infra", "t2", 60)).toBe(false);
    await s.releaseLease("infra", "t2");
    expect(await s.acquireLease("infra", "t2", 60)).toBe(false);
    await s.releaseLease("infra", "t1");
    expect(await s.acquireLease("infra", "t2", 60)).toBe(true);
    await s.db.query(`update leads set lease_until = now() - interval '1 second'`);
    expect(await s.acquireLease("infra", "t3", 60)).toBe(true);
    await s.setLeadStatus("infra", "paused");
    await s.releaseLease("infra", "t3");
    expect(await s.acquireLease("infra", "t4", 60)).toBe(false);
  });
  it("bumps the doc revision only when the doc changes", async () => {
    expect(await s.saveDoc("infra", "a", "t1")).toBe(1);
    expect(await s.saveDoc("infra", "a", "t2")).toBe(1);
    expect(await s.saveDoc("infra", "b", "t3")).toBe(2);
    expect((await s.docRevisions("infra")).map((r) => [r.rev, r.turnId])).toEqual([
      [1, "t1"],
      [2, "t3"],
    ]);
  });
  it("searches transcripts by text, lead and time", async () => {
    await s.upsertLead({ lane: "proofs", model: "fake/fake", harness: "fake" });
    await s.startTurn({ id: "a", lane: "infra", kind: "lead", harness: "fake", model: "fake/fake" });
    await s.startTurn({ id: "b", lane: "proofs", kind: "lead", harness: "fake", model: "fake/fake" });
    await s.appendParts("a", [{ type: "text", content: null, text: "the bandwidth audit regressed" }]);
    await s.appendParts("b", [{ type: "text", content: null, text: "audits of the lock" }]);
    expect((await s.searchTranscripts({ text: "audit" })).map((h) => h.turnId).sort()).toEqual(["a", "b"]);
    expect((await s.searchTranscripts({ text: "audit", lane: "proofs" })).map((h) => h.turnId)).toEqual(["b"]);
    expect(await s.searchTranscripts({ text: "audit", until: "2000-01-01T00:00:00Z" })).toEqual([]);
    expect((await s.searchTranscripts({ text: '"bandwidth audit"' })).map((h) => h.turnId)).toEqual(["a"]);
  });
  it("ticks each due lead once, however many cron invocations overlap", async () => {
    await s.db.query(`update leads set cron_every_minutes = 30, next_tick_at = now() - interval '1 minute'`);
    const [x, y] = await Promise.all([s.takeDueTicks(), s.takeDueTicks()]);
    expect([...x!, ...y!]).toEqual(["infra"]);
    expect(await s.takeDueTicks()).toEqual([]);
  });
  it("sums spend by lead and kind", async () => {
    await s.startTurn({ id: "a", lane: "infra", kind: "lead", harness: "fake", model: "fake/fake" });
    await s.startTurn({ id: "w", lane: "infra", kind: "worker", harness: "fake", model: "fake/fake" });
    await s.finishTurn("a", { status: "finished", usage: usage({ inputTokens: 100, outputTokens: 10 }), costUsd: 0.5 });
    await s.finishTurn("w", { status: "finished", usage: usage({ inputTokens: 50, outputTokens: 5 }), costUsd: 0.25 });
    expect((await s.spend()).map((r) => [r.kind, r.turns, r.costUsd, r.inputTokens])).toEqual([
      ["lead", 1, 0.5, 100],
      ["worker", 1, 0.25, 50],
    ]);
  });
  it("records grants by reference and never the token", async () => {
    const c = localCredentials(s);
    const g = await c.github("infra");
    const [grant] = await s.listGrants("infra");
    expect(grant!.kind).toBe("github-app-token");
    expect(grant!.ref).not.toContain(g!.token);
    expect(JSON.stringify(await s.listGrants())).not.toContain(g!.token);
  });
});

describe("prompt", () => {
  it("renders each event with its source, id and return address", () => {
    const e = { id: 7, lane: "infra", source: "slack" as const, kind: "message", payload: { text: "hi" }, replyTo: { channel: "C1", threadTs: "1.2" }, receivedAt: "2026-11-06T00:00:00.000Z" };
    expect(renderEvent(e)).toBe("- [slack/message] #7 at 2026-11-06T00:00:00.000Z (slack C1 thread 1.2): hi");
    expect(renderEvent({ ...e, source: "lead", kind: "ask", replyTo: { lane: "proofs" } })).toContain("(from lead proofs)");
    expect(renderEvent({ ...e, source: "mcp", replyTo: { caller: "daniel" } })).toContain("(from daniel)");
    expect(turnPrompt([e])).toContain("WORKSTREAM.md");
  });
});

describe("local sandbox", () => {
  it("reads, writes and runs inside its directory, with a HOME of its own and none of the host's environment", async () => {
    const root = await mkdtemp(join(tmpdir(), "platform-unit-"));
    const sb = localSandboxSession(root);
    await sb.writeTextFile({ path: "a/b.txt", content: "one\ntwo\nthree" });
    expect(await sb.readTextFile({ path: "a/b.txt", startLine: 2, endLine: 2 })).toBe("two");
    expect(await sb.readTextFile({ path: "missing" })).toBeNull();
    process.env.PLATFORM_UNIT_SECRET = "must-not-leak";
    const r = await sb.run({ command: 'echo "$HOME|${PLATFORM_UNIT_SECRET:-unset}|$(cat a/b.txt | wc -l)"' });
    delete process.env.PLATFORM_UNIT_SECRET;
    expect(r.stdout.trim()).toBe(`${root}|unset|2`);
    expect(await sb.deleteFile({ path: "a/b.txt" })).toBe(true);
    expect(await sb.deleteFile({ path: "a/b.txt" })).toBe(false);
  });
});
