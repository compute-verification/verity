import { beforeAll, describe, expect, it } from "vitest";
import { getRun } from "workflow/api";
import { tick } from "../src/cron.ts";
import { slackEvents } from "../src/http.ts";
import type { LeadResult } from "../workflows/lead.ts";
import { NOW, signedSlackRequest, slackBody, slackEnv, testRuntime } from "./helpers.ts";

// What happens without a human when something breaks: a failed turn is
// reported in its thread and the lead carries on; the minute cron rings a lead
// whose events wait, restarts one whose events wait too long, and reports a
// turn that has run too long, once.

let t: Awaited<ReturnType<typeof testRuntime>>;

async function say(channel: string, text: string) {
  const { ts, body } = slackBody({ channel, text });
  await slackEvents(signedSlackRequest(body), slackEnv, NOW);
  return ts;
}

const reply = (thread: string, has: string) => t.slack.waitFor((p) => p.threadTs === thread && p.text.includes(has));

beforeAll(async () => {
  t = await testRuntime();
  await t.store.upsertLead({ lane: "kernels", model: "fake/fake", harness: "fake", slackChannel: "C_KERNELS", doc: "# kernels\n" });
});

describe("recovery", () => {
  it("reports a failed turn in its thread, keeps the doc, and answers the next message", async () => {
    const thread = await say("C_KERNELS", "[[fail: harness crashed]]");
    const post = await reply(thread, "This turn failed");
    expect(post.text).toContain("harness crashed");
    const [turn] = await t.store.listTurns({ lane: "kernels", limit: 1 });
    expect(turn).toMatchObject({ status: "failed" });
    expect(turn!.error).toContain("harness crashed");
    expect((await t.store.getLead("kernels"))!.docRev).toBe(0);
    const next = await say("C_KERNELS", "are you back?");
    await reply(next, "slack/message");
  });

  it("restarts a lead whose events have waited too long, and alerts once", async () => {
    const e = await t.store.addEvent({ lane: "kernels", source: "mcp", kind: "assign", payload: { text: "stranded" }, replyTo: null });
    await t.store.db.query(`update events set received_at = now() - interval '20 minutes' where id = $1`, [e!.id]);
    const before = (await t.store.getLead("kernels"))!.runId;
    const r = await tick({ alertChannel: "C_ALERTS" });
    expect(r.restarted).toEqual(["kernels"]);
    expect(r.alerts).toHaveLength(1);
    await t.slack.waitFor((p) => p.channel === "C_ALERTS" && p.text.includes("kernels had 1 event(s) waiting"));
    await t.slack.waitFor((p) => p.channel === "C_KERNELS" && p.threadTs === undefined && p.text.includes("mcp/assign"));
    expect((await t.store.getLead("kernels"))!.runId).not.toBe(before);
  });

  it("rings a lead with pending events and reports an overlong turn once", async () => {
    await t.store.addEvent({ lane: "kernels", source: "mcp", kind: "assign", payload: { text: "fresh" }, replyTo: null });
    await t.store.db.query(
      `insert into turns (id, lane, kind, harness, model, started_at) values ('kernels-stuck', 'kernels', 'lead', 'fake', 'fake/fake', now() - interval '3 hours')`,
    );
    const r = await tick({ alertChannel: "C_ALERTS" });
    expect(r.nudged).toEqual(["kernels"]);
    expect(r.alerts).toEqual([expect.stringContaining("kernels-stuck")]);
    expect((await tick({ alertChannel: "C_ALERTS" })).alerts).toEqual([]);
    await t.slack.waitFor((p) => p.channel === "C_KERNELS" && p.text.includes("Handled 1 event(s): mcp/assign") && t.slack.posts.filter((q) => q.channel === "C_KERNELS" && q.threadTs === undefined).length >= 2);
  });

  it("moves an idle lead whose run is a day old to a fresh run", async () => {
    await t.store.db.query(`delete from turns where id = 'kernels-stuck'`);
    await t.store.db.query(`update leads set run_started_at = now() - interval '25 hours' where lane = 'kernels'`);
    const old = (await t.store.getLead("kernels"))!.runId!;
    const r = await tick();
    expect(r.rolled).toEqual(["kernels"]);
    const ended = (await getRun<LeadResult>(old).returnValue) as LeadResult;
    expect(ended).toMatchObject({ ended: "superseded" });
    expect((await tick()).rolled).toEqual([]);
    const thread = await say("C_KERNELS", "after the roll");
    await reply(thread, "slack/message");
  });
});
