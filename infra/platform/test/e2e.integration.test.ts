import { beforeAll, describe, expect, it } from "vitest";
import { slackEvents } from "../src/http.ts";
import { NOW, slackEnv, slackRequest, testRuntime } from "./helpers.ts";

let t: Awaited<ReturnType<typeof testRuntime>>;

beforeAll(async () => {
  t = await testRuntime();
  await t.store.upsertLead({ lane: "infra", model: "fake/fake", harness: "fake", slackChannel: "C_INFRA", doc: "# infra workstream\n" });
});

describe("a lead on the platform", () => {
  it("wakes on a Slack message, runs a harness turn, updates its doc and replies in the thread", async () => {
    const res = await slackEvents(slackRequest({ channel: "C_INFRA", text: "status of the queue?" }), slackEnv, NOW);
    expect(await res.json()).toEqual({ ok: true, routed: ["infra"] });
    const [post] = await t.slack.until(1);
    expect(post!.channel).toBe("C_INFRA");
    expect(post!.threadTs).toBeDefined();
    expect(post!.text).toContain("slack/message");
    const lead = (await t.store.getLead("infra"))!;
    expect(lead.docRev).toBe(1);
    expect(lead.doc).toContain("handled slack/message");
  });
});
