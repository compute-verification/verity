import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { localCredentials } from "../src/credentials.ts";
import { openDb } from "../src/db.ts";
import { memoryJobs } from "../src/jobs.ts";
import { harnessFor, localSandboxes, type Runtime, setRuntime } from "../src/runtime.ts";
import { memorySlack, signSlackBody } from "../src/slack.ts";
import { createStore } from "../src/store.ts";

export const SIGNING_SECRET = "test-signing-secret";
export const NOW = 1_900_000_000;

export async function testRuntime() {
  const store = createStore(await openDb("pglite:memory"));
  await store.migrate();
  const slack = memorySlack();
  const jobs = memoryJobs();
  const root = await mkdtemp(join(tmpdir(), "platform-test-"));
  const rt: Runtime = {
    store,
    slack,
    jobs,
    sandbox: localSandboxes(root),
    harness: harnessFor,
    credentials: localCredentials(store),
    baseUrl: "http://platform.test",
    sliceSeconds: 30,
    leaseSeconds: 120,
  };
  setRuntime(rt);
  return { rt, store, slack, jobs, root };
}

let seq = 0;

export function slackRequest(o: { channel: string; text: string; threadTs?: string; user?: string }) {
  seq++;
  const body = JSON.stringify({
    type: "event_callback",
    event_id: `Ev${seq}`,
    event: { type: "message", channel: o.channel, user: o.user ?? "U_DANIEL", text: o.text, ts: `${NOW}.${String(seq).padStart(6, "0")}`, thread_ts: o.threadTs },
  });
  const ts = String(NOW);
  return new Request("http://platform.test/api/slack/events", {
    method: "POST",
    headers: { "x-slack-request-timestamp": ts, "x-slack-signature": signSlackBody(SIGNING_SECRET, ts, body), "content-type": "application/json" },
    body,
  });
}

export const slackEnv = { SLACK_SIGNING_SECRET: SIGNING_SECRET };
