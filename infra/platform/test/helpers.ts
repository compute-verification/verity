import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { localCredentials } from "../src/credentials.ts";
import { openDb } from "../src/db.ts";
import { memoryJobs } from "../src/jobs.ts";
import { harnessFor } from "../src/harness/providers.ts";
import { localSandboxes, type Runtime, setRuntime } from "../src/runtime.ts";
import { directReply, memorySlack, signSlackBody } from "../src/slack.ts";
import { type Turn, createStore } from "../src/store.ts";

export const SIGNING_SECRET = "test-signing-secret";
export const NOW = 1_900_000_000;

export async function testRuntime() {
  const store = createStore(await openDb("pglite:memory"));
  await store.migrate();
  const slack = memorySlack();
  const jobs = memoryJobs();
  const root = await mkdtemp(join(tmpdir(), "platform-test-"));
  // A lead turn ends when it gives up its lease, after its reply is out;
  // `turnsEnded(match, n)` resolves once n ended turns match.
  const ended: Turn[] = [];
  const waiters: { done: () => boolean; resolve: () => void }[] = [];
  const release = store.releaseLease.bind(store);
  store.releaseLease = async (lane: string, turnId: string) => {
    await release(lane, turnId);
    const turn = await store.getTurn(turnId);
    if (!turn || turn.status === "running") return;
    ended.push(turn);
    for (const w of waiters.splice(0)) {
      if (w.done()) w.resolve();
      else waiters.push(w);
    }
  };
  const turnsEnded = (match: (t: Turn) => boolean, n = 1): Promise<Turn[]> => {
    const hits = () => ended.filter(match);
    if (hits().length >= n) return Promise.resolve(hits());
    return new Promise((resolve) => waiters.push({ done: () => hits().length >= n, resolve: () => resolve(hits()) }));
  };
  const turnEnded = async (match: (t: Turn) => boolean) => (await turnsEnded(match))[0]!;
  const rt: Runtime = {
    store,
    slack,
    reply: directReply(slack),
    jobs,
    sandbox: localSandboxes(root),
    harness: harnessFor,
    credentials: localCredentials(store),
    baseUrl: "http://platform.test",
    repoUrl: null,
    sliceSeconds: 30,
    leaseSeconds: 120,
  };
  setRuntime(rt);
  return { rt, store, slack, jobs, root, turnEnded, turnsEnded };
}

let seq = 0;

export const nextTs = () => `${NOW}.${String(++seq).padStart(6, "0")}`;

// A Slack Events API body for one message, with a fresh event id and ts.
export function slackBody(o: { channel: string; text: string; threadTs?: string; user?: string; botId?: string; username?: string }) {
  const ts = nextTs();
  const body = JSON.stringify({
    type: "event_callback",
    event_id: `Ev${seq}`,
    event: { type: "message", channel: o.channel, user: o.user ?? "U_DANIEL", text: o.text, ts, thread_ts: o.threadTs, bot_id: o.botId, username: o.username },
  });
  return { ts, body };
}

export function signedSlackRequest(body: string, secret = SIGNING_SECRET) {
  const ts = String(NOW);
  return new Request("http://platform.test/api/slack/events", {
    method: "POST",
    headers: { "x-slack-request-timestamp": ts, "x-slack-signature": signSlackBody(secret, ts, body), "content-type": "application/json" },
    body,
  });
}

export const slackEnv = { SLACK_SIGNING_SECRET: SIGNING_SECRET };
