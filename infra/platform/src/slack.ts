import { createHash, createHmac, timingSafeEqual } from "node:crypto";
import { REPO_DIR } from "./prompt.ts";
import { type Shell, shq } from "./shell.ts";

export type SlackPost = { channel: string; text: string; threadTs?: string; username?: string };
export type SlackClient = { post(m: SlackPost): Promise<{ ts: string }> };

// The name the platform posts its own lines under; nothing it posts wakes a lead.
export const PLATFORM_NAME = "platform";

// Slack signs every request with the app's signing secret; anything unsigned,
// mis-signed or older than five minutes is refused.
export function verifySlackSignature(o: {
  secret: string;
  timestamp: string | null;
  signature: string | null;
  body: string;
  nowSeconds: number;
}): boolean {
  if (!o.timestamp || !o.signature) return false;
  const ts = Number(o.timestamp);
  if (!Number.isFinite(ts) || Math.abs(o.nowSeconds - ts) > 300) return false;
  const expected = `v0=${createHmac("sha256", o.secret).update(`v0:${o.timestamp}:${o.body}`).digest("hex")}`;
  const a = Buffer.from(expected);
  const b = Buffer.from(o.signature);
  return a.length === b.length && timingSafeEqual(a, b);
}

export function signSlackBody(secret: string, timestamp: string, body: string): string {
  return `v0=${createHmac("sha256", secret).update(`v0:${timestamp}:${body}`).digest("hex")}`;
}

export type SlackMessage = {
  eventId: string;
  channel: string;
  user: string;
  text: string;
  ts: string;
  threadTs?: string;
  username?: string;
};

// The Events API envelope, reduced to what routing needs. Edits and other
// subtypes are dropped; a bot's post (a doorbell, an agent's) is kept with the
// name it was posted under.
export function parseSlackEvent(body: Record<string, unknown>): { challenge: string } | { message: SlackMessage } | null {
  if (body.type === "url_verification" && typeof body.challenge === "string") return { challenge: body.challenge };
  if (body.type !== "event_callback") return null;
  const e = body.event as Record<string, unknown> | undefined;
  if (!e || e.type !== "message") return null;
  if (e.subtype && e.subtype !== "bot_message") return null;
  if (typeof e.channel !== "string" || typeof e.text !== "string" || typeof e.ts !== "string") return null;
  return {
    message: {
      eventId: String(body.event_id ?? `${e.channel}:${e.ts}`),
      channel: e.channel,
      user: String(e.user ?? e.bot_id ?? ""),
      text: e.text,
      ts: e.ts,
      threadTs: typeof e.thread_ts === "string" ? e.thread_ts : undefined,
      ...(typeof e.username === "string" && { username: e.username }),
    },
  };
}

// A doorbell as `research msg` posts one in an inbox thread (its DOORBELL_RE):
// `from AUTHOR[ (URGENT)], [#CHANNEL ]thread T[, reply R]: TEXT`, where CHANNEL
// (a registry key or an id) is absent for the front door.
const DOORBELL_RE = /^from ([a-z0-9][a-z0-9._-]*)( \(URGENT\))?, (?:#([A-Za-z0-9][A-Za-z0-9._-]*) )?thread (\d+\.\d+)(?:, reply (\d+\.\d+))?: /;

export type Doorbell = { author: string; urgent: boolean; channel: string | null; thread: string; reply: string | null };

export function parseDoorbell(text: string): Doorbell | null {
  const m = DOORBELL_RE.exec(text);
  return m ? { author: m[1]!, urgent: Boolean(m[2]), channel: m[3] ?? null, thread: m[4]!, reply: m[5] ?? null } : null;
}

// The agent id a lead holds its handle's inbox under (`research msg names open
// --holder`), a UUID as research reads one. It depends on the lane alone, so it
// survives new runs and redeploys.
export function leadAgentId(lane: string): string {
  const h = createHash("sha256").update(`verity-platform-lead:${lane}`).digest("hex");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20, 32)}`;
}

// Posts a lead's reply in a thread. On the platform it is `research msg reply`
// run in the lead's sandbox, so the post rings whoever research's rule says it
// rings, exactly as it would from an agent on Cursor.
export type SlackReplier = (session: Shell, o: { lane: string; channel: string; threadTs: string; text: string }) => Promise<void>;

export const REPLY_FILE = ".platform-reply.md";
export const RESEARCH_BIN = `"$HOME/.local/bin/research"`;
const CHANNEL_ID_RE = /^[CDG][A-Z0-9]{2,}$/;

export function researchReplyCommand(o: { lane: string; channel: string; threadTs: string }): string {
  const channel = CHANNEL_ID_RE.test(o.channel) ? `--channel-id ${shq(o.channel)}` : `--channel ${shq(o.channel)}`;
  return [
    `. "$HOME/.verity-env" &&`,
    `${RESEARCH_BIN} msg reply --registry ${REPO_DIR}/.agents/skills/using-slack/registry.json`,
    `--as ${shq(o.lane)} ${channel} --thread ${shq(o.threadTs)} --file ${REPLY_FILE}`,
  ].join(" ");
}

export const researchReply: SlackReplier = async (session, o) => {
  await session.writeTextFile({ path: REPLY_FILE, content: o.text });
  const r = await session.run({ command: researchReplyCommand(o) });
  if (r.exitCode !== 0) throw new Error(`research msg reply: exit ${r.exitCode}: ${r.stderr.trim().slice(-500)}`);
};

// For tests and local runs: the reply goes straight to the client, under the lead's name.
export const directReply =
  (slack: SlackClient): SlackReplier =>
  async (_session, o) => {
    await slack.post({ channel: o.channel, threadTs: o.threadTs, text: o.text, username: o.lane });
  };

export function webSlack(token: string): SlackClient {
  return {
    async post({ channel, text, threadTs, username }) {
      const r = await fetch("https://slack.com/api/chat.postMessage", {
        method: "POST",
        headers: { authorization: `Bearer ${token}`, "content-type": "application/json; charset=utf-8" },
        body: JSON.stringify({ channel, text, thread_ts: threadTs, username: username ?? PLATFORM_NAME, unfurl_links: false }),
      });
      const j = (await r.json()) as { ok: boolean; ts?: string; error?: string };
      if (!j.ok) throw new Error(`slack chat.postMessage: ${j.error}`);
      return { ts: j.ts! };
    },
  };
}

type Posted = SlackPost & { ts: string };

// Records posts; `waitFor(match)` resolves with the first post that matches, so
// a test waits on the post itself rather than on a clock.
export function memorySlack() {
  const posts: Posted[] = [];
  const waiters: { match: (p: Posted) => boolean; resolve: (p: Posted) => void }[] = [];
  return {
    posts,
    async post(m: SlackPost) {
      const p = { ...m, ts: `${1_800_000_000 + posts.length}.000100` };
      posts.push(p);
      for (const w of waiters.splice(0)) {
        if (w.match(p)) w.resolve(p);
        else waiters.push(w);
      }
      return { ts: p.ts };
    },
    waitFor(match: (p: Posted) => boolean): Promise<Posted> {
      const hit = posts.find(match);
      if (hit) return Promise.resolve(hit);
      return new Promise((resolve) => waiters.push({ match, resolve }));
    },
  };
}
