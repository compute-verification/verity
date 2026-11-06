import { createHmac, timingSafeEqual } from "node:crypto";

export type SlackPost = { channel: string; text: string; threadTs?: string };
export type SlackClient = { post(m: SlackPost): Promise<{ ts: string }> };

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
};

// The Events API envelope, reduced to what routing needs. Bot messages, edits
// and other subtypes are dropped, so the platform never wakes on its own posts.
export function parseSlackEvent(body: Record<string, unknown>): { challenge: string } | { message: SlackMessage } | null {
  if (body.type === "url_verification" && typeof body.challenge === "string") return { challenge: body.challenge };
  if (body.type !== "event_callback") return null;
  const e = body.event as Record<string, unknown> | undefined;
  if (!e || (e.type !== "message" && e.type !== "app_mention")) return null;
  if (e.bot_id || e.subtype) return null;
  if (typeof e.channel !== "string" || typeof e.text !== "string" || typeof e.ts !== "string") return null;
  return {
    message: {
      eventId: String(body.event_id ?? `${e.channel}:${e.ts}`),
      channel: e.channel,
      user: String(e.user ?? ""),
      text: e.text,
      ts: e.ts,
      threadTs: typeof e.thread_ts === "string" ? e.thread_ts : undefined,
    },
  };
}

// A message naming a lead (`@lane` as a word, or a mention of the lead's Slack
// user group, which Slack sends as `<!subteam^ID>` with or without its handle)
// goes to that lead; otherwise it goes to the lead whose channel it is posted in.
export function mentionedLanes(text: string, leads: { lane: string; slackUsergroup?: string | null }[]): string[] {
  const found = new Set<string>();
  const lanes = new Set(leads.map((l) => l.lane));
  for (const m of text.matchAll(/<!subteam\^([A-Z0-9]+)(?:\|@?([\w-]+))?>|(?:^|[\s(])@([\w-]+)/g)) {
    const byId = m[1] ? leads.find((l) => l.slackUsergroup === m[1])?.lane : undefined;
    const name = byId ?? (m[2] ?? m[3])?.toLowerCase();
    if (name && lanes.has(name)) found.add(name);
  }
  return [...found].sort();
}

export function webSlack(token: string): SlackClient {
  return {
    async post({ channel, text, threadTs }) {
      const r = await fetch("https://slack.com/api/chat.postMessage", {
        method: "POST",
        headers: { authorization: `Bearer ${token}`, "content-type": "application/json; charset=utf-8" },
        body: JSON.stringify({ channel, text, thread_ts: threadTs, unfurl_links: false }),
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
