import { createHash, timingSafeEqual } from "node:crypto";
import { start } from "workflow/api";
import { leadWorkflow } from "../workflows/lead.ts";
import { tick } from "./cron.ts";
import { jobDone } from "./job-done.ts";
import { parseLinearEvent, verifyLinearSignature } from "./linear.ts";
import { runtime } from "./runtime.ts";
import { SERVICE_NAMES, isService } from "./services.ts";
import { PLATFORM_NAME, leadAgentId, parseDoorbell, parseSlackEvent, verifySlackSignature } from "./slack.ts";
import type { HarnessName, Lead } from "./store.ts";
import { wake } from "./wake.ts";

// The HTTP surface as plain (Request) => Response functions: the Next routes
// mount them, and tests call them directly.
type Env = Record<string, string | undefined>;

const json = (body: unknown, status = 200) => Response.json(body, { status, headers: { "cache-control": "no-store" } });

export function bearerOk(req: Request, secret: string | undefined): boolean {
  if (!secret) return false;
  const got = /^Bearer\s+(\S+)$/i.exec(req.headers.get("authorization") ?? "")?.[1] ?? "";
  const a = Buffer.from(got);
  const b = Buffer.from(secret);
  return a.length === b.length && timingSafeEqual(a, b);
}

export async function slackEvents(req: Request, env: Env = process.env, nowSeconds = Math.floor(Date.now() / 1000)) {
  const body = await req.text();
  const ok = verifySlackSignature({
    secret: env.SLACK_SIGNING_SECRET ?? "",
    timestamp: req.headers.get("x-slack-request-timestamp"),
    signature: req.headers.get("x-slack-signature"),
    body,
    nowSeconds,
  });
  if (!env.SLACK_SIGNING_SECRET || !ok) return json({ error: "bad signature" }, 401);
  const parsed = parseSlackEvent(JSON.parse(body));
  if (!parsed) return json({ ok: true, routed: [] });
  if ("challenge" in parsed) return json({ challenge: parsed.challenge });
  const m = parsed.message;
  const none = json({ ok: true, routed: [] });
  // Only a reply in a lead's inbox thread wakes it. Who a post is for is
  // `research msg`'s rule, applied by whoever sends it: it rings each name the
  // post addresses with a doorbell in that name's inbox, on Cursor and here alike.
  if (!m.threadTs || m.threadTs === m.ts) return none;
  const rt = await runtime();
  const lead = await rt.store.leadByInbox(m.channel, m.threadTs);
  if (!lead || lead.status !== "active" || m.username === lead.lane || m.username === PLATFORM_NAME) return none;
  const bell = parseDoorbell(m.text);
  await wake(lead.lane, {
    lane: lead.lane,
    source: "slack",
    kind: bell ? "doorbell" : "message",
    payload: { text: m.text, user: m.user, channel: m.channel, ts: m.ts, ...(bell && { from: bell.author, urgent: bell.urgent }) },
    replyTo: bell ? { channel: bell.channel ?? m.channel, threadTs: bell.thread } : null,
    dedupeKey: `slack:${m.eventId}`,
  });
  return json({ ok: true, routed: [lead.lane] });
}

export async function linearEvents(req: Request, env: Env = process.env, nowMs = Date.now()) {
  const body = await req.text();
  const signature = req.headers.get("linear-signature");
  if (!env.LINEAR_WEBHOOK_SECRET || !verifyLinearSignature({ secret: env.LINEAR_WEBHOOK_SECRET, signature, body, nowMs })) {
    return json({ error: "bad signature" }, 401);
  }
  const e = parseLinearEvent(JSON.parse(body));
  const rt = await runtime();
  const lead = e ? await rt.store.leadByLinearTeam(e.team) : null;
  if (!e || !lead || lead.status !== "active") return json({ ok: true, routed: [] });
  await wake(lead.lane, {
    lane: lead.lane,
    source: "linear",
    kind: e.kind,
    payload: { ...e, text: `${e.identifier} ${e.title}${e.state ? ` (${e.state})` : ""}${e.url ? ` ${e.url}` : ""}` },
    replyTo: null,
    dedupeKey: `linear:${req.headers.get("linear-delivery") ?? createHash("sha256").update(body).digest("hex")}`,
  });
  return json({ ok: true, routed: [lead.lane] });
}

// A job runner posts here when a job ends. The URL's token is the job's
// capability: it names one job and is good for nothing else.
export async function jobCallback(req: Request, token: string) {
  const rt = await runtime();
  const job = await rt.store.jobByToken(token);
  if (!job) return json({ error: "no such job" }, 404);
  const body = (await req.json().catch(() => ({}))) as { status?: string; summary?: string; result?: unknown };
  const r = await jobDone(job, { status: body.status ?? "done", summary: body.summary, result: body.result });
  return json({ ok: true, eventId: r.eventId, duplicate: r.duplicate });
}

export async function cronTick(req: Request, env: Env = process.env) {
  if (!bearerOk(req, env.CRON_SECRET)) return json({ error: "unauthorized" }, 401);
  return json(await tick({ alertChannel: env.PLATFORM_ALERT_CHANNEL }));
}

export async function status(req: Request, env: Env = process.env) {
  if (!bearerOk(req, env.PLATFORM_API_TOKEN)) return json({ error: "unauthorized" }, 401);
  const q = new URL(req.url).searchParams;
  const lane = q.get("lane") ?? undefined;
  if (lane && q.get("doc") === "1") {
    const { store } = await runtime();
    const rev = q.get("rev");
    if (rev != null) {
      const doc = await store.docAt(lane, Number(rev));
      return doc == null ? json({ error: "no such revision" }, 404) : json({ lane, docRev: Number(rev), doc });
    }
    const lead = await store.getLead(lane);
    return lead ? json({ lane, docRev: lead.docRev, doc: lead.doc, revisions: await store.docRevisions(lane) }) : json({ error: "no such lead" }, 404);
  }
  return json(await statusReport(lane));
}

type LeadChange = {
  lane: string;
  charter?: string;
  doc?: string;
  slackChannel?: string;
  slackInbox?: string;
  linearTeam?: string;
  apis?: string[];
  harness?: HarnessName;
  model?: string;
  monthlyBudgetUsd?: number;
  cronEveryMinutes?: number;
  status?: Lead["status"];
  kick?: string;
  roll?: boolean;
};

// The cutover's one write: register or change a lead, provision its gateway key
// the first time it has a budget (and move the key's limit with the budget),
// park its sandbox when it is paused, and
// optionally ring it (`kick`, an event it handles at once) or move it to a fresh
// run on this deployment (`roll`).
export async function changeLead(req: Request, env: Env = process.env) {
  if (!bearerOk(req, env.PLATFORM_API_TOKEN)) return json({ error: "unauthorized" }, 401);
  const c = (await req.json()) as LeadChange;
  if (!c.lane || !/^[a-z][a-z0-9-]{0,39}$/.test(c.lane)) return json({ error: "lane must match [a-z][a-z0-9-]*" }, 400);
  const unknown = (c.apis ?? []).filter((a) => !isService(a));
  if (unknown.length) return json({ error: `unknown apis ${unknown.join(", ")}; known: ${SERVICE_NAMES.join(", ")}` }, 400);
  const rt = await runtime();
  const before = await rt.store.getLead(c.lane);
  const model = c.model ?? before?.model ?? env.PLATFORM_DEFAULT_MODEL ?? "anthropic/claude-opus-5.5";
  const channel = c.slackChannel ?? before?.slackChannel;
  const inboxHolder = c.slackInbox && channel ? await rt.store.leadByInbox(channel, c.slackInbox) : null;
  const teamHolder = c.linearTeam ? await rt.store.leadByLinearTeam(c.linearTeam) : null;
  for (const [held, what, fix] of [
    [inboxHolder, `inbox ${channel} ${c.slackInbox}`, `swap register ${inboxHolder?.lane} --inbox <its new inbox ts>`],
    [teamHolder, `Linear team ${c.linearTeam}`, `swap register ${teamHolder?.lane} --linear <its other team>`],
  ] as const) {
    if (held && held.lane !== c.lane) return json({ error: `${what} belongs to lead ${held.lane}; move it first (${fix})` }, 409);
  }
  const { doc, ...change } = c;
  let lead = await rt.store.upsertLead({ ...change, doc: before ? undefined : doc, model });
  // A lead's doc is its memory: the imported doc and every replacement are
  // revisions like any turn's, so the doc a replacement overwrote can be read back.
  if (doc != null && doc !== before?.doc) {
    await rt.store.saveDoc(c.lane, doc, "admin");
    lead = (await rt.store.getLead(c.lane))!;
  }
  if (lead.monthlyBudgetUsd && !lead.gatewayKeyId) {
    await rt.credentials.provisionGatewayKey(lead.lane, lead.monthlyBudgetUsd);
    lead = (await rt.store.getLead(lead.lane))!;
  } else if (lead.monthlyBudgetUsd && lead.gatewayKeyId && lead.monthlyBudgetUsd !== before?.monthlyBudgetUsd) {
    await rt.credentials.setGatewayBudget(lead.lane, lead.gatewayKeyId, lead.monthlyBudgetUsd);
  }
  if (lead.status !== "active") await rt.sandbox.park(`lead-${lead.lane}`);
  let runId: string | null = null;
  if (c.roll && lead.status === "active") {
    runId = (await start(leadWorkflow, [{ lane: lead.lane, epoch: 0 }])).runId;
    await rt.store.setRunId(lead.lane, runId);
  }
  const kicked = c.kick
    ? await wake(lead.lane, { lane: lead.lane, source: "mcp", kind: "cutover", payload: { text: c.kick, from: "swap" }, replyTo: null })
    : null;
  const { doc: _doc, charter: _charter, ...shown } = lead;
  return json({ lead: { ...shown, agentId: leadAgentId(lead.lane), docBytes: lead.doc.length }, kicked, runId });
}

export async function transcripts(req: Request, env: Env = process.env) {
  if (!bearerOk(req, env.PLATFORM_API_TOKEN)) return json({ error: "unauthorized" }, 401);
  const q = new URL(req.url).searchParams;
  const rt = await runtime();
  const turn = q.get("turn");
  if (turn) return json({ turn: await rt.store.getTurn(turn), parts: await rt.store.transcript(turn) });
  return json({
    hits: await rt.store.searchTranscripts({
      text: q.get("q") ?? undefined,
      lane: q.get("lane") ?? undefined,
      since: q.get("since") ?? undefined,
      until: q.get("until") ?? undefined,
      limit: Number(q.get("limit") ?? 50),
    }),
  });
}

const monthStart = (d = new Date()) => new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), 1)).toISOString();

export async function statusReport(lane?: string) {
  const rt = await runtime();
  const leads = (await rt.store.listLeads()).filter((l) => !lane || l.lane === lane);
  const spend = await rt.store.spend({ since: monthStart() });
  const gateway = await rt.credentials.gatewaySpend().catch(() => []);
  const out = [];
  for (const l of leads) {
    const h = await rt.store.leadHealth(l.lane);
    const [last] = await rt.store.listTurns({ lane: l.lane, limit: 1 });
    const mine = spend.filter((s) => s.lane === l.lane);
    out.push({
      lane: l.lane,
      agentId: leadAgentId(l.lane),
      inbox: l.slackInbox ? { channel: l.slackChannel, threadTs: l.slackInbox } : null,
      linearTeam: l.linearTeam,
      status: l.status,
      harness: l.harness,
      model: l.model,
      runId: l.runId,
      docRev: l.docRev,
      pending: h.pending,
      turnRunning: h.leaseHeld,
      lastTurn: last ? { id: last.id, kind: last.kind, status: last.status, startedAt: last.startedAt, costUsd: last.costUsd, reply: last.finalText?.slice(0, 500) ?? null } : null,
      workersRunning: (await rt.store.listWorkers({ lane: l.lane, status: "running" })).map((w) => w.id),
      monthToDate: {
        costUsd: mine.reduce((a, s) => a + s.costUsd, 0),
        turns: mine.reduce((a, s) => a + s.turns, 0),
        byKind: mine,
        gatewayBilledUsd: gateway.find((g) => g.keyId === l.gatewayKeyId)?.spendUsd ?? null,
        budgetUsd: l.monthlyBudgetUsd,
      },
    });
  }
  return { leads: out };
}
