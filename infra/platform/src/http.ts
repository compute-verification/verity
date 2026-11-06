import { timingSafeEqual } from "node:crypto";
import { tick } from "./cron.ts";
import { runtime } from "./runtime.ts";
import { mentionedLanes, parseSlackEvent, verifySlackSignature } from "./slack.ts";
import { wake } from "./wake.ts";

// The HTTP surface as plain (Request) => Response functions: the Next routes
// mount them, and tests call them directly.
type Env = Record<string, string | undefined>;

const json = (body: unknown, status = 200) => Response.json(body, { status, headers: { "cache-control": "no-store" } });

function bearerOk(req: Request, secret: string | undefined): boolean {
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
  const rt = await runtime();
  const leads = (await rt.store.listLeads()).filter((l) => l.status === "active");
  let lanes = mentionedLanes(m.text, leads.map((l) => l.lane));
  if (lanes.length === 0 && m.threadTs) {
    const owner = await rt.store.threadLane(m.channel, m.threadTs);
    if (owner) lanes = [owner];
  }
  if (lanes.length === 0) {
    const owner = leads.find((l) => l.slackChannel === m.channel);
    if (owner) lanes = [owner.lane];
  }
  const thread = m.threadTs ?? m.ts;
  for (const lane of lanes) {
    await rt.store.bindThread(m.channel, thread, lane);
    await wake(lane, {
      lane,
      source: "slack",
      kind: "message",
      payload: { text: m.text, user: m.user, channel: m.channel, ts: m.ts },
      replyTo: { channel: m.channel, threadTs: thread },
      dedupeKey: `slack:${m.eventId}:${lane}`,
    });
  }
  return json({ ok: true, routed: lanes });
}

// A job runner posts here when a job ends. The URL's token is the job's
// capability: it names one job and is good for nothing else.
export async function jobCallback(req: Request, token: string) {
  const rt = await runtime();
  const job = await rt.store.jobByToken(token);
  if (!job) return json({ error: "no such job" }, 404);
  const body = (await req.json().catch(() => ({}))) as { status?: string; summary?: string; result?: unknown };
  const status = body.status ?? "done";
  await rt.store.finishJob(job.id, status, body.result ?? body.summary ?? null);
  const r = await wake(job.lane, {
    lane: job.lane,
    source: "job",
    kind: "job-done",
    payload: { jobId: job.id, externalId: job.externalId, status, text: `Job ${job.id} ${status}: ${body.summary ?? ""}`.trim(), result: body.result ?? null },
    replyTo: job.replyTo,
    dedupeKey: `job-done:${job.id}`,
  });
  return json({ ok: true, eventId: r.eventId, duplicate: r.duplicate });
}

export async function cronTick(req: Request, env: Env = process.env) {
  if (!bearerOk(req, env.CRON_SECRET)) return json({ error: "unauthorized" }, 401);
  return json(await tick({ alertChannel: env.PLATFORM_ALERT_CHANNEL }));
}

export async function status(req: Request, env: Env = process.env) {
  if (!bearerOk(req, env.PLATFORM_API_TOKEN)) return json({ error: "unauthorized" }, 401);
  return json(await statusReport(new URL(req.url).searchParams.get("lane") ?? undefined));
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
