import { type Db, execScript } from "./db.ts";
import { SCHEMA } from "./schema.ts";

export type HarnessName = "claude-code" | "codex" | "fake";

export type Lead = {
  lane: string;
  charter: string;
  doc: string;
  docRev: number;
  harness: HarnessName;
  model: string;
  slackChannel: string | null;
  cronEveryMinutes: number | null;
  nextTickAt: string | null;
  monthlyBudgetUsd: number | null;
  gatewayKeyId: string | null;
  status: "active" | "paused" | "retired";
  runId: string | null;
  runStartedAt: string | null;
};

// Where a turn's reply goes: a Slack thread, another lead's inbox (an answer to
// ask_lead), or nowhere but the record (an MCP caller reads it with get_answer).
export type ReplyTo = { channel: string; threadTs?: string } | { lane: string } | { caller: string };

export type InboxEvent = {
  id: number;
  lane: string;
  source: "slack" | "cron" | "job" | "worker" | "mcp" | "lead";
  kind: string;
  payload: Record<string, unknown>;
  replyTo: ReplyTo | null;
  receivedAt: string;
};

export type NewEvent = Omit<InboxEvent, "id" | "receivedAt"> & { dedupeKey?: string };

export type Usage = {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  reasoningTokens: number;
};

export type TranscriptPart = { type: string; content: unknown; text: string };

export type Turn = {
  id: string;
  lane: string;
  kind: "lead" | "worker";
  workerId: string | null;
  runId: string | null;
  status: "running" | "finished" | "failed";
  harness: string;
  model: string;
  startedAt: string;
  endedAt: string | null;
  usage: Usage;
  costUsd: number;
  finalText: string | null;
  error: string | null;
};

export type Worker = {
  id: string;
  lane: string;
  task: string;
  branch: string | null;
  harness: HarnessName;
  model: string;
  status: "running" | "finished" | "failed";
  runId: string | null;
  replyTo: ReplyTo | null;
  result: string | null;
};

export type Job = {
  id: string;
  lane: string;
  turnId: string | null;
  spec: Record<string, unknown>;
  status: string;
  callbackToken: string;
  externalId: string | null;
  replyTo: ReplyTo | null;
  result: unknown;
};

export type Grant = {
  subject: string;
  kind: string;
  scope: string;
  ref: string;
  expiresAt: string | null;
};

const iso = (v: unknown) => (v == null ? null : new Date(v as string).toISOString());
const num = (v: unknown) => (v == null ? 0 : Number(v));

function toLead(r: Record<string, unknown>): Lead {
  return {
    lane: r.lane as string,
    charter: r.charter as string,
    doc: r.doc as string,
    docRev: num(r.doc_rev),
    harness: r.harness as HarnessName,
    model: r.model as string,
    slackChannel: (r.slack_channel as string) ?? null,
    cronEveryMinutes: r.cron_every_minutes == null ? null : num(r.cron_every_minutes),
    nextTickAt: iso(r.next_tick_at),
    monthlyBudgetUsd: r.monthly_budget_usd == null ? null : num(r.monthly_budget_usd),
    gatewayKeyId: (r.gateway_key_id as string) ?? null,
    status: r.status as Lead["status"],
    runId: (r.run_id as string) ?? null,
    runStartedAt: iso(r.run_started_at),
  };
}

function toEvent(r: Record<string, unknown>): InboxEvent {
  return {
    id: num(r.id),
    lane: r.lane as string,
    source: r.source as InboxEvent["source"],
    kind: r.kind as string,
    payload: r.payload as Record<string, unknown>,
    replyTo: (r.reply_to as ReplyTo) ?? null,
    receivedAt: iso(r.received_at)!,
  };
}

function toTurn(r: Record<string, unknown>): Turn {
  return {
    id: r.id as string,
    lane: r.lane as string,
    kind: r.kind as Turn["kind"],
    workerId: (r.worker_id as string) ?? null,
    runId: (r.run_id as string) ?? null,
    status: r.status as Turn["status"],
    harness: r.harness as string,
    model: r.model as string,
    startedAt: iso(r.started_at)!,
    endedAt: iso(r.ended_at),
    usage: {
      inputTokens: num(r.input_tokens),
      outputTokens: num(r.output_tokens),
      cacheReadTokens: num(r.cache_read_tokens),
      cacheWriteTokens: num(r.cache_write_tokens),
      reasoningTokens: num(r.reasoning_tokens),
    },
    costUsd: num(r.cost_usd),
    finalText: (r.final_text as string) ?? null,
    error: (r.error as string) ?? null,
  };
}

function toWorker(r: Record<string, unknown>): Worker {
  return {
    id: r.id as string,
    lane: r.lane as string,
    task: r.task as string,
    branch: (r.branch as string) ?? null,
    harness: r.harness as HarnessName,
    model: r.model as string,
    status: r.status as Worker["status"],
    runId: (r.run_id as string) ?? null,
    replyTo: (r.reply_to as ReplyTo) ?? null,
    result: (r.result as string) ?? null,
  };
}

function toJob(r: Record<string, unknown>): Job {
  return {
    id: r.id as string,
    lane: r.lane as string,
    turnId: (r.turn_id as string) ?? null,
    spec: r.spec as Record<string, unknown>,
    status: r.status as string,
    callbackToken: r.callback_token as string,
    externalId: (r.external_id as string) ?? null,
    replyTo: (r.reply_to as ReplyTo) ?? null,
    result: r.result ?? null,
  };
}

export type Store = ReturnType<typeof createStore>;

export function createStore(db: Db) {
  return {
    db,

    migrate: () => execScript(db, SCHEMA),

    async upsertLead(l: Partial<Lead> & Pick<Lead, "lane" | "model">): Promise<Lead> {
      const { rows } = await db.query(
        `insert into leads (lane, charter, doc, harness, model, slack_channel, cron_every_minutes,
                            next_tick_at, monthly_budget_usd, status)
         values ($1, coalesce($2, ''), coalesce($3, ''), coalesce($4, 'claude-code'), $5, $6, $7,
                 case when $7::int is null then null else now() + make_interval(mins => $7::int) end,
                 $8, coalesce($9, 'active'))
         on conflict (lane) do update set
           charter = coalesce($2, leads.charter),
           doc = coalesce($3, leads.doc),
           harness = coalesce($4, leads.harness),
           model = $5,
           slack_channel = coalesce($6, leads.slack_channel),
           cron_every_minutes = coalesce($7, leads.cron_every_minutes),
           monthly_budget_usd = coalesce($8, leads.monthly_budget_usd),
           status = coalesce($9, leads.status),
           updated_at = now()
         returning *`,
        [
          l.lane,
          l.charter ?? null,
          l.doc ?? null,
          l.harness ?? null,
          l.model,
          l.slackChannel ?? null,
          l.cronEveryMinutes ?? null,
          l.monthlyBudgetUsd ?? null,
          l.status ?? null,
        ],
      );
      return toLead(rows[0]!);
    },

    async getLead(lane: string): Promise<Lead | null> {
      const { rows } = await db.query(`select * from leads where lane = $1`, [lane]);
      return rows[0] ? toLead(rows[0]) : null;
    },

    async listLeads(): Promise<Lead[]> {
      const { rows } = await db.query(`select * from leads order by lane`);
      return rows.map(toLead);
    },

    async leadByChannel(channel: string): Promise<Lead | null> {
      const { rows } = await db.query(
        `select * from leads where slack_channel = $1 and status = 'active' order by lane limit 1`,
        [channel],
      );
      return rows[0] ? toLead(rows[0]) : null;
    },

    async setLeadStatus(lane: string, status: Lead["status"]) {
      await db.query(`update leads set status = $2, updated_at = now() where lane = $1`, [lane, status]);
    },

    // The key is stored sealed (credentials.seal) and is opened only in the
    // step that hands it to the harness; it is never part of a Lead.
    async setGatewayKey(lane: string, keyId: string, sealed: string) {
      await db.query(
        `update leads set gateway_key_id = $2, gateway_key_sealed = $3, updated_at = now() where lane = $1`,
        [lane, keyId, sealed],
      );
    },

    async sealedGatewayKey(lane: string): Promise<string | null> {
      const { rows } = await db.query<{ s: string | null }>(
        `select gateway_key_sealed as s from leads where lane = $1`,
        [lane],
      );
      return rows[0]?.s ?? null;
    },

    async setRunId(lane: string, runId: string) {
      await db.query(
        `update leads set run_id = $2,
           run_started_at = case when run_id = $2 then run_started_at else now() end, updated_at = now()
         where lane = $1`,
        [lane, runId],
      );
    },

    // Returns null when the event's dedupe key was already recorded (Slack retries,
    // a callback fired twice), so a duplicate never causes a second turn.
    async addEvent(e: NewEvent): Promise<InboxEvent | null> {
      const { rows } = await db.query(
        `insert into events (lane, source, kind, dedupe_key, payload, reply_to)
         values ($1, $2, $3, $4, $5, $6)
         on conflict (dedupe_key) do nothing
         returning *`,
        [e.lane, e.source, e.kind, e.dedupeKey ?? null, JSON.stringify(e.payload), e.replyTo ? JSON.stringify(e.replyTo) : null],
      );
      return rows[0] ? toEvent(rows[0]) : null;
    },

    async pendingCount(lane: string): Promise<number> {
      const { rows } = await db.query<{ n: string }>(
        `select count(*) as n from events where lane = $1 and turn_id is null`,
        [lane],
      );
      return num(rows[0]?.n);
    },

    // One turn at a time per lead, across runs: the lease is the lock, and it
    // expires so a crashed turn cannot wedge the lead.
    async acquireLease(lane: string, turnId: string, seconds: number): Promise<boolean> {
      const { rows } = await db.query(
        `update leads set lease_turn = $2, lease_until = now() + make_interval(secs => $3)
         where lane = $1 and status = 'active'
           and (lease_turn is null or lease_turn = $2 or lease_until < now())
         returning lane`,
        [lane, turnId, seconds],
      );
      return rows.length === 1;
    },

    async releaseLease(lane: string, turnId: string) {
      await db.query(
        `update leads set lease_turn = null, lease_until = null where lane = $1 and lease_turn = $2`,
        [lane, turnId],
      );
    },

    // Idempotent per turn: a retried step gets back the events it already claimed.
    async claimEvents(lane: string, turnId: string, limit = 50): Promise<InboxEvent[]> {
      await db.query(
        `update events set turn_id = $2
         where id in (select id from events where lane = $1 and turn_id is null order by id limit $3)
           and not exists (select 1 from events where turn_id = $2)`,
        [lane, turnId, limit],
      );
      const { rows } = await db.query(`select * from events where turn_id = $1 order by id`, [turnId]);
      return rows.map(toEvent);
    },

    async renewLease(lane: string, turnId: string, seconds: number) {
      await db.query(
        `update leads set lease_until = now() + make_interval(secs => $3) where lane = $1 and lease_turn = $2`,
        [lane, turnId, seconds],
      );
    },

    async eventsForTurn(turnId: string): Promise<InboxEvent[]> {
      const { rows } = await db.query(`select * from events where turn_id = $1 order by id`, [turnId]);
      return rows.map(toEvent);
    },

    // What became of one event: pending, or the turn that took it and its reply.
    async eventOutcome(eventId: number) {
      const { rows } = await db.query(
        `select e.lane, e.turn_id, t.status, t.final_text from events e left join turns t on t.id = e.turn_id
         where e.id = $1`,
        [eventId],
      );
      const r = rows[0];
      if (!r) return null;
      return {
        lane: r.lane as string,
        turnId: (r.turn_id as string) ?? null,
        status: ((r.status as string) ?? "pending") as "pending" | Turn["status"],
        reply: (r.final_text as string) ?? null,
      };
    },

    async startTurn(t: {
      id: string;
      lane: string;
      kind: "lead" | "worker";
      workerId?: string;
      runId?: string;
      harness: string;
      model: string;
      docRevBefore?: number;
    }): Promise<boolean> {
      const { rows } = await db.query(
        `insert into turns (id, lane, kind, worker_id, run_id, harness, model, doc_rev_before)
         values ($1, $2, $3, $4, $5, $6, $7, $8)
         on conflict (id) do nothing returning id`,
        [t.id, t.lane, t.kind, t.workerId ?? null, t.runId ?? null, t.harness, t.model, t.docRevBefore ?? null],
      );
      return rows.length === 1;
    },

    async finishTurn(
      id: string,
      f: { status: "finished" | "failed"; usage: Usage; costUsd: number; finalText?: string; error?: string; docRevAfter?: number },
    ) {
      await db.query(
        `update turns set status = $2, ended_at = now(), input_tokens = $3, output_tokens = $4,
           cache_read_tokens = $5, cache_write_tokens = $6, reasoning_tokens = $7, cost_usd = $8,
           final_text = $9, error = $10, doc_rev_after = $11
         where id = $1`,
        [
          id,
          f.status,
          f.usage.inputTokens,
          f.usage.outputTokens,
          f.usage.cacheReadTokens,
          f.usage.cacheWriteTokens,
          f.usage.reasoningTokens,
          f.costUsd,
          f.finalText ?? null,
          f.error ?? null,
          f.docRevAfter ?? null,
        ],
      );
    },

    async getTurn(id: string): Promise<Turn | null> {
      const { rows } = await db.query(`select * from turns where id = $1`, [id]);
      return rows[0] ? toTurn(rows[0]) : null;
    },

    async listTurns(q: { lane?: string; since?: string; until?: string; limit?: number } = {}): Promise<Turn[]> {
      const { rows } = await db.query(
        `select * from turns
         where ($1::text is null or lane = $1)
           and ($2::timestamptz is null or started_at >= $2)
           and ($3::timestamptz is null or started_at < $3)
         order by started_at desc limit $4`,
        [q.lane ?? null, q.since ?? null, q.until ?? null, q.limit ?? 100],
      );
      return rows.map(toTurn);
    },

    // Turns whose lease outlived them: the watchdog's input.
    async staleTurns(olderThanMinutes: number): Promise<Turn[]> {
      const { rows } = await db.query(
        `select * from turns where status = 'running'
           and started_at < now() - make_interval(mins => $1) order by started_at`,
        [olderThanMinutes],
      );
      return rows.map(toTurn);
    },

    async appendParts(turnId: string, parts: TranscriptPart[]) {
      if (parts.length === 0) return;
      const { rows } = await db.query<{ n: number }>(
        `select coalesce(max(seq), -1) + 1 as n from transcript_parts where turn_id = $1`,
        [turnId],
      );
      let seq = num(rows[0]?.n);
      for (const p of parts) {
        await db.query(
          `insert into transcript_parts (turn_id, seq, type, content, text) values ($1, $2, $3, $4, $5)`,
          [turnId, seq++, p.type, JSON.stringify(p.content ?? null), p.text],
        );
      }
    },

    async transcript(turnId: string): Promise<TranscriptPart[]> {
      const { rows } = await db.query(
        `select type, content, text from transcript_parts where turn_id = $1 order by seq`,
        [turnId],
      );
      return rows as TranscriptPart[];
    },

    // Full-text search over every stored turn, filtered by lead and time.
    async searchTranscripts(q: { text?: string; lane?: string; since?: string; until?: string; limit?: number }) {
      const { rows } = await db.query(
        `select p.turn_id, p.seq, p.type, p.text, t.lane, t.kind, t.started_at
         from transcript_parts p join turns t on t.id = p.turn_id
         where ($1::text is null or p.tsv @@ websearch_to_tsquery('english', $1))
           and ($2::text is null or t.lane = $2)
           and ($3::timestamptz is null or t.started_at >= $3)
           and ($4::timestamptz is null or t.started_at < $4)
         order by t.started_at desc, p.seq limit $5`,
        [q.text ?? null, q.lane ?? null, q.since ?? null, q.until ?? null, q.limit ?? 50],
      );
      return rows.map((r) => ({
        turnId: r.turn_id as string,
        seq: num(r.seq),
        type: r.type as string,
        text: r.text as string,
        lane: r.lane as string,
        kind: r.kind as string,
        startedAt: iso(r.started_at)!,
      }));
    },

    async saveDoc(lane: string, doc: string, turnId: string): Promise<number> {
      const { rows } = await db.query<{ doc_rev: number; changed: boolean }>(
        `update leads set doc = $2, doc_rev = case when doc = $2 then doc_rev else doc_rev + 1 end,
           updated_at = now()
         where lane = $1 returning doc_rev`,
        [lane, doc],
      );
      const rev = num(rows[0]?.doc_rev);
      await db.query(
        `insert into doc_revisions (lane, rev, doc, turn_id) values ($1, $2, $3, $4) on conflict do nothing`,
        [lane, rev, doc, turnId],
      );
      return rev;
    },

    async docAt(lane: string, rev: number): Promise<string | null> {
      const { rows } = await db.query(`select doc from doc_revisions where lane = $1 and rev = $2`, [lane, rev]);
      return rows[0] ? (rows[0].doc as string) : null;
    },

    async docRevisions(lane: string) {
      const { rows } = await db.query(
        `select rev, turn_id, at from doc_revisions where lane = $1 order by rev`,
        [lane],
      );
      return rows.map((r) => ({ rev: num(r.rev), turnId: r.turn_id as string, at: iso(r.at)! }));
    },

    // Per-lead spend from the platform's own token accounting. The gateway's
    // per-key spend (credentials.gatewaySpend) is the billed figure; this one
    // also splits by turn, worker and day.
    async spend(q: { since?: string; until?: string } = {}) {
      const { rows } = await db.query(
        `select lane, kind, count(*) as turns, sum(cost_usd) as cost_usd,
                sum(input_tokens) as input_tokens, sum(output_tokens) as output_tokens,
                sum(cache_read_tokens) as cache_read_tokens
         from turns
         where ($1::timestamptz is null or started_at >= $1)
           and ($2::timestamptz is null or started_at < $2)
         group by lane, kind order by lane, kind`,
        [q.since ?? null, q.until ?? null],
      );
      return rows.map((r) => ({
        lane: r.lane as string,
        kind: r.kind as string,
        turns: num(r.turns),
        costUsd: num(r.cost_usd),
        inputTokens: num(r.input_tokens),
        outputTokens: num(r.output_tokens),
        cacheReadTokens: num(r.cache_read_tokens),
      }));
    },

    async createWorker(w: Omit<Worker, "status" | "result" | "runId">) {
      await db.query(
        `insert into workers (id, lane, task, branch, harness, model, reply_to) values ($1, $2, $3, $4, $5, $6, $7)
         on conflict (id) do nothing`,
        [w.id, w.lane, w.task, w.branch, w.harness, w.model, w.replyTo ? JSON.stringify(w.replyTo) : null],
      );
    },

    async setWorkerRun(id: string, runId: string) {
      await db.query(`update workers set run_id = $2 where id = $1`, [id, runId]);
    },

    async finishWorker(id: string, status: "finished" | "failed", result: string) {
      await db.query(`update workers set status = $2, result = $3, ended_at = now() where id = $1`, [id, status, result]);
    },

    async getWorker(id: string): Promise<Worker | null> {
      const { rows } = await db.query(`select * from workers where id = $1`, [id]);
      return rows[0] ? toWorker(rows[0]) : null;
    },

    async listWorkers(q: { lane?: string; status?: string } = {}): Promise<Worker[]> {
      const { rows } = await db.query(
        `select * from workers where ($1::text is null or lane = $1) and ($2::text is null or status = $2)
         order by started_at desc`,
        [q.lane ?? null, q.status ?? null],
      );
      return rows.map(toWorker);
    },

    async createJob(j: {
      id: string;
      lane: string;
      turnId: string | null;
      spec: Record<string, unknown>;
      callbackToken: string;
      replyTo: ReplyTo | null;
    }) {
      await db.query(
        `insert into jobs (id, lane, turn_id, spec, callback_token, reply_to) values ($1, $2, $3, $4, $5, $6)`,
        [j.id, j.lane, j.turnId, JSON.stringify(j.spec), j.callbackToken, j.replyTo ? JSON.stringify(j.replyTo) : null],
      );
    },

    async setJobExternalId(id: string, externalId: string) {
      await db.query(`update jobs set external_id = $2 where id = $1`, [id, externalId]);
    },

    async getJob(id: string): Promise<Job | null> {
      const { rows } = await db.query(`select * from jobs where id = $1`, [id]);
      return rows[0] ? toJob(rows[0]) : null;
    },

    async jobByToken(token: string): Promise<Job | null> {
      const { rows } = await db.query(`select * from jobs where callback_token = $1`, [token]);
      return rows[0] ? toJob(rows[0]) : null;
    },

    async finishJob(id: string, status: string, result: unknown) {
      await db.query(`update jobs set status = $2, result = $3, finished_at = now() where id = $1`, [
        id,
        status,
        JSON.stringify(result ?? null),
      ]);
    },

    async listJobs(q: { lane?: string } = {}): Promise<Job[]> {
      const { rows } = await db.query(
        `select * from jobs where ($1::text is null or lane = $1) order by submitted_at desc`,
        [q.lane ?? null],
      );
      return rows.map(toJob);
    },

    // A Slack thread belongs to the lead that started or answered it, so a
    // reply in the thread reaches that lead without naming it.
    async bindThread(channel: string, threadTs: string, lane: string) {
      await db.query(
        `insert into threads (channel, thread_ts, lane) values ($1, $2, $3) on conflict do nothing`,
        [channel, threadTs, lane],
      );
    },

    async threadLane(channel: string, threadTs: string): Promise<string | null> {
      const { rows } = await db.query<{ lane: string }>(
        `select lane from threads where channel = $1 and thread_ts = $2`,
        [channel, threadTs],
      );
      return rows[0]?.lane ?? null;
    },

    async recordGrant(g: Grant) {
      await db.query(
        `insert into grants (subject, kind, scope, ref, expires_at) values ($1, $2, $3, $4, $5)`,
        [g.subject, g.kind, g.scope, g.ref, g.expiresAt],
      );
    },

    async listGrants(subject?: string) {
      const { rows } = await db.query(
        `select subject, kind, scope, ref, issued_at, expires_at, revoked_at from grants
         where ($1::text is null or subject = $1) order by issued_at desc`,
        [subject ?? null],
      );
      return rows.map((r) => ({
        subject: r.subject as string,
        kind: r.kind as string,
        scope: r.scope as string,
        ref: r.ref as string,
        issuedAt: iso(r.issued_at)!,
        expiresAt: iso(r.expires_at),
        revokedAt: iso(r.revoked_at),
      }));
    },

    // True the first time a key is raised, so each alert is posted once.
    async raiseAlert(key: string, text: string): Promise<boolean> {
      const { rows } = await db.query(
        `insert into alerts (key, text) values ($1, $2) on conflict do nothing returning key`,
        [key, text],
      );
      return rows.length === 1;
    },

    async leadHealth(lane: string) {
      const { rows } = await db.query(
        `select
           (select count(*) from events where lane = $1 and turn_id is null) as pending,
           (select extract(epoch from now() - min(received_at)) from events where lane = $1 and turn_id is null) as oldest_s,
           (select lease_until > now() from leads where lane = $1) as leased,
           (select max(started_at) from turns where lane = $1 and kind = 'lead') as last_turn`,
        [lane],
      );
      const r = rows[0]!;
      return {
        pending: num(r.pending),
        oldestPendingSeconds: r.oldest_s == null ? null : num(r.oldest_s),
        leaseHeld: r.leased === true,
        lastTurnAt: iso(r.last_turn),
      };
    },

    // Leads whose cron tick is due; advancing next_tick_at in the same statement
    // makes two overlapping cron invocations tick each lead once.
    async takeDueTicks(): Promise<string[]> {
      const { rows } = await db.query<{ lane: string }>(
        `update leads set next_tick_at = now() + make_interval(mins => cron_every_minutes)
         where status = 'active' and cron_every_minutes is not null
           and (next_tick_at is null or next_tick_at <= now())
         returning lane`,
      );
      return rows.map((r) => r.lane).sort();
    },
  };
}
