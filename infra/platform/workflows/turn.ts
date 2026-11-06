import { HarnessAgent } from "@ai-sdk/harness/agent";
import { type HarnessWorkflowState, createHarnessWorkflowState, runHarnessAgentTimeSlice } from "@ai-sdk/workflow-harness";
import { tapped } from "../src/harness/tap.ts";
import { costUsd } from "../src/prices.ts";
import { DOC_PATH, REPO_DIR, leadInstructions, turnPrompt, workerInstructions } from "../src/prompt.ts";
import { type Runtime, type SandboxSession, runtime } from "../src/runtime.ts";
import { GITHUB_PLACEHOLDER, envExports, serviceEnv } from "../src/services.ts";
import { type Shell, mustRun, shq } from "../src/shell.ts";
import { leadAgentId } from "../src/slack.ts";
import type { HarnessName, InboxEvent, ReplyTo, Usage, Worker } from "../src/store.ts";
import { hostTools } from "../src/tools.ts";
import { finalText, foldChunks } from "../src/transcript.ts";
import { wake } from "../src/wake.ts";

export type TurnCtx = {
  turnId: string;
  lane: string;
  kind: "lead" | "worker";
  workerId: string | null;
  harness: HarnessName;
  model: string;
  sandboxName: string;
  subject: string;
  instructions: string;
  replyTo: ReplyTo | null;
  branch: string | null;
  apis: string[];
};

export type Begun = { ctx: TurnCtx; state: HarnessWorkflowState };

export type WorkerInput = Pick<Worker, "id" | "lane" | "task" | "branch" | "harness" | "model" | "replyTo">;

const SLACK_LIMIT = 3500;
const NO_REPLY = "NO_REPLY";
const UV_VERSION = "0.13.7";

const openSandbox = (rt: Runtime, ctx: TurnCtx) =>
  rt.sandbox.open(ctx.sandboxName, { lane: ctx.lane, subject: ctx.subject, credentials: rt.credentials, apis: ctx.apis });

// `research` (no dependencies of its own) from the checkout, editable so each
// fetch updates it, on a pinned uv; a checkout without it gets nothing.
const INSTALL_RESEARCH = [
  `test -d ${REPO_DIR}/tools/research || exit 0`,
  `test -x "$HOME/.local/bin/uv" || curl -LsSf https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh | INSTALLER_NO_MODIFY_PATH=1 sh -s -- -q`,
  `"$HOME/.local/bin/uv" tool install -q --python 3.12 --editable ./${REPO_DIR}/tools/research`,
].join("; ");

// What a lead's shell exports: the granted APIs' placeholders, the notes
// token's (research reads routes with it), and the agent id research posts
// as; the env file holds no secret.
const leadEnv = (rt: Runtime, ctx: TurnCtx): Record<string, string> => ({
  ...serviceEnv(ctx.apis),
  RESEARCH_NOTES_TOKEN: GITHUB_PLACEHOLDER,
  RESEARCH_AGENT_URL: `${rt.baseUrl}/api/leads?lane=${ctx.lane}&agent=${leadAgentId(ctx.lane)}`,
});

// The checkout is cloned once per sandbox (the harness marks a sandbox as
// bootstrapped per bootstrapHash, and a persistent sandbox keeps the marker
// and the clone) and fetched at the start of every turn; a worker's sits on
// its own branch.
async function buildAgent(rt: Runtime, ctx: TurnCtx) {
  const repo = rt.repoUrl;
  const checkout = (branch: string) =>
    `git -C ${REPO_DIR} checkout -q ${shq(branch)} 2>/dev/null || git -C ${REPO_DIR} checkout -q -b ${shq(branch)} origin/HEAD`;
  const env = ctx.kind === "lead" ? leadEnv(rt, ctx) : {};
  return new HarnessAgent({
    harness: await rt.harness(ctx.harness, { auth: await rt.credentials.modelAuth(ctx.lane) }),
    model: ctx.model,
    instructions: ctx.instructions,
    tools: hostTools({ turnId: ctx.turnId, lane: ctx.lane, kind: ctx.kind, replyTo: ctx.replyTo }),
    permissionMode: "allow-all",
    sandboxConfig: {
      workDir: ".",
      ...(repo && {
        bootstrapHash: `checkout:${repo}:uv-${UV_VERSION}`,
        onBootstrap: async ({ session }) => {
          await mustRun(session, `test -d ${REPO_DIR}/.git || git clone -q --filter=blob:none ${shq(repo)} ${REPO_DIR}`);
          if (ctx.kind === "lead") await mustRun(session, INSTALL_RESEARCH);
        },
      }),
      // git and gh inside the sandbox send a placeholder; the sandbox firewall
      // swaps in the short-lived token on the way out.
      onSession: async ({ session }) => {
        await mustRun(
          session,
          `git config --global credential.helper '!f() { echo username=x-access-token; echo password=${GITHUB_PLACEHOLDER}; }; f' && git config --global user.name verity-platform && git config --global user.email platform@verity.invalid`,
        );
        if (Object.keys(env).length > 0) {
          await mustRun(
            session,
            `printf '%s\\n' ${shq(envExports(env))} > "$HOME/.verity-env" && for f in "$HOME/.bashrc" "$HOME/.profile"; do grep -qs verity-env "$f" || echo '. "$HOME/.verity-env"' >> "$f"; done`,
          );
        }
        if (!repo) return;
        await session.run({ command: `git -C ${REPO_DIR} fetch -q origin` });
        if (ctx.branch) await mustRun(session, checkout(ctx.branch));
      },
    },
  });
}

export async function registerRun(lane: string, runId: string) {
  "use step";
  const rt = await runtime();
  await rt.store.setRunId(lane, runId);
}

// Claims everything pending for the lead and opens a turn on it, or returns
// null when there is nothing to do or another run's turn holds the lease.
export async function beginLeadTurn(lane: string, turnId: string, runId: string): Promise<Begun | null> {
  "use step";
  const rt = await runtime();
  const lead = await rt.store.getLead(lane);
  if (!lead || lead.status !== "active") return null;
  if (!(await rt.store.acquireLease(lane, turnId, rt.leaseSeconds))) return null;
  const events = await rt.store.claimEvents(lane, turnId);
  if (events.length === 0) {
    await rt.store.releaseLease(lane, turnId);
    return null;
  }
  const ctx: TurnCtx = {
    turnId,
    lane,
    kind: "lead",
    workerId: null,
    harness: lead.harness,
    model: lead.model,
    sandboxName: `lead-${lane}`,
    subject: lane,
    instructions: leadInstructions(lead, { checkout: rt.repoUrl != null }),
    replyTo: events.find((e) => e.replyTo && "channel" in e.replyTo)?.replyTo ?? null,
    branch: null,
    apis: lead.apis,
  };
  const prompt = turnPrompt(events);
  if (await rt.store.startTurn({ id: turnId, lane, kind: "lead", runId, harness: lead.harness, model: lead.model, docRevBefore: lead.docRev })) {
    await rt.store.appendParts(turnId, [
      { type: "instructions", content: null, text: ctx.instructions },
      { type: "prompt", content: { events: events.map((e) => e.id), docRev: lead.docRev }, text: prompt },
    ]);
  }
  const sandbox = await openSandbox(rt, ctx);
  await sandbox.writeTextFile({ path: DOC_PATH, content: lead.doc });
  return { ctx, state: createHarnessWorkflowState({ sessionId: turnId, prompt }) };
}

export async function beginWorkerTurn(w: WorkerInput, runId: string): Promise<Begun> {
  "use step";
  const rt = await runtime();
  const turnId = `${w.id}-t1`;
  const ctx: TurnCtx = {
    turnId,
    lane: w.lane,
    kind: "worker",
    workerId: w.id,
    harness: w.harness,
    model: w.model,
    sandboxName: `worker-${w.id}`,
    subject: w.id,
    instructions: workerInstructions(w, { checkout: rt.repoUrl != null }),
    replyTo: w.replyTo,
    branch: w.branch,
    apis: [],
  };
  if (await rt.store.startTurn({ id: turnId, lane: w.lane, kind: "worker", workerId: w.id, runId, harness: w.harness, model: w.model })) {
    await rt.store.appendParts(turnId, [
      { type: "instructions", content: null, text: ctx.instructions },
      { type: "prompt", content: { workerId: w.id }, text: w.task },
    ]);
  }
  return { ctx, state: createHarnessWorkflowState({ sessionId: turnId, prompt: w.task }) };
}

// One time-boxed slice of a harness turn. A turn longer than a function may
// run is several slices, each its own step; the harness suspends and resumes
// the runtime in the sandbox between them.
export async function sliceTurn(ctx: TurnCtx, state: HarnessWorkflowState): Promise<{ state: HarnessWorkflowState; usage: Usage; text: string }> {
  "use step";
  const rt = await runtime();
  const sandboxSession = await openSandbox(rt, ctx);
  const t = tapped(await buildAgent(rt, ctx));
  const chunks: { type: string }[] = [];
  const writable = new WritableStream<{ type: string }>({
    write(c) {
      chunks.push(c);
    },
  });
  const next = await runHarnessAgentTimeSlice({
    agent: t.agent,
    state,
    sandboxSession: sandboxSession as never,
    writable,
    timeSliceSeconds: rt.sliceSeconds,
  });
  const usage = await t.usage();
  const parts = foldChunks(chunks);
  await rt.store.appendParts(ctx.turnId, parts);
  if (ctx.kind === "lead") await rt.store.renewLease(ctx.lane, ctx.turnId, rt.leaseSeconds);
  return { state: next, usage, text: finalText(parts) };
}

type Thread = { channel: string; threadTs: string };

// A reply that can't be posted (research refuses one past its length, say) is
// recorded and handed back to the lead once, never retried.
async function replyOnce(rt: Runtime, ctx: TurnCtx, session: Shell, t: Thread, text: string, events: InboxEvent[]) {
  const same = (c: unknown) => (c as Thread | null)?.channel === t.channel && (c as Thread | null)?.threadTs === t.threadTs;
  if ((await rt.store.transcript(ctx.turnId)).some((p) => p.type === "post" && same(p.content))) return;
  const body = text.length > SLACK_LIMIT ? `${text.slice(0, SLACK_LIMIT)}\n… (full reply in transcript ${ctx.turnId})` : text;
  try {
    await rt.reply(session, { lane: ctx.lane, channel: t.channel, threadTs: t.threadTs, text: body });
    await rt.store.appendParts(ctx.turnId, [{ type: "post", content: t, text: body }]);
  } catch (err) {
    const why = err instanceof Error ? err.message : String(err);
    await rt.store.appendParts(ctx.turnId, [{ type: "post-error", content: t, text: why }]);
    if (events.every((e) => e.kind === "reply-failed")) return;
    await wake(ctx.lane, {
      lane: ctx.lane,
      source: "slack",
      kind: "reply-failed",
      payload: { ...t, text: `Your reply in ${t.channel} thread ${t.threadTs} was not posted (${why}). Post it yourself with research msg reply if it is still needed.` },
      replyTo: null,
      dedupeKey: `reply-failed:${ctx.turnId}:${t.channel}:${t.threadTs}`,
    });
  }
}

// The reply goes to every Slack thread the turn's events came from, posted
// from the lead's sandbox (Runtime.reply), and back to every lead that asked.
// A turn on events with no return address (ticks, say) posts nothing; an MCP
// caller reads its answer with get_answer.
async function deliver(rt: Runtime, ctx: TurnCtx, events: InboxEvent[], text: string) {
  const threads = new Map<string, Thread>();
  const asks: { lane: string; e: InboxEvent }[] = [];
  for (const e of events) {
    const r = e.replyTo;
    if (r && "channel" in r) threads.set(`${r.channel}/${r.threadTs}`, { channel: r.channel, threadTs: r.threadTs });
    if (r && "lane" in r) asks.push({ lane: r.lane, e });
  }
  const quiet = text === "" || text === NO_REPLY;
  if (!quiet && threads.size > 0) {
    const session: SandboxSession = await openSandbox(rt, ctx);
    for (const t of threads.values()) await replyOnce(rt, ctx, session as Shell, t, text, events);
  }
  for (const { lane, e } of asks) {
    await wake(lane, {
      lane,
      source: "lead",
      kind: "answer",
      payload: { from: ctx.lane, question: e.payload.text ?? null, text: quiet ? "(no answer)" : text },
      replyTo: (e.payload.askerReplyTo as ReplyTo | undefined) ?? null,
      dedupeKey: `answer:${ctx.turnId}:${e.id}`,
    });
  }
}

export async function endTurn(
  ctx: TurnCtx,
  outcome: { status: HarnessWorkflowState["status"]; error?: string },
  usage: Usage,
  text: string,
): Promise<{ costUsd: number; text: string }> {
  "use step";
  const rt = await runtime();
  let docRevAfter: number | undefined;
  if (ctx.kind === "lead") {
    const sandbox = await openSandbox(rt, ctx);
    const doc = await sandbox.readTextFile({ path: DOC_PATH });
    if (doc != null) docRevAfter = await rt.store.saveDoc(ctx.lane, doc, ctx.turnId);
  }
  const failed = outcome.status === "failed";
  const cost = costUsd(ctx.model, usage);
  const detail = failed ? (await rt.store.transcript(ctx.turnId)).filter((p) => p.type === "error").at(-1)?.text : undefined;
  const error = failed ? [outcome.error ?? "unknown error", detail].filter(Boolean).join(": ") : undefined;
  const reply = failed ? `This turn failed (${error}). Its transcript is ${ctx.turnId}.` : text;
  await rt.store.finishTurn(ctx.turnId, {
    status: failed ? "failed" : "finished",
    usage,
    costUsd: cost,
    finalText: reply,
    error,
    docRevAfter,
  });
  if (ctx.kind === "lead") {
    await deliver(rt, ctx, await rt.store.eventsForTurn(ctx.turnId), reply);
    await rt.store.releaseLease(ctx.lane, ctx.turnId);
  }
  return { costUsd: cost, text: reply };
}

export async function reportToLead(w: WorkerInput, outcome: { failed: boolean; text: string }) {
  "use step";
  const rt = await runtime();
  const status = outcome.failed ? "failed" : "finished";
  await rt.store.finishWorker(w.id, status, outcome.text);
  await wake(w.lane, {
    lane: w.lane,
    source: "worker",
    kind: "worker-done",
    payload: { workerId: w.id, status, branch: w.branch, text: `Worker ${w.id} ${status}: ${outcome.text}` },
    replyTo: w.replyTo,
    dedupeKey: `worker-done:${w.id}`,
  });
  await rt.sandbox.park(`worker-${w.id}`);
}

// Stops an idle lead's sandbox. Parking holds the lead's lease, so it never
// stops the sandbox under a turn that a newer run has begun; it returns true
// when events arrived meanwhile, so the caller drains again.
export async function parkLead(lane: string, runId: string): Promise<boolean> {
  "use step";
  const rt = await runtime();
  const key = `park-${runId}`;
  if (!(await rt.store.acquireLease(lane, key, 120))) return false;
  try {
    await rt.sandbox.park(`lead-${lane}`);
  } finally {
    await rt.store.releaseLease(lane, key);
  }
  return (await rt.store.pendingCount(lane)) > 0;
}
