import { HarnessAgent } from "@ai-sdk/harness/agent";
import {
  type HarnessWorkflowAgent,
  type HarnessWorkflowState,
  createHarnessWorkflowState,
  runHarnessAgentTimeSlice,
} from "@ai-sdk/workflow-harness";
import { costUsd } from "../src/prices.ts";
import { DOC_PATH, leadInstructions, turnPrompt, workerInstructions } from "../src/prompt.ts";
import { GITHUB_PLACEHOLDER, type Runtime, runtime } from "../src/runtime.ts";
import type { HarnessName, InboxEvent, ReplyTo, Usage, Worker } from "../src/store.ts";
import { hostTools } from "../src/tools.ts";
import { ZERO_USAGE, addUsage, finalText, foldChunks, fromSdkUsage } from "../src/transcript.ts";
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
};

export type Begun = { ctx: TurnCtx; state: HarnessWorkflowState };

export type WorkerInput = Pick<Worker, "id" | "lane" | "task" | "branch" | "harness" | "model" | "replyTo">;

const SLACK_LIMIT = 3500;
const NO_REPLY = "NO_REPLY";

const openSandbox = (rt: Runtime, ctx: TurnCtx) =>
  rt.sandbox.open(ctx.sandboxName, { lane: ctx.lane, subject: ctx.subject, credentials: rt.credentials });

async function buildAgent(rt: Runtime, ctx: TurnCtx) {
  return new HarnessAgent({
    harness: await rt.harness(ctx.harness, { auth: await rt.credentials.modelAuth(ctx.lane) }),
    model: ctx.model,
    instructions: ctx.instructions,
    tools: hostTools({ turnId: ctx.turnId, lane: ctx.lane, kind: ctx.kind, replyTo: ctx.replyTo }),
    permissionMode: "allow-all",
    sandboxConfig: {
      workDir: ".",
      // git and gh inside the sandbox send a placeholder; the sandbox firewall
      // swaps in the short-lived token on the way out.
      onSession: async ({ session }) => {
        await session.run({
          command: `git config --global credential.helper '!f() { echo username=x-access-token; echo password=${GITHUB_PLACEHOLDER}; }; f' && git config --global user.name verity-platform && git config --global user.email platform@verity.invalid`,
        });
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
    instructions: leadInstructions(lead),
    replyTo: events.find((e) => e.replyTo && "channel" in e.replyTo)?.replyTo ?? null,
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
    instructions: workerInstructions(w),
    replyTo: w.replyTo,
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
  const agent = await buildAgent(rt, ctx);
  const results: { totalUsage: PromiseLike<unknown> }[] = [];
  const tapped: HarnessWorkflowAgent = {
    hasOutput: agent.hasOutput,
    createSession: (o) => agent.createSession(o as never),
    stream: async (o) => {
      const r = await agent.stream(o as never);
      results.push(r);
      return r as never;
    },
    continueStream: async (o) => {
      const r = await agent.continueStream(o as never);
      results.push(r);
      return r as never;
    },
  };
  const chunks: { type: string }[] = [];
  const writable = new WritableStream<{ type: string }>({
    write(c) {
      chunks.push(c);
    },
  });
  const next = await runHarnessAgentTimeSlice({
    agent: tapped,
    state,
    sandboxSession: sandboxSession as never,
    writable,
    timeSliceSeconds: rt.sliceSeconds,
  });
  let usage = ZERO_USAGE;
  for (const r of results) usage = addUsage(usage, fromSdkUsage(await Promise.resolve(r.totalUsage).catch(() => undefined)));
  const parts = foldChunks(chunks);
  await rt.store.appendParts(ctx.turnId, parts);
  if (ctx.kind === "lead") await rt.store.renewLease(ctx.lane, ctx.turnId, rt.leaseSeconds);
  return { state: next, usage, text: finalText(parts) };
}

async function postOnce(rt: Runtime, ctx: TurnCtx, channel: string, threadTs: string | undefined, text: string) {
  const already = (await rt.store.transcript(ctx.turnId)).some((p) => {
    const c = p.content as { channel?: string; threadTs?: string } | null;
    return p.type === "post" && c?.channel === channel && c?.threadTs === threadTs;
  });
  if (already) return;
  const body = text.length > SLACK_LIMIT ? `${text.slice(0, SLACK_LIMIT)}\n… (full reply in transcript ${ctx.turnId})` : text;
  const { ts } = await rt.slack.post({ channel, text: body, threadTs });
  await rt.store.bindThread(channel, threadTs ?? ts, ctx.lane);
  await rt.store.appendParts(ctx.turnId, [{ type: "post", content: { channel, threadTs, ts }, text: body }]);
}

// The reply goes to every Slack thread the turn's events came from, and back to
// every lead that asked (into the thread it asked from). A turn on events with
// no return address (ticks, say) posts in the lead's channel; an MCP caller's
// event gets no post and is read back with get_answer.
async function deliver(rt: Runtime, ctx: TurnCtx, events: InboxEvent[], text: string) {
  const lead = (await rt.store.getLead(ctx.lane))!;
  const threads = new Map<string, { channel: string; threadTs?: string }>();
  const asks: { lane: string; e: InboxEvent }[] = [];
  for (const e of events) {
    const r = e.replyTo;
    if (r && "channel" in r) threads.set(`${r.channel}/${r.threadTs}`, r);
    if (r && "lane" in r) asks.push({ lane: r.lane, e });
  }
  const quiet = text === "" || text === NO_REPLY;
  if (!quiet) {
    for (const t of threads.values()) await postOnce(rt, ctx, t.channel, t.threadTs, text);
    if (events.every((e) => !e.replyTo) && lead.slackChannel) await postOnce(rt, ctx, lead.slackChannel, undefined, text);
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
  const reply = failed ? `This turn failed: ${outcome.error ?? "unknown error"}. Its transcript is ${ctx.turnId}.` : text;
  await rt.store.finishTurn(ctx.turnId, {
    status: failed ? "failed" : "finished",
    usage,
    costUsd: cost,
    finalText: reply,
    error: outcome.error,
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
