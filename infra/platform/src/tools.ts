import { randomBytes } from "node:crypto";
import { type ToolSet, tool } from "ai";
import { start } from "workflow/api";
import { z } from "zod";
import { workerWorkflow } from "../workflows/worker.ts";
import { digest } from "./credentials.ts";
import { runtime } from "./runtime.ts";
import type { HarnessName, ReplyTo } from "./store.ts";
import { wake } from "./wake.ts";

// replyTo is where the turn's own reply goes; a worker's or job's outcome is
// delivered to the lead with the same target, so the lead answers in the
// thread that asked for the work.
export type ToolCtx = { turnId: string; lane: string; kind: "lead" | "worker"; replyTo: ReplyTo | null };

// Host tools run in the step, outside the sandbox, with the platform's
// credentials. Ids derive from (turn, tool call), so a retried step that runs a
// tool again finds its earlier worker or job instead of making a second one.
export async function dispatchWorker(
  ctx: ToolCtx,
  input: { task: string; branch?: string; harness?: HarnessName; model?: string },
  toolCallId: string,
) {
  const rt = await runtime();
  const lead = (await rt.store.getLead(ctx.lane))!;
  const hex = digest(`${ctx.turnId}:${toolCallId}`).slice(0, 6);
  const id = `${ctx.lane}-w-${hex}`;
  const existing = await rt.store.getWorker(id);
  if (existing) return { workerId: id, branch: existing.branch, status: existing.status };
  const w = {
    id,
    lane: ctx.lane,
    task: input.task,
    branch: input.branch ?? `cursor/${ctx.lane}-${hex}`,
    harness: input.harness ?? lead.harness,
    model: input.model ?? lead.model,
    replyTo: ctx.replyTo,
  };
  await rt.store.createWorker(w);
  const run = await start(workerWorkflow, [w]);
  await rt.store.setWorkerRun(id, run.runId);
  return { workerId: id, branch: w.branch, status: "running" };
}

export async function submitJob(
  ctx: ToolCtx,
  input: { command: string; requirements?: Record<string, unknown>; leaseSeconds?: number },
  toolCallId: string,
) {
  const rt = await runtime();
  const id = `${ctx.lane}-j-${digest(`${ctx.turnId}:${toolCallId}`).slice(0, 8)}`;
  const existing = await rt.store.getJob(id);
  if (existing) return { jobId: id, externalId: existing.externalId, status: existing.status };
  const callbackToken = randomBytes(24).toString("hex");
  await rt.store.createJob({ id, lane: ctx.lane, turnId: ctx.turnId, spec: input, callbackToken, replyTo: ctx.replyTo });
  const { externalId } = await rt.jobs.submit({
    id,
    lane: ctx.lane,
    spec: input,
    callbackUrl: `${rt.baseUrl}/api/jobs/callback/${callbackToken}`,
  });
  await rt.store.setJobExternalId(id, externalId);
  return { jobId: id, externalId, status: "submitted" };
}

export async function askLead(ctx: ToolCtx, input: { lane: string; question: string }, toolCallId: string) {
  const r = await wake(input.lane, {
    lane: input.lane,
    source: "lead",
    kind: "ask",
    payload: { text: input.question, from: ctx.lane },
    replyTo: { lane: ctx.lane },
    dedupeKey: `ask:${ctx.turnId}:${toolCallId}`,
  });
  return { queued: !r.duplicate, eventId: r.eventId };
}

export function hostTools(ctx: ToolCtx): ToolSet {
  const jobTool = tool({
    description:
      "Submit heavy compute (GPU work, Lean or circuit builds, test suites, anything over a few CPU-minutes) to the team's job queue. Returns at once; you are woken with the job's outcome.",
    inputSchema: z.object({
      command: z.string().describe("The command, as `research run` would take it."),
      requirements: z.record(z.string(), z.unknown()).optional().describe("Job requirements, e.g. {min_mem_gb: 64}."),
      leaseSeconds: z.number().int().optional(),
    }),
    execute: (input, { toolCallId }) => submitJob(ctx, input, toolCallId),
  });
  const askTool = tool({
    description: "Ask another lead a question. Returns at once; their answer arrives as an event in a later turn.",
    inputSchema: z.object({ lane: z.string(), question: z.string() }),
    execute: (input, { toolCallId }) => askLead(ctx, input, toolCallId),
  });
  if (ctx.kind === "worker") return { submit_job: jobTool, ask_lead: askTool };
  return {
    dispatch_worker: tool({
      description:
        "Start a worker: a separate coding agent with its own sandbox and branch, for a self-contained task. Returns at once; you are woken with its report.",
      inputSchema: z.object({
        task: z.string().describe("The complete brief: the worker sees nothing else."),
        branch: z.string().optional(),
        harness: z.enum(["claude-code", "codex"]).optional(),
        model: z.string().optional(),
      }),
      execute: (input, { toolCallId }) => dispatchWorker(ctx, input, toolCallId),
    }),
    submit_job: jobTool,
    ask_lead: askTool,
  };
}
