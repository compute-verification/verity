import { createHook, getWorkflowMetadata } from "workflow";
import { start } from "workflow/api";
import { HookForceClaimedError } from "workflow/errors";
import { addUsage, ZERO_USAGE } from "../src/transcript.ts";
import { beginLeadTurn, endTurn, parkLead, registerRun, sliceTurn } from "./turn.ts";

export type LeadState = { lane: string; epoch: number };

export type LeadResult =
  | { lane: string; epoch: number; turns: number; ended: "continued"; next: string }
  | { lane: string; epoch: number; turns: number; ended: "superseded"; by: string };

// A run handles at most this many turns, then hands the lead to a fresh run on
// the latest deployment: runs stay short (bounded event logs, current code) and
// the lead itself never ends.
export const TURNS_PER_RUN = 25;

// One lead: a durable loop that owns the hook `lead:<lane>`. Each wake drains
// the lead's inbox one turn at a time; a turn takes every event pending when
// it starts. The newest run always owns the hook (experimental_force), so a
// stuck or outdated run is replaced by starting another.
export async function leadWorkflow(state: LeadState): Promise<LeadResult> {
  "use workflow";
  const { workflowRunId } = getWorkflowMetadata();
  const hook = createHook<{ eventId: number | null }>({ token: `lead:${state.lane}`, experimental_force: true });
  await registerRun(state.lane, workflowRunId);
  const tag = workflowRunId.slice(-8);
  let turns = 0;

  const drain = async () => {
    while (turns < TURNS_PER_RUN) {
      const turnId = `${state.lane}-${state.epoch}-${tag}-${turns + 1}`;
      const begun = await beginLeadTurn(state.lane, turnId, workflowRunId);
      if (!begun) return;
      turns++;
      let s = begun.state;
      let usage = ZERO_USAGE;
      let text = "";
      for (;;) {
        const r = await sliceTurn(begun.ctx, s);
        s = r.state;
        usage = addUsage(usage, r.usage);
        if (r.text) text = r.text;
        if (s.status !== "ready_for_next_step") break;
      }
      await endTurn(begun.ctx, { status: s.status, error: s.error }, usage, text);
    }
  };

  const settle = async () => {
    do await drain();
    while (turns < TURNS_PER_RUN && (await parkLead(state.lane, workflowRunId)));
  };

  try {
    await settle();
    if (turns < TURNS_PER_RUN) {
      for await (const _wake of hook) {
        await settle();
        if (turns >= TURNS_PER_RUN) break;
      }
    }
  } catch (e) {
    if (HookForceClaimedError.is(e)) return { lane: state.lane, epoch: state.epoch, turns, ended: "superseded", by: e.claimedByRunId };
    throw e;
  }
  const next = await start(leadWorkflow, [{ lane: state.lane, epoch: state.epoch + 1 }], { deploymentId: "latest" });
  return { lane: state.lane, epoch: state.epoch, turns, ended: "continued", next: next.runId };
}
