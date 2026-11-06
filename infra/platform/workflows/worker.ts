import { getWorkflowMetadata } from "workflow";
import { addUsage, ZERO_USAGE } from "../src/transcript.ts";
import { type WorkerInput, beginWorkerTurn, endTurn, reportToLead, sliceTurn } from "./turn.ts";

// A worker: one harness turn in its own sandbox, as many slices as it needs,
// then its report is delivered to its lead's inbox, which wakes the lead.
export async function workerWorkflow(w: WorkerInput) {
  "use workflow";
  const { workflowRunId } = getWorkflowMetadata();
  const begun = await beginWorkerTurn(w, workflowRunId);
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
  const done = await endTurn(begun.ctx, { status: s.status, error: s.error }, usage, text);
  await reportToLead(w, { failed: s.status === "failed", text: done.text });
  return { workerId: w.id, status: s.status, costUsd: done.costUsd };
}
