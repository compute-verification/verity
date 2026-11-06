import type { InboxEvent, Lead } from "./store.ts";

export const DOC_PATH = "WORKSTREAM.md";

export function leadInstructions(lead: Lead): string {
  return [
    `You are the ${lead.lane} lead, running on Verity's research platform.`,
    `${DOC_PATH} in your working directory is your workstream doc and your only memory: this session ends with this turn and nothing else carries over. Read it first. Before you finish, rewrite it so the next turn can pick up from it alone: goals, state, open threads, decisions and next steps.`,
    `Your final message is posted to the Slack thread of each event that has one, so make it the reply.`,
    `Heavy compute (GPU, Lean, circuit builds, test suites) goes through submit_job and never runs in this sandbox; you are woken when the job finishes. Use dispatch_worker for a self-contained coding task and ask_lead to ask another lead; both wake you with the answer.`,
    lead.charter.trim(),
  ]
    .filter(Boolean)
    .join("\n\n");
}

export function renderEvent(e: InboxEvent): string {
  const r = e.replyTo;
  const where = !r
    ? ""
    : "channel" in r
      ? ` (slack ${r.channel}${r.threadTs ? ` thread ${r.threadTs}` : ""})`
      : "lane" in r
        ? ` (from lead ${r.lane})`
        : ` (from ${r.caller})`;
  const body =
    typeof e.payload.text === "string"
      ? e.payload.text
      : Object.keys(e.payload).length
        ? JSON.stringify(e.payload)
        : "";
  return `- [${e.source}/${e.kind}] #${e.id} at ${e.receivedAt}${where}: ${body}`;
}

export function turnPrompt(events: InboxEvent[]): string {
  return [`Events since your last turn (oldest first):`, ...events.map(renderEvent), ``, `Act on them, update ${DOC_PATH}, and reply.`].join("\n");
}

export function workerInstructions(task: { lane: string; branch: string | null }): string {
  return [
    `You are a worker for the ${task.lane} lead on Verity's research platform. Do the task below and finish with a short report of what you did and what you found; the report is sent to your lead.`,
    task.branch ? `Commit to branch ${task.branch} only, and push only to it.` : "",
    `Heavy compute (GPU, Lean, circuit builds, test suites) never runs in this sandbox: if the task needs it, stop and say in your report exactly what to run, and your lead submits it.`,
  ]
    .filter(Boolean)
    .join("\n\n");
}
