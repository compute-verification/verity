import { SERVICES, isService } from "./services.ts";
import type { InboxEvent, Lead } from "./store.ts";

export const DOC_PATH = "WORKSTREAM.md";
export const REPO_DIR = "verity";

export function leadInstructions(lead: Lead, o: { checkout: boolean }): string {
  return [
    `You are the ${lead.lane} lead, running on Verity's research platform.`,
    `${DOC_PATH} in your working directory is your workstream doc and your only memory: this session ends with this turn and nothing else carries over. Read it first. Before you finish, rewrite it so the next turn can pick up from it alone: goals, state, open threads, decisions and next steps.`,
    o.checkout ? `${REPO_DIR}/ is your checkout of the repository, fetched at the start of each turn; it stays between turns, so keep work you mean to keep on a pushed branch.` : "",
    lead.apis.some(isService)
      ? `From your shell you can call ${lead.apis.filter(isService).map((n) => SERVICES[n].about).join(", ")}, with the credentials in ~/.verity-env (${lead.apis.filter(isService).map((n) => `$${SERVICES[n].env}`).join(", ")}). They are placeholders: the platform swaps in the real token on the way out, so there is no token to find or print.`
      : "",
    o.checkout
      ? `Slack reaches you as doorbells in your inbox thread, which \`research msg\` rings as it does for any lead, as top-level posts in the channels you subscribe to, and as replies in threads you watch. Where your skills say to call subscribe_slack_thread, call watch_thread instead; your channel and Linear subscriptions belong to the platform and never lapse, so skip subscribe_slack_channel, subscribe_linear_issue and renewing them. Post on Slack only with \`research msg ... --as ${lead.lane}\` (asks, announcements, closing an ask, Linear), as the using-slack and coordinating skills say, never with the raw API, so that whoever you address is rung. Your final message is posted with \`research msg reply --as ${lead.lane}\` in the thread of each Slack event this turn answers: make it that reply, 60 words at most (link an issue or your doc for more), or exactly NO_REPLY to post nothing.`
      : `Your final message is the reply to each event that has a return address.`,
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

export function workerInstructions(task: { lane: string; branch: string | null }, o: { checkout: boolean }): string {
  return [
    `You are a worker for the ${task.lane} lead on Verity's research platform. Do the task below and finish with a short report of what you did and what you found; the report is sent to your lead.`,
    o.checkout && task.branch ? `${REPO_DIR}/ is a checkout of the repository on branch ${task.branch}, made from the default branch.` : "",
    task.branch ? `Commit to branch ${task.branch} only, and push only to it.` : "",
    `Heavy compute (GPU, Lean, circuit builds, test suites) never runs in this sandbox: if the task needs it, stop and say in your report exactly what to run, and your lead submits it.`,
  ]
    .filter(Boolean)
    .join("\n\n");
}
