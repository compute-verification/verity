import { resumeHook, start } from "workflow/api";
import { HookNotFoundError } from "workflow/errors";
import { leadWorkflow } from "../workflows/lead.ts";
import { runtime } from "./runtime.ts";
import type { NewEvent } from "./store.ts";

export const leadToken = (lane: string) => `lead:${lane}`;

export type Wake = { eventId: number | null };

// Every trigger (Slack, cron, a job's callback, a worker's end, an MCP call)
// goes through here: record the event, then ring the lead. The event row is
// the message; the hook only says "look". A lead with no live run is started,
// so a trigger is also the liveness check.
export async function wake(lane: string, event: NewEvent | null): Promise<{ eventId: number | null; duplicate: boolean; started: string | null }> {
  const rt = await runtime();
  const lead = await rt.store.getLead(lane);
  if (!lead) throw new Error(`no lead ${lane}`);
  let eventId: number | null = null;
  if (event) {
    const e = await rt.store.addEvent(event);
    if (!e) return { eventId: null, duplicate: true, started: null };
    eventId = e.id;
  }
  if (lead.status !== "active") return { eventId, duplicate: false, started: null };
  try {
    await resumeHook<Wake>(leadToken(lane), { eventId });
    return { eventId, duplicate: false, started: null };
  } catch (e) {
    if (!HookNotFoundError.is(e)) throw e;
  }
  const run = await start(leadWorkflow, [{ lane, epoch: 0 }]);
  await rt.store.setRunId(lane, run.runId);
  return { eventId, duplicate: false, started: run.runId };
}
