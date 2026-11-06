import { start } from "workflow/api";
import { leadWorkflow } from "../workflows/lead.ts";
import { jobDone } from "./job-done.ts";
import { FINAL_STATES } from "./jobs.ts";
import { runtime } from "./runtime.ts";
import { wake } from "./wake.ts";

export const STUCK_PENDING_SECONDS = 15 * 60;
export const STALE_TURN_MINUTES = 120;
export const ROLL_AFTER_HOURS = 24;

// The one scheduled entry point, every minute: due ticks become events, any
// lead with pending events is rung (which starts it if no run holds its hook),
// a lead whose events wait too long with no turn running gets a fresh run, an
// idle lead whose run is a day old moves to a fresh run on the current
// deployment, a turn running too long is reported once, and jobs that ended
// are delivered. Nothing here needs a human.
export async function tick(o: { alertChannel?: string; now?: Date } = {}) {
  const rt = await runtime();
  const now = o.now ?? new Date();
  const minute = now.toISOString().slice(0, 16);
  const ticked = await rt.store.takeDueTicks();
  for (const lane of ticked) {
    await wake(lane, {
      lane,
      source: "cron",
      kind: "tick",
      payload: { text: "Scheduled tick: review your workstream doc and act on whatever is due." },
      replyTo: null,
      dedupeKey: `tick:${lane}:${minute}`,
    });
  }
  const nudged: string[] = [];
  const restarted: string[] = [];
  const rolled: string[] = [];
  const alerts: string[] = [];
  const alert = async (key: string, text: string) => {
    if (!(await rt.store.raiseAlert(key, text))) return;
    alerts.push(text);
    if (o.alertChannel) await rt.slack.post({ channel: o.alertChannel, text });
  };
  const fresh = async (lane: string) => {
    const run = await start(leadWorkflow, [{ lane, epoch: 0 }]);
    await rt.store.setRunId(lane, run.runId);
    return run.runId;
  };
  for (const lead of await rt.store.listLeads()) {
    if (lead.status !== "active") continue;
    const h = await rt.store.leadHealth(lead.lane);
    if (h.leaseHeld) continue;
    if (h.pending === 0) {
      const age = lead.runStartedAt ? (now.getTime() - Date.parse(lead.runStartedAt)) / 3600_000 : 0;
      if (age > ROLL_AFTER_HOURS) {
        await fresh(lead.lane);
        rolled.push(lead.lane);
      }
      continue;
    }
    if ((h.oldestPendingSeconds ?? 0) > STUCK_PENDING_SECONDS) {
      const runId = await fresh(lead.lane);
      restarted.push(lead.lane);
      await alert(`stuck:${lead.lane}:${minute}`, `platform: ${lead.lane} had ${h.pending} event(s) waiting over ${STUCK_PENDING_SECONDS / 60} min with no turn running; started run ${runId}.`);
    } else {
      await wake(lead.lane, null);
      nudged.push(lead.lane);
    }
  }
  for (const t of await rt.store.staleTurns(STALE_TURN_MINUTES)) {
    await alert(`stale:${t.id}`, `platform: turn ${t.id} of ${t.lane} has run over ${STALE_TURN_MINUTES} min.`);
  }
  // A job's end is also read from the queue, so a runner that never calls back
  // (or a callback that was lost) still wakes the lead.
  const jobsEnded: string[] = [];
  for (const job of await rt.store.openJobs()) {
    const s = await rt.jobs.status(job.externalId!).catch(() => null);
    if (!s || !FINAL_STATES.has(s.state)) continue;
    await jobDone(job, { status: s.state, summary: s.result?.summary, result: s.result });
    jobsEnded.push(job.id);
  }
  return { ticked, nudged, restarted, rolled, alerts, jobsEnded };
}
