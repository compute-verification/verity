import { start } from "workflow/api";
import { leadWorkflow } from "../workflows/lead.ts";
import { runtime } from "./runtime.ts";
import { wake } from "./wake.ts";

export const STUCK_PENDING_SECONDS = 15 * 60;
export const STALE_TURN_MINUTES = 120;

// The one scheduled entry point, every minute: due ticks become events, any
// lead with pending events is rung (which starts it if no run holds its hook),
// a lead whose events wait too long with no turn running gets a fresh run, and
// a turn running too long is reported once. Nothing here needs a human.
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
  const alerts: string[] = [];
  const alert = async (key: string, text: string) => {
    if (!(await rt.store.raiseAlert(key, text))) return;
    alerts.push(text);
    if (o.alertChannel) await rt.slack.post({ channel: o.alertChannel, text });
  };
  for (const lead of await rt.store.listLeads()) {
    if (lead.status !== "active") continue;
    const h = await rt.store.leadHealth(lead.lane);
    if (h.pending === 0 || h.leaseHeld) continue;
    if ((h.oldestPendingSeconds ?? 0) > STUCK_PENDING_SECONDS) {
      const run = await start(leadWorkflow, [{ lane: lead.lane, epoch: 0 }]);
      await rt.store.setRunId(lead.lane, run.runId);
      restarted.push(lead.lane);
      await alert(`stuck:${lead.lane}:${minute}`, `platform: ${lead.lane} had ${h.pending} event(s) waiting over ${STUCK_PENDING_SECONDS / 60} min with no turn running; started run ${run.runId}.`);
    } else {
      await wake(lead.lane, null);
      nudged.push(lead.lane);
    }
  }
  for (const t of await rt.store.staleTurns(STALE_TURN_MINUTES)) {
    await alert(`stale:${t.id}`, `platform: turn ${t.id} of ${t.lane} has run over ${STALE_TURN_MINUTES} min.`);
  }
  return { ticked, nudged, restarted, alerts };
}
