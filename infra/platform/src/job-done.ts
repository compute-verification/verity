import { runtime } from "./runtime.ts";
import type { Job } from "./store.ts";
import { wake } from "./wake.ts";

// A job's end, however it was learned (the runner's callback, or the cron
// reading the job's state): recorded once, and delivered to the lead in the
// thread that submitted it. Learning it twice wakes the lead once.
export async function jobDone(job: Job, o: { status: string; summary?: string; result?: unknown }) {
  const rt = await runtime();
  await rt.store.finishJob(job.id, o.status, o.result ?? o.summary ?? null);
  return wake(job.lane, {
    lane: job.lane,
    source: "job",
    kind: "job-done",
    payload: { jobId: job.id, externalId: job.externalId, status: o.status, text: `Job ${job.id} ${o.status}: ${o.summary ?? ""}`.trim(), result: o.result ?? null },
    replyTo: job.replyTo,
    dedupeKey: `job-done:${job.id}`,
  });
}
