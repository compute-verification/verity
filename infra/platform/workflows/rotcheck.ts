import { sleep } from "workflow";
import { pollRotCheck, reportRotCheck, startRotCheck } from "./rotcheck-steps.ts";

const MAX_MINUTES = 40;

// Weekly: does the platform still build and pass its tests against the newest
// SDK releases? It runs scripts/rot-check.sh in a fresh Vercel Sandbox cloned
// from the platform's own repository and reports to the alert channel.
export async function rotCheckWorkflow(o: { name: string }) {
  "use workflow";
  const started = await startRotCheck(o.name);
  for (let i = 0; i < MAX_MINUTES; i++) {
    await sleep("1m");
    const r = await pollRotCheck(started);
    if (r.done) return reportRotCheck(o.name, r);
  }
  return reportRotCheck(o.name, { done: true, exitCode: null, tail: `no result after ${MAX_MINUTES} min` });
}
