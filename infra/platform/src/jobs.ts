// Heavy work never runs in a sandbox: it goes to the team's job queue (the
// `research-run` kind of the jobs API that `research run` uses), and the job's
// end comes back as an inbox event through the callback URL.
export type JobSpec = {
  command: string;
  requirements?: Record<string, unknown>;
  leaseSeconds?: number;
};

export type JobState = { state: string; result: { outcome?: string; summary?: string; run_id?: string } | null };

export type JobRunner = {
  submit(j: { id: string; lane: string; spec: JobSpec; callbackUrl: string }): Promise<{ externalId: string }>;
  status(externalId: string): Promise<JobState>;
};

// The queue's states a job doesn't leave.
export const FINAL_STATES = new Set(["done", "failed", "dead", "cancelled"]);

export function jobsApi(o: { url: string; token: string; line?: string }): JobRunner {
  return {
    async status(externalId) {
      const r = await fetch(`${o.url.replace(/\/$/, "")}/${encodeURIComponent(externalId)}`, { headers: { authorization: `Bearer ${o.token}` } });
      if (!r.ok) throw new Error(`jobs API: ${r.status} ${(await r.text()).slice(0, 300)}`);
      const j = (await r.json()) as JobState;
      return { state: j.state, result: j.result ?? null };
    },
    async submit({ id, lane, spec, callbackUrl }) {
      const r = await fetch(o.url, {
        method: "POST",
        headers: { authorization: `Bearer ${o.token}`, "content-type": "application/json" },
        body: JSON.stringify({
          kind: "research-run",
          payload: { command: spec.command, lane, platform: { job_id: id, callback_url: callbackUrl } },
          requirements: spec.requirements ?? {},
          lease_s: spec.leaseSeconds ?? 900,
          dedupe_key: `platform:${id}`,
          ...(o.line ? { line: o.line } : {}),
        }),
      });
      if (!r.ok) throw new Error(`jobs API: ${r.status} ${(await r.text()).slice(0, 300)}`);
      const j = (await r.json()) as { id: number | string };
      return { externalId: String(j.id) };
    },
  };
}

// A queue in memory: a job stays queued until a test `end`s it.
export function memoryJobs() {
  const submitted: { id: string; lane: string; spec: JobSpec; callbackUrl: string }[] = [];
  const states = new Map<string, JobState>();
  return {
    submitted,
    async submit(j: { id: string; lane: string; spec: JobSpec; callbackUrl: string }) {
      submitted.push(j);
      const externalId = `local-${submitted.length}`;
      states.set(externalId, { state: "queued", result: null });
      return { externalId };
    },
    async status(externalId: string): Promise<JobState> {
      const s = states.get(externalId);
      if (!s) throw new Error(`no job ${externalId}`);
      return s;
    },
    end(externalId: string, state: string, result: JobState["result"]) {
      states.set(externalId, { state, result });
    },
  } satisfies JobRunner & Record<string, unknown>;
}
