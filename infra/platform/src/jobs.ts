// Heavy work never runs in a sandbox: it goes to the team's job queue (the
// `research-run` kind of the jobs API that `research run` uses), and the job's
// end comes back as an inbox event through the callback URL.
export type JobSpec = {
  command: string;
  requirements?: Record<string, unknown>;
  leaseSeconds?: number;
};

export type JobRunner = {
  submit(j: { id: string; lane: string; spec: JobSpec; callbackUrl: string }): Promise<{ externalId: string }>;
};

export function jobsApi(o: { url: string; token: string; line?: string }): JobRunner {
  return {
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

export function memoryJobs(): JobRunner & { submitted: { id: string; lane: string; spec: JobSpec; callbackUrl: string }[] } {
  const submitted: { id: string; lane: string; spec: JobSpec; callbackUrl: string }[] = [];
  return {
    submitted,
    async submit(j) {
      submitted.push(j);
      return { externalId: `local-${submitted.length}` };
    },
  };
}
