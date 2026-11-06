import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import type { HarnessV1, HarnessV1NetworkSandboxSession } from "@ai-sdk/harness";
import type { Experimental_SandboxSession } from "@ai-sdk/provider-utils";
import { type Credentials, type ModelAuth, localCredentials, vercelCredentials } from "./credentials.ts";
import { openDb } from "./db.ts";
import { harnessFor } from "./harness/providers.ts";
import { type JobRunner, jobsApi, memoryJobs } from "./jobs.ts";
import { localSandboxSession } from "./sandbox/local.ts";
import { credentialRules } from "./services.ts";
import { type SlackClient, type SlackReplier, directReply, memorySlack, researchReply, webSlack } from "./slack.ts";
import { type HarnessName, type Store, createStore } from "./store.ts";

export type SandboxSession = HarnessV1NetworkSandboxSession | Experimental_SandboxSession;

export type SandboxProvider = {
  open(name: string, o: { lane: string; subject: string; credentials: Credentials; apis: string[] }): Promise<SandboxSession>;
  park(name: string): Promise<void>;
};

export type HarnessProvider = (name: HarnessName, o: { auth: ModelAuth }) => Promise<HarnessV1>;

// Everything a step touches outside the workflow. Steps run from bundles the
// workflow compiler builds, so they reach these through one process-wide slot
// rather than imports; tests fill the slot with in-memory backends.
export type Runtime = {
  store: Store;
  slack: SlackClient;
  reply: SlackReplier;
  jobs: JobRunner;
  sandbox: SandboxProvider;
  harness: HarnessProvider;
  credentials: Credentials;
  baseUrl: string;
  // Cloned into each sandbox the first time a harness uses it; null leaves sandboxes empty.
  repoUrl: string | null;
  sliceSeconds: number;
  leaseSeconds: number;
};

const SLOT = Symbol.for("verity.platform.runtime");
type Slot = { [SLOT]?: Promise<Runtime> };

export function setRuntime(rt: Runtime | Promise<Runtime>) {
  (globalThis as Slot)[SLOT] = Promise.resolve(rt);
}

// A failed start (the database unreachable, say) is not cached: the next call tries again.
export function runtime(): Promise<Runtime> {
  const g = globalThis as Slot;
  if (!g[SLOT]) {
    const p = fromEnv(process.env);
    g[SLOT] = p;
    p.catch(() => {
      if (g[SLOT] === p) delete g[SLOT];
    });
  }
  return g[SLOT];
}

// A directory per sandbox. It issues the same per-open credentials the Vercel
// provider does, and like it never writes them inside.
export function localSandboxes(root: string): SandboxProvider {
  return {
    async open(name, { subject, credentials, apis }) {
      await credentialRules(credentials, subject, apis);
      const dir = join(root, name);
      await mkdir(dir, { recursive: true });
      return localSandboxSession(dir);
    },
    async park() {},
  };
}

export async function vercelSandboxes(o: { vcpus: number; timeoutMs: number }): Promise<SandboxProvider> {
  const { DEFAULT_ALLOW, openVercelSandbox } = await import("./sandbox/vercel.ts");
  const { Sandbox } = await import("@vercel/sandbox");
  return {
    async open(name, { subject, credentials, apis }) {
      const { rules, hosts } = await credentialRules(credentials, subject, apis);
      return openVercelSandbox(name, { vcpus: o.vcpus, timeoutMs: o.timeoutMs, allowHosts: [...DEFAULT_ALLOW, ...hosts], inject: rules });
    },
    async park(name) {
      const s = await Sandbox.get({ name }).catch(() => null);
      await s?.stop().catch(() => {});
    },
  };
}

export async function fromEnv(env: NodeJS.ProcessEnv): Promise<Runtime> {
  const store = createStore(await openDb(env.DATABASE_URL ?? "pglite:.data/db"));
  await store.migrate();
  const { getVercelOidcToken } = await import("@vercel/oidc");
  const credentials = env.PLATFORM_SECRET
    ? vercelCredentials({
        store,
        secret: env.PLATFORM_SECRET,
        vercelToken: env.VERCEL_TOKEN,
        teamId: env.VERCEL_TEAM_ID,
        githubBrokerUrl: env.PLATFORM_GITHUB_BROKER_URL,
        services: { slack: env.SLACK_BOT_TOKEN, linear: env.LINEAR_API_KEY, notion: env.NOTION_API_KEY },
        oidcToken: () => getVercelOidcToken(),
      })
    : localCredentials(store);
  if (env.VERCEL && env.PLATFORM_SANDBOX !== "vercel") throw new Error("PLATFORM_SANDBOX=vercel is required on Vercel: local sandboxes isolate nothing");
  if (env.VERCEL && !env.PLATFORM_REPO_URL) throw new Error("PLATFORM_REPO_URL is required on Vercel: a lead replies on Slack with research msg from its checkout");
  const sandbox =
    env.PLATFORM_SANDBOX === "vercel"
      ? await vercelSandboxes({ vcpus: Number(env.PLATFORM_SANDBOX_VCPUS ?? 2), timeoutMs: Number(env.PLATFORM_SANDBOX_TIMEOUT_MINUTES ?? 45) * 60_000 })
      : localSandboxes(env.PLATFORM_SANDBOX_DIR ?? ".data/sandboxes");
  const baseUrl =
    env.PLATFORM_BASE_URL ??
    (env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${env.VERCEL_PROJECT_PRODUCTION_URL}` : "http://localhost:3000");
  const slack = env.SLACK_BOT_TOKEN ? webSlack(env.SLACK_BOT_TOKEN) : memorySlack();
  return {
    store,
    slack,
    reply: env.PLATFORM_REPO_URL ? researchReply : directReply(slack),
    jobs: env.PLATFORM_JOBS_URL && env.PLATFORM_JOBS_TOKEN ? jobsApi({ url: env.PLATFORM_JOBS_URL, token: env.PLATFORM_JOBS_TOKEN, line: env.PLATFORM_JOBS_LINE }) : memoryJobs(),
    sandbox,
    harness: harnessFor,
    credentials,
    baseUrl,
    repoUrl: env.PLATFORM_REPO_URL ?? null,
    sliceSeconds: Number(env.PLATFORM_SLICE_SECONDS ?? 600),
    leaseSeconds: Number(env.PLATFORM_LEASE_SECONDS ?? 900),
  };
}
