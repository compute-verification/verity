import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import type { HarnessV1, HarnessV1NetworkSandboxSession } from "@ai-sdk/harness";
import type { Experimental_SandboxSession } from "@ai-sdk/provider-utils";
import { type Credentials, type ModelAuth, localCredentials, vercelCredentials } from "./credentials.ts";
import { openDb } from "./db.ts";
import { fakeHarness } from "./harness/fake.ts";
import { type JobRunner, jobsApi, memoryJobs } from "./jobs.ts";
import { localSandboxSession } from "./sandbox/local.ts";
import { type SlackClient, memorySlack, webSlack } from "./slack.ts";
import { type HarnessName, type Store, createStore } from "./store.ts";

export type SandboxSession = HarnessV1NetworkSandboxSession | Experimental_SandboxSession;

export type SandboxProvider = {
  open(name: string, o: { lane: string; subject: string; credentials: Credentials }): Promise<SandboxSession>;
  park(name: string): Promise<void>;
};

export type HarnessProvider = (name: HarnessName, o: { auth: ModelAuth }) => Promise<HarnessV1>;

// Everything a step touches outside the workflow. Steps run from bundles the
// workflow compiler builds, so they reach these through one process-wide slot
// rather than imports; tests fill the slot with in-memory backends.
export type Runtime = {
  store: Store;
  slack: SlackClient;
  jobs: JobRunner;
  sandbox: SandboxProvider;
  harness: HarnessProvider;
  credentials: Credentials;
  baseUrl: string;
  sliceSeconds: number;
  leaseSeconds: number;
};

const SLOT = Symbol.for("verity.platform.runtime");
type Slot = { [SLOT]?: Promise<Runtime> };

export function setRuntime(rt: Runtime | Promise<Runtime>) {
  (globalThis as Slot)[SLOT] = Promise.resolve(rt);
}

export function runtime(): Promise<Runtime> {
  const g = globalThis as Slot;
  g[SLOT] ??= fromEnv(process.env);
  return g[SLOT];
}

export function localSandboxes(root: string): SandboxProvider {
  return {
    async open(name) {
      const dir = join(root, name);
      await mkdir(dir, { recursive: true });
      return localSandboxSession(dir);
    },
    async park() {},
  };
}

const DEFAULT_ALLOW = [
  "ai-gateway.vercel.sh",
  "github.com",
  "api.github.com",
  "codeload.github.com",
  "objects.githubusercontent.com",
  "registry.npmjs.org",
  "pypi.org",
  "files.pythonhosted.org",
];

export const GITHUB_PLACEHOLDER = "platform-github-placeholder";

export async function vercelSandboxes(o: { vcpus: number; timeoutMs: number }): Promise<SandboxProvider> {
  const { openVercelSandbox } = await import("./sandbox/vercel.ts");
  const { Sandbox } = await import("@vercel/sandbox");
  return {
    async open(name, { subject, credentials }) {
      // git sends Basic x-access-token:<token>, gh and the API send Bearer; the
      // sandbox is configured with the placeholder for both.
      const gh = await credentials.github(subject);
      const basic = (t: string) => `Basic ${Buffer.from(`x-access-token:${t}`).toString("base64")}`;
      const inject = gh
        ? ["github.com", "api.github.com"].flatMap((host) => [
            { host, sent: `Bearer ${GITHUB_PLACEHOLDER}`, real: `Bearer ${gh.token}` },
            { host, sent: basic(GITHUB_PLACEHOLDER), real: basic(gh.token) },
          ])
        : [];
      return openVercelSandbox(name, { vcpus: o.vcpus, timeoutMs: o.timeoutMs, allowHosts: DEFAULT_ALLOW, inject });
    },
    async park(name) {
      const s = await Sandbox.get({ name }).catch(() => null);
      await s?.stop().catch(() => {});
    },
  };
}

export async function harnessFor(name: HarnessName, o: { auth: ModelAuth }): Promise<HarnessV1> {
  if (name === "fake") return fakeHarness();
  if (name === "codex") {
    const { createCodex } = await import("@ai-sdk/harness-codex");
    return createCodex({ auth: o.auth }) as unknown as HarnessV1;
  }
  const { createClaudeCode } = await import("@ai-sdk/harness-claude-code");
  return createClaudeCode({ auth: o.auth, effort: "high" }) as unknown as HarnessV1;
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
        oidcToken: () => getVercelOidcToken(),
      })
    : localCredentials(store);
  const sandbox =
    env.PLATFORM_SANDBOX === "vercel"
      ? await vercelSandboxes({ vcpus: Number(env.PLATFORM_SANDBOX_VCPUS ?? 2), timeoutMs: 45 * 60_000 })
      : localSandboxes(env.PLATFORM_SANDBOX_DIR ?? ".data/sandboxes");
  const baseUrl =
    env.PLATFORM_BASE_URL ??
    (env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${env.VERCEL_PROJECT_PRODUCTION_URL}` : "http://localhost:3000");
  return {
    store,
    slack: env.SLACK_BOT_TOKEN ? webSlack(env.SLACK_BOT_TOKEN) : memorySlack(),
    jobs: env.PLATFORM_JOBS_URL && env.PLATFORM_JOBS_TOKEN ? jobsApi({ url: env.PLATFORM_JOBS_URL, token: env.PLATFORM_JOBS_TOKEN, line: env.PLATFORM_JOBS_LINE }) : memoryJobs(),
    sandbox,
    harness: harnessFor,
    credentials,
    baseUrl,
    sliceSeconds: Number(env.PLATFORM_SLICE_SECONDS ?? 600),
    leaseSeconds: Number(env.PLATFORM_LEASE_SECONDS ?? 900),
  };
}
