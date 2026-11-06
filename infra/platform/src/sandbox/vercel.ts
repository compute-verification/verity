import type { HarnessV1NetworkSandboxSession } from "@ai-sdk/harness";
import { createCredentialRequestTransformation } from "@ai-sdk/harness/utils";
import { createVercelNetworkSandboxSessionFromNativeSandbox } from "@ai-sdk/sandbox-vercel";
import { Sandbox } from "@vercel/sandbox";

export type VercelSandboxOptions = {
  vcpus: number;
  timeoutMs: number;
  // Hosts the sandbox may reach; everything not listed is refused.
  allowHosts: string[];
  // Header rules applied outside the sandbox: the sandbox sends a placeholder
  // and the firewall swaps in the real credential, so no key ever lives inside.
  inject?: { host: string; sent: string; real: string }[];
};

export const BRIDGE_PORT = 4000;

export const DEFAULT_ALLOW = [
  "ai-gateway.vercel.sh",
  "github.com",
  "api.github.com",
  "codeload.github.com",
  "objects.githubusercontent.com",
  "registry.npmjs.org",
  "pypi.org",
  "files.pythonhosted.org",
];

// One persistent named sandbox per lead (and per worker): stopping it snapshots
// its filesystem and the next turn resumes from that snapshot, so checkouts and
// caches survive between turns without anything running in between.
export async function openVercelSandbox(name: string, o: VercelSandboxOptions): Promise<HarnessV1NetworkSandboxSession> {
  const sandbox = await Sandbox.getOrCreate({
    name,
    persistent: true,
    snapshotExpiration: 0,
    keepLastSnapshots: { count: 3, expiration: 0 },
    ports: [BRIDGE_PORT],
    runtime: "node24",
    resources: { vcpus: o.vcpus },
    timeout: o.timeoutMs,
    networkPolicy: { allow: o.allowHosts },
    tags: { platform: "verity", sandbox: name.slice(0, 60) },
    resume: true,
  });
  const session = createVercelNetworkSandboxSessionFromNativeSandbox(sandbox);
  // The policy is one value that the harness extends with its model
  // credential rule, and it refuses to extend rules it can't attribute, so the
  // GitHub rules go through the same session. Replacing them on every open
  // drops the previous turn's (expired) tokens, the model rule included; the
  // harness adds a fresh one when it starts.
  await session.setRequestTransformations!(
    (o.inject ?? []).map((r) =>
      createCredentialRequestTransformation({ matchUrl: `https://${r.host}`, matchHeaders: { authorization: r.sent }, transformHeaders: { authorization: r.real } }),
    ),
  );
  await session.setNetworkPolicy!({ mode: "custom", allowedHosts: o.allowHosts });
  return session;
}
