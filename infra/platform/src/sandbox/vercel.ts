import type { HarnessV1NetworkSandboxSession } from "@ai-sdk/harness";
import { createVercelNetworkSandboxSessionFromNativeSandbox } from "@ai-sdk/sandbox-vercel";
import { type NetworkPolicyRule, Sandbox } from "@vercel/sandbox";

export type VercelSandboxOptions = {
  vcpus: number;
  timeoutMs: number;
  // Hosts the sandbox may reach. The harness adds its model endpoint's
  // credential rule itself; everything not listed is refused.
  allowHosts: string[];
  // Header rules applied outside the sandbox: the sandbox sends a placeholder
  // and the firewall swaps in the real credential, so no key ever lives inside.
  inject?: { host: string; sent: string; real: string }[];
};

export const BRIDGE_PORT = 4000;

// One persistent named sandbox per lead (and per worker): stopping it snapshots
// its filesystem and the next turn resumes from that snapshot, so checkouts and
// caches survive between turns without anything running in between.
export async function openVercelSandbox(name: string, o: VercelSandboxOptions): Promise<HarnessV1NetworkSandboxSession> {
  const allow: Record<string, NetworkPolicyRule[]> = {};
  for (const h of o.allowHosts) allow[h] = [];
  for (const r of o.inject ?? []) {
    (allow[r.host] ??= []).push({
      match: { headers: [{ key: { exact: "authorization" }, value: { exact: r.sent } }] },
      transform: [{ headers: { authorization: r.real } }],
    });
  }
  const sandbox = await Sandbox.getOrCreate({
    name,
    persistent: true,
    snapshotExpiration: 0,
    keepLastSnapshots: { count: 3, expiration: 0 },
    ports: [BRIDGE_PORT],
    runtime: "node24",
    resources: { vcpus: o.vcpus },
    timeout: o.timeoutMs,
    networkPolicy: { allow },
    tags: { platform: "verity", sandbox: name.slice(0, 60) },
    resume: true,
  });
  // An existing sandbox keeps the policy it was created with; the injected
  // tokens are short-lived, so every open installs fresh ones.
  await sandbox.updateNetworkPolicy({ allow });
  return createVercelNetworkSandboxSessionFromNativeSandbox(sandbox);
}
