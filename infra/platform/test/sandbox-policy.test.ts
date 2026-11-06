import { createCredentialRequestTransformation } from "@ai-sdk/harness/utils";
import { createVercelNetworkSandboxSessionFromNativeSandbox } from "@ai-sdk/sandbox-vercel";
import type { NetworkPolicy } from "@vercel/sandbox";
import { describe, expect, it } from "vitest";
import type { Credentials } from "../src/credentials.ts";
import { brokeredSession } from "../src/sandbox/vercel.ts";
import { credentialRules, placeholder } from "../src/services.ts";

// The parts of a native Vercel sandbox the harness's policy manager reads and
// writes. The real API reports a rule's transformed headers without their
// values, so the fake keeps the policy as the platform sent it and the tests
// read only hosts and rule counts.
function nativeSandbox(policy: NetworkPolicy) {
  const s = { policy, updates: 0 };
  const sandbox = {
    name: "lead-infra",
    routes: [{ port: 4000 }],
    currentSession: () => ({ sessionId: "s1", cwd: "/vercel/sandbox", networkPolicy: s.policy }),
    listSessions: async () =>
      (async function* () {
        yield { id: "s1", networkPolicy: s.policy };
      })(),
    update: async (u: { networkPolicy?: NetworkPolicy }) => {
      if (u.networkPolicy) {
        s.policy = u.networkPolicy;
        s.updates++;
      }
    },
  };
  return { s, sandbox: sandbox as unknown as Parameters<typeof createVercelNetworkSandboxSessionFromNativeSandbox>[0] };
}

const ALLOW = ["ai-gateway.vercel.sh", "github.com", "api.github.com"];
const GITHUB = [{ host: "api.github.com", sent: "Bearer placeholder", real: "Bearer gh-1" }];
const gatewayRule = (real: string) =>
  createCredentialRequestTransformation({
    matchUrl: "https://ai-gateway.vercel.sh",
    matchHeaders: { "x-api-key": "sandbox-placeholder" },
    transformHeaders: { "x-api-key": real },
  });

const rulesByHost = (p: NetworkPolicy) =>
  typeof p === "object" && p.allow && !Array.isArray(p.allow)
    ? Object.fromEntries(Object.entries(p.allow).map(([h, r]) => [h, r.length]))
    : p;

describe("sandbox network policy", () => {
  it("lets the harness add its model rule beside the GitHub rules", async () => {
    const { s, sandbox } = nativeSandbox({ allow: ALLOW });
    const session = await brokeredSession(sandbox, { allowHosts: ALLOW, inject: GITHUB });
    await session.addRequestTransformations!([gatewayRule("key-1")]);
    expect(rulesByHost(s.policy)).toEqual({ "ai-gateway.vercel.sh": 1, "github.com": 0, "api.github.com": 1 });
  });

  it("replaces the previous turn's rules when a resumed sandbox is reopened", async () => {
    const first = nativeSandbox({ allow: ALLOW });
    const one = await brokeredSession(first.sandbox, { allowHosts: ALLOW, inject: GITHUB });
    await one.addRequestTransformations!([gatewayRule("key-1")]);
    const { s, sandbox } = nativeSandbox(first.s.policy);
    const two = await brokeredSession(sandbox, { allowHosts: ALLOW, inject: [{ ...GITHUB[0]!, real: "Bearer gh-2" }] });
    expect(rulesByHost(s.policy)).toEqual({ "ai-gateway.vercel.sh": 0, "github.com": 0, "api.github.com": 1 });
    await two.addRequestTransformations!([gatewayRule("key-2")]);
    const allow = (s.policy as { allow: Record<string, { transform?: { headers?: Record<string, string> }[] }[]> }).allow;
    expect(allow["api.github.com"]![0]!.transform![0]!.headers).toEqual({ authorization: "Bearer gh-2" });
    expect(allow["ai-gateway.vercel.sh"]![0]!.transform![0]!.headers).toEqual({ "x-api-key": "key-2" });
  });

  it("refuses the model rule when the GitHub rules were installed beside the manager", async () => {
    const { sandbox } = nativeSandbox({
      allow: {
        "ai-gateway.vercel.sh": [],
        "api.github.com": [
          {
            match: { headers: [{ key: { exact: "authorization" }, value: { exact: "Bearer placeholder" } }] },
            transform: [{ headers: { authorization: "Bearer gh-1" } }],
          },
        ],
      },
    });
    const session = createVercelNetworkSandboxSessionFromNativeSandbox(sandbox);
    await expect(session.addRequestTransformations!([gatewayRule("key-1")])).rejects.toThrow(/cannot be attributed/);
  });

  it("brokers each granted API the host holds a token for, in its own header form, and no other", async () => {
    const issued: string[] = [];
    const credentials = {
      github: async () => ({ token: "gh-1", expiresAt: "" }),
      service: async (_subject: string, name: string) => {
        issued.push(name);
        return ({ slack: "xoxb-1", linear: "lin-1" } as Record<string, string>)[name] ?? null;
      },
    } as unknown as Credentials;
    const { rules, hosts } = await credentialRules(credentials, "infra", ["slack", "linear", "notion", "bogus"]);
    expect(issued).toEqual(["slack", "linear", "notion"]);
    expect(hosts).toEqual(["slack.com", "api.linear.app"]);
    const rule = (host: string) => rules.filter((r) => r.host === host).map(({ sent, real }) => ({ sent, real }));
    expect(rule("slack.com")).toEqual([{ sent: `Bearer ${placeholder("slack")}`, real: "Bearer xoxb-1" }]);
    expect(rule("api.linear.app")).toEqual([{ sent: placeholder("linear"), real: "lin-1" }]);
    expect(rule("api.github.com").map((r) => r.real)).toEqual(["Bearer gh-1", `Basic ${Buffer.from("x-access-token:gh-1").toString("base64")}`]);
    expect(rules.some((r) => r.host === "api.notion.com")).toBe(false);

    const allowHosts = [...ALLOW, ...hosts];
    const { s, sandbox } = nativeSandbox({ allow: ALLOW });
    const session = await brokeredSession(sandbox, { allowHosts, inject: rules });
    await session.addRequestTransformations!([gatewayRule("key-1")]);
    expect(rulesByHost(s.policy)).toEqual({ "ai-gateway.vercel.sh": 1, "github.com": 2, "api.github.com": 2, "slack.com": 1, "api.linear.app": 1 });
  });
});
