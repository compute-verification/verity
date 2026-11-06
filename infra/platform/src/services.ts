import type { Credentials } from "./credentials.ts";
import { shq } from "./shell.ts";

export const placeholder = (name: string) => `platform-${name}-placeholder`;
export const GITHUB_PLACEHOLDER = placeholder("github");

// The APIs a lead may call from its sandbox besides GitHub. Each token stays on
// the host: the sandbox's environment holds the placeholder under `env`, and
// the firewall swaps the header for the real one on the way to `hosts`.
export const SERVICES = {
  slack: { hosts: ["slack.com"], env: "SLACK_BOT_TOKEN", header: (t: string) => `Bearer ${t}`, about: "Slack's Web API (https://slack.com/api/…, as Verity's bot)" },
  linear: { hosts: ["api.linear.app"], env: "LINEAR_API_KEY", header: (t: string) => t, about: "Linear's GraphQL API (https://api.linear.app/graphql)" },
  notion: { hosts: ["api.notion.com"], env: "NOTION_API_KEY", header: (t: string) => `Bearer ${t}`, about: "Notion's API (https://api.notion.com/v1/…)" },
} as const;

export type ServiceName = keyof typeof SERVICES;
export const SERVICE_NAMES = Object.keys(SERVICES) as ServiceName[];
export const DEFAULT_APIS: ServiceName[] = ["slack", "linear"];
export const isService = (name: string): name is ServiceName => Object.hasOwn(SERVICES, name);

export type HeaderRule = { host: string; sent: string; real: string };

// The header rules and extra hosts for a sandbox opened for `subject`: GitHub
// in git's Basic form and the API's Bearer form (raw files too, where research
// reads the notes repository's routes), and each granted API whose token the
// host holds. Each credential is issued (and recorded) per open.
export async function credentialRules(credentials: Credentials, subject: string, apis: string[]): Promise<{ rules: HeaderRule[]; hosts: string[] }> {
  const rules: HeaderRule[] = [];
  const hosts: string[] = [];
  const gh = await credentials.github(subject);
  if (gh) {
    const basic = (t: string) => `Basic ${Buffer.from(`x-access-token:${t}`).toString("base64")}`;
    for (const host of ["github.com", "api.github.com", "raw.githubusercontent.com"]) {
      rules.push({ host, sent: `Bearer ${GITHUB_PLACEHOLDER}`, real: `Bearer ${gh.token}` });
      rules.push({ host, sent: basic(GITHUB_PLACEHOLDER), real: basic(gh.token) });
    }
  }
  for (const name of apis.filter(isService)) {
    const token = await credentials.service(subject, name);
    if (!token) continue;
    const s = SERVICES[name];
    hosts.push(...s.hosts);
    for (const host of s.hosts) rules.push({ host, sent: s.header(placeholder(name)), real: s.header(token) });
  }
  return { rules, hosts };
}

export const serviceEnv = (apis: string[]): Record<string, string> =>
  Object.fromEntries(apis.filter(isService).map((n) => [SERVICES[n].env, placeholder(n)]));

export const envExports = (vars: Record<string, string>) =>
  Object.entries(vars)
    .map(([k, v]) => `export ${k}=${shq(v)}`)
    .join("\n");
