import { createCipheriv, createDecipheriv, createHash, randomBytes } from "node:crypto";
import type { Store } from "./store.ts";

// Every credential an agent's work uses is issued here, on the host, and
// recorded in `grants` by reference (a key id or a digest, never the value).
// A sandbox never holds one: model keys and GitHub tokens reach the wire
// through the sandbox firewall's header rules, with a placeholder inside.
export type ModelAuth = "ai-gateway" | Record<string, string>;

export type Credentials = {
  modelAuth(lane: string): Promise<ModelAuth>;
  github(subject: string): Promise<{ token: string; expiresAt: string } | null>;
  provisionGatewayKey(lane: string, monthlyUsd: number): Promise<{ keyId: string }>;
  gatewaySpend(): Promise<{ keyId: string; name: string; spendUsd: number; limitUsd: number | null }[]>;
};

export const digest = (s: string) => createHash("sha256").update(s).digest("hex").slice(0, 16);

export function seal(secret: string, plaintext: string): string {
  const key = createHash("sha256").update(secret).digest();
  const iv = randomBytes(12);
  const c = createCipheriv("aes-256-gcm", key, iv);
  const body = Buffer.concat([c.update(plaintext, "utf8"), c.final()]);
  return [iv, c.getAuthTag(), body].map((b) => b.toString("base64url")).join(".");
}

export function unseal(secret: string, sealed: string): string {
  const [iv, tag, body] = sealed.split(".").map((p) => Buffer.from(p, "base64url"));
  const key = createHash("sha256").update(secret).digest();
  const d = createDecipheriv("aes-256-gcm", key, iv!);
  d.setAuthTag(tag!);
  return Buffer.concat([d.update(body!), d.final()]).toString("utf8");
}

const API = "https://api.vercel.com";

export function vercelCredentials(o: {
  store: Store;
  secret: string;
  vercelToken?: string;
  teamId?: string;
  githubBrokerUrl?: string;
  oidcToken: () => Promise<string>;
}): Credentials {
  const team = o.teamId ? `teamId=${encodeURIComponent(o.teamId)}` : "";
  const vercel = async (method: string, path: string, body?: unknown) => {
    if (!o.vercelToken) throw new Error("VERCEL_TOKEN is not set: per-lead gateway keys need a token that can manage AI Gateway keys");
    const sep = path.includes("?") ? "&" : "?";
    const r = await fetch(`${API}${path}${team ? sep + team : ""}`, {
      method,
      headers: { authorization: `Bearer ${o.vercelToken}`, "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`${method} ${path}: ${r.status} ${(await r.text()).slice(0, 300)}`);
    return (await r.json()) as Record<string, unknown>;
  };

  return {
    // A lead without its own key runs on the deployment's OIDC token, fetched
    // per turn: the copy in a long-lived instance's environment can be stale.
    async modelAuth(lane) {
      const sealed = await o.store.sealedGatewayKey(lane);
      return { AI_GATEWAY_API_KEY: sealed ? unseal(o.secret, sealed) : await o.oidcToken() };
    },

    async github(subject) {
      if (!o.githubBrokerUrl) return null;
      const r = await fetch(o.githubBrokerUrl, {
        method: "POST",
        headers: { authorization: `Bearer ${await o.oidcToken()}`, "x-verity-agent": subject, "content-type": "application/json" },
        body: JSON.stringify({ subject }),
      });
      if (!r.ok) throw new Error(`github broker: ${r.status}`);
      const j = (await r.json()) as { token: string; expires_at: string };
      await o.store.recordGrant({ subject, kind: "github-app-token", scope: "verity-agents", ref: digest(j.token), expiresAt: j.expires_at });
      return { token: j.token, expiresAt: j.expires_at };
    },

    async provisionGatewayKey(lane, monthlyUsd) {
      const created = await vercel("POST", "/v1/api-keys", {
        purpose: "ai-gateway",
        name: `platform-lead-${lane}`,
        metadata: { spendAttribution: "user" },
      });
      const keyId = String(created.id ?? (created.apiKey as Record<string, unknown>)?.id);
      const value = String(created.key ?? created.token ?? (created.apiKey as Record<string, unknown>)?.key);
      await vercel("PATCH", `/v1/api-keys/${encodeURIComponent(keyId)}/quota`, { limitAmount: monthlyUsd, refreshPeriod: "monthly" });
      await o.store.setGatewayKey(lane, keyId, seal(o.secret, value));
      await o.store.recordGrant({ subject: lane, kind: "ai-gateway-key", scope: `monthly $${monthlyUsd}`, ref: keyId, expiresAt: null });
      return { keyId };
    },

    async gatewaySpend() {
      const j = await vercel("GET", "/ai-gateway/budgets/list?scopeType=api-key");
      const items = (j.budgets ?? j.items ?? []) as Record<string, unknown>[];
      return items.map((b) => ({
        keyId: String(b.scopeId ?? b.apiKeyId ?? ""),
        name: String(b.name ?? ""),
        spendUsd: Number(b.spend ?? b.currentSpend ?? 0),
        limitUsd: b.limitAmount == null ? null : Number(b.limitAmount),
      }));
    },
  };
}

// For tests and local runs: the platform's own gateway auth for everyone, and a
// GitHub token that is a recognisable non-secret.
export function localCredentials(store: Store): Credentials {
  return {
    async modelAuth() {
      return "ai-gateway";
    },
    async github(subject) {
      const token = `local-github-token-for-${subject}`;
      const expiresAt = new Date(Date.now() + 3600_000).toISOString();
      await store.recordGrant({ subject, kind: "github-app-token", scope: "local", ref: digest(token), expiresAt });
      return { token, expiresAt };
    },
    async provisionGatewayKey(lane, monthlyUsd) {
      const keyId = `local-key-${lane}`;
      await store.setGatewayKey(lane, keyId, "");
      await store.recordGrant({ subject: lane, kind: "ai-gateway-key", scope: `monthly $${monthlyUsd}`, ref: keyId, expiresAt: null });
      return { keyId };
    },
    async gatewaySpend() {
      return [];
    },
  };
}
