import type { Usage } from "./store.ts";

// USD per million tokens, from the AI Gateway model list
// (https://ai-gateway.vercel.sh/v1/models). The gateway's per-key spend is the
// billed figure; this table only attributes it to turns before the bill arrives.
export type Price = { input: number; output: number; cacheRead?: number; cacheWrite?: number };

export const PRICES: Record<string, Price> = {
  "anthropic/claude-opus-5.5": { input: 4, output: 20, cacheRead: 0.2, cacheWrite: 5 },
  "anthropic/claude-sonnet-5.5": { input: 2, output: 10, cacheRead: 0.1, cacheWrite: 2.5 },
  "openai/gpt-5.5": { input: 5, output: 30, cacheRead: 0.5 },
  "openai/gpt-6-sol": { input: 2, output: 10, cacheRead: 0.2, cacheWrite: 2.5 },
  "openai/gpt-6.1-sol": { input: 2, output: 10, cacheRead: 0.1, cacheWrite: 2.5 },
  "openai/gpt-6-astra": { input: 10, output: 50, cacheRead: 1, cacheWrite: 12.5 },
  "openai/gpt-5.3-codex": { input: 1.75, output: 14, cacheRead: 0.175 },
  "fake/fake": { input: 0, output: 0 },
};

const norm = (s: string) => s.toLowerCase().replace(/^[a-z]+\//, "").replace(/[._]/g, "-");

export function priceFor(model: string): Price | null {
  if (PRICES[model]) return PRICES[model]!;
  const m = norm(model);
  for (const [id, p] of Object.entries(PRICES)) if (norm(id) === m) return p;
  return null;
}

// inputTokens counts every prompt token, cached or not; cache reads and writes
// are priced at their own rates and the rest at the input rate.
export function costUsd(model: string, u: Usage): number {
  const p = priceFor(model);
  if (!p) return 0;
  const uncached = Math.max(0, u.inputTokens - u.cacheReadTokens - u.cacheWriteTokens);
  const usd =
    (uncached * p.input +
      u.cacheReadTokens * (p.cacheRead ?? p.input) +
      u.cacheWriteTokens * (p.cacheWrite ?? p.input) +
      u.outputTokens * p.output) /
    1e6;
  return Math.round(usd * 1e6) / 1e6;
}
