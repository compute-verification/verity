import { createHash } from "node:crypto";
import { HarnessAgent } from "@ai-sdk/harness/agent";
import { type HarnessWorkflowState, createHarnessWorkflowState, runHarnessAgentTimeSlice } from "@ai-sdk/workflow-harness";
import { getVercelOidcToken } from "@vercel/oidc";
import { Sandbox } from "@vercel/sandbox";
import { harnessFor } from "../../src/harness/providers.ts";
import { tapped } from "../../src/harness/tap.ts";
import { costUsd } from "../../src/prices.ts";
import { DEFAULT_ALLOW, openVercelSandbox } from "../../src/sandbox/vercel.ts";
import type { HarnessName } from "../../src/store.ts";
import { ZERO_USAGE, addUsage, finalText, foldChunks } from "../../src/transcript.ts";

const DOC = "WORKSTREAM.md";
const GITHUB_PLACEHOLDER = "platform-github-placeholder";
const BUDGET_MS = 270_000;

// One live lead turn as the platform runs it, without the database: a
// persistent sandbox with the platform's network policy and a brokered GitHub
// rule, a real harness turn in time slices (each slice reopens the sandbox and
// rebuilds the agent, as each workflow step does), the doc read back, a check
// that the model credential never entered the sandbox, then the sandbox deleted.
export async function GET(req: Request): Promise<Response> {
  const q = new URL(req.url).searchParams;
  // The deployment holds only the key's digest, so the key never passes through
  // whatever sets the project's environment.
  const want = process.env.SMOKE_KEY_SHA256;
  if (!want || createHash("sha256").update(q.get("key") ?? "").digest("hex") !== want) return new Response("not found", { status: 404 });
  const harness = (q.get("harness") ?? "claude-code") as HarnessName;
  const model = q.get("model") ?? (harness === "codex" ? "openai/gpt-5.3-codex" : "anthropic/claude-sonnet-5.5");
  const sliceSeconds = Number(q.get("slice") ?? 200);
  const t0 = Date.now();
  const log: string[] = [];
  const at = (m: string) => log.push(`${((Date.now() - t0) / 1000).toFixed(1)}s ${m}`);
  const name = `platform-smoke-${t0}`;
  const open = () =>
    openVercelSandbox(name, {
      vcpus: 2,
      timeoutMs: 15 * 60_000,
      allowHosts: DEFAULT_ALLOW,
      inject: [{ host: "api.github.com", sent: `Bearer ${GITHUB_PLACEHOLDER}`, real: "Bearer smoke-not-a-token" }],
    });

  const out: Record<string, unknown> = { harness, model, sliceSeconds, sandbox: name };
  let modelToken = "";
  try {
    let session = await open();
    at("sandbox open");
    await session.writeTextFile({ path: DOC, content: "# smoke\n\n- created by the platform smoke\n" });
    const prompt = `Append the line "- smoke ok" to ${DOC} in the current directory, then reply with exactly PONG followed by the number of lines now in the file.`;
    let state: HarnessWorkflowState = createHarnessWorkflowState({ sessionId: name, prompt });
    let usage = ZERO_USAGE;
    const chunks: { type: string }[] = [];
    let slices = 0;
    while (Date.now() - t0 < BUDGET_MS) {
      if (slices > 0) session = await open();
      modelToken = await getVercelOidcToken();
      const t = tapped(
        new HarnessAgent({
          harness: await harnessFor(harness, { auth: { AI_GATEWAY_API_KEY: modelToken } }),
          model,
          instructions: "You are a smoke test. Do exactly what the prompt says.",
          permissionMode: "allow-all",
          sandboxConfig: { workDir: "." },
        }),
      );
      const writable = new WritableStream<{ type: string }>({ write: (c) => void chunks.push(c) });
      state = await runHarnessAgentTimeSlice({
        agent: t.agent,
        state,
        sandboxSession: session as never,
        writable,
        timeSliceSeconds: Math.min(sliceSeconds, Math.floor((BUDGET_MS - (Date.now() - t0)) / 1000)),
      });
      usage = addUsage(usage, await t.usage());
      slices++;
      at(`slice ${slices}: ${state.status}${state.error ? ` (${state.error})` : ""}`);
      if (state.status !== "ready_for_next_step") break;
    }
    const parts = foldChunks(chunks);
    const native = await Sandbox.get({ name });
    const policy = native.currentSession().networkPolicy;
    const rules =
      policy && typeof policy === "object" && policy.allow && !Array.isArray(policy.allow)
        ? Object.fromEntries(Object.entries(policy.allow).filter(([, r]) => r.length > 0).map(([h, r]) => [h, r.length]))
        : policy;
    // The model credential in the sandbox: compared by digest against every
    // process's environment and every file the harness wrote, never sent in.
    const digests = await session.run({
      command:
        "{ for f in /proc/[0-9]*/environ; do tr '\\0' '\\n' < \"$f\"; done 2>/dev/null | sed 's/^[^=]*=//'; " +
        "find ~ /tmp . -type f -size -2M 2>/dev/null | grep -v '/node_modules/\\|/.pnpm-store/' | xargs -r cat 2>/dev/null | tr -s ' \\t\"=:,' '\\n'; } " +
        "| sort -u | while IFS= read -r v; do printf %s \"$v\" | sha256sum; done | cut -c1-64",
    });
    const tokenDigest = createHash("sha256").update(modelToken).digest("hex");
    Object.assign(out, {
      status: state.status,
      error: state.error ?? null,
      slices,
      text: finalText(parts),
      doc: await session.readTextFile({ path: DOC }),
      usage,
      costUsd: costUsd(model, usage),
      partTypes: [...new Set(parts.map((p) => p.type))],
      chunkTypes: [...new Set(chunks.map((c) => c.type))],
      rulesByHost: rules,
      credentialInSandbox: digests.stdout.split("\n").includes(tokenDigest),
      valuesScanned: digests.stdout.split("\n").length,
    });
  } catch (e) {
    out.status = "error";
    out.error = String((e as Error)?.stack ?? e).slice(0, 2000);
  } finally {
    const s = await Sandbox.get({ name }).catch(() => null);
    await s?.stop().catch(() => {});
    await s?.delete().catch((e: unknown) => at(`delete failed: ${String(e)}`));
    at("sandbox deleted");
  }
  out.log = log;
  out.seconds = (Date.now() - t0) / 1000;
  return Response.json(out);
}
