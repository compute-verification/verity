import type { HarnessV1, HarnessV1PromptControl, HarnessV1Session, HarnessV1StreamPart } from "@ai-sdk/harness";
import { DOC_PATH } from "../prompt.ts";

// A scripted HarnessV1 for tests: the real HarnessAgent drives it exactly as it
// drives Claude Code or Codex. It reads and edits the workstream doc in the
// sandbox, and turns directives in its prompt into host tool calls:
//   [[dispatch: <task>]]   -> dispatch_worker
//   [[job: <command>]]     -> submit_job
//   [[ask <lane>: <text>]] -> ask_lead
//   [[fail: <reason>]]     -> the turn throws, as a crashed harness would
const DIRECTIVE = /\[\[(dispatch|job|ask ([\w-]+)):\s*([^\]]+)\]\]/g;
const FAIL = /\[\[fail:\s*([^\]]+)\]\]/;

function promptText(prompt: unknown): string {
  if (typeof prompt === "string") return prompt;
  const content = (prompt as { content?: unknown }).content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((p) => (p as { text?: string }).text ?? "").join("\n");
  return "";
}

const usage = (input: number, output: number) => ({
  inputTokens: { total: input, noCache: input, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: output, text: output, reasoning: 0 },
});

export function fakeHarness(): HarnessV1 {
  let counter = 0;
  return {
    specificationVersion: "harness-v1",
    harnessId: "fake",
    builtinTools: {},
    async doStart({ sessionId, sandboxSession, sessionWorkDir, resumeFrom, continueFrom }) {
      const docPath = `${sessionWorkDir}/${DOC_PATH}`;
      const state = (type: "resume-session" | "continue-turn") => ({
        type,
        harnessId: "fake",
        specificationVersion: "harness-v1" as const,
        data: { sessionId },
      });
      const session: HarnessV1Session = {
        sessionId,
        isResume: Boolean(resumeFrom || continueFrom),
        async doPromptTurn({ prompt, tools, emit }) {
          const text = promptText(prompt);
          const pending = new Map<string, (v: { output: unknown; isError?: boolean }) => void>();
          const control: HarnessV1PromptControl = {
            async submitToolResult({ toolCallId, output, isError }) {
              pending.get(toolCallId)?.({ output, isError });
            },
            done: (async () => {
              emit({ type: "stream-start", modelId: "fake" });
              const fail = FAIL.exec(text);
              if (fail) throw new Error(fail[1]!.trim());
              const calls: { id: string; name: string; input: Record<string, string> }[] = [];
              for (const m of text.matchAll(DIRECTIVE)) {
                const id = `call_${++counter}`;
                const arg = m[3]!.trim();
                if (m[1] === "dispatch") calls.push({ id, name: "dispatch_worker", input: { task: arg } });
                else if (m[1] === "job") calls.push({ id, name: "submit_job", input: { command: arg } });
                else calls.push({ id, name: "ask_lead", input: { lane: m[2]!, question: arg } });
              }
              const available = new Set(tools.map((t) => t.name));
              const results: string[] = [];
              if (calls.length > 0) {
                for (const c of calls) {
                  if (!available.has(c.name)) continue;
                  const result = new Promise<{ output: unknown; isError?: boolean }>((r) => pending.set(c.id, r));
                  emit({ type: "tool-call", toolCallId: c.id, toolName: c.name, input: JSON.stringify(c.input) } as HarnessV1StreamPart);
                  const r = await result;
                  results.push(`${c.name} -> ${JSON.stringify(r.output)}`);
                }
                emit({ type: "finish-step", finishReason: { unified: "tool-calls", raw: "tool_use" }, usage: usage(10, 5) });
              }
              const before = (await sandboxSession.readTextFile({ path: docPath })) ?? "";
              const events = [...text.matchAll(/^- \[(\w+)\/([\w-]+)\]/gm)].map((m) => `${m[1]}/${m[2]}`);
              const line = `- ${sessionId}: handled ${events.join(", ") || "nothing"}${results.length ? `; ${results.join("; ")}` : ""}`;
              await sandboxSession.writeTextFile({ path: docPath, content: `${before.trimEnd()}\n${line}\n` });
              emit({ type: "file-change", event: "modify", path: DOC_PATH });
              const reply = `Handled ${events.length} event(s): ${events.join(", ")}.${results.length ? ` ${results.join("; ")}` : ""}`;
              emit({ type: "text-start", id: "t1" });
              emit({ type: "text-delta", id: "t1", delta: reply });
              emit({ type: "text-end", id: "t1" });
              const u = usage(Math.ceil(text.length / 4), Math.ceil(reply.length / 4));
              emit({ type: "finish-step", finishReason: { unified: "stop", raw: "end_turn" }, usage: u });
              emit({ type: "finish", finishReason: { unified: "stop", raw: "end_turn" }, totalUsage: u });
            })(),
          };
          return control;
        },
        async doContinueTurn({ emit }) {
          return {
            async submitToolResult() {},
            done: (async () => {
              emit({ type: "finish", finishReason: { unified: "stop", raw: "end_turn" }, totalUsage: usage(0, 0) });
            })(),
          };
        },
        async doCompact() {},
        async doSuspendTurn() {
          return state("continue-turn") as never;
        },
        async doDetach() {
          return state("resume-session") as never;
        },
        async doStop() {
          return state("resume-session") as never;
        },
        async doDestroy() {},
      };
      return session;
    },
  };
}
