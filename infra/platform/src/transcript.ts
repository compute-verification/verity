import type { TranscriptPart, Usage } from "./store.ts";

type Chunk = { type: string; [k: string]: unknown };

const MAX_TEXT = 100_000;
const str = (v: unknown) => (typeof v === "string" ? v : JSON.stringify(v ?? null)).slice(0, MAX_TEXT);

// A turn's UI message stream, folded into the parts a reader wants: one part
// per text or reasoning block, one per tool call and per tool result, and every
// error. Nothing is summarised; deltas are only joined.
export function foldChunks(chunks: Chunk[]): TranscriptPart[] {
  const parts: TranscriptPart[] = [];
  const open = new Map<string, TranscriptPart>();
  for (const c of chunks) {
    switch (c.type) {
      case "text-start":
      case "reasoning-start": {
        const p: TranscriptPart = { type: c.type === "text-start" ? "text" : "reasoning", content: null, text: "" };
        open.set(`${p.type}:${c.id}`, p);
        parts.push(p);
        break;
      }
      case "text-delta":
      case "reasoning-delta": {
        const kind = c.type === "text-delta" ? "text" : "reasoning";
        let p = open.get(`${kind}:${c.id}`);
        if (!p) {
          p = { type: kind, content: null, text: "" };
          open.set(`${kind}:${c.id}`, p);
          parts.push(p);
        }
        p.text += String(c.delta ?? "");
        break;
      }
      case "tool-input-available":
        parts.push({
          type: "tool-call",
          content: { toolCallId: c.toolCallId, toolName: c.toolName, input: c.input, providerExecuted: c.providerExecuted ?? false },
          text: `${c.toolName} ${str(c.input)}`,
        });
        break;
      case "tool-output-available":
        parts.push({ type: "tool-result", content: { toolCallId: c.toolCallId, output: c.output }, text: str(c.output) });
        break;
      case "tool-output-error":
      case "tool-input-error":
        parts.push({ type: "tool-error", content: { toolCallId: c.toolCallId, errorText: c.errorText }, text: str(c.errorText) });
        break;
      case "tool-approval-request":
        parts.push({ type: "approval-request", content: c, text: "" });
        break;
      case "error":
        parts.push({ type: "error", content: { errorText: c.errorText }, text: str(c.errorText) });
        break;
      case "file":
        parts.push({ type: "file", content: { url: c.url, mediaType: c.mediaType }, text: "" });
        break;
      case "finish":
        parts.push({ type: "finish", content: { finishReason: c.finishReason ?? null }, text: "" });
        break;
      default:
        if (c.type.startsWith("data-")) parts.push({ type: c.type, content: c.data ?? null, text: "" });
    }
  }
  for (const p of parts) if (p.type === "text" || p.type === "reasoning") p.content = { length: p.text.length };
  return parts;
}

// The reply is the text after the last tool activity: what the agent said once
// it was done.
export function finalText(parts: TranscriptPart[]): string {
  let last = -1;
  parts.forEach((p, i) => {
    if (p.type.startsWith("tool-")) last = i;
  });
  return parts
    .slice(last + 1)
    .filter((p) => p.type === "text")
    .map((p) => p.text)
    .join("\n")
    .trim();
}

export const ZERO_USAGE: Usage = { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, reasoningTokens: 0 };

export function addUsage(a: Usage, b: Usage): Usage {
  return {
    inputTokens: a.inputTokens + b.inputTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    cacheReadTokens: a.cacheReadTokens + b.cacheReadTokens,
    cacheWriteTokens: a.cacheWriteTokens + b.cacheWriteTokens,
    reasoningTokens: a.reasoningTokens + b.reasoningTokens,
  };
}

// AI SDK LanguageModelUsage -> the platform's flat usage.
export function fromSdkUsage(u: unknown): Usage {
  const x = (u ?? {}) as {
    inputTokens?: number;
    outputTokens?: number;
    inputTokenDetails?: { cacheReadTokens?: number; cacheWriteTokens?: number };
    outputTokenDetails?: { reasoningTokens?: number };
  };
  return {
    inputTokens: x.inputTokens ?? 0,
    outputTokens: x.outputTokens ?? 0,
    cacheReadTokens: x.inputTokenDetails?.cacheReadTokens ?? 0,
    cacheWriteTokens: x.inputTokenDetails?.cacheWriteTokens ?? 0,
    reasoningTokens: x.outputTokenDetails?.reasoningTokens ?? 0,
  };
}
