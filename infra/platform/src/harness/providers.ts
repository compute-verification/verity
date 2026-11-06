import type { HarnessV1 } from "@ai-sdk/harness";
import type { ModelAuth } from "../credentials.ts";
import type { HarnessName } from "../store.ts";
import { fakeHarness } from "./fake.ts";

export async function harnessFor(name: HarnessName, o: { auth: ModelAuth }): Promise<HarnessV1> {
  if (name === "fake") return fakeHarness();
  if (name === "codex") {
    const { createCodex } = await import("@ai-sdk/harness-codex");
    return createCodex({ auth: o.auth }) as unknown as HarnessV1;
  }
  const { createClaudeCode } = await import("@ai-sdk/harness-claude-code");
  return createClaudeCode({ auth: o.auth, effort: "high" }) as unknown as HarnessV1;
}
