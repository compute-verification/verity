import type { HarnessAgent } from "@ai-sdk/harness/agent";
import type { HarnessWorkflowAgent } from "@ai-sdk/workflow-harness";
import type { Usage } from "../store.ts";
import { ZERO_USAGE, addUsage, fromSdkUsage } from "../transcript.ts";

// The UI stream reports every error as "An error occurred." unless given an
// onError; the transcript and the failure reply want the harness's own words.
function unmasked<R extends object>(r: R): R {
  const errorText = (e: unknown) => String((e as Error)?.message ?? e).slice(0, 500);
  return new Proxy(r, {
    get(target, prop) {
      const v = Reflect.get(target, prop, target);
      if (prop === "toUIMessageStream" && typeof v === "function")
        return (o?: Record<string, unknown>) => v.call(target, { ...o, onError: errorText });
      return typeof v === "function" ? v.bind(target) : v;
    },
  });
}

// The agent as runHarnessAgentTimeSlice drives it, keeping every stream it
// opens so the slice's usage can be summed afterwards.
export function tapped(agent: Pick<HarnessAgent, "hasOutput" | "createSession" | "stream" | "continueStream">): {
  agent: HarnessWorkflowAgent;
  usage(): Promise<Usage>;
} {
  const results: { totalUsage: PromiseLike<unknown> }[] = [];
  const tap = <R extends { totalUsage: PromiseLike<unknown> }>(r: R): R => {
    results.push(r);
    return unmasked(r);
  };
  return {
    agent: {
      hasOutput: agent.hasOutput,
      createSession: (o) => agent.createSession(o as never),
      stream: async (o) => tap(await agent.stream(o as never)) as never,
      continueStream: async (o) => tap(await agent.continueStream(o as never)) as never,
    },
    async usage() {
      let u = ZERO_USAGE;
      for (const r of results) u = addUsage(u, fromSdkUsage(await Promise.resolve(r.totalUsage).catch(() => undefined)));
      return u;
    },
  };
}
