import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { z } from "zod";
import { bearerOk, statusReport } from "./http.ts";
import { runtime } from "./runtime.ts";
import { wake } from "./wake.ts";

const MAX_WAIT_SECONDS = 50;

const result = (value: unknown) => ({ content: [{ type: "text" as const, text: JSON.stringify(value, null, 2) }] });

async function answerOf(eventId: number, waitSeconds: number) {
  const rt = await runtime();
  const deadline = Date.now() + Math.min(waitSeconds, MAX_WAIT_SECONDS) * 1000;
  for (;;) {
    const o = await rt.store.eventOutcome(eventId);
    if (!o) return { eventId, status: "unknown" as const };
    if (o.status === "finished" || o.status === "failed" || Date.now() >= deadline) return { eventId, ...o };
    await new Promise((r) => setTimeout(r, 1000));
  }
}

// The platform's tools for anyone outside it (Daniel's editor, another agent,
// a script): ask a lead, hand it work, read its state and its transcripts.
export function platformMcpServer() {
  const server = new McpServer({ name: "verity-platform", version: "0.1.0" });

  server.registerTool(
    "ask_lead",
    {
      description:
        "Ask a lead a question. It is answered in the lead's next turn; wait_seconds (at most 50) waits for that turn, and get_answer reads the answer later.",
      inputSchema: { lane: z.string(), question: z.string(), from: z.string().optional(), wait_seconds: z.number().min(0).optional() },
    },
    async ({ lane, question, from, wait_seconds }) => {
      const caller = from ?? "mcp";
      const r = await wake(lane, { lane, source: "mcp", kind: "ask", payload: { text: question, from: caller }, replyTo: { caller } });
      return result(await answerOf(r.eventId!, wait_seconds ?? 0));
    },
  );

  server.registerTool(
    "assign",
    {
      description:
        "Hand a lead a task. The lead replies in the given Slack thread, or in its own channel when none is given.",
      inputSchema: {
        lane: z.string(),
        task: z.string(),
        from: z.string().optional(),
        slack_channel: z.string().optional(),
        thread_ts: z.string().optional(),
      },
    },
    async ({ lane, task, from, slack_channel, thread_ts }) => {
      const r = await wake(lane, {
        lane,
        source: "mcp",
        kind: "assign",
        payload: { text: task, from: from ?? "mcp" },
        replyTo: slack_channel ? { channel: slack_channel, threadTs: thread_ts } : null,
      });
      return result({ eventId: r.eventId, started: r.started });
    },
  );

  server.registerTool(
    "get_answer",
    { description: "What became of an event: pending, or the turn that took it and its reply.", inputSchema: { event_id: z.number().int() } },
    async ({ event_id }) => result(await answerOf(event_id, 0)),
  );

  server.registerTool(
    "status",
    {
      description: "Each lead's state: pending events, the running turn, its last turn and reply, running workers, and month-to-date spend against its budget.",
      inputSchema: { lane: z.string().optional() },
    },
    async ({ lane }) => result(await statusReport(lane)),
  );

  server.registerTool(
    "search_transcripts",
    {
      description: "Full-text search over every turn's transcript (prompts, tool calls, results, replies), by lead and time.",
      inputSchema: {
        query: z.string().optional(),
        lane: z.string().optional(),
        since: z.string().optional().describe("ISO time"),
        until: z.string().optional().describe("ISO time"),
        limit: z.number().int().min(1).max(200).optional(),
      },
    },
    async ({ query, lane, since, until, limit }) => {
      const rt = await runtime();
      return result(await rt.store.searchTranscripts({ text: query, lane, since, until, limit }));
    },
  );

  return server;
}

// Stateless streamable HTTP: one server and transport per request, JSON replies.
export async function mcpHandler(req: Request, env: Record<string, string | undefined> = process.env): Promise<Response> {
  if (!bearerOk(req, env.PLATFORM_API_TOKEN)) return Response.json({ error: "unauthorized" }, { status: 401 });
  const server = platformMcpServer();
  const transport = new WebStandardStreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
  await server.connect(transport);
  return transport.handleRequest(req);
}
