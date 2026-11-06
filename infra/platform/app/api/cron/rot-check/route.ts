import { start } from "workflow/api";
import { bearerOk } from "../../../../src/http.ts";
import { rotCheckWorkflow } from "../../../../workflows/rotcheck.ts";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  if (!bearerOk(req, process.env.CRON_SECRET)) return Response.json({ error: "unauthorized" }, { status: 401 });
  const name = `platform-rotcheck-${new Date().toISOString().slice(0, 10)}`;
  const run = await start(rotCheckWorkflow, [{ name }]);
  return Response.json({ name, runId: run.runId });
}
