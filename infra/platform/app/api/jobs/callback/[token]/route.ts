import { jobCallback } from "../../../../../src/http.ts";

export const dynamic = "force-dynamic";

export async function POST(req: Request, ctx: { params: Promise<{ token: string }> }) {
  return jobCallback(req, (await ctx.params).token);
}
