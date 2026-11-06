import { cronTick } from "../../../../src/http.ts";

export const dynamic = "force-dynamic";

// Vercel Cron calls with GET and `Authorization: Bearer $CRON_SECRET`.
export const GET = (req: Request) => cronTick(req);
