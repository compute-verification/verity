import { slackEvents } from "../../../../src/http.ts";

export const dynamic = "force-dynamic";

export const POST = (req: Request) => slackEvents(req);
