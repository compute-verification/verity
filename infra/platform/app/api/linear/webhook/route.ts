import { linearEvents } from "../../../../src/http.ts";

export const dynamic = "force-dynamic";

export const POST = (req: Request) => linearEvents(req);
