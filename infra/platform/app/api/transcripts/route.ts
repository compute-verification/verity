import { transcripts } from "../../../src/http.ts";

export const dynamic = "force-dynamic";

export const GET = (req: Request) => transcripts(req);
