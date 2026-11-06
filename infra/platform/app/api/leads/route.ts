import { changeLead, status } from "../../../src/http.ts";

export const dynamic = "force-dynamic";

export const GET = (req: Request) => status(req);
export const POST = (req: Request) => changeLead(req);
